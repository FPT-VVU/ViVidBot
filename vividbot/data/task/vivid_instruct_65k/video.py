import json
import logging
import os
import time
from pathlib import Path
from typing import List

import google.generativeai as genai
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from langfuse.callback import CallbackHandler
from tqdm import tqdm

from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.huggingface import HuggingFaceProcessor
from vividbot.data.task.vivid_instruct_65k.utils.chains import GENERATE_QA_PAIRS_CHAIN
from vividbot.data.task.vivid_instruct_65k.utils.notifications import (
  send_completion_message,
  send_process_shard_success_message,
)
from vividbot.data.task.vivid_instruct_65k.utils.prompts import DESCRIBE_VIDEO_PROMPT

load_dotenv()
BASE_DATA_PATH = f"{Path.home()}/data"


logger = logging.getLogger(__name__)
logging.basicConfig(
  filename=f"{BASE_DATA_PATH}/run.log",
  filemode="w",
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


hf_processor = HuggingFaceProcessor()
yt_downloader = YoutubeDownloader()


def _process(batch: dict):
  processed_dataset = None

  if os.path.exists(
    f"{BASE_DATA_PATH}/output/metadata/shard_{batch['shard_id'][0]}.jsonl"
  ):
    processed_dataset = load_dataset(
      "json",
      data_files=f"{BASE_DATA_PATH}/output/metadata/shard_{batch['shard_id'][0]}.jsonl",
    )["train"]

  for start, end, video_id_with_chunk_id, shard_id in tqdm(
    zip(batch["start"], batch["end"], batch["id"], batch["shard_id"])
  ):
    video_id, chunk_id = video_id_with_chunk_id.split(".")

    if (
      processed_dataset is not None
      and processed_dataset.filter(lambda x: x["id"] == video_id_with_chunk_id).num_rows
      > 0
    ):
      logger.info(f"Video {video_id_with_chunk_id} already processed. Skipping...")
      continue

    if not os.path.exists(
      f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
    ):
      try:
        logger.info(f"Downloading video {video_id_with_chunk_id}...")
        yt_downloader.process(
          video_id=video_id,
          video_id_with_chunk_id=video_id_with_chunk_id,
          start=start,
          end=end,
          path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}",
        )

      except Exception as e:
        logger.error(f"Error downloading video {video_id_with_chunk_id}: {e}")
        with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
          data = {
            "id": video_id_with_chunk_id,
            "reason": str(e),
            "timestamp": round(time.time()),
          }
          f.write(json.dumps(data) + "\n")

        continue

    if not os.path.exists(
      f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
    ):
      logger.error(f"Error downloading video {video_id_with_chunk_id}.")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {
          "id": video_id_with_chunk_id,
          "reason": "Video not found after download.",
          "timestamp": round(time.time()),
        }
        f.write(json.dumps(data) + "\n")

      continue

    try:
      logger.info(f"Generating metadata for video {video_id_with_chunk_id}...")

      video_file = None
      google_file_name = f"files/{shard_id}-{video_id}-{chunk_id}".lower().replace(
        "_", "-"
      )
      try:
        video_file = genai.get_file(name=google_file_name)
      except Exception as e:
        logger.warning(f"Couldn't get video file {google_file_name}: {e}")

      if video_file is None or not video_file.state.name == "ACTIVE":
        if video_file and not video_file.state.name == "ACTIVE":
          genai.delete_file(name=video_file.name)

        video_file = genai.upload_file(
          path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4",
          name=google_file_name,
          display_name=video_id_with_chunk_id,
        )

      while video_file and video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

      if video_file and video_file.state.name == "FAILED":
        logger.error(
          f"Error uploading video {video_id_with_chunk_id}: {video_file.error}"
        )
        with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
          data = {
            "id": video_id_with_chunk_id,
            "reason": f"Error uploading video: {video_file.error}",
            "timestamp": round(time.time()),
          }
          f.write(json.dumps(data) + "\n")

        if os.path.exists(
          f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
        ):
          os.remove(
            f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
          )

      elif video_file and video_file.state.name == "ACTIVE":
        describer = genai.GenerativeModel(
          "models/gemini-1.5-flash",
          generation_config={
            "temperature": 0.25,
            "max_output_tokens": 1536,
          },
          safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
          },
          system_instruction=DESCRIBE_VIDEO_PROMPT,
        )
        describer_response = describer.generate_content(
          [video_file, "Describe the video as instructed."],
          request_options={
            "timeout": 60,
          },
        )
        try:
          langfuse_handler = CallbackHandler(
            secret_key="sk-lf-bfa7365e-2871-4669-bda2-4818fcab68de",
            public_key="pk-lf-b6e63d27-8f4b-4911-acd2-5838400c1dfb",
            host="https://langfuse.formularizer.com",
            tags=["vividbot"],
            session_id=video_id_with_chunk_id,
          )

          qa_pairs: List = GENERATE_QA_PAIRS_CHAIN.invoke(
            {"message": describer_response.text.strip()},
            {"callbacks": [langfuse_handler]},
          )

          if not qa_pairs:
            logger.error(
              f"Couldn't generate QA pairs for video {video_id_with_chunk_id}: QA pairs could not be generated."
            )
            with open(
              f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a"
            ) as f:
              data = {
                "id": video_id_with_chunk_id,
                "reason": "QA pairs could not be generated.",
                "timestamp": round(time.time()),
              }
              f.write(json.dumps(data) + "\n")

            if os.path.exists(
              f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
            ):
              os.remove(
                f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
              )

            continue

          conversations = []

          for qa in qa_pairs:
            if not qa.get("question") or not qa.get("answer"):
              continue

            human_value = str(qa["question"]).strip()
            gpt_value = str(qa["answer"]).strip()
            if np.random.random() < 0.5:
              human_value = human_value + "\n<video>"
            else:
              human_value = "<video>\n" + human_value

            conversations.append({"from": "human", "value": human_value})
            conversations.append({"from": "gpt", "value": gpt_value})

          with open(
            f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
            "a",
          ) as f:
            data = {
              "id": video_id_with_chunk_id,
              "video": f"shard_{shard_id}/{video_id_with_chunk_id}.mp4",
              "timestamp": round(time.time()),
              "start": start,
              "end": end,
              "description": describer_response.text.strip(),
              "conversations": conversations,
            }

            f.write(
              json.dumps(
                data,
                ensure_ascii=False,
              )
              + "\n"
            )

            logger.info(
              f"Generated metadata for video {video_id_with_chunk_id}: {json.dumps(data, ensure_ascii=False)}"
            )

        except Exception as e:
          logger.error(
            f"Couldn't generate QA pairs for video {video_id_with_chunk_id}: {str(e)} - Response: {describer_response}"
          )
          with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
            data = {
              "id": video_id_with_chunk_id,
              "reason": f"{str(e)} - Response: {describer_response}",
              "timestamp": round(time.time()),
            }
            f.write(json.dumps(data) + "\n")

          if os.path.exists(
            f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
          ):
            os.remove(
              f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
            )

      else:
        logger.error(f"Error uploading video {video_id_with_chunk_id}: {video_file}")
        with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
          data = {
            "id": video_id_with_chunk_id,
            "reason": f"Error uploading video: {video_file}",
            "timestamp": round(time.time()),
          }
          f.write(json.dumps(data) + "\n")

        if os.path.exists(
          f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
        ):
          os.remove(
            f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
          )

    except Exception as e:
      logger.error(f"Error generating metadata for video {video_id_with_chunk_id}: {e}")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {
          "id": video_id_with_chunk_id,
          "reason": str(e),
          "timestamp": round(time.time()),
        }
        f.write(json.dumps(data) + "\n")


def _delete_video(batch: dict):
  for video_id_with_chunk_id, shard_id in tqdm(zip(batch["id"], batch["shard_id"])):
    video_id, chunk_id = video_id_with_chunk_id.split(".")

    google_file_name = f"files/{shard_id}-{video_id}-{chunk_id}".lower().replace(
      "_", "-"
    )
    try:
      genai.delete_file(name=google_file_name)
    except Exception as e:
      logger.error(f"Error deleting video file {google_file_name}: {e}")

    try:
      os.remove(
        f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
      )
    except Exception as e:
      logger.error(f"Error deleting video {video_id_with_chunk_id}: {e}")


def process(shard_file_name: str):
  """
  shard_file_name: str = "shard_0.jsonl"
  """
  start_time = time.time()

  shard = shard_file_name.split(".")[0]

  logger.info(f"Processing videos for shard {shard}...")

  if hf_processor.check_file_exists(
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"videos/{shard}.zip",
    repo_type="dataset",
  ) and not os.path.exists(f"{BASE_DATA_PATH}/output/videos/{shard}"):
    logger.info(f"Downloading videos for shard {shard} from Hugging Face...")
    hf_processor.download_and_unzip_file(
      repo_id="Vividbot/vividbot_video",
      filename=f"videos/{shard}.zip",
      local_dir=f"{BASE_DATA_PATH}/output",
      extract_dir=f"{BASE_DATA_PATH}/output/videos",
    )
  else:
    os.makedirs(f"{BASE_DATA_PATH}/output/videos/{shard}", exist_ok=True)

  if hf_processor.check_file_exists(
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"metadata/{shard}.jsonl",
    repo_type="dataset",
  ) and not os.path.exists(f"{BASE_DATA_PATH}/output/metadata/{shard}.jsonl"):
    logger.info(f"Downloading metadata for shard {shard} from Hugging Face...")
    hf_processor.download_file(
      repo_id="Vividbot/vividbot_video",
      filename=f"metadata/{shard}.jsonl",
      local_dir=f"{BASE_DATA_PATH}/output",
    )

  # if hf_processor.check_file_exists(
  #   repo_id="Vividbot/vividbot_video",
  #   path_in_repo=f"errors/{shard}.jsonl",
  #   repo_type="dataset",
  # ) and not os.path.exists(f"{BASE_DATA_PATH}/output/errors/{shard}.jsonl"):
  #   hf_processor.download_file(
  #     repo_id="Vividbot/vividbot_video",
  #     filename=f"errors/{shard}.jsonl",
  #     local_dir=f"{BASE_DATA_PATH}/output",
  #   )

  dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/vivid_instruct_65k/{shard_file_name}"
  )["train"]

  dataset.map(
    _process,
    batched=True,
    batch_size=128,
    num_proc=os.cpu_count(),
  )

  if os.path.exists(f"{BASE_DATA_PATH}/output/videos/{shard}"):
    # remove partial video files *.part
    logger.info(f"Removing partially downloaded video files for shard {shard}...")
    for file in os.listdir(f"{BASE_DATA_PATH}/output/videos/{shard}"):
      if file.endswith(".part"):
        os.remove(f"{BASE_DATA_PATH}/output/videos/{shard}/{file}")

    logger.info(f"Zipping and uploading videos for shard {shard}...")
    hf_processor.zip_and_upload_dir(
      dir_path=f"{BASE_DATA_PATH}/output/videos/{shard}",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"videos/{shard}.zip",
      repo_type="dataset",
      overwrite=True,
    )

  final_datas = []
  ids = set()
  if os.path.exists(f"{BASE_DATA_PATH}/output/metadata/{shard}.jsonl"):
    logger.info(f"Filtering out duplicate videos for shard {shard}...")
    # filter out duplicate ids by taking only the last occurrence
    with open(f"{BASE_DATA_PATH}/output/metadata/{shard}.jsonl", "r") as f:
      lines = f.readlines()
      for line in reversed(lines):
        data = json.loads(line)
        if data["id"] not in ids:
          ids.add(data["id"])
          final_datas.append(data)

    with open(f"{BASE_DATA_PATH}/output/metadata/{shard}.jsonl", "w") as f:
      for data in reversed(final_datas):
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info(f"Found {len(final_datas)} unique videos for shard {shard}.")

    logger.info(f"Uploading metadata for shard {shard}...")
    hf_processor.upload_file(
      file_path=f"{BASE_DATA_PATH}/output/metadata/{shard}.jsonl",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"metadata/{shard}.jsonl",
      repo_type="dataset",
      overwrite=True,
    )

  if os.path.exists(f"{BASE_DATA_PATH}/output/errors/{shard}.jsonl"):
    logger.info(f"Uploading errors for shard {shard}...")
    hf_processor.upload_file(
      file_path=f"{BASE_DATA_PATH}/output/errors/{shard}.jsonl",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"errors/{shard}.jsonl",
      repo_type="dataset",
      overwrite=True,
    )

  # remove video file from google cloud
  try:
    logger.info(f"Cleaning up shard {shard}...")
    dataset.map(
      _delete_video,
      batched=True,
      batch_size=16,
      num_proc=os.cpu_count(),
    )
  except Exception as e:
    logger.error(f"Error deleting videos for shard {shard}: {e}")

  end_time = time.time()
  duration = round(end_time - start_time, 2)

  send_process_shard_success_message(shard, duration, count=len(final_datas))


def prepare():
  os.makedirs(f"{BASE_DATA_PATH}/output", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/videos", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/errors", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/metadata", exist_ok=True)


def main():
  prepare()
  shard_files = os.listdir(f"{BASE_DATA_PATH}/vivid_instruct_65k")
  shard_files = sorted(
    shard_files,
    key=lambda x: int(x.split(".")[0].split("_")[1]),
  )

  last_successful_shard = 44
  # only process shards after the last successful shard
  shard_files = shard_files[last_successful_shard + 1 :]

  logger.info(f"Processing shards: {shard_files}")

  for shard in tqdm(
    shard_files,
    desc="Processing shards",
    unit="shard",
    unit_scale=True,
    unit_divisor=1024,
  ):
    process(shard)

  send_completion_message()


if __name__ == "__main__":
  main()
