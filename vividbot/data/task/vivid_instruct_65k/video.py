import json
import logging
import os
import sys
import time

from tqdm import tqdm

from vividbot.data.task.vivid_instruct_65k.utils.chains import GENERATE_QA_PAIRS_CHAIN
from vividbot.data.task.vivid_instruct_65k.utils.common import (
  find_first_list_from_response,
)
from vividbot.data.task.vivid_instruct_65k.utils.notifications import (
  send_process_shard_success_message,
)
from vividbot.data.task.vivid_instruct_65k.utils.prompts import DESCRIBE_VIDEO_PROMPT

sys.path.append(os.getcwd())

from pathlib import Path

import google.generativeai as genai
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from yt_dlp.utils import DownloadError

from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.huggingface import HuggingFaceProcessor

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
  for start, end, video_id_with_chunk_id, shard_id in tqdm(
    zip(batch["start"], batch["end"], batch["id"], batch["shard_id"])
  ):
    video_id, chunk_id = video_id_with_chunk_id.split(".")

    try:
      if not os.path.exists(
        f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
      ) and not hf_processor.check_file_exists(
        repo_id="Vividbot/vividbot_video",
        path_in_repo=f"videos/shard_{shard_id}.zip",
        repo_type="dataset",
      ):
        logger.info(f"Downloading video {video_id_with_chunk_id}...")
        yt_downloader.process(
          video_id=video_id,
          video_id_with_chunk_id=video_id_with_chunk_id,
          start=start,
          end=end,
          path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}",
        )

    except DownloadError as e:
      logger.error(f"Error downloading video {video_id_with_chunk_id}: {e}")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {"id": video_id_with_chunk_id, "reason": str(e)}
        f.write(json.dumps(data) + "\n")

    try:
      if not hf_processor.check_file_exists(
        repo_id="Vividbot/vividbot_video",
        path_in_repo=f"metadata/shard_{shard_id}.jsonl",
        repo_type="dataset",
      ):
        logger.info(f"Generating metadata for video {video_id_with_chunk_id}...")
        if not os.path.exists(
          f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
        ):
          logger.info(f"Video {video_id_with_chunk_id} not found. Skipping...")
          continue

        video_file = None
        google_file_name = f"files/{shard_id}-{video_id}-{chunk_id}".lower().replace(
          "_", "-"
        )
        try:
          video_file = genai.get_file(name=google_file_name)
        except Exception as e:
          logger.error(f"Error getting video file {google_file_name}: {e}")

        if video_file is None or not video_file.state.name == "ACTIVE":
          if video_file and not video_file.state.name == "ACTIVE":
            genai.delete_file(name=video_file.name)
          video_file = genai.upload_file(
            path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4",
            name=google_file_name,
            display_name=video_id_with_chunk_id,
          )

        while video_file and video_file.state.name == "PROCESSING":
          time.sleep(5)
          video_file = genai.get_file(video_file.name)

        if video_file and video_file.state.name == "FAILED":
          logger.error(
            f"Error uploading video {video_id_with_chunk_id}: {video_file.error}"
          )
          with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
            data = {
              "id": video_id_with_chunk_id,
              "reason": f"Error uploading video: {video_file.error}",
            }
            f.write(json.dumps(data) + "\n")
        elif video_file and video_file.state.name == "ACTIVE":
          describer = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            generation_config={
              "temperature": 0,
              "max_output_tokens": 512,
            },
          )
          describer_response = describer.generate_content(
            [video_file, DESCRIBE_VIDEO_PROMPT],
            request_options={
              "timeout": 60,
            },
          )
          try:
            response: str = GENERATE_QA_PAIRS_CHAIN.invoke(
              {"message": describer_response.text.strip()}
            )
            if not response.startswith("["):
              response = find_first_list_from_response(response)

            qa_pairs = json.loads(response)
            conversations = []

            for qa in qa_pairs:
              rand_num = np.random.random()
              human_value = qa["question"]
              gpt_value = qa["answer"]
              if rand_num < 0.5:
                human_value = human_value + "\n<video>"
              else:
                human_value = "<video>\n" + human_value
              conversations.append({"from": "human", "value": human_value})
              conversations.append({"from": "gpt", "value": gpt_value})

            data = {
              "id": video_id_with_chunk_id,
              "video": f"shard_{shard_id}/{video_id_with_chunk_id}.mp4",
              "generator": "chain",
              "description": describer_response.text.strip(),
              "conversations": conversations,
            }

            with open(
              f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
              "a",
            ) as f:
              f.write(
                json.dumps(
                  data,
                  ensure_ascii=False,
                )
                + "\n"
              )
          except Exception as e:
            logger.error(
              f"Couldn't generate QA pairs for video {video_id_with_chunk_id}: {str(e)}."
            )
            with open(
              f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a"
            ) as f:
              data = {"id": video_id_with_chunk_id, "reason": str(e)}
              f.write(json.dumps(data) + "\n")
        else:
          logger.error(f"Error uploading video {video_id_with_chunk_id}: {video_file}")
          with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
            data = {
              "id": video_id_with_chunk_id,
              "reason": f"Error uploading video: {video_file}",
            }
            f.write(json.dumps(data) + "\n")

    except Exception as e:
      logger.error(f"Error generating metadata for video {video_id_with_chunk_id}: {e}")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {"id": video_id_with_chunk_id, "reason": str(e)}
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


def process(shard: str):
  """
  shard: str = "shard_0.json"
  """
  start_time = time.time()

  logger.info(f"Processing videos for shard {shard}...")

  if hf_processor.check_file_exists(
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"videos/{shard}.zip",
    repo_type="dataset",
  ):
    logger.info(f"Shard {shard} already uploaded. Skipping...")

  else:
    os.makedirs(f"{BASE_DATA_PATH}/output/videos/{shard}", exist_ok=True)

  dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/vivid_instruct_65k/{shard}"
  )["train"]

  dataset.map(
    _process,
    batched=True,
    batch_size=128,
    num_proc=os.cpu_count(),
  )

  hf_processor.zip_and_upload_dir(
    dir_path=f"{BASE_DATA_PATH}/output/videos/{shard}",
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"videos/{shard}.zip",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_file(
    file_path=f"{BASE_DATA_PATH}/output/metadata/{shard}.jsonl",
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"metadata/{shard}.jsonl",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_file(
    file_path=f"{BASE_DATA_PATH}/output/errors/{shard}.jsonl",
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"errors/{shard}.jsonl",
    repo_type="dataset",
    overwrite=True,
  )

  # remove video file from google cloud
  logger.info(f"Cleaning up shard {shard}...")
  try:
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

  send_process_shard_success_message(shard, duration)


def prepare():
  os.makedirs(f"{BASE_DATA_PATH}/output", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/videos", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/errors", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/metadata", exist_ok=True)


def main():
  prepare()

  for shard in tqdm(
    sorted(
      os.listdir(f"{BASE_DATA_PATH}/vivid_instruct_65k"),
      key=lambda x: int(x.split(".")[0].split("_")[1]),
    )
  ):
    process(shard)


if __name__ == "__main__":
  main()
