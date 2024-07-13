import json
import os
import shutil
import sys
from time import time

from tqdm import tqdm

sys.path.append(os.getcwd())
import datetime
from datetime import timezone
from pathlib import Path

import google.generativeai as genai
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from yt_dlp.utils import DownloadError

from vividbot.data.discord.discord import DiscordNotifier
from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.executor import Executor
from vividbot.data.processor.upload_hf import Uploader

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_DATA_PATH = f"{Path.home()}/data"
DESCRIBE_VIDEO_PROMPT = "Describe only the visual content of the video without using its audio or transcript so that a person without vision can fully understand it. Remember to use Vietnamese language to describe the video."
GENERATE_QA_PROMPT = """Generate 5 different pairs of questions and answers based on the description of the video. The questions should be relevant to the video content and the answers should be correct. Also, diversify the types of questions and answers as much as possible.
Remember to use Vietnamese language to generate the questions and answers.
Examples of questions:
- What's the video about?
- What are key points in the video?
- What is the color of the object?
- What is the person doing?
- What is the person in the video holding and what are its characteristics?
- How does the person in the video look?
- What is the position of the object in the video?
And more questions that can be asked about the video content (what, where, when, why, how, etc.) with varying levels of complexity.
Return the questions and answers in the following format:
[{"question": "Q1","answer": "A1"},{"question": "Q2","answer": "A2"},...]
"""

genai.configure(api_key=GOOGLE_API_KEY)
notifier = DiscordNotifier(
  "https://discord.com/api/webhooks/1255505460040040508/n-QCTqNgp3RrsNc1hBRnXH4dfOejeH8iPTd8lqGevbSb_wAovD4xxv5ZVkVJBfVLF8vN"
)
uploader = Uploader()
downloader = YoutubeDownloader()


def send_process_shard_success_message(shard_count, duration):
  notifier.send(
    body={
      "embeds": [
        {
          "title": f"✅ ViVid Instruct 65k: Processed shard {shard_count}!",
          "description": f"Processed shard {shard_count} \
of {len(os.listdir(f'{BASE_DATA_PATH}/output/videos/shard_{shard_count}'))} clips \
in {duration}(s). \
Visit at https://huggingface.co/datasets/Vividbot/vividbot_video/tree/main/videos.",
          "color": 2278494,
          "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }
      ]
    }
  )


def _process(batch: dict):
  for start, end, video_id_with_chunk_id, shard_id in tqdm(
    zip(batch["start"], batch["end"], batch["id"], batch["shard_id"])
  ):
    video_id = video_id_with_chunk_id.split(".")[0]

    try:
      if not os.path.exists(
        f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
      ) and not uploader.check_file_exists(
        repo_id="Vividbot/vividbot_video",
        path_in_repo=f"videos/shard_{shard_id}.zip",
        repo_type="dataset",
      ):
        print(f"Downloading video {video_id_with_chunk_id}...")
        downloader.process(
          video_id=video_id,
          video_id_with_chunk_id=video_id_with_chunk_id,
          start=start,
          end=end,
          path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}",
        )

    except DownloadError as e:
      print(f"Error downloading video {video_id_with_chunk_id}: {e}")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {"id": video_id_with_chunk_id, "reason": str(e)}
        f.write(json.dumps(data) + "\n")

    try:
      if not uploader.check_file_exists(
        repo_id="Vividbot/vividbot_video",
        path_in_repo=f"metadata/shard_{shard_id}.jsonl",
        repo_type="dataset",
      ):
        print(f"Generating metadata for video {video_id_with_chunk_id}...")
        if not os.path.exists(
          f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
        ):
          print(f"Video {video_id_with_chunk_id} not found. Skipping...")
          continue

        video_file = genai.upload_file(
          path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
        )

        while video_file.state.name == "PROCESSING":
          time.sleep(10)
          video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
          print(f"Error generating metadata for video {video_id_with_chunk_id}: {video_file.error}")
          with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
            data = {"id": video_id_with_chunk_id, "reason": video_file.error}
            f.write(json.dumps(data) + "\n")
        elif video_file.state.name == "ACTIVE":
          describer = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            generation_config={
              "temperature": 0.1,
            },
          )
          describer_response = describer.generate_content(
            [video_file, DESCRIBE_VIDEO_PROMPT],
          )

          qa_generator = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
              "response_mime_type": "application/json",
              "temperature": 1,
            },
          )
          full_prompt = f"""{GENERATE_QA_PROMPT}

  VIDEO CONTENT: {describer_response.text.strip()}"""

          qa_generator_response = qa_generator.generate_content(full_prompt)
          qa_pairs = json.loads(qa_generator_response.text)
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
      print(f"Error generating metadata for video {video_id_with_chunk_id}: {e}")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {"id": video_id_with_chunk_id, "reason": str(e)}
        f.write(json.dumps(data) + "\n")


def process(shard: str):
  """
  shard: str = "shard_0.json"
  """
  start_time = time()

  shard_id = int(shard.split(".")[0].split("_")[1])

  print(f"Processing videos for shard {shard_id}...")

  os.makedirs(f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}", exist_ok=True)

  dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/vivid_instruct_65k/{shard}"
  )["train"]

  dataset.map(
    _process,
    batched=True,
    batch_size=200,
    num_proc=os.cpu_count(),
  )

  if uploader.check_file_exists(
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"videos/shard_{shard_id}.zip",
    repo_type="dataset",
  ):
    print(f"Shard {shard_id} already uploaded. Skipping...")
  else:
    uploader.zip_and_upload_dir(
      dir_path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"videos/shard_{shard_id}.zip",
      repo_type="dataset",
      overwrite=True,
    )

  uploader.upload_file(
    file_path=f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl",
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"errors/shard_{shard_id}.jsonl",
    repo_type="dataset",
    overwrite=True,
  )

  if uploader.check_file_exists(
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"metadata/shard_{shard_id}.jsonl",
    repo_type="dataset",
  ):
    print(f"Metadata for shard {shard_id} already uploaded. Skipping...")
  else:
    uploader.upload_file(
      file_path=f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"metadata/shard_{shard_id}.jsonl",
      repo_type="dataset",
      overwrite=True,
    )

  end_time = time()
  duration = round(end_time - start_time, 2)

  send_process_shard_success_message(shard_id, duration)

  shutil.rmtree(f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}")
  os.remove(f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl")
  os.remove(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl")


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
