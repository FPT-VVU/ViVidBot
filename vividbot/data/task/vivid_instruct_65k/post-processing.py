import json
import logging
import os
from pathlib import Path
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.huggingface import HuggingFaceProcessor
from vividbot.data.task.vivid_instruct_65k.utils.notifications import (
  send_completion_message,
)
from vividbot.data.task.vivid_instruct_65k.utils.prompts import (
  get_describe_video_prompt_vi,
)

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


def process(shard_files: List[str]):
  """
  shard_filename: str = "shard_0.jsonl"
  """

  # download all metadatas from hugingface
  # 1. rename to *.json
  # 2. combine all json files into one
  # finally, upload to huggingface

  combined_data = []

  for shard_filename in shard_files:
    shard = shard_filename.split(".")[0]
    hf_processor.download_file(
      repo_id="Vividbot/vividbot_video",
      filename=f"metadata/{shard}.jsonl",
      local_dir=f"{BASE_DATA_PATH}/post-processing",
    )

    # remove all fields except those fields: id, video, conversations

    data = []
    with open(f"{BASE_DATA_PATH}/post-processing/{shard}.jsonl", "r") as f:
      for line in f:
        data.append(json.loads(line))

    for i, d in enumerate(data):
      id = d["id"]
      video = d["video"]
      conversations = []
      description = d.get("description", None)

      if description:
        conversations.append(
          {
            "from": "human",
            "value": get_describe_video_prompt_vi(),
          }
        )

        conversations.append(
          {
            "from": "gpt",
            "value": description,
          }
        )

      for j, c in enumerate(d["conversations"]):
        if j > 0 and j % 2 == 0:
          # remove the strings "<video>\n" or "\n<video>" from the value
          c["value"] = c["value"].replace("\n<video>", "").replace("<video>\n", "")

        conversations.append(c)

      data[i] = {
        "id": id,
        "video": video,
        "conversations": conversations,
      }

    combined_data.extend(data)

    # sort data by id
    data = sorted(data, key=lambda x: x["id"])

    with open(f"{BASE_DATA_PATH}/post-processing/{shard}.json", "w") as f:
      for d in data:
        f.write(json.dumps(d) + "\n")

    # upload to huggingface
    hf_processor.upload_file(
      repo_id="Vividbot/vividbot_video",
      filename=f"metadata-training/{shard}.json",
      local_dir=f"{BASE_DATA_PATH}/post-processing",
    )

  # sort combined data by video, id
  combined_data = sorted(
    combined_data,
    key=lambda x: (x["video"], x["id"]),
  )

  # save combined data
  with open(f"{BASE_DATA_PATH}/post-processing/metadata-training.json", "w") as f:
    for d in combined_data:
      f.write(json.dumps(d) + "\n")

  # upload to huggingface
  hf_processor.upload_file(
    repo_id="Vividbot/vividbot_video",
    filename="metadata-training.json",
    local_dir=f"{BASE_DATA_PATH}/post-processing",
  )


def prepare():
  os.makedirs(f"{BASE_DATA_PATH}/post-processing", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/post-processing/metadata", exist_ok=True)


def main():
  prepare()
  shard_files = os.listdir(f"{BASE_DATA_PATH}/vivid_instruct_65k")
  shard_files = sorted(
    shard_files,
    key=lambda x: int(x.split(".")[0].split("_")[1]),
  )

  logger.info(f"Processing shards: {shard_files}")

  process(shard_files)

  send_completion_message()


if __name__ == "__main__":
  main()
