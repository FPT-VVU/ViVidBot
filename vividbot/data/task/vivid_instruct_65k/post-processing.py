import json
import logging
import os
import time
from pathlib import Path
from typing import List

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.huggingface import HuggingFaceProcessor
from vividbot.data.task.vivid_instruct_65k.utils.chains import (
  get_dedup_description_chain,
)
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

  for shard_filename in tqdm(shard_files):
    shard = shard_filename.split(".")[0]

    if not os.path.exists(f"{BASE_DATA_PATH}/post-processing/metadata/{shard}.jsonl"):
      hf_processor.download_file(
        repo_id="Vividbot/vividbot_video",
        filename=f"metadata/{shard}.jsonl",
        local_dir=f"{BASE_DATA_PATH}/post-processing",
      )

    # remove all fields except those fields: id, video, conversations

    data = []
    with open(f"{BASE_DATA_PATH}/post-processing/metadata/{shard}.jsonl", "r") as f:
      for line in f:
        data.append(json.loads(line))

    for i, d in tqdm(enumerate(data)):
      id = d["id"]
      video = d["video"]
      conversations = []
      description = d.get("description", None)

      logger.info(f"Processing video: {video}")

      if description:
        if len(description) > 30000:
          logger.info(f"Found malformed description: {description[:1000]}...")

          dedup_chain = get_dedup_description_chain()

          description = dedup_chain.invoke(
            {
              "message": description[:5000],
            }
          )

          logger.info(f"Deduped description: {description}")

        human_value = get_describe_video_prompt_vi()
        if np.random.rand() < 0.5:
          human_value = f"{human_value}\n<video>"
        else:
          human_value = f"<video>\n{human_value}"

        conversations.append(
          {
            "from": "human",
            "value": human_value,
          }
        )
        conversations.append(
          {
            "from": "gpt",
            "value": description,
          }
        )

        for j, c in enumerate(d["conversations"]):
          if j % 2 == 0:
            # remove the strings "<video>\n" or "\n<video>" from the value
            c["value"] = c["value"].replace("\n<video>", "").replace("<video>\n", "")

          conversations.append(c)
      else:
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

    with open(
      f"{BASE_DATA_PATH}/post-processing/metadata-training/{shard}.json", "a"
    ) as f:
      for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(f"{BASE_DATA_PATH}/post-processing/metadata-training.json", "a") as f:
      for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # upload to huggingface
    hf_processor.upload_file(
      file_path=f"{BASE_DATA_PATH}/post-processing/metadata-training/{shard}.json",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"metadata-training/{shard}.json",
      repo_type="dataset",
      overwrite=True,
    )

    # wait for 120 seconds to avoid rate limit
    logger.info("Waiting for 120 seconds...")
    time.sleep(120)

  # upload to huggingface
  hf_processor.upload_file(
    file_path=f"{BASE_DATA_PATH}/output/post-processing/metadata-training.json",
    repo_id="Vividbot/vividbot_video",
    path_in_repo="metadata-training.json",
    repo_type="dataset",
    overwrite=True,
  )


def prepare():
  os.makedirs(f"{BASE_DATA_PATH}/post-processing", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/post-processing/metadata", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/post-processing/metadata-training", exist_ok=True)


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
