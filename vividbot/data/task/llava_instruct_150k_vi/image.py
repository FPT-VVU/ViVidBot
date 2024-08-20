import json
import logging
import os
import threading
import time
from pathlib import Path

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from vividbot.data.processor.huggingface import HuggingFaceProcessor
from vividbot.data.task.vivid_instruct_65k.utils.notifications import (
  send_completion_message,
  send_process_shard_success_message,
)

load_dotenv()
BASE_DATA_PATH = f"{Path.home()}/llava-instruct-150k-vi"
COCO_BASE_URL = "http://images.cocodataset.org/train2014/COCO_train2014_"

logger = logging.getLogger(__name__)
logging.basicConfig(
  filename=f"{BASE_DATA_PATH}/run.log",
  filemode="w",
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


hf_processor = HuggingFaceProcessor()


def download_image(image_url: str, image: str):
  if os.path.exists(f"{BASE_DATA_PATH}/images/{image}"):
    logger.info(f"Image {image} already exists, skipping...")
    return

  logger.info(f"Downloading image {image}...")

  max_retries = 3
  while max_retries > 0:
    max_retries -= 1
    try:
      response = requests.get(image_url)
      if response.status_code == 200:
        with open(f"{BASE_DATA_PATH}/images/{image}", "wb") as f:
          f.write(response.content)

        break
    except Exception as e:
      logger.error(f"Failed to download image {image}: {e} - Retrying...")

  if not os.path.exists(f"{BASE_DATA_PATH}/images/{image}"):
    logger.error(f"Failed to download image {image}.")
    with open(f"{BASE_DATA_PATH}/errors.json", "a") as f:
      f.write(json.dumps({"image": image, "url": image_url}) + "\n")


def _process(batch: dict):
  threads: dict = {}
  for id, image in tqdm(zip(batch["id"], batch["image"])):
    # image: str = "shard_0/000000000000.jpg"
    # id: str = "000000000000"

    image_url = f"{COCO_BASE_URL}{id}.jpg"
    threads[image] = threading.Thread(target=download_image, args=(image_url, image))

  for thread in threads.values():
    thread.start()

  for thread in threads.values():
    thread.join()


def process_shard(shard_filename: str):
  """
  shard_filename: str = "shard_0.json"
  """
  start_time = time.time()

  shard = shard_filename.split(".")[0]
  if (
    hf_processor.check_file_exists(
      repo_id="Vividbot/llava-instruct-150k-vi",
      path_in_repo=f"images/{shard}.zip",
      repo_type="dataset",
    )
    and len(os.listdir(f"{BASE_DATA_PATH}/images/{shard}")) == 5000
  ):
    logger.info(f"Images for shard {shard} already processed, skipping...")
    return

  os.makedirs(f"{BASE_DATA_PATH}/images/{shard}", exist_ok=True)

  logger.info(f"Processing images for shard {shard}...")

  dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/shards/{shard_filename}"
  )["train"]

  dataset.map(
    _process,
    batched=True,
    batch_size=32,
    num_proc=os.cpu_count(),
  )

  if os.path.exists(f"{BASE_DATA_PATH}/images/{shard}"):
    logger.info(f"Zipping and uploading images for shard {shard}...")

    hf_processor.zip_and_upload_dir(
      dir_path=f"{BASE_DATA_PATH}/images/{shard}",
      repo_id="Vividbot/llava-instruct-150k-vi",
      path_in_repo=f"images/{shard}.zip",
      repo_type="dataset",
      overwrite=True,
    )

  end_time = time.time()
  duration = round(end_time - start_time, 2)

  logger.info(f"Processed shard {shard} in {duration} seconds.")

  send_process_shard_success_message(
    shard, duration, len(os.listdir(f"{BASE_DATA_PATH}/images/{shard}"))
  )


def prepare():
  os.makedirs(f"{BASE_DATA_PATH}/images", exist_ok=True)
  with open(f"{BASE_DATA_PATH}/errors.json", "w") as f:
    f.write("")


def process():
  shard_files = os.listdir(f"{BASE_DATA_PATH}/shards")
  shard_files = sorted(
    shard_files,
    key=lambda x: int(x.split(".")[0].split("_")[1]),
  )

  last_successful_shard = -1
  # only process shards after the last successful shard
  shard_files = shard_files[last_successful_shard + 1 :]

  logger.info(f"Processing shards: {shard_files}")

  for shard in tqdm(
    shard_files,
    desc="Processing shards",
    unit="shard",
    unit_scale=True,
  ):
    process_shard(shard)


def main():
  prepare()
  process()
  send_completion_message()


if __name__ == "__main__":
  main()
