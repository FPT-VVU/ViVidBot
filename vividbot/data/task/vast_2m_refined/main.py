import json
import logging
import os
import time
from pathlib import Path

import google.generativeai as genai
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from vividbot.data.processor.huggingface import HuggingFaceProcessor
from vividbot.data.schemas import VideoMetadata
from vividbot.data.task.common.chains import (
  get_rewrite_vast_caps_chain,
)
from vividbot.data.task.vivid_instruct_65k.utils.notifications import (
  send_completion_message,
  send_process_shard_success_message,
)

load_dotenv()
BASE_DATA_PATH = f"{Path.home()}/data/vast-2m"
REFINED_METADATA_PATH = f"{BASE_DATA_PATH}/metadata-refined"
TEMP_REFINED_METADATA_PATH = f"{BASE_DATA_PATH}/metadata-refined-temp"

_QUESTIONS = [
  "Video này nói về điều gì?",
  "Nội dung chính của video này là gì?",
  "Video này truyền tải thông điệp gì?",
  "Chủ đề chính được đề cập trong video là gì?",
  "Bạn có thể tóm tắt nội dung của video này không?",
  "Video này đang cố gắng giải thích điều gì?",
  "Ý tưởng chính được thể hiện trong video là gì?",
  "Video này đang trình bày về vấn đề gì?",
  "Nội dung cốt lõi mà video muốn truyền đạt là gì?",
  "Bạn có thể mô tả ngắn gọn về nội dung của video không?",
  "Điểm chính mà video này muốn người xem hiểu là gì?",
  "Có những sự kiện gì xảy ra trong video?",
  "Video này ghi lại những hoạt động nào?",
  "Bạn có thể liệt kê các sự kiện quan trọng trong video không?",
  "Những diễn biến nào được thể hiện trong nội dung video?",
  "Video này mô tả chuỗi sự kiện nào?",
  "Các tình huống chính xuất hiện trong video là gì?",
  "Những hành động đáng chú ý nào được thể hiện trong video?",
  "Video này trình bày trình tự các sự kiện như thế nào?",
  "Bạn có thể tóm tắt các sự việc xảy ra trong video không?",
  "Video này ghi lại quá trình diễn ra của những sự kiện nào?",
  "Có những tình tiết quan trọng nào được thể hiện trong video?",
  "Bạn có thể mô tả các hoạt động chính trong video không?",
  "Hãy đưa ra mô tả chi tiết về nội dung của video.",
]

logger = logging.getLogger(__name__)
logging.basicConfig(
  filename=f"{BASE_DATA_PATH}/run.log",
  filemode="w",
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  force=True,
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


hf_processor = HuggingFaceProcessor()


def get_random_question(id: int) -> str:
  if id:
    return _QUESTIONS[id % len(_QUESTIONS)]
  else:
    return np.random.choice(_QUESTIONS)


def prepare():
  os.makedirs(BASE_DATA_PATH, exist_ok=True)
  os.makedirs(REFINED_METADATA_PATH, exist_ok=True)


def _process(batch: dict):
  ids = batch["id"]
  videos = batch["video"]
  vision_caps = batch["vision_cap"]
  shard = videos[0].split("/")[0]
  processed_metadata = None

  # check if id already exists
  if os.path.exists(f"{REFINED_METADATA_PATH}/{shard}.jsonl"):
    processed_metadata = load_dataset(
      "json",
      data_files=f"{REFINED_METADATA_PATH}/{shard}.jsonl",
      split="train",
    )

  for i, (id, video, vision_cap) in tqdm(enumerate(zip(ids, videos, vision_caps))):
    if (
      processed_metadata
      and processed_metadata.filter(lambda x: x["id"] == id).num_rows > 0
    ):
      logger.info(f"Skipping video {id} in shard {shard} as it already exists.")
      continue

    metadata = VideoMetadata(
      id=id,
      video=video,
      conversations=[],
    )

    translate_chain = get_rewrite_vast_caps_chain()

    question = get_random_question(id=i)
    answer = translate_chain.invoke(
      {
        "captions": json.dumps(vision_cap, separators=(",", ":"), ensure_ascii=False),
        "question": question,
      }
    )

    if np.random.rand() < 0.5:
      question = f"{question}\n<video>"
    else:
      question = f"<video>\n{question}"

    metadata["conversations"].append(
      {
        "from": "human",
        "value": question,
      }
    )
    metadata["conversations"].append(
      {
        "from": "gpt",
        "value": answer,
      }
    )

    metadata_json = json.dumps(metadata, ensure_ascii=False)

    logger.info(f"{metadata_json}")

    with open(f"{REFINED_METADATA_PATH}/{shard}.jsonl", "a") as f:
      f.write(metadata_json + "\n")


def process_shard(shard_filename: str):
  """
  shard_filename: str = "shard_0.jsonl"
  """
  start_time = time.time()

  shard = shard_filename.split(".")[0]

  if os.path.exists(f"{REFINED_METADATA_PATH}/{shard}.jsonl"):
    logger.info(f"Skipping shard {shard} as it already exists.")
    return

  logger.info(f"Processing shard {shard}...")

  temp_metadata = load_dataset(
    "json",
    data_files=f"{TEMP_REFINED_METADATA_PATH}/{shard_filename}",
    split="train",
  )

  temp_metadata.map(_process, num_proc=os.cpu_count(), batched=True, batch_size=200)

  processed_metadata = load_dataset(
    "json",
    data_files=f"{REFINED_METADATA_PATH}/{shard}.jsonl",
    split="train",
  )

  end_time = time.time()
  duration = round(end_time - start_time, 2)

  send_process_shard_success_message(shard, duration, count=processed_metadata.num_rows)


def post_process():
  logger.info("Post-processing metadata...")
  # read all metadata files and combine json lines to a list of json objects
  metadata_files = os.listdir(REFINED_METADATA_PATH)
  metadata_files = sorted(
    metadata_files,
    key=lambda x: int(x.split(".")[0].split("_")[1]),
  )

  logger.info(f"Combining metadata files: {metadata_files}")
  metadata = []

  for metadata_file in metadata_files:
    with open(f"{REFINED_METADATA_PATH}/{metadata_file}", "r") as f:
      for line in f:
        metadata.append(json.loads(line))

  with open(f"{BASE_DATA_PATH}/vast_2m_vi_refined_all.json", "w") as f:
    f.write(json.dumps(metadata, ensure_ascii=False, indent=2))

  logger.info("Uploading metadata to Hugging Face...")
  hf_processor.upload_file(
    file_path=f"{BASE_DATA_PATH}/vast_2m_vi_refined_all.json",
    path_in_repo="vast_2m_vi_refined_all.json",
    repo_id="Vividbot/vast-2m-vi",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_dir(
    dir_path=REFINED_METADATA_PATH,
    path_in_repo="metadata-refined",
    repo_id="Vividbot/vast-2m-vi",
    repo_type="dataset",
    overwrite=True,
  )


def process():
  shard_files = os.listdir(TEMP_REFINED_METADATA_PATH)
  shard_files = sorted(
    shard_files,
    key=lambda x: int(x.split(".")[0].split("_")[1]),
  )

  last_successful_shard = 299
  shard_files = shard_files[last_successful_shard + 1 :]

  logger.info(f"Processing shards: {shard_files}")

  for shard in tqdm(
    shard_files,
    desc="Processing shards",
    unit="shard",
    unit_scale=True,
    unit_divisor=1,
  ):
    process_shard(shard)


def main():
  prepare()
  process()
  post_process()
  send_completion_message()


if __name__ == "__main__":
  main()
