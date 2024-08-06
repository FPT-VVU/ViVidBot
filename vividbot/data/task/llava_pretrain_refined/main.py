import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from vividbot.data.processor.huggingface import HuggingFaceProcessor
from vividbot.data.schemas import ImageMetadata
from vividbot.data.task.common.chains import (
  get_rewrite_image_caps_chain,
)
from vividbot.data.task.vivid_instruct_65k.utils.notifications import (
  send_completion_message,
  send_process_shard_success_message,
)

load_dotenv()
BASE_DATA_PATH = f"{Path.home()}/data/llava-pretrain"
REFINED_METADATA_FILE = f"{BASE_DATA_PATH}/llava_pretrain_vi.jsonl"
REFINED_METADATA_FILE_ALT = f"{BASE_DATA_PATH}/llava_pretrain_vi_all.json"
TEMP_REFINED_METADATA_FILE = f"{BASE_DATA_PATH}/llava_pretrain_en.jsonl"

_QUESTIONS = [
  "Hình này nói về điều gì?",
  "Đây là gì?",
  "Bạn mô tả được gì từ hình ảnh này?",
  "Hình ảnh này thể hiện điều gì?",
  "Nội dung chính của bức ảnh này là gì?",
  "Bạn nhìn thấy gì trong hình này?",
  "Hình ảnh này đang minh họa cho điều gì?",
  "Chủ đề chính của bức ảnh này là gì?",
  "Bạn có thể giải thích ý nghĩa của hình ảnh này không?",
  "Hình ảnh này đang cố gắng truyền tải thông điệp gì?",
  "Những yếu tố chính nào bạn nhận thấy trong hình này?",
  "Bạn có thể tóm tắt nội dung của hình ảnh này không?",
  "Hình ảnh này đang kể câu chuyện gì?",
  "Điểm nổi bật nhất trong bức ảnh này là gì?",
  "Bạn có thể mô tả ngắn gọn về hình ảnh này không?",
  "Có những gì đáng chú ý trong hình ảnh này?",
  "Hãy đặt tựa đề cho bức ảnh này.",
  "Nếu bạn phải mô tả hình ảnh này bằng một câu, bạn sẽ nói gì?",
  "Đề xuất một chú thích ngắn gọn cho hình ảnh này.",
  "Đề xuất một tiêu đề ngắn gọn dựa trên nội dung của hình ảnh này.",
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


def get_random_question(id: Optional[int]) -> str:
  if id:
    return _QUESTIONS[id % len(_QUESTIONS)]
  else:
    return np.random.choice(_QUESTIONS)


def prepare():
  os.makedirs(BASE_DATA_PATH, exist_ok=True)


def _process(batch: dict):
  batch_ids = batch["id"]
  batch_images = batch["image"]
  batch_conversations = batch["conversations"]
  processed_ids = []

  # check if id already exists
  if os.path.exists(f"{Path.home()}/data/llava-pretrain/process_ids.json"):
    processed_ids = json.load(
      open(f"{Path.home()}/data/llava-pretrain/process_ids.json")
    )

  for i, (id, image, conversations) in tqdm(
    enumerate(zip(batch_ids, batch_images, batch_conversations))
  ):
    if id in processed_ids:
      logger.info(f"Skipping video {id} as it already exists.")
      continue

    metadata = ImageMetadata(
      id=id,
      image=image,
      conversations=[],
    )

    rewrite_chain = get_rewrite_image_caps_chain()

    question = get_random_question(id=i)
    answer = rewrite_chain.invoke(
      {
        "captions": conversations[1]["value"],
        "question": question,
      },
    )

    if np.random.rand() < 0.5:
      question = f"{question}\n<image>"
    else:
      question = f"<image>\n{question}"

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

    metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

    logger.info(f"{metadata_json}")

    with open(REFINED_METADATA_FILE, "a") as f:
      f.write(metadata_json + "\n")


def process():
  start_time = time.time()

  temp_metadata = load_dataset(
    "json",
    data_files=TEMP_REFINED_METADATA_FILE,
    split="train",
  )

  temp_metadata.map(_process, num_proc=os.cpu_count(), batched=True, batch_size=480)

  processed_metadata = load_dataset(
    "json",
    data_files=REFINED_METADATA_FILE,
    split="train",
  )

  end_time = time.time()
  duration = round(end_time - start_time, 2)

  send_process_shard_success_message("X", duration, count=processed_metadata.num_rows)


def post_process():
  logger.info("Post-processing metadata...")

  metadata = load_dataset(
    "json",
    data_files=REFINED_METADATA_FILE,
    split="train",
  )

  metadata_alt = [d for d in metadata]

  with open(REFINED_METADATA_FILE_ALT, "w") as f:
    f.write(json.dumps(metadata_alt, ensure_ascii=False, indent=2))

  logger.info("Uploading metadata to Hugging Face...")
  hf_processor.upload_file(
    file_path=REFINED_METADATA_FILE,
    path_in_repo="llava_pretrain_vi.jsonl",
    repo_id="Vividbot/llava-pretrain-vi",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_file(
    file_path=REFINED_METADATA_FILE_ALT,
    path_in_repo="llava_pretrain_vi_all.json",
    repo_id="Vividbot/llava-pretrain-vi",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_file(
    file_path=TEMP_REFINED_METADATA_FILE,
    path_in_repo="llava_pretrain_en.jsonl",
    repo_id="Vividbot/llava-pretrain-vi",
    repo_type="dataset",
    overwrite=True,
  )


def main():
  prepare()
  process()
  post_process()
  send_completion_message()


if __name__ == "__main__":
  main()
