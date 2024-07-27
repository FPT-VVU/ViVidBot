import json
import logging
import os
import time
from pathlib import Path

import google.generativeai as genai
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from tqdm import tqdm

from vividbot.data.processor.huggingface import HuggingFaceProcessor
from vividbot.data.task.vivid_instruct_65k.utils import pinscrape
from vividbot.data.task.vivid_instruct_65k.utils.notifications import (
  send_completion_message,
)
from vividbot.data.task.vivid_instruct_65k.utils.prompts import (
  get_generate_qa_from_image_prompt,
)

load_dotenv()
BASE_DATA_PATH = f"{Path.home()}/data/images"


logger = logging.getLogger(__name__)
logging.basicConfig(
  filename=f"{BASE_DATA_PATH}/run.log",
  filemode="w",
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


hf_processor = HuggingFaceProcessor()


def _process(batch: dict):
  for keyword, category in tqdm(zip(batch["keyword"], batch["category"])):
    logger.info(f"Processing images for keyword {keyword}...")
    details = pinscrape.scraper.scrape(
      keyword, f"{BASE_DATA_PATH}/output/images", {}, 4, 50
    )

    # 'urls_list': ['https://i.pinimg.com/originals/12/8e/6e/128e6e5e651dab7aae7f135e24b74a08.jpg',
    # 'https://i.pinimg.com/originals/6d/8c/ef/6d8cef114f2613042765e3f7fc8262d0.jpg'],

    for url in tqdm(details.get("urls_list", [])):
      image_filename = url.split("/")[-1]
      image_id = image_filename.split(".")[0]
      google_file_name = f"files/{image_filename}".lower().replace("_", "-")

      image_file = genai.upload_file(
        path=f"{BASE_DATA_PATH}/output/images/{image_filename}",
        name=google_file_name,
        display_name=image_id,
      )

      describer = genai.GenerativeModel(
        "models/gemini-1.5-pro",
        generation_config={
          "temperature": 0.5,
          "max_output_tokens": 2048,
        },
        safety_settings={
          HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
          HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
          HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
          HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
        system_instruction=get_generate_qa_from_image_prompt(),
      )
      describer_response = describer.generate_content(
        [image_file, "Generate QA pairs as instructed."],
        request_options={
          "timeout": 60,
        },
      )
      try:
        qa_pairs = json.loads(describer_response.text.strip())
        conversations = []

        for qa in qa_pairs:
          if not qa.get("question") or not qa.get("answer"):
            continue

          human_value = str(qa["question"]).strip()
          gpt_value = str(qa["answer"]).strip()
          if np.random.random() < 0.5:
            human_value = human_value + "\n<image>"
          else:
            human_value = "<image>\n" + human_value

          conversations.append({"from": "human", "value": human_value})
          conversations.append({"from": "gpt", "value": gpt_value})

        with open(
          f"{BASE_DATA_PATH}/metadata.jsonl",
          "a",
        ) as f:
          data = {
            "id": image_id,
            "image": image_id,
            "conversations": conversations,
            "image_filename": image_filename,
            "timestamp": round(time.time()),
            "description": describer_response.text.strip(),
            "keyword": keyword,
            "category": category,
          }

          f.write(
            json.dumps(
              data,
              ensure_ascii=False,
            )
            + "\n"
          )

          logger.info(
            f"Generated metadata for image {image_id}: {json.dumps(data, ensure_ascii=False)}"
          )
      except Exception as e:
        logger.error(
          f"Couldn't generate QA pairs for image {image_id}: {str(e)} - Response: {describer_response}"
        )
        with open(f"{BASE_DATA_PATH}/errors.jsonl", "a") as f:
          data = {
            "id": image_id,
            "reason": f"{str(e)} - Response: {describer_response}",
            "timestamp": round(time.time()),
          }
          f.write(json.dumps(data) + "\n")

        if os.path.exists(f"{BASE_DATA_PATH}/{image_filename}"):
          os.remove(f"{BASE_DATA_PATH}/{image_filename}")


def process():
  start_time = time.time()

  logger.info("Processing images...")

  dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/flattened_keywords.jsonl"
  )["train"]

  dataset.map(
    _process,
    batched=True,
    batch_size=32,
    num_proc=2,
  )

  hf_processor.zip_and_upload_dir(
    dir_path=f"{BASE_DATA_PATH}/output/images",
    repo_id="Vividbot/vividbot_image",
    path_in_repo="images.zip",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_file(
    file_path=f"{BASE_DATA_PATH}/metadata.jsonl",
    repo_id="Vividbot/vividbot_image",
    path_in_repo="metadata.jsonl",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_file(
    file_path=f"{BASE_DATA_PATH}/errors.jsonl",
    repo_id="Vividbot/vividbot_image",
    path_in_repo="errors.jsonl",
    repo_type="dataset",
    overwrite=True,
  )

  # remove files that are not processed
  processed_dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/metadata.jsonl"
  )["train"]

  processed_images = set([item["image_filename"] for item in processed_dataset])
  all_images = set(
    [item.split("/")[-1] for item in os.listdir(f"{BASE_DATA_PATH}/output/images")]
  )

  for image in all_images - processed_images:
    os.remove(f"{BASE_DATA_PATH}/output/images/{image}")

  end_time = time.time()
  duration = round(end_time - start_time, 2)

  logger.info(f"Processed images in {duration} seconds.")


def prepare():
  os.makedirs(BASE_DATA_PATH, exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/images", exist_ok=True)

  # create run.log file
  with open(f"{BASE_DATA_PATH}/run.log", "w") as f:
    f.write("")


def main():
  prepare()
  process()
  send_completion_message()


if __name__ == "__main__":
  main()
