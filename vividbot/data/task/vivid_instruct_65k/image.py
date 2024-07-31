import json
import logging
import os
import time
from pathlib import Path
from typing import Literal, Union

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

PROMPT_TYPE: Union[Literal["conversation"], Literal["reasoning"], Literal["detail"]] = (
  "detail"
)


def _process(batch: dict):
  processed_dataset = None

  if os.path.exists(f"{BASE_DATA_PATH}/metadata_{PROMPT_TYPE}.json"):
    processed_dataset = load_dataset(
      "json", data_files=f"{BASE_DATA_PATH}/metadata_{PROMPT_TYPE}.json"
    )["train"]

  for keyword, category in tqdm(zip(batch["keyword"], batch["category"])):
    logger.info(f"Processing images for keyword {keyword}...")
    details = pinscrape.scraper.scrape(
      keyword, f"{BASE_DATA_PATH}/output/images_extended", {}, 12, 100
    )

    for url in tqdm(details.get("urls_list", [])):
      image_filename = url.split("/")[-1]
      if not os.path.exists(
        f"{BASE_DATA_PATH}/output/images_extended/{image_filename}"
      ):
        continue

      image_id = image_filename.split(".")[0]

      # skip if image_id already exists in metadata
      if (
        processed_dataset is not None
        and processed_dataset.filter(lambda x: x["id"] == image_id).num_rows > 0
      ):
        logger.info(f"Image {image_id} already processed. Skipping...")
        continue

      google_filename = f"files/{image_id}".lower().replace("_", "-")

      image_file = None
      try:
        image_file = genai.get_file(name=google_filename)
      except Exception as e:
        logger.warning(f"Couldn't get file {google_filename}: {str(e)}")

      if not image_file:
        try:
          image_file = genai.upload_file(
            path=f"{BASE_DATA_PATH}/output/images_extended/{image_filename}",
            name=google_filename,
            display_name=image_id,
          )
        except Exception as e:
          logger.error(f"Couldn't upload file {google_filename}: {str(e)}")

      if not image_file:
        logger.error(f"Couldn't upload file {google_filename}")
        with open(f"{BASE_DATA_PATH}/errors.json", "a") as f:
          data = {
            "id": image_id,
            "reason": "Couldn't upload file",
            "timestamp": round(time.time()),
          }
          f.write(json.dumps(data) + "\n")
        continue

      describer = genai.GenerativeModel(
        "models/gemini-1.5-flash",
        generation_config={
          "temperature": 0.45,
          "max_output_tokens": 1024,
          "top_p": 0.7,
        },
        safety_settings={
          HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
          HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
          HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
          HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
        system_instruction=get_generate_qa_from_image_prompt(type=PROMPT_TYPE),
      )
      describer_response = describer.generate_content(
        [image_file, "Generate QA pairs as instructed."],
        request_options={
          "timeout": 60,
        },
      )

      json_text = None

      try:
        json_text = (
          describer_response.text
          if describer_response.text
          else json.dumps(
            json.loads(describer_response.parts[0].text), ensure_ascii=False
          )
          if describer_response.parts and len(describer_response.parts) > 0
          else json.dumps(
            json.loads(describer_response.candidates[0].content.parts[0].text),
            ensure_ascii=False,
          )
          .encode("utf-8")
          .decode("utf-8")
          if describer_response.candidates
          and len(describer_response.candidates) > 0
          and len(describer_response.candidates[0].content.parts) > 0
          else None
        )
      except Exception as e:
        logger.error(
          f"Couldn't generate QA pairs for image {image_id}: {str(e)} - Response: {describer_response}"
        )
        with open(f"{BASE_DATA_PATH}/errors.json", "a") as f:
          data = {
            "id": image_id,
            "reason": f"{str(e)} - Response: {describer_response}",
            "timestamp": round(time.time()),
          }
          f.write(json.dumps(data) + "\n")

        # if os.path.exists(f"{BASE_DATA_PATH}/{image_filename}"):
        #   os.remove(f"{BASE_DATA_PATH}/{image_filename}")
        continue

      if not json_text:
        logger.error(
          f"Couldn't generate QA pairs for image {image_id}: {describer_response}"
        )
        with open(f"{BASE_DATA_PATH}/errors.json", "a") as f:
          data = {
            "id": image_id,
            "reason": f"Couldn't generate QA pairs - Response: {describer_response}",
            "timestamp": round(time.time()),
          }
          f.write(json.dumps(data) + "\n")

        # if os.path.exists(f"{BASE_DATA_PATH}/{image_filename}"):
        #   os.remove(f"{BASE_DATA_PATH}/{image_filename}")
        continue

      try:
        qa_pairs = json.loads(json_text.strip())
        conversations = []

        for qa in qa_pairs:
          if not qa.get("question") or not qa.get("answer"):
            continue

          human_value = str(qa["question"]).strip()
          gpt_value = str(qa["answer"]).strip()
          if len(conversations) == 0:
            if np.random.random() < 0.5:
              human_value = human_value + "\n<image>"
            else:
              human_value = "<image>\n" + human_value

          conversations.append({"from": "human", "value": human_value})
          conversations.append({"from": "gpt", "value": gpt_value})

        with open(
          f"{BASE_DATA_PATH}/metadata_{PROMPT_TYPE}.json",
          "a",
        ) as f:
          data = {
            "id": image_id,
            "image": f"images_extended/{image_filename}",
            "conversations": conversations,
            "timestamp": round(time.time()),
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
        with open(f"{BASE_DATA_PATH}/errors.json", "a") as f:
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

  logger.info(f"Processing images for prompt type: {PROMPT_TYPE}")

  dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/image_flattened_keywords.json"
  ).shuffle(seed=2024)["train"]

  dataset.map(
    _process,
    batched=True,
    batch_size=128,
    num_proc=6,
  )

  # remove files in output/images that are not in metadata
  # processed_dataset = load_dataset(
  #   "json", data_files=f"{BASE_DATA_PATH}/metadata_{PROMPT_TYPE}.json"
  # )["train"]

  # processed_image_ids = [item["id"] for item in processed_dataset]
  # for image_filename in os.listdir(f"{BASE_DATA_PATH}/output/images_exte"):
  #   image_id = image_filename.split(".")[0]
  #   if image_id not in processed_image_ids:
  #     os.remove(f"{BASE_DATA_PATH}/output/images/{image_filename}")

  hf_processor.zip_and_upload_dir(
    dir_path=f"{BASE_DATA_PATH}/output/images_extended",
    repo_id="Vividbot/vividbot_image",
    path_in_repo="images_extended.zip",
    repo_type="dataset",
    overwrite=True,
  )

  hf_processor.upload_file(
    file_path=f"{BASE_DATA_PATH}/metadata_{PROMPT_TYPE}.json",
    repo_id="Vividbot/vividbot_image",
    path_in_repo=f"metadata_{PROMPT_TYPE}.json",
    repo_type="dataset",
    overwrite=True,
  )

  # hf_processor.upload_file(
  #   file_path=f"{BASE_DATA_PATH}/errors.json",
  #   repo_id="Vividbot/vividbot_image",
  #   path_in_repo="errors.json",
  #   repo_type="dataset",
  #   overwrite=True,
  # )

  end_time = time.time()
  duration = round(end_time - start_time, 2)

  logger.info(f"Processed images in {duration} seconds.")


def prepare():
  os.makedirs(BASE_DATA_PATH, exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/images_extended", exist_ok=True)

  # create run.log file
  with open(f"{BASE_DATA_PATH}/run.log", "w") as f:
    f.write("")


def main():
  prepare()
  process()
  send_completion_message()


if __name__ == "__main__":
  main()
