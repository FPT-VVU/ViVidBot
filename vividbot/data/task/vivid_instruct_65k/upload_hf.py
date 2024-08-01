import os
import time
from pathlib import Path

from tqdm import tqdm

from vividbot.data.processor.huggingface import HuggingFaceProcessor

hf_processor = HuggingFaceProcessor()

BASE_DATA_PATH = f"{Path.home()}/data/images"


def main():
  while True:
    print("Uploading metadata files to Hugging Face every 1 minutes...")

    # merge 3 files into 1
    conversation_lines = open(
      f"{BASE_DATA_PATH}/metadata_conversation.json"
    ).readlines()
    detail_lines = open(f"{BASE_DATA_PATH}/metadata_detail.json").readlines()
    reasoning_lines = open(f"{BASE_DATA_PATH}/metadata_reasoning.json").readlines()
    with open(f"{BASE_DATA_PATH}/metadata_extended_2.json", "w") as f:
      for conversation_line in conversation_lines:
        f.write(conversation_line)
      for detail_line in detail_lines:
        f.write(detail_line)
      for reasoning_line in reasoning_lines:
        f.write(reasoning_line)

    hf_processor.upload_file(
      file_path=f"{BASE_DATA_PATH}/metadata_extended_2.json",
      repo_id="Vividbot/vividbot_image",
      path_in_repo="metadata_extended_2.json",
      repo_type="dataset",
      overwrite=True,
    )

    # hf_processor.upload_file(
    #   file_path=f"{BASE_DATA_PATH}/metadata_conversation.json",
    #   repo_id="Vividbot/vividbot_image",
    #   path_in_repo="metadata_conversation.json",
    #   repo_type="dataset",
    #   overwrite=True,
    # )

    # hf_processor.upload_file(
    #   file_path=f"{BASE_DATA_PATH}/metadata_detail.json",
    #   repo_id="Vividbot/vividbot_image",
    #   path_in_repo="metadata_detail.json",
    #   repo_type="dataset",
    #   overwrite=True,
    # )

    # hf_processor.upload_file(
    #   file_path=f"{BASE_DATA_PATH}/metadata_reasoning.json",
    #   repo_id="Vividbot/vividbot_image",
    #   path_in_repo="metadata_reasoning.json",
    #   repo_type="dataset",
    #   overwrite=True,
    # )

    hf_processor.zip_and_upload_dir(
      dir_path=f"{BASE_DATA_PATH}/output/images_extended_2",
      repo_id="Vividbot/vividbot_image",
      path_in_repo="images_extended_2/images_extended_2.zip",
      repo_type="dataset",
      overwrite=True,
    )

    for _ in tqdm(range(60)):
      time.sleep(1)


if __name__ == "__main__":
  main()
