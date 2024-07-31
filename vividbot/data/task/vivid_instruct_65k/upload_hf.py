import time
from pathlib import Path

from tqdm import tqdm

from vividbot.data.processor.huggingface import HuggingFaceProcessor

hf_processor = HuggingFaceProcessor()

BASE_DATA_PATH = f"{Path.home()}/data/images"


def main():
  while True:
    print("Uploading metadata files to Hugging Face every 10 minutes...")

    # merge 3 files into 1
    conversation_lines = open(
      f"{BASE_DATA_PATH}/metadata_conversation.json"
    ).readlines()
    detail_lines = open(f"{BASE_DATA_PATH}/metadata_detail.json").readlines()
    reasoning_lines = open(f"{BASE_DATA_PATH}/metadata_reasoning.json").readlines()
    with open(f"{BASE_DATA_PATH}/metadata_extended.json", "w") as f:
      for conversation_line in conversation_lines:
        f.write(conversation_line)
      for detail_line in detail_lines:
        f.write(detail_line)
      for reasoning_line in reasoning_lines:
        f.write(reasoning_line)

    hf_processor.upload_file(
      file_path=f"{BASE_DATA_PATH}/metadata_extended.json",
      repo_id="Vividbot/vividbot_image",
      path_in_repo="metadata_extended.json",
      repo_type="dataset",
      overwrite=True,
    )

    hf_processor.zip_and_upload_dir(
      dir_path=f"{BASE_DATA_PATH}/output/images_extended",
      repo_id="Vividbot/vividbot_image",
      path_in_repo="images_extended/images_extended.zip",
      repo_type="dataset",
      overwrite=True,
    )

    for _ in tqdm(range(60 * 10)):
      time.sleep(1)


if __name__ == "__main__":
  main()
