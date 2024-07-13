import os
import sys

sys.path.append(os.getcwd())
import datetime
from datetime import timezone
from pathlib import Path

import google.generativeai as genai
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


def send_upload_shard_success_message(shard_count):
  notifier.send(
    body={
      "embeds": [
        {
          "title": f"✅ ViVid Instruct 65k: Uploaded shard {shard_count}!",
          "description": f"Uploaded shard {shard_count} with {len(os.listdir(f'{BASE_DATA_PATH}/videos/shard_{shard_count}'))} clips at https://huggingface.co/datasets/Vividbot/vividbot_video.",
          "color": 2278494,
          "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }
      ]
    }
  )


def send_upload_shard_failure_message(shard_count, e):
  notifier.send(
    body={
      "embeds": [
        {
          "title": f"❌ ViVid Instruct 65k: Failed to upload shard {shard_count}!",
          "description": f"Failed to upload shard {shard_count} with {len(os.listdir(f'{BASE_DATA_PATH}/videos/shard_{shard_count}'))} clips at https://huggingface.co/datasets/Vividbot/vividbot_video.",
          "color": 16711680,
          "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
          "fields": [
            {
              "name": "Error",
              "value": str(e),
            }
          ],
        }
      ]
    }
  )


def send_upload_metadata_success_message(shard_count):
  notifier.send(
    body={
      "embeds": [
        {
          "title": f"✅ ViVid Instruct 65k: Uploaded metadata for shard shard_{shard_count}!",
          "description": f"Uploaded metadata for shard shard_{shard_count} at https://huggingface.co/datasets/Vividbot/vividbot_video.",
          "color": 2278494,
          "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }
      ]
    }
  )


def send_upload_metadata_failure_message(shard_count, e):
  notifier.send(
    body={
      "embeds": [
        {
          "title": f"❌ ViVid Instruct 65k: Failed to upload metadata for shard shard_{shard_count}!",
          "description": f"Failed to upload metadata for shard shard_{shard_count} at https://huggingface.co/datasets/Vividbot/vividbot_video.",
          "color": 16711680,
          "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
          "fields": [
            {
              "name": "Error",
              "value": str(e),
            }
          ],
        }
      ]
    }
  )


def _download(batch: dict, path: str):
  error_list = {"url_error": []}

  for clip in batch.items():
    start = clip["start"]
    end = clip["end"]
    clip_id = clip["id"]
    video_id = clip_id.split(".")[0]
    try:
      downloader.process(clip_id=clip_id, start=start, end=end, path=path)
    except DownloadError:
      error_list["url_error"].append(video_id)
  return error_list


def prepare():
  os.makedirs(f"{BASE_DATA_PATH}/videos", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/metadata", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/error", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/video", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/metadata", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/cache", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/cache/temp", exist_ok=True)


def download(executor: Executor):
  name = os.path.basename(executor.file_path).split(".")[0]
  path_out_chunks = f"{BASE_DATA_PATH}/output/{name}"

  executor.process(
    map_fn=_download,
    task="download",
    batch_size=16,
    num_proc=os.cpu_count(),
    save=True,
    remove_columns=[],
    name_out=f"error/error_{name}",
    fn_kwargs={"path": path_out_chunks},
  )

  uploader.zip_and_upload_dir(
    dir_path=path_out_chunks,
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"video/{name}.zip",
    repo_type="dataset",
    overwrite=True,
  )
  uploader.upload_file(
    file_path=f"{BASE_DATA_PATH}/output/error/error_{name}.json",
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"error/error_{name}.json",
    repo_type="dataset",
    overwrite=True,
  )


def main():
  prepare()

  executor = Executor(
    file_path=f"{BASE_DATA_PATH}/vivid_instruct_65k_result.jsonl",
    cache_dir=f"{BASE_DATA_PATH}/cache",
    output_dir=f"{BASE_DATA_PATH}/output",
    num_shards=500,
  )

  download(executor)


if __name__ == "__main__":
  main()

# for video in tqdm(videos):
#   print(f"Uploading video {video} to Gemini...")
#   video_path = f"{BASE_DATA_PATH}/videos/shard_{shard_count}/{video}"
#   video_file = genai.upload_file(path=video_path)

#   print(f'Generating finetuning data for video "{video}"...', end="")
#   while video_file.state.name == "PROCESSING":
#     print(".", end="")
#     time.sleep(10)
#     video_file = genai.get_file(video_file.name)

#   if video_file.state.name == "FAILED":
#     print(ValueError(video_file.state.name))
#     with open(f"{BASE_DATA_PATH}/generation_error_ids.txt", "a") as f:
#       f.write(video[:-4] + "\n")
#   elif video_file.state.name == "ACTIVE":
#     try:
#       describer = genai.GenerativeModel(
#         "models/gemini-1.5-flash",
#         generation_config={
#           "temperature": 0.1,
#         },
#       )
#       describer_response = describer.generate_content(
#         [video_file, DESCRIBE_VIDEO_PROMPT],
#       )

#       qa_generator = genai.GenerativeModel(
#         "gemini-1.5-flash",
#         generation_config={
#           "response_mime_type": "application/json",
#           "temperature": 1,
#         },
#       )
#       full_prompt = f"""{GENERATE_QA_PROMPT}

#   VIDEO CONTENT: {describer_response.text.strip()}"""

#       qa_generator_response = qa_generator.generate_content(full_prompt)
#       qa_pairs = json.loads(qa_generator_response.text)
#       conversations = []

#       for qa in qa_pairs:
#         rand_num = np.random.random()
#         human_value = qa["question"]
#         if rand_num < 0.5:
#           human_value += "\n<video>"
#         else:
#           human_value = "<video>\n" + human_value
#         conversations.append({"from": "human", "value": human_value})
#         conversations.append({"from": "gpt", "value": qa["answer"]})

#       dataset = {
#         "id": video[:-4],
#         "video": f"shard_{shard_count}/{video}",
#         "description": describer_response.text.strip(),
#         "conversations": conversations,
#       }

#       with open(
#         f"{BASE_DATA_PATH}/metadata/shard_{shard_count}.json",
#         "a",
#       ) as f:
#         f.write(
#           json.dumps(
#             dataset,
#             ensure_ascii=False,
#           )
#           + "\n"
#         )

#       print(f"Generated finetuning data for video {video}.")
#     except Exception as e:
#       print(str(e))
#       with open(f"{BASE_DATA_PATH}/generation_error_ids.txt", "a") as f:
#         f.write(video[:-4] + "\n")

#       raise e
