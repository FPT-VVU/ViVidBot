import json
import logging
import os
import shutil
import sys
import time

from tqdm import tqdm

sys.path.append(os.getcwd())
import datetime
from datetime import timezone
from pathlib import Path

import anthropic
import google.generativeai as genai
import numpy as np
import openai
from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq
from yt_dlp.utils import DownloadError

from vividbot.data.discord.discord import DiscordNotifier
from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.upload_hf import Uploader

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_DATA_PATH = f"{Path.home()}/data"
DESCRIBE_VIDEO_PROMPT = """Describe only the visual content of the video without using the audio or transcript so that a normal people can interpret what is happening in the video.
Don't use the audio or transcript of the video to describe the video content. Use only the visual content.
Remember to use Vietnamese language to describe the video."""
GENERATE_QA_PROMPT = """Generate 5 different pairs of questions and answers in JSON format based on the description of the video (in which the description is generated for person without vision can understand the video content).
The questions should be relevant to the video content and the answers should be correct.
Also, diversify the types of questions and answers as much as possible.
Remember to use Vietnamese language to generate the questions and answers.
Examples of questions (do not need to follow the order and these are just examples, you must generate your own questions based on the video content):
- What's the video about?
- What are key points in the video?
- What is the color of the object?
- What is the person doing?
- What is the person in the video holding and what are its characteristics?
- How does the person in the video look?
- What is the position of the object in the video?
And more questions that can be asked about the video content (what, where, when, why, how, etc.) with varying levels of complexity.
All questions should be relevant to the video content and the answers should be FULLY informative and correct. The answer should be a complete sentence or a complete phrase.
Only return the list of pair of questions and answers in the following JSON format:
[{"question":"Q1","answer":"A1"},{"question":"Q2","answer":"A2"},...]
Your response should be only the JSON list without narrative or additional information.
"""

logger = logging.getLogger(__name__)
logging.basicConfig(
  filename="/root/ViVidBot/run.log",
  filemode="a",
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

genai.configure(api_key=GOOGLE_API_KEY)
groq_client = Groq(
  api_key=os.getenv("GROQ_API_KEY"),
  max_retries=0,
)
together_client = openai.OpenAI(
  base_url="https://api.together.xyz/v1",
  api_key=os.getenv("TOGETHER_API_KEY"),
  max_retries=0,
)
anthropic_client = anthropic.Anthropic(
  api_key=os.getenv("ANTHROPIC_API_KEY"),
)
notifier = DiscordNotifier(
  "https://discord.com/api/webhooks/1255505460040040508/n-QCTqNgp3RrsNc1hBRnXH4dfOejeH8iPTd8lqGevbSb_wAovD4xxv5ZVkVJBfVLF8vN"
)
uploader = Uploader()
downloader = YoutubeDownloader()


def process_response_content(response_content: str) -> str:
  # the response content may not begin with a list, so we need to find the first list
  response_content = response_content.strip()
  response_content = response_content[response_content.index("[") :]
  return response_content


def send_process_shard_success_message(shard_count, duration):
  logger.info(f"Processed shard {shard_count} in {duration}(s).")
  notifier.send(
    body={
      "embeds": [
        {
          "title": f"✅ ViVid Instruct 65k: Processed shard {shard_count}!",
          "description": f"Processed shard {shard_count} \
of {len(os.listdir(f'{BASE_DATA_PATH}/output/videos/shard_{shard_count}'))} clips \
in {duration}(s). \
Visit at https://huggingface.co/datasets/Vividbot/vividbot_video/tree/main/videos.",
          "color": 2278494,
          "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }
      ]
    }
  )


def _process(batch: dict):
  for start, end, video_id_with_chunk_id, shard_id in tqdm(
    zip(batch["start"], batch["end"], batch["id"], batch["shard_id"])
  ):
    video_id, chunk_id = video_id_with_chunk_id.split(".")

    try:
      if not os.path.exists(
        f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
      ) and not uploader.check_file_exists(
        repo_id="Vividbot/vividbot_video",
        path_in_repo=f"videos/shard_{shard_id}.zip",
        repo_type="dataset",
      ):
        logger.info(f"Downloading video {video_id_with_chunk_id}...")
        downloader.process(
          video_id=video_id,
          video_id_with_chunk_id=video_id_with_chunk_id,
          start=start,
          end=end,
          path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}",
        )

    except DownloadError as e:
      logger.error(f"Error downloading video {video_id_with_chunk_id}: {e}")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {"id": video_id_with_chunk_id, "reason": str(e)}
        f.write(json.dumps(data) + "\n")

    try:
      if not uploader.check_file_exists(
        repo_id="Vividbot/vividbot_video",
        path_in_repo=f"metadata/shard_{shard_id}.jsonl",
        repo_type="dataset",
      ):
        logger.info(f"Generating metadata for video {video_id_with_chunk_id}...")
        if not os.path.exists(
          f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
        ):
          logger.info(f"Video {video_id_with_chunk_id} not found. Skipping...")
          continue

        video_file = None
        google_file_name = f"files/{shard_id}-{video_id}-{chunk_id}".lower().replace(
          "_", "-"
        )
        try:
          video_file = genai.get_file(name=google_file_name)
        except Exception as e:
          logger.error(f"Error getting video file {google_file_name}: {e}")

        if video_file is None or not video_file.state.name == "ACTIVE":
          video_file = genai.upload_file(
            path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4",
            name=google_file_name,
            display_name=video_id_with_chunk_id,
          )

        while video_file.state.name == "PROCESSING":
          time.sleep(5)
          video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
          logger.error(
            f"Error uploading video {video_id_with_chunk_id}: {video_file.error}"
          )
          with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
            data = {
              "id": video_id_with_chunk_id,
              "reason": f"Error uploading video: {video_file.error}",
            }
            f.write(json.dumps(data) + "\n")
        elif video_file.state.name == "ACTIVE":
          describer = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            generation_config={
              "temperature": 0,
              "max_output_tokens": 512,
            },
          )
          describer_response = describer.generate_content(
            [video_file, DESCRIBE_VIDEO_PROMPT],
            request_options={"timeout": 30, "retry": 2},
          )

          try:
            logger.info(
              f"Generating QA pairs for video {video_id_with_chunk_id} with Groq..."
            )
            chat_completion = groq_client.chat.completions.create(
              messages=[
                {
                  "role": "system",
                  "content": GENERATE_QA_PROMPT,
                },
                {
                  "role": "user",
                  "content": f"{describer_response.text.strip()}",
                },
              ],
              model="llama3-70b-8192",
              temperature=1,
              stream=False,
              max_tokens=8192,
            )
            response_content = process_response_content(
              chat_completion.choices[0].message.content
            )
            logger.info(
              f"Groq response for video {video_id_with_chunk_id}: {response_content}"
            )
            qa_pairs = json.loads(response_content)
            conversations = []

            for qa in qa_pairs:
              rand_num = np.random.random()
              human_value = qa["question"]
              gpt_value = qa["answer"]
              if rand_num < 0.5:
                human_value = human_value + "\n<video>"
              else:
                human_value = "<video>\n" + human_value
              conversations.append({"from": "human", "value": human_value})
              conversations.append({"from": "gpt", "value": gpt_value})

            data = {
              "id": video_id_with_chunk_id,
              "video": f"shard_{shard_id}/{video_id_with_chunk_id}.mp4",
              "generator": "groq/llama3-70b-8192",
              "description": describer_response.text.strip(),
              "conversations": conversations,
            }

            with open(
              f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
              "a",
            ) as f:
              f.write(
                json.dumps(
                  data,
                  ensure_ascii=False,
                )
                + "\n"
              )

          except Exception as e:
            logger.error(
              f"Couldn't generate QA pairs for video {video_id_with_chunk_id}: {str(e)}. Retrying with Together..."
            )
            try:
              response = together_client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[
                  {
                    "role": "system",
                    "content": GENERATE_QA_PROMPT,
                  },
                  {
                    "role": "user",
                    "content": f"{describer_response.text.strip()}",
                  },
                ],
                temperature=1,
                stream=False,
                max_tokens=4096,
              )
              response_content = process_response_content(
                response.choices[0].message.content
              )
              logger.info(
                f"Together response for video {video_id_with_chunk_id}: {response_content}"
              )
              qa_pairs = json.loads(response_content)
              conversations = []

              for qa in qa_pairs:
                rand_num = np.random.random()
                human_value = qa["question"]
                gpt_value = qa["answer"]
                if rand_num < 0.5:
                  human_value = human_value + "\n<video>"
                else:
                  human_value = "<video>\n" + human_value
                conversations.append({"from": "human", "value": human_value})
                conversations.append({"from": "gpt", "value": gpt_value})

              data = {
                "id": video_id_with_chunk_id,
                "video": f"shard_{shard_id}/{video_id_with_chunk_id}.mp4",
                "generator": "together/meta-llama/Llama-3-70b-chat-hf",
                "description": describer_response.text.strip(),
                "conversations": conversations,
              }

              with open(
                f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
                "a",
              ) as f:
                f.write(
                  json.dumps(
                    data,
                    ensure_ascii=False,
                  )
                  + "\n"
                )

            except Exception as e:
              logger.error(
                f"Couldn't generate QA pairs for video {video_id_with_chunk_id}: {str(e)}. Retrying with Anthropic..."
              )
              try:
                message = anthropic_client.messages.create(
                  model="claude-3-haiku-20240307",
                  system=GENERATE_QA_PROMPT,
                  messages=[
                    {
                      "role": "user",
                      "content": f"{describer_response.text.strip()}",
                    },
                  ],
                  stream=False,
                  max_tokens=4096,
                  temperature=1,
                )
                response_content = message.content[0].text
                logger.info(
                  f"Anthropic response for video {video_id_with_chunk_id}: {response_content}"
                )
                qa_pairs = json.loads(response_content)
                conversations = []

                for qa in qa_pairs:
                  rand_num = np.random.random()
                  human_value = qa["question"]
                  gpt_value = qa["answer"]
                  if rand_num < 0.5:
                    human_value = human_value + "\n<video>"
                  else:
                    human_value = "<video>\n" + human_value
                  conversations.append({"from": "human", "value": human_value})
                  conversations.append({"from": "gpt", "value": gpt_value})

                data = {
                  "id": video_id_with_chunk_id,
                  "video": f"shard_{shard_id}/{video_id_with_chunk_id}.mp4",
                  "generator": "anthropic/claude-3-haiku-20240307",
                  "description": describer_response.text.strip(),
                  "conversations": conversations,
                }

                with open(
                  f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
                  "a",
                ) as f:
                  f.write(
                    json.dumps(
                      data,
                      ensure_ascii=False,
                    )
                    + "\n"
                  )
              except Exception as e:
                logger.error(
                  f"Couldn't generate QA pairs for video {video_id_with_chunk_id}: {str(e)}. Retrying with Gemini..."
                )
                qa_generator = genai.GenerativeModel(
                  "gemini-1.5-flash",
                  generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 1,
                  },
                  system_instruction=[GENERATE_QA_PROMPT],
                )
                full_prompt = f"{describer_response.text.strip()}"

                qa_generator_response = qa_generator.generate_content(
                  full_prompt,
                )
                qa_pairs = json.loads(qa_generator_response.text)
                conversations = []

                for qa in qa_pairs:
                  rand_num = np.random.random()
                  human_value = qa["question"]
                  gpt_value = qa["answer"]
                  if rand_num < 0.5:
                    human_value = human_value + "\n<video>"
                  else:
                    human_value = "<video>\n" + human_value
                  conversations.append({"from": "human", "value": human_value})
                  conversations.append({"from": "gpt", "value": gpt_value})

                data = {
                  "id": video_id_with_chunk_id,
                  "video": f"shard_{shard_id}/{video_id_with_chunk_id}.mp4",
                  "generator": "google/gemini-1.5-flash",
                  "description": describer_response.text.strip(),
                  "conversations": conversations,
                }

                with open(
                  f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
                  "a",
                ) as f:
                  f.write(
                    json.dumps(
                      data,
                      ensure_ascii=False,
                    )
                    + "\n"
                  )
    except Exception as e:
      logger.error(f"Error generating metadata for video {video_id_with_chunk_id}: {e}")
      with open(f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl", "a") as f:
        data = {"id": video_id_with_chunk_id, "reason": str(e)}
        f.write(json.dumps(data) + "\n")


def _delete_video(batch: dict):
  for video_id_with_chunk_id, shard_id in tqdm(zip(batch["id"], batch["shard_id"])):
    video_id, chunk_id = video_id_with_chunk_id.split(".")

    google_file_name = f"files/{shard_id}-{video_id}-{chunk_id}".lower().replace(
      "_", "-"
    )
    try:
      genai.delete_file(name=google_file_name)
    except Exception as e:
      logger.error(f"Error deleting video file {google_file_name}: {e}")

    try:
      os.remove(
        f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}/{video_id_with_chunk_id}.mp4"
      )
    except Exception as e:
      logger.error(f"Error deleting video {video_id_with_chunk_id}: {e}")


def process(shard: str):
  """
  shard: str = "shard_0.json"
  """
  start_time = time.time()

  shard_id = int(shard.split(".")[0].split("_")[1])

  logger.info(f"Processing videos for shard {shard_id}...")

  os.makedirs(f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}", exist_ok=True)

  dataset = load_dataset(
    "json", data_files=f"{BASE_DATA_PATH}/vivid_instruct_65k/{shard}"
  )["train"]

  dataset.map(
    _process,
    batched=True,
    batch_size=128,
    num_proc=os.cpu_count(),
  )

  if uploader.check_file_exists(
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"videos/shard_{shard_id}.zip",
    repo_type="dataset",
  ):
    logger.info(f"Shard {shard_id} already uploaded. Skipping...")
  else:
    uploader.zip_and_upload_dir(
      dir_path=f"{BASE_DATA_PATH}/output/videos/shard_{shard_id}",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"videos/shard_{shard_id}.zip",
      repo_type="dataset",
      overwrite=True,
    )

  uploader.upload_file(
    file_path=f"{BASE_DATA_PATH}/output/errors/shard_{shard_id}.jsonl",
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"errors/shard_{shard_id}.jsonl",
    repo_type="dataset",
    overwrite=True,
  )

  if uploader.check_file_exists(
    repo_id="Vividbot/vividbot_video",
    path_in_repo=f"metadata/shard_{shard_id}.jsonl",
    repo_type="dataset",
  ):
    logger.info(f"Metadata for shard {shard_id} already uploaded. Skipping...")
  else:
    uploader.upload_file(
      file_path=f"{BASE_DATA_PATH}/output/metadata/shard_{shard_id}.jsonl",
      repo_id="Vividbot/vividbot_video",
      path_in_repo=f"metadata/shard_{shard_id}.jsonl",
      repo_type="dataset",
      overwrite=True,
    )

  # remove video file from google cloud
  logger.info(f"Cleaning up shard {shard_id}...")
  try:
    dataset.map(
      _delete_video,
      batched=True,
      batch_size=16,
      num_proc=os.cpu_count(),
    )
  except Exception as e:
    logger.error(f"Error deleting videos for shard {shard_id}: {e}")
    pass

  end_time = time.time()
  duration = round(end_time - start_time, 2)

  send_process_shard_success_message(shard_id, duration)


def prepare():
  os.makedirs(f"{BASE_DATA_PATH}/output", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/videos", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/errors", exist_ok=True)
  os.makedirs(f"{BASE_DATA_PATH}/output/metadata", exist_ok=True)


def main():
  prepare()
  last_successful_shard = 9
  for shard in tqdm(
    sorted(
      os.listdir(f"{BASE_DATA_PATH}/vivid_instruct_65k"),
      key=lambda x: int(x.split(".")[0].split("_")[1]),
    )
  ):
    shard_id = int(shard.split(".")[0].split("_")[1])
    if shard_id <= last_successful_shard:
      continue
    process(shard)


if __name__ == "__main__":
  main()
