import os
import shutil
import sys

sys.path.append(os.getcwd())


import datetime
import json
import time
from datetime import timezone

import google.generativeai as genai
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, download_range_func

from vividbot.data.discord.discord import DiscordNotifier
from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.upload_hf import Uploader

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_DATA_PATH = "/root/data"
RANDOM_DURATIONS = json.load(open("/root/ViVidBot/random_durations.json"))
DESCRIBE_VIDEO_PROMPT = "Describe only the visual content of the video without using its audio or transcript so that a person without vision can fully understand it. Remember to use Vietnamese language to describe the video."
GENERATE_QA_PROMPT = """Generate 5 different pairs of questions and answers based on the description of the video. The questions should be relevant to the video content and the answers should be correct. Also, diversify the types of questions and answers as much as possible.
Remember to use Vietnamese language to generate the questions and answers.
Some examples of questions:
- What is the video about?
- What are key points in the video?
- What is the color of the object in the video?
- What is the person in the video doing?
- What is the object in the video?
- What is the person in the video holding?
- How does the person in the video look?
- Where is the object in the video?
- What is the position of the object in the video?
And more questions that can be asked about the video content.
Question length and complexity should be varied.
Return the questions and answers in the following format:
[
    {
        "question": "Question 1",
        "answer": "Answer 1"
    },
    {
        "question": "Question 2",
        "answer": "Answer 2"
    },
]
"""

genai.configure(api_key=GOOGLE_API_KEY)
notifier = DiscordNotifier(
    "https://discord.com/api/webhooks/1255505460040040508/n-QCTqNgp3RrsNc1hBRnXH4dfOejeH8iPTd8lqGevbSb_wAovD4xxv5ZVkVJBfVLF8vN"
)
uploader = Uploader()
downloader = YoutubeDownloader()


def download():
    shard_count = 0
    total_clip_count = 0
    random_durations_index = 0
    data = load_dataset(
        "json",
        data_files="vivid_instruct_65k_unprocessed.jsonl",
    )

    while os.path.exists(f"{BASE_DATA_PATH}/videos/shard_{shard_count}"):
        shard_count += 1
        total_clip_count += len(
            os.listdir(f"{BASE_DATA_PATH}/videos/shard_{shard_count}")
        )

    for row in tqdm(data["train"]):
        video_id = row["id"]
        video_duration = row["duration"]

        start = 0
        clip_count = 0

        # increase shard_count every 5000 clips
        if (
            os.path.exists(f"{BASE_DATA_PATH}/videos/shard_{shard_count}")
            and len(os.listdir(f"{BASE_DATA_PATH}/videos/shard_{shard_count}")) >= 10
        ):
            try:
                uploader = Uploader()
                uploader.zip_and_upload_dir(
                    f"{BASE_DATA_PATH}/videos/shard_{shard_count}",
                    "Vividbot/vividbot_video",
                    f"videos/shard_{shard_count}.zip",
                    overwrite=True,
                )

                notifier.send(
                    body={
                        "embeds": [
                            {
                                "title": f"✅ ViVid Instruct 65k: Uploaded shard {shard_count}!",
                                "description": f"Uploaded shard {shard_count} with {total_clip_count} clips at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                                "color": 2278494,
                                "timestamp": datetime.datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            }
                        ]
                    }
                )

            except Exception as e:
                notifier.send(
                    body={
                        "embeds": [
                            {
                                "title": f"❌ ViVid Instruct 65k: Failed to upload shard {shard_count}!",
                                "description": f"Failed to upload shard {shard_count} with {total_clip_count} clips at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                                "color": 16711680,
                                "timestamp": datetime.datetime.now(
                                    timezone.utc
                                ).isoformat(),
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
                raise e

            videos = os.listdir(f"{BASE_DATA_PATH}/videos/shard_{shard_count}")
            for video in tqdm(videos):
                print(f"Uploading video {video} to Gemini...")
                video_path = f"{BASE_DATA_PATH}/videos/shard_{shard_count}/{video}"
                video_file = genai.upload_file(path=video_path)

                print(f'Generating finetuning data for video "{video}"...', end="")
                while video_file.state.name == "PROCESSING":
                    print(".", end="")
                    time.sleep(10)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    print(ValueError(video_file.state.name))
                    with open(f"{BASE_DATA_PATH}/generation_error_ids.txt", "a") as f:
                        f.write(video[:-4] + "\n")
                elif video_file.state.name == "ACTIVE":
                    try:
                        describer = genai.GenerativeModel(
                            "models/gemini-1.5-flash",
                            generation_config={
                                "temperature": 0.1,
                            },
                        )
                        describer_response = describer.generate_content(
                            [video_file, DESCRIBE_VIDEO_PROMPT],
                        )

                        qa_generator = genai.GenerativeModel(
                            "gemini-1.5-flash",
                            generation_config={
                                "response_mime_type": "application/json",
                                "temperature": 1,
                            },
                        )
                        full_prompt = f"""{GENERATE_QA_PROMPT}

VIDEO CONTENT: {describer_response.text.strip()}"""

                        qa_generator_response = qa_generator.generate_content(
                            full_prompt
                        )
                        qa_pairs = json.loads(qa_generator_response.text)
                        conversations = []

                        for qa in qa_pairs:
                            conversations.append(
                                {"from": "human", "value": qa["question"]}
                            )
                            conversations.append({"from": "gpt", "value": qa["answer"]})

                        data = {
                            "id": video[:-4],
                            "video": f"shard_{shard_count}/{video}",
                            "description": describer_response.text.strip(),
                            "conversations": conversations,
                        }

                        with open(
                            f"{BASE_DATA_PATH}/metadata/shard_{shard_count}.json",
                            "a",
                        ) as f:
                            f.write(
                                json.dumps(
                                    data,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                        print(f"Generated finetuning data for video {video}.")
                        shutil.rmtree(video_path)
                    except Exception as e:
                        print(str(e))
                        with open(
                            f"{BASE_DATA_PATH}/generation_error_ids.txt", "a"
                        ) as f:
                            f.write(video[:-4] + "\n")

            try:
                print(
                    f"Uploading metadata for shard shard_{shard_count} to Hugging Face..."
                )

                uploader = Uploader()
                uploader.upload_file(
                    file_path=f"{BASE_DATA_PATH}/metadata/shard_{shard_count}.json",
                    repo_id="Vividbot/vividbot_video",
                    path_in_repo=f"metadata/shard_{shard_count}.json",
                    repo_type="dataset",
                    overwrite=True,
                )

                notifier.send(
                    body={
                        "embeds": [
                            {
                                "title": f"✅ ViVid Instruct 65k: Uploaded metadata for shard shard_{shard_count}!",
                                "description": f"Uploaded metadata for shard shard_{shard_count} at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                                "color": 2278494,
                                "timestamp": datetime.datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            }
                        ]
                    }
                )
            except Exception as e:
                print(str(e))
                notifier.send(
                    body={
                        "embeds": [
                            {
                                "title": f"❌ ViVid Instruct 65k: Failed to upload metadata for shard shard_{shard_count}!",
                                "description": f"Failed to upload metadata for shard shard_{shard_count} at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                                "color": 16711680,
                                "timestamp": datetime.datetime.now(
                                    timezone.utc
                                ).isoformat(),
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

                raise e

        shard_count += 1
        os.makedirs(
            f"{BASE_DATA_PATH}/videos/shard_{shard_count}",
            exist_ok=True,
        )

        if random_durations_index >= len(RANDOM_DURATIONS):
            break

        while start + RANDOM_DURATIONS[random_durations_index] < video_duration:
            end = start + RANDOM_DURATIONS[random_durations_index]

            if not os.path.exists(
                f"{BASE_DATA_PATH}/videos/shard_{shard_count}/{video_id}.{clip_count}.mp4"
            ):
                try:
                    print(
                        f"Downloading clip {clip_count} of video {video_id} starting at {start} and ending at {end}..."
                    )

                    with YoutubeDL(
                        params={
                            "format": "best[ext=mp4]",
                            "quiet": True,
                            "outtmpl": f"{BASE_DATA_PATH}/videos/shard_{shard_count}/{video_id}.{clip_count}.mp4",
                            "download_ranges": download_range_func(
                                None, [(start, end)]
                            ),
                            "fixup": "never",
                            "no_warnings": True,
                            "force_keyframes_at_cuts": True,
                            "downloader": "aria2c",
                        }
                    ) as ydl2:
                        ydl2.download(
                            f"https://www.youtube.com/watch?v={video_id}",
                        )
                    # downloader.process(url_id, start, end, path)

                except DownloadError:
                    print(f"Failed to download clip {clip_count} of video {video_id}.")
                    with open(f"{BASE_DATA_PATH}/download_error_ids.txt", "a") as f:
                        f.write(video_id + "\n")

            start = end
            random_durations_index += 1
            clip_count += 1
            total_clip_count += 1

            if random_durations_index >= len(RANDOM_DURATIONS):
                break

    # upload error ids
    if os.path.exists(f"{BASE_DATA_PATH}/download_error_ids.txt"):
        uploader = Uploader()
        uploader.upload_file(
            file_path=f"{BASE_DATA_PATH}/download_error_ids.txt",
            repo_id="Vividbot/vividbot_video",
            path_in_repo="download_error_ids.txt",
            repo_type="dataset",
            overwrite=True,
        )

    if os.path.exists(f"{BASE_DATA_PATH}/generation_error_ids.txt"):
        uploader = Uploader()
        uploader.upload_file(
            file_path=f"{BASE_DATA_PATH}/generation_error_ids.txt",
            repo_id="Vividbot/vividbot_video",
            path_in_repo="generation_error_ids.txt",
            repo_type="dataset",
            overwrite=True,
        )

    # upload the last shard
    if os.path.exists(f"{BASE_DATA_PATH}/videos/shard_{shard_count}"):
        try:
            uploader = Uploader()
            uploader.zip_and_upload_dir(
                f"{BASE_DATA_PATH}/videos/shard_{shard_count}",
                "Vividbot/vividbot_video",
                f"videos/shard_{shard_count}.zip",
                overwrite=True,
            )
            notifier.send(
                body={
                    "embeds": [
                        {
                            "title": f"✅ ViVid Instruct 65k: Uploaded shard {shard_count}!",
                            "description": f"Uploaded shard {shard_count} with {total_clip_count} clips at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                            "color": 2278494,
                            "timestamp": datetime.datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    ]
                }
            )
        except Exception as e:
            notifier.send(
                body={
                    "embeds": [
                        {
                            "title": f"❌ ViVid Instruct 65k: Failed to upload shard {shard_count}!",
                            "description": f"Failed to upload shard {shard_count} with {total_clip_count} clips at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                            "color": 16711680,
                            "timestamp": datetime.datetime.now(
                                timezone.utc
                            ).isoformat(),
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

            raise e

        videos = os.listdir(f"{BASE_DATA_PATH}/videos/shard_{shard_count}")
        for video in tqdm(videos):
            print(f"Uploading video {video} to Gemini...")
            video_path = f"{BASE_DATA_PATH}/videos/shard_{shard_count}/{video}"
            video_file = genai.upload_file(path=video_path)

            print(f'Generating finetuning data for video "{video}"...', end="")
            while video_file.state.name == "PROCESSING":
                print(".", end="")
                time.sleep(10)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                print(ValueError(video_file.state.name))
                with open(f"{BASE_DATA_PATH}/generation_error_ids.txt", "a") as f:
                    f.write(video[:-4] + "\n")
            elif video_file.state.name == "ACTIVE":
                try:
                    describer = genai.GenerativeModel(
                        "models/gemini-1.5-flash",
                        generation_config={
                            "temperature": 0.1,
                        },
                    )
                    describer_response = describer.generate_content(
                        [video_file, DESCRIBE_VIDEO_PROMPT],
                    )

                    qa_generator = genai.GenerativeModel(
                        "gemini-1.5-flash",
                        generation_config={
                            "response_mime_type": "application/json",
                            "temperature": 1,
                        },
                    )
                    full_prompt = f"""{GENERATE_QA_PROMPT}

VIDEO CONTENT: {describer_response.text.strip()}"""

                    qa_generator_response = qa_generator.generate_content(full_prompt)
                    qa_pairs = json.loads(qa_generator_response.text)
                    conversations = []

                    for qa in qa_pairs:
                        conversations.append({"from": "human", "value": qa["question"]})
                        conversations.append({"from": "gpt", "value": qa["answer"]})

                    data = {
                        "id": video[:-4],
                        "video": f"shard_{shard_count}/{video}",
                        "description": describer_response.text.strip(),
                        "conversations": conversations,
                    }

                    with open(
                        f"{BASE_DATA_PATH}/metadata/shard_{shard_count}.json",
                        "a",
                    ) as f:
                        f.write(
                            json.dumps(
                                data,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    print(f"Generated finetuning data for video {video}.")
                    os.remove(video_path)
                except Exception as e:
                    print(str(e))
                    with open(f"{BASE_DATA_PATH}/generation_error_ids.txt", "a") as f:
                        f.write(video[:-4] + "\n")

                    raise e

        try:
            print(
                f"Uploading metadata for shard shard_{shard_count} to Hugging Face..."
            )

            uploader = Uploader()
            uploader.upload_file(
                file_path=f"{BASE_DATA_PATH}/metadata/shard_{shard_count}.json",
                repo_id="Vividbot/vividbot_video",
                path_in_repo=f"metadata/shard_{shard_count}.json",
                repo_type="dataset",
                overwrite=True,
            )

            notifier.send(
                body={
                    "embeds": [
                        {
                            "title": f"✅ ViVid Instruct 65k: Uploaded metadata for shard shard_{shard_count}!",
                            "description": f"Uploaded metadata for shard shard_{shard_count} at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                            "color": 2278494,
                            "timestamp": datetime.datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    ]
                }
            )
        except Exception as e:
            print(str(e))
            notifier.send(
                body={
                    "embeds": [
                        {
                            "title": f"❌ ViVid Instruct 65k: Failed to upload metadata for shard shard_{shard_count}!",
                            "description": f"Failed to upload metadata for shard shard_{shard_count} at https://huggingface.co/datasets/Vividbot/vividbot_video.",
                            "color": 16711680,
                            "timestamp": datetime.datetime.now(
                                timezone.utc
                            ).isoformat(),
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

            raise e


def prepare():
    os.makedirs(f"{BASE_DATA_PATH}/videos", exist_ok=True)
    os.makedirs(f"{BASE_DATA_PATH}/metadata", exist_ok=True)


def main():
    prepare()
    download()


if __name__ == "__main__":
    main()
