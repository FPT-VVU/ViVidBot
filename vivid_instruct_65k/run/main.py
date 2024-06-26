import json
import os
import time
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

BASE_DATA_PATH = "data"

FILES_NAME_DICT = {
    "YOUTUBE_SEARCH_RESULTS": "youtube_search_results.json",
    "RANDOM_DURATIONS": "random_durations.json",
}
RANDOM_DURATIONS = json.load(
    open(f"{BASE_DATA_PATH}/{FILES_NAME_DICT['RANDOM_DURATIONS']}")
)
YOUTUBE_SEARCH_RESULTS = json.load(
    open(f"{BASE_DATA_PATH}/{FILES_NAME_DICT['YOUTUBE_SEARCH_RESULTS']}")
)

DESCRIBE_VIDEO_PROMPT = "Describe only the visual content of the video without using its audio or transcript so that a person without vision can fully understand it. Remember to use Vietnamese language to describe the video."
GENERATE_QA_PROMPT = """Generate 5 different pairs of questions and answers based on the description of the video. The questions should be relevant to the video content and the answers should be correct. Also, diversify the types of questions and answers as much as possible.
Remember to use Vietnamese language to generate the questions and answers.
Some examples of questions:
- What is the video about? What is the action at the second n?
- What is the color of the object in the video?
- What is the person in the video doing?
- What is the object in the video?
- What is the person in the video wearing?
- What is the person in the video holding?
- What is the person in the video saying?
And more questions like that.
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

SHARD_COUNT = -1
TOTAL_CLIP_COUNT = 0

def download_videos():
    random_durations_index = 0

    for category in YOUTUBE_SEARCH_RESULTS.keys():
        for video_id in YOUTUBE_SEARCH_RESULTS[category]:
            if random_durations_index >= len(RANDOM_DURATIONS):
                break
            # increase SHARD_COUNT every 5000 clips
            if TOTAL_CLIP_COUNT % 5000 == 0:
                SHARD_COUNT += 1
                os.makedirs(f"{BASE_DATA_PATH}/videos/shard_{SHARD_COUNT}", exist_ok=True)
                
            with YoutubeDL(params={"format": "best[ext=mp4]", "quiet": True}) as ydl:
                # ydl.download(f"https://www.youtube.com/watch?v={video_id}")

                video_info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
                video_duration = video_info["duration"]
                start = 0
                clip_count = 0

                while start + RANDOM_DURATIONS[random_durations_index] < video_duration:
                    end = start + RANDOM_DURATIONS[random_durations_index]

                    if not os.path.exists(
                        f"{BASE_DATA_PATH}/videos/shard_{SHARD_COUNT}/{video_id}.{clip_count}.mp4"
                    ):
                        with YoutubeDL(
                            params={
                                "format": "best[ext=mp4]",
                                "quiet": True,
                                "outtmpl": f"{BASE_DATA_PATH}/videos/{video_id}.{clip_count}.%(ext)s",
                                "download_ranges": download_range_func(
                                    None, [(start, end)]
                                ),
                            }
                        ) as ydl2:
                            ydl2.download(
                                f"https://www.youtube.com/watch?v={video_id}",
                            )
                            
                            start = end
                            random_durations_index += 1
                            clip_count += 1
                            TOTAL_CLIP_COUNT += 1


def generate_finetuning_data():
    videos = os.listdir(f"{BASE_DATA_PATH}/videos")
    data_dict = {}

    for video in videos:
        video_path = f"{BASE_DATA_PATH}/videos/{video}"
        video_file = genai.upload_file(path=video_path)

        print(f'Generating finetuning data for video "{video}"...', end="")
        while video_file.state.name == "PROCESSING":
            print(".", end="")
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            print(ValueError(video_file.state.name))
        elif video_file.state.name == "ACTIVE":
            describer = genai.GenerativeModel(
                "models/gemini-1.5-flash",
                generation_config={
                    "temperature": 0.1,
                },
            )
            describer_response = describer.generate_content(
                [video_file, DESCRIBE_VIDEO_PROMPT],
            )

            print(describer_response.text)

            qa_generator = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 1,
                },
            )
            full_prompt = f"""{GENERATE_QA_PROMPT}

VIDEO CONTENT: {describer_response.text}"""

            qa_generator_response = qa_generator.generate_content(full_prompt)

            data_dict[video] = {
                "description": describer_response.text,
                "qa_pairs": json.loads(qa_generator_response.text),
            }

def prepare():
    os.makedirs(f"{BASE_DATA_PATH}/videos", exist_ok=True)
    
def main():
    prepare()
    download_videos()
    generate_finetuning_data()


if __name__ == "__main__":
    main()
