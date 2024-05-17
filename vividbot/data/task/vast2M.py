import os
import sys
import argparse
import json


sys.path.append(os.getcwd())
# turn of warning 
import warnings
warnings.filterwarnings("ignore")

from yt_dlp.utils import DownloadError
from datasets import disable_progress_bar

from vividbot.data.processor.translator import GGTranslator
from vividbot.data.processor.question_selection import QuestionSelection
from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.upload_hf import Uploader
from vividbot.data.processor.executor import Executor

from huggingface_hub import HfApi, HfFolder, HfFileSystem

def parse_args() -> argparse.Namespace:
    """
    Get arguments from command line.
    :return:    Arguments from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Available tasks: download, generate",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        required=True,
        help="file path to load data. Note: just support json file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save data.",
    )

    parser.add_argument(
        "--select",
        type=int,
        default=-1,
        help="number of process sample",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number batch_size for process sample",
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="number processing",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="repo id for upload to hungingface",
    )
    parser.add_argument(
        "--num-shards",
        default=-1,
        type=int,
        help="number of shard for generate data",
    )
    parser.add_argument(
        "--upload-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload to hub after processing.",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove all output files except for metadata after processing.",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(os.getcwd(), ".cache"),
        help="Cache directory",
    )
    return parser.parse_args()

question_list = QuestionSelection("vividbot/data/stuff/questions.txt")
translator = GGTranslator()
uploader = Uploader()
downloader = YoutubeDownloader()
error_list = []

def generate(batch: dict):
    result_translate = translator.process(batch["vast_cap"], src="en", dest="vi") 
    result_question = question_list.process(len(batch["vast_cap"]))
    new_result = {"id" : batch["clip_id"],
                    "video": [item + ".mp4" for item in batch["clip_id"]],
                    "conversation": [[{"from": "human", "value" : question}, {"from" : "gpt", "value": answer}] 
                                    for question, answer in zip(result_question, result_translate)]}
    return new_result
def download(batch: dict, path: str, upload_to_hub: bool, repo_id: str, clean_output: bool):
    error_list = {"url_error": []}
    for url_id, span in zip(batch["clip_id"], batch["clip_span"]):
        try:
            downloader.process(url_id, span[0], span[1], path)
            if upload_to_hub:
                uploader.upload_file(file_path=f"{path}/{url_id}.mp4", 
                                    repo_id=repo_id, 
                                    path_in_repo=f"video/{url_id}.mp4", 
                                    repo_type="dataset",
                                    overwrite=False)
            if clean_output:
                os.remove(f"{path}/{url_id}.mp4")
        except DownloadError:
            error_list["url_error"].append(url_id)
    return error_list
def change_format(batch, **kwargs):
    pass

def remove_sample(batch, **kwargs):
    pass

def main(args: argparse.Namespace):
    executor = Executor(file_path=args.file_path,
                        cache_dir=args.cache_dir,
                        output_dir=args.output_dir,
                        select=args.select,
                        num_shards=args.num_shards)
    if args.task == "generate":
        executor.process(map_fn=generate,
                         task=args.task,
                         batch_size=args.batch_size, 
                         num_proc=args.num_proc,
                         name_out="test.json",
                         save=True,
                         remove_columns=['clip_id', 'clip_span', 'url', 'vision_cap', 'audio_cap', 'subtitle', 'vast_cap'])
    if args.task == "download":
        executor.process(map_fn=download,
                         task=args.task,
                         batch_size=args.batch_size,
                         num_proc=args.num_proc,
                         save=True,
                         remove_columns=['clip_id', 'clip_span', 'url', 'vision_cap', 'audio_cap', 'subtitle', 'vast_cap'],
                         name_out="error.json",
                         fn_kwargs={"path": args.output_dir, "repo_id": args.repo_id, "upload_to_hub": args.upload_to_hub, "clean_output": args.clean_output})
if __name__ == '__main__':
    main(parse_args())