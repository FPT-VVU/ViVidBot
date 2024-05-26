import os
import sys
import argparse
import random
import time
import shutil


sys.path.append(os.getcwd())
# turn of warning
import warnings

warnings.filterwarnings("ignore")

from datasets import load_dataset

from vividbot.data.processor.translator import GGTranslator
from vividbot.data.processor.question_selection import QuestionSelection
from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.upload_hf import Uploader
from vividbot.data.processor.executor import Executor



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
        help="Available tasks: translate, rename column, remove sample, divide dataset",
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
        "--name-out",
        type=str,
        default=None,
        help="Name of output after processing.",
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
        default=1,
        type=int,
        help="number of shard for process data",
    )
    parser.add_argument(
        "--list-old-name",
        default=None,
        type=str,
        help='For task "rename column", please provide list old name that you want to rename as format "column1,column2"',
    )

    parser.add_argument(
        "--list-new-name",
        default=None,
        type=str,
        help='For task "rename column", please provide list new name that you want to rename to as format "column1,column2"',
    )

    parser.add_argument(
        "--error-file-path",
        default="",
        type=str,
        help='For task "remove sample", please provide error file path (json file or folder contains json files)',
    )

    parser.add_argument(
        "--upload-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload to hub after processing.",
    )

    parser.add_argument(
        "--clean-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove all input files after processing.",
    )

    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove all output files after processing.",
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing files in huggingface hub.",
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


def _translate(batch):
    result_question = question_list.process(len(batch["conversations"]))
    for item, question in zip(batch["conversations"], result_question):
        item[0]["value"] = question
        item[1]["value"] = translator.process(item[1]["value"], src="en", dest="vi")
    return batch
def translate(args: argparse.Namespace, executor: Executor):
    name_file_out = (
        os.path.basename(executor.file_path).split(".")[0]
        if args.name_out is None
        else args.name_out
    )
    executor.process(
        map_fn=_translate,
        task=args.task,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        name_out=name_file_out,
        save=True,
    )
    if args.upload_to_hub:
        uploader.upload_file(
            file_path=f"{args.output_dir}/{name_file_out + ".json"}",
            repo_id=args.repo_id,
            path_in_repo=f"{name_file_out}",
            repo_type="dataset",
            overwrite=args.overwrite,
        )
    if args.clean_output:
        os.remove(f"{args.output_dir}/{name_file_out}")

    if args.clean_input:
        os.remove(args.file_path)

    return

def rename_column(args: argparse.Namespace, executor: Executor):
    name_file_out = (
        os.path.basename(executor.file_path)
        if args.name_out is None
        else args.name_out
    )
    list_old_name = args.list_old_name.split(",")
    list_new_name = args.list_new_name.split(",")
    executor.rename_column(
        list_old_name=list_old_name,
        list_new_name=list_new_name,
        name_file=name_file_out,
    )
    if args.upload_to_hub:
        uploader.upload_file(
            file_path=f"{args.output_dir}/{name_file_out + ".json"}",
            repo_id=args.repo_id,
            path_in_repo=f"{name_file_out}",
            repo_type="dataset",
            overwrite=args.overwrite,
        )
    if args.clean_input:
        os.remove(args.file_path)

    return

def remove_sample(args: argparse.Namespace, executor: Executor):
    name_file_out = (
        os.path.basename(executor.file_path)
        if args.name_out is None
        else args.name_out
    )
    if args.error_file_path.split("/")[-1].split(".")[-1] == "json":
        error_list = load_dataset("json", data_files=args.error_file_path)["train"]
    else:
        error_list = load_dataset("json", data_files=f"{args.error_file_path}/*.json")[
            "train"
        ]

    error_list = error_list["url_error"]
    executor.remove_sample(error_list=error_list, name_file=name_file_out)
    if args.upload_to_hub:
        uploader.upload_file(
            file_path=f"{args.output_dir}/{name_file_out + ".json"}",
            repo_id=args.repo_id,
            path_in_repo=f"{name_file_out}",
            repo_type="dataset",
            overwrite=args.overwrite,
        )
    if args.clean_input:
        os.remove(args.file_path)

    return

def divide_dataset(args: argparse.Namespace, executor: Executor):
    name_file_out = (
        os.path.basename(executor.file_path).split(".")[0]
        if args.name_out is None
        else args.name_out
    )
    executor.divide_shard_json()
    if args.upload_to_hub:
        uploader.upload_file(
            file_path=f"{args.output_dir}/{name_file_out + ".json"}",
            repo_id=args.repo_id,
            path_in_repo=f"{name_file_out}",
            repo_type="dataset",
            overwrite=args.overwrite,
        )
    if args.clean_input:
        os.remove(args.file_path)

    return

def main(args: argparse.Namespace):
    support_tasks = [
        "translate",
        "rename column",
        "remove sample",
        "divide dataset",
    ]
    if args.task not in support_tasks:
        print(f"task {args.task} not support")
        return

    if not os.path.exists(args.cache_dir):
        try:
            os.mkdir(args.cache_dir)
        except Exception as e:
            print(e)

    if not os.path.exists(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except Exception as e:
            print(e)

    if os.path.isdir(args.file_path):
        for json_file in sorted(
            os.listdir(args.file_path), key=lambda x: int(x.split(".")[0].split("_")[1])
        ):
            executor = Executor(
                file_path=f"{args.file_path}/{json_file}",
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                select=args.select,
                num_shards=args.num_shards,
            )
            if args.task == "translate":
                translate(args=args, executor=executor)
            if args.task == "rename column":
                rename_column(args=args, executor=executor)
            if args.task == "remove sample":
                remove_sample(args=args, executor=executor)
            if args.task == "divide dataset":
                divide_dataset(args=args, executor=executor)
    else:
        executor = Executor(
            file_path=args.file_path,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            select=args.select,
            num_shards=args.num_shards,
        )
        if args.task == "translate":
            translate(args=args, executor=executor)
        if args.task == "rename column":
            rename_column(args=args, executor=executor)
        if args.task == "remove sample":
            remove_sample(args=args, executor=executor)
        if args.task == "divide dataset":
            divide_dataset(args=args, executor=executor)
    
if __name__ == "__main__":
    main(parse_args())
