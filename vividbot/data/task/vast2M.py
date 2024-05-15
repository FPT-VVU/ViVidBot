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

from datasets import load_dataset, load_from_disk, concatenate_datasets

from vividbot.data.processor.translator import GGTranslator
from vividbot.data.processor.question_selection import QuestionSelection
from vividbot.data.processor.download import YoutubeDownloader
from vividbot.data.processor.upload_hf import Uploader

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
        "--output-dir",
        type=str,
        required=True,
        help="Path to save data.",
    )
    # parser.add_argument(
    #     "--channel-names",
    #     type=str,
    #     default=None,
    #     help="A channel name or path to file containing channel names.",
    # )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--num-shards",
        default=-1,
        type=int,
        help="number of shard for generate data",
    )
    # parser.add_argument(
    #     "--upload-to-hub",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Upload to hub after processing.",
    # )
    # parser.add_argument(
    #     "--clean-input",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Remove all downloaded input files after processing.",
    # )
    # parser.add_argument(
    #     "--clean-output",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Remove all output files except for metadata after processing.",
    # )
    # parser.add_argument(
    #     "--version",
    #     type=int,
    #     default=1,
    #     help="Version of the dataset.",
    # )
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

def map_func(batch):
    result_translate = translator.process(batch["vast_cap"], src="en", dest="vi") 
    result_question = question_list.process(len(batch["vast_cap"]))
    new_result = {"id" : batch["clip_id"],
                    "video": [item + ".mp4" for item in batch["clip_id"]],
                    "conversation": [[{"from": "human", "value" : question}, {"from" : "gpt", "value": answer}] 
                                    for question, answer in zip(result_question, result_translate)]}
    return new_result

def main(args: argparse.Namespace):
    
    # load dataset
    if not os.path.exists(args.file_path):
        print("Can not find the file, Can you please check it again?")
        return
    

    if not os.path.exists(f"{args.cache_dir}/temp") or not os.path.exists(f"{args.cache_dir}/result"):
        os.mkdir(f"{args.cache_dir}/result")
        os.mkdir(f"{args.cache_dir}/temp")

    if args.num_shards > 0 and len(os.listdir(f"{args.cache_dir}/temp")) == 0 and len(os.listdir(f"{args.cache_dir}/result")) == 0:
        try:
            dataset = load_dataset("json", data_files=args.file_path, cache_dir=args.cache_dir)["train"]
        except:
            print("Please provide json file")
            return
        if args.select > 0:
            dataset = dataset.select(range(args.select))

        for shard_idx in range(args.num_shards):
            shard = dataset.shard(num_shards=args.num_shards, index=shard_idx, contiguous=True)
            shard.save_to_disk(f"{args.cache_dir}/temp/shard_{shard_idx}")
    if args.num_shards < 0:
        try:
            dataset = load_dataset("json", data_files=args.file_path, cache_dir=args.cache_dir)["train"]
        except:
            print("Please provide json file")
            return
        if args.select > 0:
            dataset = dataset.select(range(args.select))

    if args.task == "download":
        downloader = YoutubeDownloader()
        dataset.map(downloader.process, 
                    fn_kwargs={"key_url": "clip_id", "key_span": "clip_span", "path": args.output_dir},
                    batched=True, batch_size=args.batch_size, num_proc=args.num_proc, 
                    load_from_cache_file= not args.overwrite,
                    desc="Download data from youtube")
    if args.task == "generate":
        # read number folder in a folder
        if len(os.listdir(f"{args.cache_dir}/result")) <= args.num_shards:
            for shard_idx in os.listdir(args.cache_dir + "/temp"):
                shard = load_from_disk(f"{args.cache_dir}/temp/{shard_idx}")
                # try:
                shard = shard.map(map_func, 
                                        batched=True, batch_size=args.batch_size, num_proc=args.num_proc, 
                                        remove_columns=list(dataset.features.keys()),
                                        load_from_cache_file=True,
                                        desc=f"Generate data to a new format and translate them from English to Vietnamese at shard {shard_idx}")
                shard.save_to_disk(f"{args.cache_dir}/result/{shard_idx}")
                # remove shard from memory
                shutil.rmtree(f"{args.cache_dir}/temp/{shard_idx}")
                # except: 
                #     print(f"shard {shard_idx} has errors, please check it")
        
        if len(os.listdir(f"{args.cache_dir}/temp")) == 0 and len(os.listdir(f"{args.cache_dir}/result")) == args.num_shards:    
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            # to json with encoding utf-8
            ds = concatenate_datasets([
                                        load_from_disk(f"{args.cache_dir}/result/{shard_idx}")
                                        for shard_idx in os.listdir(args.cache_dir + "/result")
                                    ])
            shutil.rmtree(f"{args.cache_dir}/temp")
            shutil.rmtree(f"{args.cache_dir}/result")
            ds.to_json(args.output_dir + "/vast2M_vi.json", orient="records", lines=True, force_ascii=False)

if __name__ == '__main__':
    main(parse_args())