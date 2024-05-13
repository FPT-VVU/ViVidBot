import os
import sys
import argparse

sys.path.append(os.getcwd())

from datasets import load_dataset

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
    # parser.add_argument(
    #     "--overwrite",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Overwrite existing files.",
    # )
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
    # parser.add_argument(
    #     "--cache-dir",
    #     type=str,
    #     default=os.path.join(os.getcwd(), ".cache"),
    #     help="Cache directory.",
    # )
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
    try:
        dataset = load_dataset("json", data_files=args.file_path)["train"]
    except:
        print("Please provide json file")
        return
    if args.select > 0:
        dataset = dataset.select(range(args.select))

    if args.task == "download":
        downloader = YoutubeDownloader()
        dataset.map(downloader.download, 
                    fn_kwargs={"key_url": "clip_id", "key_span": "clip_span", "path": args.output_dir},
                    batched=True, batch_size=args.batch_size, num_proc=args.num_proc, 
                    desc="Download data from youtube")
    if args.task == "generate":
        dataset = dataset.map(map_func, 
                                batched=True, batch_size=args.batch_size, num_proc=args.num_proc, 
                                remove_columns=list(dataset.features.keys()),
                                desc="Generate data to a new format and translate them from English to Vietnamese")
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        # to json with encoding utf-8
        dataset.to_json(args.output_dir + "/vast2M_vi.json", orient="records", lines=True, force_ascii=False)

if __name__ == '__main__':
    main(parse_args())