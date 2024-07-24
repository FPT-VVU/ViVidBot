import os
import shutil
from typing import Callable

import tqdm
from datasets import (
    concatenate_datasets,
    disable_progress_bar,
    load_dataset,
    load_from_disk,
)

from vividbot.data.processor.base import BaseProcessor
from vividbot.data.processor.huggingface import HuggingFaceProcessor


class Executor(BaseProcessor):
    def __init__(
        self,
        file_path: str,
        cache_dir: str,
        output_dir: str,
        select: int = -1,
        num_shards: int = -1,
    ) -> None:
        self.file_path = file_path
        self.cache_dir = cache_dir
        if not os.path.exists(f"{self.cache_dir}/temp") or not os.path.exists(
            f"{self.cache_dir}/result"
        ):
            try:
                os.mkdir(f"{self.cache_dir}/result")
                os.mkdir(f"{self.cache_dir}/temp")
            except Exception as e:
                print(e)

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            try:
                os.mkdir(self.output_dir)
            except Exception as e:
                print(e)

        self.select = select
        self.num_shards = num_shards
        self.num_file_temp = len(os.listdir(f"{self.cache_dir}/temp"))
        self.num_file_result = len(os.listdir(f"{self.cache_dir}/result"))
        self.dataset = (
            self.load_dataset()
            if self.num_file_temp == 0 and self.num_file_result == 0
            else None
        )
        self.uploader = HuggingFaceProcessor()

    def load_dataset(self):
        if (
            not os.path.exists(self.file_path)
            or self.file_path.split("/")[-1].split(".")[-1] != "json"
        ):
            print(
                "Can not find the file or the file is not json, Can you please check it again?"
            )
            exit(0)
        dataset = load_dataset("json", data_files=self.file_path)["train"]
        if self.select > 0:
            dataset = dataset.select(range(self.select))
        return dataset

    def rename_column(self, list_old_name: list, list_new_name: list, name_file: str):
        dataset = self.load_dataset()
        assert len(list_old_name) == len(
            list_new_name
        ), "list_old_name and list_new_name must have the same length"
        for old_name, new_name in zip(list_old_name, list_new_name):
            dataset = dataset.rename_column(old_name, new_name)
        dataset.to_json(
            self.output_dir + "/" + name_file,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        print(f"Save to {self.output_dir}/{name_file}")

    def remove_sample(self, error_list: list, name_file: str):
        dataset = self.load_dataset()
        dataset = dataset.filter(lambda x: x["id"] not in error_list)
        dataset.to_json(
            self.output_dir + "/" + name_file,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        print(f"Save to {self.output_dir}/{name_file}")

    def save(self, name_file: str, save: bool) -> None:
        if (
            len(os.listdir(f"{self.cache_dir}/temp")) == 0
            and len(os.listdir(f"{self.cache_dir}/result")) == self.num_shards
        ):
            if save:
                # to json with encoding utf-8
                ds = concatenate_datasets(
                    [
                        load_from_disk(f"{self.cache_dir}/result/{shard_idx}")
                        for shard_idx in os.listdir(self.cache_dir + "/result")
                    ]
                )
                shutil.rmtree(f"{self.cache_dir}/temp")
                shutil.rmtree(f"{self.cache_dir}/result")
                ds.to_json(
                    self.output_dir + "/" + name_file,
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )
                print(f"Save to {self.output_dir}/{name_file}")
            else:
                shutil.rmtree(f"{self.cache_dir}/temp")
                shutil.rmtree(f"{self.cache_dir}/result")

    def divide_shard(self) -> None:
        if self.num_file_temp == 0 and self.num_file_result == 0:
            for shard_idx in range(self.num_shards):
                shard = self.dataset.shard(
                    num_shards=self.num_shards, index=shard_idx, contiguous=True
                )
                shard.save_to_disk(f"{self.cache_dir}/temp/shard_{shard_idx}")

    def divide_shard_json(self) -> None:
        dataset = self.load_dataset()
        for shard_idx in range(self.num_shards):
            shard = dataset.shard(
                num_shards=self.num_shards, index=shard_idx, contiguous=True
            )
            shard.to_json(
                f"{self.output_dir}/shard_{shard_idx}.json",
                orient="records",
                lines=True,
                force_ascii=False,
            )
            print(f"save to {self.output_dir}/shard_{shard_idx}.json")

    def process(
        self,
        map_fn: Callable,
        task: str,
        batch_size: int = 1,
        num_proc: int = 1,
        name_out: str = "result",
        save: bool = True,
        remove_columns: list = None,
        fn_kwargs: dict = None,
    ):
        self.divide_shard()

        print("-" * 50 + f"Have processed {self.num_file_result} shards" + "-" * 50)

        pbar = tqdm.tqdm(
            total=len(os.listdir(self.cache_dir + "/temp")),
            desc=f"Processing {len(os.listdir(self.cache_dir + '/temp'))} shards - {task} task",
        )
        if self.num_file_result < self.num_shards:
            for shard_idx in os.listdir(self.cache_dir + "/temp"):
                shard = load_from_disk(f"{self.cache_dir}/temp/{shard_idx}")
                disable_progress_bar()
                shard = shard.map(
                    map_fn,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=num_proc,
                    remove_columns=remove_columns,
                )
                shard.save_to_disk(f"{self.cache_dir}/result/{shard_idx}")
                shutil.rmtree(f"{self.cache_dir}/temp/{shard_idx}")
                pbar.update(1)
            pbar.close()

        self.save(name_file=name_out, save=save)
