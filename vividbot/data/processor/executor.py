import os
import shutil
import tqdm
from typing import Callable

from datasets import load_dataset, load_from_disk, concatenate_datasets, disable_progress_bar
from vividbot.data.processor.base import BaseProcessor
from vividbot.data.processor.upload_hf import Uploader
class Executor(BaseProcessor):
    def __init__(self, file_path: str, 
                    cache_dir: str,
                    output_dir: str,
                    select: int= -1,
                    num_shards: int = -1) -> None:
        self.cache_dir = cache_dir
        if not os.path.exists(f"{self.cache_dir}/temp") or not os.path.exists(f"{self.cache_dir}/result"):
            os.mkdir(f"{self.cache_dir}/result")
            os.mkdir(f"{self.cache_dir}/temp")
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        self.select = select
        self.num_shards = num_shards
        self.num_file_temp = len(os.listdir(f"{self.cache_dir}/temp"))
        self.num_file_result = len(os.listdir(f"{self.cache_dir}/result"))
        self.dataset = self.load_dataset(file_path=file_path) if self.num_file_temp == 0 and self.num_file_result == 0 else None
        self.uploader = Uploader()

    
    def load_dataset(self, file_path : str):
        if not os.path.exists(file_path) or file_path.split("/")[-1].split(".")[-1] != "json":
            print("Can not find the file or the file is not json, Can you please check it again?")
            exit(0)
        dataset = load_dataset("json", data_files=file_path)["train"]
        if self.select > 0:
            dataset = dataset.select(range(self.select))
        return dataset

    def save(self, name_file: str, save: bool) -> None:
        if len(os.listdir(f"{self.cache_dir}/temp")) == 0 and len(os.listdir(f"{self.cache_dir}/result")) == self.num_shards:  
            if save:
                # to json with encoding utf-8
                ds = concatenate_datasets([
                                            load_from_disk(f"{self.cache_dir}/result/{shard_idx}")
                                            for shard_idx in os.listdir(self.cache_dir + "/result")
                                        ])
                shutil.rmtree(f"{self.cache_dir}/temp")
                shutil.rmtree(f"{self.cache_dir}/result")
                ds.to_json(self.output_dir + "/" + name_file, orient="records", lines=True, force_ascii=False)
                print(f"Save to {self.output_dir}/{name_file}")
            else:
                shutil.rmtree(f"{self.cache_dir}/temp")
                shutil.rmtree(f"{self.cache_dir}/result")

    def divide_shard(self) -> None:
        if self.num_file_temp == 0 and self.num_file_result == 0:
            for shard_idx in range(self.num_shards):
                shard = self.dataset.shard(num_shards=self.num_shards, index=shard_idx, contiguous=True)
                shard.save_to_disk(f"{self.cache_dir}/temp/shard_{shard_idx}")

    def process(self, map_fn: Callable, task: str, batch_size: int = 1, 
                num_proc: int = 1, name_out: str = "result.json", 
                save: bool = True, remove_columns: list = None, fn_kwargs: dict = None):
        self.divide_shard()

        print("-"*50 + f"Have processed {self.num_file_result} shards" + "-"*50)
        disable_progress_bar()

        pbar = tqdm.tqdm(total=len(os.listdir(self.cache_dir + '/temp')),
                        desc=f"Processing {len(os.listdir(self.cache_dir + '/temp'))} shards - {task} task")
        if self.num_file_result < self.num_shards:
            for shard_idx in os.listdir(self.cache_dir + "/temp"):
                shard = load_from_disk(f"{self.cache_dir}/temp/{shard_idx}")
                shard = shard.map(map_fn, 
                                fn_kwargs=fn_kwargs,
                                batched=True, batch_size=batch_size, num_proc=num_proc, 
                                remove_columns=remove_columns)
                shard.save_to_disk(f"{self.cache_dir}/result/{shard_idx}")
                shutil.rmtree(f"{self.cache_dir}/temp/{shard_idx}")
                pbar.update(1)
            pbar.close()
    
        self.save(name_file=name_out, save=save)
        
            
        

