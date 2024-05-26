import os
import sys
import argparse
import json
from datasets import load_dataset

path_chunk_en = "/home/duytran/Downloads/output_ds/vast_2m_chunk_en"
vi_dataset_pạth = "/home/duytran/Downloads/output_ds/vast2M_vi.json"
output_dir = "/home/duytran/Downloads/output_ds/temp"

for file in os.listdir(path_chunk_en):
    name_file = file.split(".")[0]
    chunk_data_en = load_dataset("json", data_files=f"{path_chunk_en}/{file}")
    chunk_data_en = chunk_data_en["train"].map(lambda x: {"clip_id": name_file + "/" + x["clip_id"] + ".mp4"}, num_proc=16)
    chunk_data_en.to_json(
            output_dir + "/" + file,
            orient="records",
            lines=True,
            force_ascii=False,
        )

# vi_data = load_dataset("json", data_files=vi_dataset_pạth)
# chunk_list_file = os.listdir(path_chunk_en)
# for file in chunk_list_file:
#     name_file = file.split(".")[0]
#     chunk_data_en = load_dataset("json", data_files=f"{path_chunk_en}/{file}")
#     chunk_data_vi = vi_data["train"].filter(lambda x: x["id"] in chunk_data_en["train"]["clip_id"], num_proc=16)
#     chunk_data_vi.to_json(f"{output_dir}/{name_file}.json")
