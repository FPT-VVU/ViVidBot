import os
import sys
from datasets import load_dataset

sys.path.append(os.getcwd())

from vividbot.data.processor.upload_hf import Uploader

uploader = Uploader()
path = "/home/duytran/Downloads/output_ds/vast_2m_final_all.json"

# uploader.upload_dir(dir_path=path, 
#                     repo_id="Vividbot/vast2m_vi", 
#                     path_in_repo="metadata", 
#                     repo_type="dataset", overwrite=True)

uploader.upload_file(file_path=path,
                     repo_id= "Vividbot/vast2m_vi",
                     path_in_repo= "vast2m_vi_all.json",
                     repo_type= "dataset",
                     overwrite= True)

# for folder in os.listdir(path):
#     uploader.zip_and_upload_dir(
#         dir_path=f"{path}/{folder}",
#         repo_id="Vividbot/instruct500k_vi",
#         path_in_repo=f"images/{folder}.zip",
#         repo_type="dataset",
#         overwrite=False,
#     )
# from huggingface_hub import HfFileSystem
# import os
# import zipfile
# from datasets import load_dataset

# from tqdm import tqdm

# fs = HfFileSystem()
# input = "/home/duytran/Downloads/output_ds/vast_2m_chunk_en2vi"
# output = "/home/duytran/Downloads/output_ds/vast_2m_final"

# def rename_path(batch):
#         batch['video'] = [file_name + "/" + item for item in batch['video']]
#         return batch
# error = []
# for file_name in tqdm(os.listdir(input)):
#     path = os.path.join(input, file_name)
    
#     file_name = os.path.basename(path).split(".")[0]
#     print("-"*50, file_name, "-"*50)
#     dataset = load_dataset("json", data_files=path)["train"]
#     dataset = dataset.map(rename_path, 
#                         batched=True,
#                         num_proc=8)
    
#     path_hf = "datasets/Vividbot/vast2m_vi/video/" + file_name + ".zip"
#     zip_hf = fs.open(path_hf)
#     with zipfile.ZipFile(zip_hf, 'r') as zip_ref:
#         list_files = zip_ref.namelist()[1:]
#         len_zip = len(list_files)
#         new_data = dataset.filter(lambda x: x["video"] in list_files, num_proc=8)
#         print("length of new data: ", len(new_data))
#         print("length of zip: ", len_zip)
#         assert len(new_data) <= len_zip, "Not enough video in zip"
#         new_data.to_json(output + "/" + file_name + ".json", force_ascii=False)
#         print(f"Save to {output}/{file_name}.json")
#     zip_hf.close()


    