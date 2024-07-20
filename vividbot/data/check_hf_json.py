from huggingface_hub import HfFileSystem
import os
import zipfile
from datasets import load_dataset

from tqdm import tqdm

fs = HfFileSystem()
input = "/home/duytran/Downloads/output_ds/videoinstruck_100k_chunk_vi"
output = "/home/duytran/Downloads/output_ds/videoinstruck_100k_chunk_vi_final"


def rename_path(batch, file_name):
        batch['video'] = [file_name + "/" + item for item in batch['video']]
        print(batch["video"])
        return batch
error = []
for file_name in tqdm(os.listdir(input)):
    path = os.path.join(input, file_name)
    
    file_name = os.path.basename(path).split(".")[0]
    print("-"*50, file_name, "-"*50)
    dataset = load_dataset("json", data_files=path)["train"]
    dataset = dataset.map(rename_path, 
                        fn_kwargs={"file_name": file_name},
                        batched=True,
                        num_proc=8)

    path_hf = "datasets/Vividbot/videoinstruck100k/video/" + file_name + ".zip"
    zip_hf = fs.open(path_hf)
    with zipfile.ZipFile(zip_hf, 'r') as zip_ref:
        list_files = zip_ref.namelist()[1:]
        #print(list_files)
        len_zip = len(list_files)
        new_data = dataset.filter(lambda x: x["video"] in list_files, num_proc=8)
        print("length of new data: ", len(new_data))
        print("length of zip: ", len_zip)
        #assert len(new_data) <= len_zip, "Not enough video in zip"
        #print(new_data)
        new_data.to_json(output + "/" + file_name + ".json", force_ascii=False)
        print(f"Save to {output}/{file_name}.json")
    zip_hf.close()