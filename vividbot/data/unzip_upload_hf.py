from huggingface_hub import hf_hub_download
import os
import sys
import shutil

sys.path.append(os.getcwd())
from vividbot.data.processor.upload_hf import Uploader

uploader = Uploader()

repo_id = "Vividbot/vast2m_vi"
start = 0
end = 499

path_out = "/home/duytran/Downloads/output_video"

for i in range(start, end + 1):
    if len(os.listdir(path_out+"/video")) == 10:
        # remove all files
        shutil.rmtree(path_out+"/video")

    file_name = f"shard_{i}.zip"
    if uploader.check_file_exist(repo_type="dataset", repo_id=repo_id, path_in_repo="video2/"+file_name):
        continue
    else:
        if not os.path.exists(path_out+f"/video/{file_name}"):
            hf_hub_download(repo_id=repo_id, filename="video/"+file_name, repo_type="dataset", local_dir=path_out)
        if not os.path.exists(path_out+f"/video/{file_name.split('.')[0]}"):
            unzip_command = f"unzip {path_out}/video/{file_name} -d {path_out}/video/"
            # run unzip command
            os.system(unzip_command)
        uploader.upload_dir(dir_path=path_out+f"/video/{file_name.split('.')[0]}", 
                        repo_id=repo_id, 
                        path_in_repo="video2", 
                        repo_type="dataset", overwrite=True)
        print("Have uploaded", file_name)

    