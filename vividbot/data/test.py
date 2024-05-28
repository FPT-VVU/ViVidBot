import os
import sys

sys.path.append(os.getcwd())

from vividbot.data.processor.upload_hf import Uploader

uploader = Uploader()
path = "/home/duytran/Downloads/output/instruct500k_vi.json"

uploader.upload_file(file_path=path,
                     repo_id= "Vividbot/instruct500k_vi",
                     path_in_repo= "instruct500k_vi.json",
                     repo_type= "dataset",
                     overwrite= True,)

# for folder in os.listdir(path):
#     uploader.zip_and_upload_dir(
#         dir_path=f"{path}/{folder}",
#         repo_id="Vividbot/instruct500k_vi",
#         path_in_repo=f"images/{folder}.zip",
#         repo_type="dataset",
#         overwrite=False,
#     )
