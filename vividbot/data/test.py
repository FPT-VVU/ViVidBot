import os
import sys
import argparse
import json


sys.path.append(os.getcwd())
from vividbot.data.processor.upload_hf import Uploader

uploader = Uploader()
uploader.upload_file(
    file_path="/home/duytran/Downloads/output_video/error/error_shard_1.json",
    repo_id="Vividbot/vast2m_vi",
    path_in_repo=f"error/error_shard_1.json",
    repo_type="dataset",
    overwrite=True,
)
