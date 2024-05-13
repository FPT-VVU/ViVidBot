import os
from huggingface_hub import HfApi, HfFolder, HfFileSystem
from vividbot.data.processor.base import BaseProcessor

class Uploader(BaseProcessor):
    TOKEN = HfFolder.get_token()

    def __init__(self) -> None:
        """
        Initialize processor.
        """
        self.api = HfApi()
        self.fs = HfFileSystem()

    def upload_file(
        self, file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str = "dataset",
        overwrite: bool = True,
    ) -> None:
        """
        Upload file to the hub.
        :param file_path:       Path to file.
        :param repo_id:         Repository id.
        :param path_in_repo:    Path to file in repository.
        :param repo_type:       Repository type.
        """
        if overwrite or not self.fs.exists(f"{repo_type}s/{repo_id}/{path_in_repo}"):
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=self.TOKEN,
                repo_type=repo_type,
            )