import os
import zipfile

from huggingface_hub import HfApi, HfFileSystem, HfFolder

from vividbot.data.processor.base import BaseProcessor
from vividbot.data.utils.file import zip_dir


class HuggingFaceProcessor(BaseProcessor):
  TOKEN = HfFolder.get_token()

  def __init__(self) -> None:
    """
    Initialize processor.
    """
    self.api = HfApi()
    self.fs = HfFileSystem()

  def zip_and_upload_dir(
    self,
    dir_path: str,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    overwrite: bool = True,
  ) -> None:
    """
    Zip directory and upload it to the hub.
    :param dir_path:        Path to directory.
    :param repo_id:         Repository id.
    :param path_in_repo:    Path to directory in repository.
    :param repo_type:       Repository type.
    """
    if overwrite or not self.fs.exists(f"{repo_type}s/{repo_id}/{path_in_repo}"):
      self.api.upload_file(
        path_or_fileobj=zip_dir(dir_path, overwrite=True),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=self.TOKEN,
        repo_type=repo_type,
      )
      os.remove(dir_path + ".zip")

  def check_file_exists(self, repo_type: str, repo_id: str, path_in_repo: str) -> bool:
    """
    Check if file exist in the hub.
    :param repo_id:         Repository id.
    :param path_in_repo:    Path to file in repository.
    :return:                Whether file exist or not.
    """
    return self.fs.exists(f"{repo_type}s/{repo_id}/{path_in_repo}")

  def upload_dir(
    self,
    dir_path: str,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    overwrite: bool = True,
  ) -> None:
    """
    Upload directory to the hub.
    :param dir_path:        Path to directory.
    :param repo_id:         Repository id.
    :param path_in_repo:    Path to directory in repository.
    :param repo_type:       Repository type.
    """
    if overwrite or not self.fs.exists(f"{repo_type}s/{repo_id}/{path_in_repo}"):
      self.api.upload_folder(
        folder_path=dir_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=self.TOKEN,
        repo_type=repo_type,
      )

  def upload_file(
    self,
    file_path: str,
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

  def download_file(
    self,
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    local_dir: str = ".",
  ) -> None:
    """
    Download file from the hub.
    :param repo_id:         Repository id.
    :param filename:        Filename.
    :param repo_type:       Repository type.
    """
    self.api.hf_hub_download(
      repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir
    )

  def download_and_unzip_file(
    self,
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    local_dir: str = None,
    extract_dir: str = None,
  ) -> None:
    """
    Download and unzip file from the hub.
    :param repo_id:         Repository id.
    :param filename:        Filename.
    :param repo_type:       Repository type.
    """
    self.download_file(
      repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir
    )
    with zipfile.ZipFile(os.path.join(local_dir, filename), "r") as zip_ref:
      zip_ref.extractall(extract_dir)
