import os
import shutil


def zip_dir(zip_dir: str, overwrite: bool = False) -> str:
    """
    Zip directory.
    :param zip_dir:     Path to directory.
    :param overwrite:   Whether to overwrite existing zip file.
    :return:            Path to zip file.
    """
    if overwrite and os.path.exists(zip_dir + ".zip"):
        os.remove(zip_dir + ".zip")
    shutil.make_archive(
        zip_dir, "zip", os.path.dirname(zip_dir), os.path.basename(zip_dir)
    )
    return zip_dir + ".zip"
