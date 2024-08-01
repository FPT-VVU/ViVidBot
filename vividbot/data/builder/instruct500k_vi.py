import os

import datasets
from huggingface_hub import HfFileSystem

logger = datasets.logging.get_logger(__name__)
fs = HfFileSystem()


_CITATION = """
"""
_DESCRIPTION = """
"""
_HOMEPAGE = "https://github.com/FPT-VVU/ViVidBot"
_REPO_ID = "datasets/Vividbot/instruct500k_vi"
_REPO_URL = f"https://huggingface.co/{_REPO_ID}/resolve/main"
_URLS = {
    "meta": f"{_REPO_URL}/instruct500k_vi.json",
    "image": f"{_REPO_URL}/images/" + "{shard}.zip",
}

_CONFIGS = ["all"]
if fs.exists(_REPO_ID + "/images"):
    _CONFIGS.extend(
        [
            os.path.basename(file_name).split(".")[0]
            for file_name in fs.listdir(_REPO_ID + "/images", detail=False)
            if file_name.endswith(".zip")
        ]
    )


class Instruct500k_ViConfig(datasets.BuilderConfig):
    """BuilderConfig for Instruct500k_ViConfig."""

    def __init__(self, name, **kwargs):
        """
        :param name:    Name of subset.
        :param kwargs:  Arguments.
        """
        super().__init__(
            name=name,
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class Instruck500k_Vi(datasets.GeneratorBasedBuilder):
    """Instruct500k Vi dataset."""

    BUILDER_CONFIGS = [Instruct500k_ViConfig(name) for name in _CONFIGS]

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "image": datasets.Value("binary"),
                "conversations": [
                    {
                        "from": datasets.Value("string"),
                        "value": datasets.Value("string"),
                    }
                ],
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """
        Get splits.
        :param dl_manager:  Download manager.
        :return:            Splits.
        """
        config_names = _CONFIGS[1:] if self.config.name == "all" else [self.config.name]

        metadata_paths = dl_manager.download(_URLS["meta"])
        dataset = datasets.load_dataset(
            "json",
            data_files=metadata_paths,
            split="train",
        )
        dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        train_set = dataset["train"]
        val_test_set = dataset["test"].train_test_split(test_size=0.5)
        val_set = val_test_set["train"]
        test_set = val_test_set["test"]

        split_dict = {
            datasets.Split.TRAIN: train_set,
            datasets.Split.VALIDATION: val_set,
            datasets.Split.TEST: test_set,
        }

        image_dirs = dl_manager.download_and_extract(
            [_URLS["image"].format(shard=shard) for shard in config_names]
        )
        image_dict = {
            shard: image_dir for shard, image_dir in zip(config_names, image_dirs)
        }

        return [
            datasets.SplitGenerator(
                name=name,
                gen_kwargs={
                    "split": split,
                    "image_dict": image_dict,
                },
            )
            for name, split in split_dict.items()
        ]

    def _generate_examples(
        self,
        split: datasets.Dataset,
        image_dict: dict,
    ) -> tuple[int, dict]:
        """
        Generate examples.
        :param split:                   Split.
        :param image_dict:              Paths to directory containing image files.
        :return:                        Example.
        """
        for i, sample in enumerate(split):
            shard = sample["image"].split("/")[0]
            image_path = os.path.join(
                image_dict[shard], shard, sample["image"].split("/")[1]
            )

            yield i, {
                "id": sample["id"],
                "image": self.__get_binary_data(image_path),
                "conversations": sample["conversations"],
            }

    def __get_binary_data(self, path: str) -> bytes:
        """
        Get binary data from path.
        :param path:    Path to file.
        :return:        Binary data.
        """
        with open(path, "rb") as f:
            return f.read()

    def __get_text_data(self, path: str) -> str:
        """
        Get transcript from path.
        :param path:     Path to transcript.
        :return:         Transcript.
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
