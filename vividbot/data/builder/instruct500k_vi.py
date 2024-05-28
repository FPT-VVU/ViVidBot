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
_REPO_URL = "https://huggingface.co/datasets/Vividbot/instruct500k_vi"
_URLS = {
    "meta": f"{_REPO_URL}/instruct500k_vi.json",
    "image": f"{_REPO_URL}/images" + "{shard}.zip",
}

_CONFIGS = [
    os.path.basename(file_name).split(".")[0]
    for file_name in fs.listdir(_REPO_URL + "/images", detail=False)
    if file_name.endswith(".zip")
]

class Instruct500k_ViConfig(datasets.BuilderConfig):
    """BuilderConfig for Vast2M_Vi."""

    def __init__(self, **kwargs):
        """
        :param kwargs:  Arguments.
        """
        super().__init__(
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
            **kwargs,
        )


class Instruck500k_Vi(datasets.GeneratorBasedBuilder):
    """Vast2M Vi dataset."""

    BUILDER_CONFIGS = Instruct500k_ViConfig()

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "image": datasets.Value("binary"),
                "conversations": datasets.Value(
                    "dict",
                ),
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
            [_URLS["image"].format(shard=shard) for shard in _CONFIGS]
        )
        image_dict = {
            shard: visual_dir
            for shard, visual_dir in zip(_CONFIGS, image_dirs)
        }

        return [
            datasets.SplitGenerator(
                gen_kwargs={
                    "split": split,
                    "image_dict": image_dict,
                },
            )
            for split in split_dict.items()
        ]

    def _generate_examples(
        self,
        split: datasets.Dataset,
        image_dict: dict,
    ) -> tuple[int, dict]:
        """
        Generate examples.
        :param split:                   Split.
        :param visual_dict:             Paths to directory containing visual files.
        :param audio_dict:              Paths to directory containing audio files.
        :param transcript_dict:         Paths to directory containing transcripts.
        :return:                        Example.
        """
        for i, sample in enumerate(split):
            shard = sample["image"].split("/")[-1].split(".")[0]
            image_path = os.path.join(
                image_dict[shard], sample["image"]
            )

            yield i, {
                "id": sample["id"],
                "video": self.__get_binary_data(image_path),
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