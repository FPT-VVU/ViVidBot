import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Union

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from vividbot.valley.util.config import IGNORE_INDEX
from vividbot.valley.util.constants import (
  FALLBACK_HF_IMAGE_PATHS,
  FALLBACK_HF_VIDEO_PATHS,
)
from vividbot.valley.util.data_util import (
  load_image_hf,
  load_video,
  load_video_hf,
  preprocess,
  preprocess_multimodal_multiimage,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
  filename="model/output/stage2/trainer.log",
  filemode="a",
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class HybridDataset(Dataset):
  """Dataset for supervised fine-tuning."""

  def __init__(
    self,
    data_path: str,
    video_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    multimodal_cfg: dict,
    **kwargs,
  ):
    super(HybridDataset, self).__init__()
    logging.warning("Loading data...")
    list_data_dict = []
    list_video_dict = []
    list_fashion_dict = []
    if multimodal_cfg["fast_epoch"]:
      if data_path is not None:
        for path in data_path:
          list_data_dict.extend(json.load(open(path, "r"))[0:10])
      if video_path is not None:
        for path in video_path:
          list_video_dict.extend(json.load(open(path, "r"))[0:10])
      if multimodal_cfg["use_fashion"]:
        list_fashion_dict = json.load(open(kwargs["fashion_data_path"]))[0:100]
    else:
      if data_path is not None:
        for path in data_path:
          list_data_dict.extend(json.load(open(path, "r")))
      if video_path is not None:
        for path in video_path:
          list_video_dict.extend(json.load(open(path, "r")))
      if multimodal_cfg["use_fashion"]:
        list_fashion_dict = json.load(open(kwargs["fashion_data_path"]))
    logging.warning("Formatting inputs...Skip in lazy mode")
    self.tokenizer = tokenizer
    self.list_data_dict = (
      list_video_dict + list_data_dict + list_fashion_dict
      if multimodal_cfg["use_fashion"]
      else list_video_dict + list_data_dict
    )
    # self.list_data_dict = list_data_dict
    random.shuffle(self.list_data_dict)
    self.multimodal_cfg = multimodal_cfg
    self.header_mode = multimodal_cfg["conv_mode"]

  def __len__(self):
    return len(self.list_data_dict)

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    sources = self.list_data_dict[i]
    data_type: Union[Literal["image", "video"], None] = (
      "image" if "image" in sources[0] else "video" if "video" in sources[0] else None
    )
    image = None
    video = None

    try:
      if isinstance(i, int):
        sources = [sources]
      assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
      if data_type == "image":
        processor = self.multimodal_cfg["image_processor"]
        # multi image preprocess
        if isinstance(self.list_data_dict[i]["image"], list):
          image_file_lsit = self.list_data_dict[i]["image"] 
          image = [Image.open(image_file) for image_file in image_file_lsit]
          image = processor.preprocess(image, return_tensors="pt")["pixel_values"]
          # FIXME: 14 is hardcoded patch size
          cur_token_len = (image[0].shape[1] // 14) * (image[0].shape[2] // 14)
          sources = preprocess_multimodal_multiimage(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.multimodal_cfg,
            cur_token_len,
            image.shape[0],
          )
        else:
          image_file = self.list_data_dict[i]["image"]
          image_folder = self.multimodal_cfg["image_folder"]
          if "train2014" in image_folder:
            image_file = "COCO_train2014_" + image_file
          img_path = os.path.join(image_folder, image_file)
          if os.path.exists(img_path):
            image = Image.open(os.path.join(image_folder, image_file))
          else:
            if self.multimodal_cfg["hf_repo_image"] is None:
              raise ValueError("Please specify the HF repo where the image is stored")
            image = None
            for repo_id in self.multimodal_cfg["hf_repo_image"]:
              try:
                image = load_image_hf(repo_path=repo_id, hf_image_path=image_file)
                break
              except Exception:
                continue
            if image is None:
              raise ValueError(f"Image {image_file} not found")

          if self.multimodal_cfg["image_aspect_ratio"] == "keep":
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = processor.preprocess(
              image,
              return_tensors="pt",
              do_center_crop=False,
              size={"shortest_edge": shortest_edge},
            )["pixel_values"][0]
          else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
          if self.multimodal_cfg["multi_image"]:
            image = image.unsqueeze(0)
          if len(image.shape) == 3:
            # FIXME: 14 is hardcoded patch size
            cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)
          elif len(image.shape) == 4:
            # FIXME: 14 is hardcoded patch size
            cur_token_len = (image.shape[2] // 14) * (image.shape[3] // 14)
          sources = preprocess_multimodal_multiimage(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.multimodal_cfg,
            cur_token_len,
            image.shape[0],
          )
      elif data_type == "video":
        video_file = (
          self.list_data_dict[i]["video"]
          if ".mp4" in self.list_data_dict[i]["video"]
          else self.list_data_dict[i]["video"] + ".mp4"
        )
        if "source" in self.list_data_dict[i]:
          video_file_source = self.list_data_dict[i]["source"]
          video_folder = self.multimodal_cfg["video_folder"] + "/" + video_file_source
        else:
          video_folder = self.multimodal_cfg["video_folder"] + "/"
        video_path = video_folder + "/" + video_file
        if os.path.exists(video_path):
          video = load_video(video_path)
        else:
          if self.multimodal_cfg["hf_repo_video"] is None:
            raise ValueError("Please specify the HF repo where the video is stored")
          video = None
          for repo_id in self.multimodal_cfg["hf_repo_video"]:
            try:
              video = load_video_hf(repo_path=repo_id, hf_video_path=video_file)
              break
            except Exception:
              continue
          if video is None:
            raise ValueError(f"Video {video_file} not found")
        # print(video.shape)
        video = video.permute(1, 0, 2, 3)
        # FIXME: 14 is hardcoded patch size
        cur_token_len = (video[0].shape[1] // 14) * (video[0].shape[2] // 14)
        sources = preprocess_multimodal_multiimage(
          copy.deepcopy([e["conversations"] for e in sources]),
          self.multimodal_cfg,
          cur_token_len,
          video.shape[0],
        )
      else:
        sources = copy.deepcopy([e["conversations"] for e in sources])
      data_dict = preprocess(sources, self.tokenizer, self.header_mode)
      if isinstance(i, int):
        data_dict = dict(
          input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
        )

      # image exist in the data
      if data_type == "image":
        data_dict["image"] = image
      elif data_type == "video":
        data_dict["image"] = video
      elif self.multimodal_cfg["is_multimodal"]:
        # image does not exist in the data, but the model is multimodal
        crop_size = self.multimodal_cfg["image_processor"].crop_size
        data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
      return data_dict
    except Exception as e:
      try:
        logger.error(
          f"Error processing data {self.list_data_dict[i]}: {e} - Retrying with fallbacks..."
        )

        fallback_sources = None

        if data_type == "image":
          processor = self.multimodal_cfg["image_processor"]
          # multi image preprocess
          for image_file in random.sample(
            FALLBACK_HF_IMAGE_PATHS, len(FALLBACK_HF_IMAGE_PATHS)
          ):
            try:
              if self.multimodal_cfg["hf_repo_image"] is None:
                raise ValueError("Please specify the HF repo where the image is stored")

              image = load_image_hf(
                repo_path="Vividbot/vividbot_image/images", hf_image_path=image_file
              )

              if image is None:
                raise ValueError(f"Image {image_file} not found")

              if self.multimodal_cfg["image_aspect_ratio"] == "keep":
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(
                  image,
                  return_tensors="pt",
                  do_center_crop=False,
                  size={"shortest_edge": shortest_edge},
                )["pixel_values"][0]
              else:
                image = processor.preprocess(image, return_tensors="pt")[
                  "pixel_values"
                ][0]
              if self.multimodal_cfg["multi_image"]:
                image = image.unsqueeze(0)
              if len(image.shape) == 3:
                # FIXME: 14 is hardcoded patch size
                cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)
              elif len(image.shape) == 4:
                # FIXME: 14 is hardcoded patch size
                cur_token_len = (image.shape[2] // 14) * (image.shape[3] // 14)

              fallback_sources = json.load(
                open("/content/vividbot_image_56k_all.json", "r")
              )
              fallback_sources = [
                e for e in fallback_sources if "image" in e and e["image"] == image_file
              ]
              if len(fallback_sources) == 0:
                raise ValueError(f"Image {image_file} not found in fallback source.")

              fallback_sources = random.sample(fallback_sources, 1)
              fallback_sources = preprocess_multimodal_multiimage(
                copy.deepcopy([e["conversations"] for e in fallback_sources]),
                self.multimodal_cfg,
                cur_token_len,
                image.shape[0],
              )
              break
            except Exception as e:
              logger.error(f"Couldn't process fallback image {image_file}: {e}")
              continue
        elif data_type == "video":
          # repo: Vividbot/vividbot_video/videos
          for video_file in random.sample(
            FALLBACK_HF_VIDEO_PATHS, len(FALLBACK_HF_VIDEO_PATHS)
          ):
            try:
              if self.multimodal_cfg["hf_repo_video"] is None:
                raise ValueError("Please specify the HF repo where the video is stored")

              video = load_video_hf(
                repo_path="Vividbot/vividbot_video/videos", hf_video_path=video_file
              )

              if video is None:
                raise ValueError(f"Video {video_file} not found")

              video = video.permute(1, 0, 2, 3)
              # FIXME: 14 is hardcoded patch size
              cur_token_len = (video[0].shape[1] // 14) * (video[0].shape[2] // 14)

              # create fallback_sources containing one item in which the list is read from "/content/vividbot_video_65k_all.json" and filtered by "video" == video_file
              fallback_sources = json.load(
                open("/content/vivid_video_instruct_128k_all.json", "r")
              )
              fallback_sources = [
                e for e in fallback_sources if "video" in e and e["video"] == video_file
              ]
              if len(fallback_sources) == 0:
                raise ValueError(f"Video {video_file} not found in fallback source.")

              fallback_sources = random.sample(fallback_sources, 1)
              fallback_sources = preprocess_multimodal_multiimage(
                copy.deepcopy([e["conversations"] for e in fallback_sources]),
                self.multimodal_cfg,
                cur_token_len,
                video.shape[0],
              )
              break
            except Exception as e:
              logger.error(f"Couldn't process fallback video {video_file}: {e}")
              continue
        else:
          fallback_sources = json.load(
            open("/content/vividbot_image_56k_all.json", "r")
          )
          fallback_sources = random.sample(fallback_sources, 1)
          fallback_sources = copy.deepcopy(
            [e["conversations"] for e in fallback_sources]
          )

        data_dict = preprocess(fallback_sources, self.tokenizer, self.header_mode)
        if isinstance(i, int):
          data_dict = dict(
            input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
          )

        # image exist in the data
        if data_type == "image":
          data_dict["image"] = image
        elif data_type == "video":
          data_dict["image"] = video
        elif self.multimodal_cfg["is_multimodal"]:
          # image does not exist in the data, but the model is multimodal
          crop_size = self.multimodal_cfg["image_processor"].crop_size
          data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return data_dict
      except Exception as e:
        logger.error(
          f"Error processing fallback data for {self.list_data_dict[i]}: {e}"
        )
        return ("fail", sources)


@dataclass
class DataCollatorForSupervisedDataset(object):
  """Collate examples for supervised fine-tuning."""

  tokenizer: transformers.PreTrainedTokenizer

  def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    instances_no_error = []
    for ins in instances:
      if type(ins) is not tuple:
        instances_no_error.append(ins)
    instances = instances_no_error
    input_ids, labels = tuple(
      [instance[key] for instance in instances] for key in ("input_ids", "labels")
    )
    input_ids = torch.nn.utils.rnn.pad_sequence(
      input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
    )
    # print(input_ids.shape)
    labels = torch.nn.utils.rnn.pad_sequence(
      labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    batch = dict(
      input_ids=input_ids,
      labels=labels,
      attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
    )

    if "image" in instances[0]:
      images = [instance["image"].half() for instance in instances]
      if all(x is not None and x.shape == images[0].shape for x in images):
        batch["images"] = torch.stack(images)
      else:
        batch["images"] = images

    return batch


def make_video_supervised_data_module(
  tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
  """Make dataset and collator for supervised fine-tuning."""
  train_dataset = HybridDataset(
    data_args.data_path,
    data_args.video_data_path,
    tokenizer,
    dict(
      conv_mode=data_args.conv_mode,
      fast_epoch=data_args.fast_epoch,
      use_fashion=data_args.use_fashion,
      multi_image=data_args.multi_image,
      num_image=data_args.num_image,
      is_multimodal=data_args.is_multimodal,
      image_token_len=data_args.image_token_len,
      image_folder=data_args.image_folder,
      hf_repo_image=data_args.hf_repo_image,
      video_folder=data_args.video_folder,
      hf_repo_video=data_args.hf_repo_video,
      image_aspect_ratio=data_args.image_aspect_ratio,
      use_im_start_end=getattr(data_args, "mm_use_im_start_end", False),
      image_processor=getattr(data_args, "image_processor", None),
    ),
    fashion_data_path=data_args.fashion_data_path,
  )

  data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
  return dict(
    train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
  )


if __name__ == "__main__":
  pass
