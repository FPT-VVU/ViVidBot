import copy
import io
import logging
import os
import random
import zipfile
from pathlib import Path
from typing import Dict, Sequence

import av
import decord
import numpy as np
import torch
import transformers
from huggingface_hub import HfFileSystem
from PIL import Image
from torchvision import transforms
from transformers import StoppingCriteria

from vividbot.valley import conversation as conversation_lib
from vividbot.valley.data import video_transform
from vividbot.valley.util.config import (
  DEFAULT_IM_END_TOKEN,
  DEFAULT_IM_START_TOKEN,
  DEFAULT_IMAGE_PATCH_TOKEN,
  DEFAULT_IMAGE_TOKEN,
  DEFAULT_VI_END_TOKEN,
  DEFAULT_VI_START_TOKEN,
  DEFAULT_VIDEO_FRAME_TOKEN,
  DEFAULT_VIDEO_TOKEN,
  IGNORE_INDEX,
)
from vividbot.valley.util.constants import FALLBACK_HF_VIDEO_PATHS

logger = logging.getLogger(__name__)
# logging.basicConfig(
#   filename="model/output/stage2/trainer.log",
#   filemode="a",
#   level=logging.INFO,
#   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )


def collate_wrapper(batch):
  image_list = [b[0] for b in batch]
  prompt_list = [b[2] for b in batch]
  # input_ids = pad_sequence(prompt_list, padding_value = 0, batch_first = True)
  conv_list = [b[3] for b in batch]
  label_list = [b[1] for b in batch]
  return prompt_list, image_list, conv_list, label_list


def collate_process_image_text(batch, tokenizer, image_processor):
  batch_prompt, batch_image, conv_list, label_list = batch
  batch_prompt = tokenizer(batch_prompt, padding=True)
  input_ids, attention_mask = batch_prompt.input_ids, batch_prompt.attention_mask
  input_ids = torch.as_tensor(input_ids)
  attention_mask = torch.as_tensor(attention_mask)
  videos = []
  for this_batch_images in batch_image:
    video = image_processor.preprocess(this_batch_images, return_tensors="pt")[
      "pixel_values"
    ]
    videos.append(video)
  return input_ids, attention_mask, videos, conv_list, label_list


class KeywordsStoppingCriteria(StoppingCriteria):
  def __init__(self, keywords, tokenizer, input_ids):
    self.keywords = keywords
    self.tokenizer = tokenizer
    self.start_len = None
    self.input_ids = input_ids

  def __call__(
    self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
  ) -> bool:
    if self.start_len is None:
      self.start_len = self.input_ids.shape[1]
    else:
      outputs = self.tokenizer.batch_decode(
        output_ids[:, self.start_len :], skip_special_tokens=True
      )[0]
      for keyword in self.keywords:
        if keyword in outputs:
          return True
    return False


# for finetune


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
  """Collects the state dict and dump to disk."""

  if trainer.args.lora:
    if trainer.args.should_save:
      trainer.model.save_pretrained(output_dir)
  else:
    if trainer.deepspeed:
      print("saving deepspeed model...")
      torch.cuda.synchronize()
      trainer.save_model(output_dir)
      return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
      cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
      del state_dict
      trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
  special_tokens_dict: Dict,
  tokenizer: transformers.PreTrainedTokenizer,
  model: transformers.PreTrainedModel,
):
  """Resize tokenizer and embedding.

  Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
  """
  num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))

  if num_new_tokens > 0:
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
      dim=0, keepdim=True
    )

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
  strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
  """Tokenize a list of strings."""
  tokenized_list = [
    tokenizer(
      text,
      return_tensors="pt",
      padding="longest",
      max_length=tokenizer.model_max_length,
      truncation=True,
    )
    for text in strings
  ]
  input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
  input_ids_lens = labels_lens = [
    tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
    for tokenized in tokenized_list
  ]
  return dict(
    input_ids=input_ids,
    labels=labels,
    input_ids_lens=input_ids_lens,
    labels_lens=labels_lens,
  )


def _mask_targets(target, tokenized_lens, speakers, only_mask_system):
  # cur_idx = 0
  cur_idx = tokenized_lens[0]
  tokenized_lens = tokenized_lens[1:]
  target[:cur_idx] = IGNORE_INDEX
  if not only_mask_system:
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
      if speaker == "human":
        target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
      cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
  """Add speaker and start/end signal on each round."""
  BEGIN_SIGNAL = "### "
  END_SIGNAL = "\n"
  conversation = header
  for sentence in source:
    from_str = sentence["from"]
    if from_str.lower() == "human":
      from_str = conversation_lib.default_conversation.roles[0]
    elif from_str.lower() == "gpt":
      from_str = conversation_lib.default_conversation.roles[1]
    else:
      from_str = "unknown"
    sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
    if get_conversation:
      conversation += sentence["value"]
  conversation += BEGIN_SIGNAL
  return conversation


def preprocess_multimodal(
  sources: Sequence[str],
  multimodal_cfg: dict,
  cur_token_len: int,
) -> Dict:
  is_multimodal = multimodal_cfg["is_multimodal"]
  # image_token_len = multimodal_cfg['image_token_len']
  image_token_len = cur_token_len
  if not is_multimodal:
    return sources

  for source in sources:
    for sentence in source:
      replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
      if multimodal_cfg["use_im_start_end"]:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
      sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

  return sources


def preprocess_multimodal_multiimage(
  sources: Sequence[str], multimodal_cfg: dict, cur_token_len: int, num_image: int
) -> Dict:
  is_multimodal = multimodal_cfg["is_multimodal"]
  # image_token_len = multimodal_cfg['image_token_len']
  image_token_len = cur_token_len
  if not is_multimodal:
    return sources

  for source in sources:
    for sentence in source:
      if multimodal_cfg["use_im_start_end"]:
        replace_token = (
          DEFAULT_IM_START_TOKEN
          + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
          + DEFAULT_IM_END_TOKEN
        )
        replace_token = (
          replace_token
          + DEFAULT_VI_START_TOKEN
          + DEFAULT_VIDEO_FRAME_TOKEN * num_image
          + DEFAULT_VI_END_TOKEN
        )
      sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
      sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)
  return sources


def preprocess(
  sources: Sequence[str],
  tokenizer: transformers.PreTrainedTokenizer,
  conv_mode,
  only_mask_system=False,
) -> Dict:
  """
  Given a list of sources, each is a conversation list. This transform:
  1. Add signal '### ' at the beginning each sentence, with end signal '\n';
  2. Concatenate conversations together;
  3. Tokenize the concatenated conversation;
  4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
  """
  # add end signal and concatenate together
  conversations = []
  for source in sources:
    header = f"{conversation_lib.conv_templates[conv_mode].system}\n\n"
    conversation = _add_speaker_and_signal(header, source)
    conversations.append(conversation)
  # tokenize conversations
  conversations_tokenized = _tokenize_fn(conversations, tokenizer)
  input_ids = conversations_tokenized["input_ids"]
  targets = copy.deepcopy(input_ids)
  for target, source in zip(targets, sources):
    tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)[
      "input_ids_lens"
    ]
    speakers = [sentence["from"] for sentence in source]
    _mask_targets(target, tokenized_lens, speakers, only_mask_system)

  return dict(input_ids=input_ids, labels=targets)


def load_video(
  path,
  image_processer=None,
  frame_mode="fixed",
  fixed_frame_number=8,
  fps_number=0.5,
  frame_process_method="centercrop",
):
  if os.path.isfile(path):
    video_reader = decord.VideoReader(path, num_threads=1, ctx=decord.cpu(0))
    decord.bridge.set_bridge("torch")
    video_len = len(video_reader)

    if frame_mode == "fixed":
      video = video_reader.get_batch(
        np.linspace(0, video_len - 1, fixed_frame_number).astype(np.int_)
      ).byte()  # 8, height,width,3
      video = video.permute(3, 0, 1, 2)  # 3 x 8 x height x width
    elif frame_mode == "fps":
      fps_offset = int(round(video_reader.get_avg_fps()) / fps_number)
      video = video_reader.get_batch(range(0, video_len, fps_offset)).byte()
      video = video.permute(3, 0, 1, 2)  # 3 x n x height x width
    input_mean = [
      0.48145466,
      0.4578275,
      0.40821073,
    ]  # Consistent with clilp preprocessing
    input_std = [
      0.26862954,
      0.26130258,
      0.27577711,
    ]  # Consistent with clilp preprocessing
    crop_size, scale_size = 224, 256
    trans = transforms.Compose(
      [
        video_transform.TensorToNumpy(),
        video_transform.Resize(scale_size),
        video_transform.CenterCrop(crop_size),
        video_transform.ClipToTensor(channel_nb=3),
        video_transform.Normalize(mean=input_mean, std=input_std),
      ]
    )
    video = trans(video)
  else:
    video_frames = list(Path(path).rglob("*"))
    if frame_mode == "fixed":
      video_frames = [
        video_frames[i]
        for i in np.linspace(0, len(video_frames) - 1, fixed_frame_number).astype(
          np.int_
        )
      ]
    elif frame_mode == "fps":
      raise ValueError("Input folder is not support this frame mode")
    else:
      raise ValueError('Frame mode only supports "fps" or "fixed"')
    video_frames = [Image.open(str(path)) for path in video_frames]

    if frame_process_method == "resize":
      min_length = min(video_frames[0].size)
      resize = transforms.Resize([min_length, min_length])
      video_frames = [resize(frame) for frame in video_frames]
      # test_frame = video_frames[0]

    video = image_processer.preprocess(video_frames, return_tensors="pt")[
      "pixel_values"
    ]
    video = video.permute(1, 0, 2, 3)
  return video


def extract_frames(video_bytes, num_frames=8):
  # Create a memory-mapped file from the bytes
  container = av.open(io.BytesIO(video_bytes))

  # Find the video stream
  visual_stream = next(iter(container.streams.video), None)
  if not visual_stream:
    return None

  # Extract video properties
  total_frames = visual_stream.frames

  # Calculate the interval to capture the frames
  interval = max(total_frames // num_frames, 1)

  # Initialize arrays to store frames
  frames_array = []
  frame_indices = set(range(0, total_frames, interval))  # Indices of frames to capture
  frame_counter = 0

  # Extract frames
  for packet in container.demux([visual_stream]):
    for frame in packet.decode():
      if frame_counter in frame_indices:
        img_array = np.array(frame.to_image())
        frames_array.append(img_array)
        if len(frames_array) >= num_frames:
          break
      frame_counter += 1
    if len(frames_array) >= num_frames:
      break

  return np.array(frames_array)


def load_video_hf(repo_path, hf_video_path, frame_mode="fixed", fixed_frame_number=8):
  # hf_path = "datasets/Vividbot/vast2m_vi/video/shard_0/video_mp4"
  fs = HfFileSystem()
  zip_path = f"datasets/{repo_path}/{hf_video_path.split('/')[0]}.zip"
  zip_folder = fs.open(zip_path)
  zip_ref = zipfile.ZipFile(zip_folder, "r")
  video = zip_ref.read(hf_video_path)

  if frame_mode == "fixed":
    video = extract_frames(video, fixed_frame_number)  # 8, height,width,3
    video = torch.from_numpy(video)
    video = video.permute(3, 0, 1, 2)  # 3 x 8 x height x width
  elif frame_mode == "fps":
    raise ValueError("Input folder is not support this frame mode")
  else:
    raise ValueError('Frame mode only supports "fps" or "fixed"')
  input_mean = [
    0.48145466,
    0.4578275,
    0.40821073,
  ]  # Consistent with clilp preprocessing
  input_std = [
    0.26862954,
    0.26130258,
    0.27577711,
  ]  # Consistent with clilp preprocessing
  crop_size, scale_size = 224, 256
  trans = transforms.Compose(
    [
      video_transform.TensorToNumpy(),
      video_transform.Resize(scale_size),
      video_transform.CenterCrop(crop_size),
      video_transform.ClipToTensor(channel_nb=3),
      video_transform.Normalize(mean=input_mean, std=input_std),
    ]
  )
  video = trans(video)
  return video


def load_video_hf_with_fallback(
  repo_path, hf_video_path, frame_mode="fixed", fixed_frame_number=8
):
  # try to load the video from the hf path
  # if it fails, iterate through the shuffled fallback paths and try to load the video from each until one succeeds
  try:
    return load_video_hf(repo_path, hf_video_path, frame_mode, fixed_frame_number)
  except Exception as e:
    logger.warning(
      f"Failed to load video from hf path {hf_video_path}: {e} - Retrying with fallback paths..."
    )
    for fallback_path in random.sample(
      FALLBACK_HF_VIDEO_PATHS, len(FALLBACK_HF_VIDEO_PATHS)
    ):
      try:
        return load_video_hf(repo_path, fallback_path, frame_mode, fixed_frame_number)
      except Exception as e:
        logger.warning(f"Failed to load video from fallback path {fallback_path}: {e}")
        continue


def load_image_hf(repo_path, hf_image_path):
  fs = HfFileSystem()
  zip_path = f"datasets/{repo_path}/{hf_image_path.split('/')[0]}.zip"
  zip_folder = fs.open(zip_path)
  zip_ref = zipfile.ZipFile(zip_folder, "r")
  image = zip_ref.read(hf_image_path)
  image = Image.open(io.BytesIO(image))
  return image
