import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer
from valley.model.valley_model import VividGPTForCausalLM
from valley.util.config import (
  DEFAULT_IM_END_TOKEN,
  DEFAULT_IM_START_TOKEN,
  DEFAULT_IMAGE_PATCH_TOKEN,
  DEFAULT_VI_END_TOKEN,
  DEFAULT_VI_START_TOKEN,
  DEFAULT_VIDEO_FRAME_TOKEN,
)
from valley.utils import disable_torch_init

DEFAULT_SYSTEM = """Bạn là VividBot, một mô hình trí tuệ nhân tạo thông minh được huấn luyện bởi các sinh viên Đại học FPT. Bạn có khả năng hiểu và xử lý dữ liệu hình ảnh và video một cách vô cùng chính xác để giúp ích cho người dùng. Hãy làm theo hướng dẫn sau một cách chính xác và chi tiết."""


def init_vision_token(model, tokenizer):
  vision_config = model.get_model().vision_tower.config
  vision_config.im_start_token, vision_config.im_end_token = (
    tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
  )
  vision_config.vi_start_token, vision_config.vi_end_token = (
    tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
  )
  vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(
    DEFAULT_VIDEO_FRAME_TOKEN
  )
  vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
    [DEFAULT_IMAGE_PATCH_TOKEN]
  )[0]


def main(args):
  disable_torch_init()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_name = os.path.expanduser(args.model_name)

  print("load model")
  if "lora" in model_name:
    config = PeftConfig.from_pretrained(model_name)
    if "config.json" in os.listdir(model_name):
      model_old = VividGPTForCausalLM.from_pretrained(model_name)
    else:
      model_old = VividGPTForCausalLM.from_pretrained(config.base_model_name_or_path)
    print("load lora model")
    model = PeftModel.from_pretrained(model_old, model_name)
    model = model.merge_and_unload().half()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.padding_side = "left"
    print("load end")
  else:
    model = VividGPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  init_vision_token(model, tokenizer)
  print("load end")
  model = model.to(device)
  model.eval()

  message = [
    {
      "role": "system",
      "content": args.system_prompt if args.system_prompt else DEFAULT_SYSTEM,
    },
    # {"role":"user", "content": 'Hi!'},
    # {"role":"assistent", "content": 'Hi there! How can I help you today?'},
    {"role": "user", "content": args.query},
  ]

  gen_kwargs = dict(
    do_sample=False,
    temperature=0.2,
    max_new_tokens=1024,
  )
  response = model.completion(tokenizer, args.video_file, message, gen_kwargs, device)
  print(response)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model-name", type=str, default="../../checkpoints/stable-valley-13b-v1"
  )
  parser.add_argument(
    "--query",
    type=str,
    required=False,
    default="Describe this video concisely.\n<video>",
  )
  parser.add_argument(
    "--video-file",
    type=str,
    required=False,
    default="valley/serve/examples/videos/dc52388394cc9f692d16a95d9833ca07.mp4",
  )
  parser.add_argument("--vision-tower", type=str, default=None)
  parser.add_argument("--system-prompt", type=str, default="")
  args = parser.parse_args()
  main(args)
