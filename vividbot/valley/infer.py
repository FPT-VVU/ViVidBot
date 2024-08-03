import argparse
import os
import sys
from transformers import BitsAndBytesConfig
import torch

from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer

sys.path.append("/ViVidBot")
from vividbot.valley.model.valley_model import VividGPTForCausalLM
from vividbot.valley.util.config import (
  DEFAULT_IM_END_TOKEN,
  DEFAULT_IM_START_TOKEN,
  DEFAULT_IMAGE_PATCH_TOKEN,
  DEFAULT_VI_END_TOKEN,
  DEFAULT_VI_START_TOKEN,
  DEFAULT_VIDEO_FRAME_TOKEN,
)
from vividbot.valley.utils import disable_torch_init

DEFAULT_SYSTEM = (
  "Bạn là VividBot, một trợ lý ảo AI được huấn luyện bởi nhóm đồ án của Minh, Duy, và Thạch từ trường Đại học FPT TPHCM."
  "Bạn có khả năng hiểu và xử lý thông tin từ hình ảnh và video một cách vô cùng chính xác để giúp ích cho người dùng."
  "Bạn có nhiệm vụ trò chuyện và cung cấp những câu trả lời chính xác, hữu ích và rõ ràng cho người dùng."
)


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
  # device = "cuda"
  model_name = os.path.expanduser(args.model_name)
  # bnb_config = BitsAndBytesConfig(
  #     load_in_8bit=True)
      # bnb_8bit_use_double_quant=True,
      # bnb_8bit_quant_type="nf4",
      # bnb_8bit_compute_dtype=torch.float16,)
  print("load model")
  # if "lora" in model_name:
  config = PeftConfig.from_pretrained(model_name)
  if "config.json" in os.listdir(model_name):
    model_old = VividGPTForCausalLM.from_pretrained(model_name, device_map=device)
  else:
    model_old = VividGPTForCausalLM.from_pretrained(config.base_model_name_or_path, device_map=device)
  print("load lora model")
  model = PeftModel.from_pretrained(model_old, model_name, device_map=device)
  model = model.merge_and_unload()
  tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
  tokenizer.padding_side = "left"
  print("load end")
  # else:
  #     model = VividGPTForCausalLM.from_pretrained(
  #         model_name, torch_dtype=torch.float16
  #     )
  #     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  init_vision_token(model, tokenizer)
  print("load end")
  # model = model.to(device)
  model.eval()

  message = [
    {
      "role": "system",
      "content": DEFAULT_SYSTEM,
    },
    # {"role":"user", "content": 'Hi!'},
    # {"role":"assistent", "content": 'Hi there! How can I help you today?'},
    {"role": "user", "content": args.query},
  ]

  gen_kwargs = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
  )
  response = model.completion(tokenizer, args.video_file, message, gen_kwargs, device)
  print(response)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model-name", type=str, default="/ViVidBot/model/output/stage2/lora"
  )
  parser.add_argument(
    "--query",
    type=str,
    required=False,
    default="Tỉ số của trận đấu này là bao nhiêu?\n<video>",
  )
  parser.add_argument(
    "--video-file",
    type=str,
    required=False,
    default="/content/t.mp4",
  )
  parser.add_argument("--vision-tower", type=str, default=None)
  parser.add_argument("--system-prompt", type=str, default="")
  args = parser.parse_args()
  main(args)
