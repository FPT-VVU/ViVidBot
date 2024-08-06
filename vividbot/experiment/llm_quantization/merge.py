import os
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import BitsAndBytesConfig

from vividbot.valley.model.valley_model import VividMptForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_name = "Vividbot-vivid-4b-instruct"
bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16,
)

config = PeftConfig.from_pretrained(model_name)
print(config)

model = VividMptForCausalLM.from_pretrained(
  model_name,
  quantization_config=bnb_config,
  device_map="auto",
  torch_dtype=torch.bfloat16,
  use_safetensors=True,
).to(device)
print(model)

model = PeftModel.from_pretrained(
  model, model_name, device_map="auto", torch_dtype=torch.bfloat16
).to(device)
print(model)

model = model.merge_and_unload()
os.makedirs(f"{Path.home()}/vivid-4b-merged", exist_ok=True)
model.save_pretrained(f"{Path.home()}/vivid-4b-merged")
