import torch
from peft import PeftConfig, PeftModel

from vividbot.valley.model.valley_model import VividGPTForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_name = "/content/stage2/steps_15000"

config = PeftConfig.from_pretrained(model_name)
print(config)

model = VividGPTForCausalLM.from_pretrained(
  model_name, device_map="auto", torch_dtype=torch.float16
).to(device)
print(model)

model = PeftModel.from_pretrained(
  model, model_name, device_map="auto", torch_dtype=torch.float16
).to(device)
print(model)

model = model.merge_and_unload().half()
model.save_pretrained("/content/vivid-4b")
