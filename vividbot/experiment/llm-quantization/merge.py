from peft import PeftModel

from vividbot.valley.model.valley_model import VividGPTForCausalLM

base_model_name = "/home/ct-minhvu/vivid-4b"
adapter_model_name = "/home/ct-minhvu/vivid-4b"
model = VividGPTForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)

model = model.merge_and_unload()
model.save_pretrained("merged_adapters")
