from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()

model_id = "Vividbot/vivid-pretrained-4b-gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
  path_or_fileobj=f"{Path.home()}/vividbot_pretrain/vivid-4b-q8_0.gguf",
  path_in_repo="vivid-4b-q8_0.gguf",
  repo_id=model_id,
)
