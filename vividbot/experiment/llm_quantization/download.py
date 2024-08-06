from pathlib import Path

from huggingface_hub import snapshot_download

model_id = "Vividbot/vividbot_finetune"
snapshot_download(
  repo_id=model_id,
  local_dir=f"{Path.home()}/vividbot_finetune",
  local_dir_use_symlinks=False,
  revision="main",
  # ignore_patterns=["*.zip"],
)
