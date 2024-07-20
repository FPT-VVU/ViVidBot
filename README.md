# VIVIDBOT
```python
pip install -r requirements.txt

huggingface-cli login --token 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

## download data for pretrain stage
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Vividbot/vast2m_vi", filename="vast2m_vi_all.json", repo_type="dataset", local_dir="/content")
hf_hub_download(repo_id="Vividbot/instruct500k_vi", filename="instruct500k_vi_all.json", repo_type="dataset", local_dir="/content")

# download model pretrain
hf_hub_download(repo_id="Vividbot/vividbot_pretrain", filename="output/vividbot-pretrained.zip", repo_type="model", local_dir="/content")

unzip /content/vividbot-pretrained.zip -d /content/model_pretrain

# download data finetune
hf_hub_download(repo_id="Vividbot/videoinstruck100k", filename="videoinstruck100_vi_all.json", repo_type="dataset", local_dir="/content")

bash valley/train/train.sh valley/configs/experiment/valley_stage1.yaml
```
