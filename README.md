# VIVIDBOT
```python
pip install -r requirements.txt

sudo apt-get install unzip

huggingface-cli login --token ...

## download data for pretrain stage
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Vividbot/vast2m_vi", filename="vast2m_vi_all.json", repo_type="dataset", local_dir="/content")
hf_hub_download(repo_id="Vividbot/instruct500k_vi", filename="instruct500k_vi_all.json", repo_type="dataset", local_dir="/content")

# download model pretrain
hf_hub_download(repo_id="Vividbot/vividbot_pretrain", filename="output/vividbot-pretrained.zip", repo_type="model", local_dir="/content")

unzip /content/output/vividbot-pretrained.zip -d /content/model_pretrain

# download data finetune
hf_hub_download(repo_id="Vividbot/videoinstruck100k", filename="videoinstruck100_vi_all.json", repo_type="dataset", local_dir="/content")
hf_hub_download(repo_id="Vividbot/vividbot_video", filename="vividbot_video_65k_all.json", repo_type="dataset", local_dir="/content")
hf_hub_download(repo_id="Vividbot/vividbot_image", filename="vividbot_image_56k_all.json", repo_type="dataset", local_dir="/content")
hf_hub_download(repo_id="Vividbot/llava-instruct-150k-vi", filename="llava_instruck_150k_all.json", repo_type="dataset", local_dir="/content")

# run pretrained
bash valley/train/train.sh valley/configs/experiment/valley_stage1.yaml

# run finetuned
bash valley/train/train.sh valley/configs/experiment/valley_stage2.yaml
```
