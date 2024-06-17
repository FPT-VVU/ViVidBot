```python
pip install -r requirements.txt

huggingface-cli login --token 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

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Vividbot/vast2m_vi", filename="vast2m_vi_all.json", repo_type="dataset", local_dir="/content")

bash valley/train/train.sh valley/configs/experiment/valley_stage1.yaml
```
