# Processing Data
Run this before starting any task
```python
pip install requirements.txt
```
# VAST2M Vi
## Download from youtube
1. Go to this link to download folder chunk: https://drive.google.com/drive/folders/1RjCk2hf52xTKX2zBjwyEbVvhBJ4EjSRN?usp=drive_link
2. Install ffmpeg and login to hugging face account
```python
sudo apt install ffmpeg
huggingface-cli login --token your token --add-to-git-credential True
```
3. Edit file in path: vividbot/data/scripts as bellow
```python
python vividbot/data/task/vast2M.py --task "download" \
                                    --file-path "folder (files) of chunks" \
                                    --batch-size 100 \
                                    --repo-id "Vividbot/vast2m_vi" \
                                    --upload-to-hub \
                                    --num-shards 10 \
                                    --output-dir "output folder" \
                                    --cache-dir "cached folder" \
                                    --num-proc "adjust number base on your computer"
```
4. Run
```python
bash vividbot/data/scripts/vast2m.sh
```
5. Check status in the sheet: https://docs.google.com/spreadsheets/d/1xIwos2TttQYi-iVgxru15Ydc0L65ou3OC9ev5CzsusE/edit#gid=0
