sudo apt update -y && \
sudo apt install python3-pip unzip ffmpeg pipx -y && \
pipx install poetry && pipx ensurepath && \
source ~/.bashrc && \
pip install "huggingface_hub[cli]" && \
huggingface-cli login --token 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 && \
git clone https://dminhvu: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@github.com/FPT-VVU/ViVidBot && \
cd ViVidBot && \
git checkout data/llava-instruct-150k-vi && \
poetry install && \
cd ~/ && mkdir llava-instruct-150k-vi && cd llava-instruct-150k-vi && \
huggingface-cli download Vividbot/llava-instruct-150k-vi shards.zip --repo-type dataset --local-dir ./ && \
unzip shards.zip && rm shards/shard_all.json \
cd ~/ViVidBot

poetry run python3 vividbot/data/task/llava_instruct_150k_vi/main.py