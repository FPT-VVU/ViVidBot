sudo apt update -y && \
sudo apt install python3-pip unzip ffmpeg pipx -y && \
pipx install poetry && pipx ensurepath && \
source ~/.bashrc && \
pip install "huggingface_hub[cli]" && \
huggingface-cli login --token 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 && \
git clone https://dminhvu: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@github.com/FPT-VVU/ViVidBot && \
cd ViVidBot && \
git checkout data/vivid-instruct-65k && \
poetry install && \
cd ~/ && mkdir data && cd data && wget https://vividbot.s3.ap-southeast-1.amazonaws.com/vivid_instruct_65k.zip && \
unzip vivid_instruct_65k.zip -d vivid_instruct_65k && \
cd ~/ViVidBot

export $(grep -v '^#' .env | xargs)

poetry run python3 vividbot/data/task/vivid_instruct_65k/video.py