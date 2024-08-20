import json
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

from vividbot.data.processor.huggingface import HuggingFaceProcessor

hf_processor = HuggingFaceProcessor()
data = []

with open(f"{Path.home()}/data/MetaMathQA-40K_vi.json", "rb") as f:
  raw_data = orjson.loads(f.read())
  np.random.seed(2103)
  # # pick random 25k samples
  # raw_data = np.random.choice(raw_data, 100000, replace=False)
  # raw_data = np.random.choice(raw_data, 40000, replace=False)
  print(len(raw_data))

  for i in tqdm(range(10000)):
    while True:
      # pick random item
      rand_int = np.random.randint(0, len(raw_data))
      item = raw_data[rand_int]
      # print(item)
      human_message = item["query_vi"]
      gpt_message = item["response_vi"]

      if len(human_message.split(" ")) + len(gpt_message.split(" ")) < 2500:
        data.append(
          {
            "id": len(data),
            "conversations": [
              {"from": "human", "value": human_message},
              {"from": "gpt", "value": gpt_message},
            ],
          }
        )
        break

with open(f"{Path.home()}/data/vietnamese_meta_math.json", "w") as f:
  json.dump(data, f, ensure_ascii=False, indent=2)
