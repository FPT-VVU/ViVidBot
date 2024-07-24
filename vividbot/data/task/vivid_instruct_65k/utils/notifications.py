import datetime
import logging
from datetime import timezone

import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
  def __init__(self, webhook_url):
    self.webhook_url = webhook_url

  def send(self, body):
    requests.post(self.webhook_url, json=body)


notifier = DiscordNotifier(
  "https://discord.com/api/webhooks/1255505460040040508/n-QCTqNgp3RrsNc1hBRnXH4dfOejeH8iPTd8lqGevbSb_wAovD4xxv5ZVkVJBfVLF8vN"
)


def send_process_shard_success_message(shard: str, duration: float):
  logger.info(f"Processed shard {shard} in {duration}(s).")
  notifier.send(
    body={
      "embeds": [
        {
          "title": f"✅ ViVid Instruct 65k: Processed shard {shard}!",
          "description": f"Processed shard {shard} \
in {duration}(s). \
Visit at https://huggingface.co/datasets/Vividbot/vividbot_video/tree/main/videos.",
          "color": 2278494,
          "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        }
      ]
    }
  )
