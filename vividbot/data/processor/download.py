import datetime
import os
import time

import yt_dlp
from yt_dlp.utils import download_range_func

from vividbot.data.processor.base import BaseProcessor


class YoutubeDownloader(BaseProcessor):
  """
  This class and its children are used to download youtube videos.
  """

  def __init__(
    self,
  ):
    self.opts = {
      "format": "best[ext=mp4]/best",
      "quiet": True,
      "fixup": "never",
      "no_warnings": True,
      "force_keyframes_at_cuts": True,
      "downloader": "aria2c",
      "geo_bypass_country": "VN",
    }

  def process(
    self,
    video_id: str | None,
    video_id_with_chunk_id: str | None,
    start: str | int = None,  # if int, it is the start time
    end: str | int = None,  # if int, it is the end time
    path: str = "",
  ):
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=True)

    url_yt = "https://www.youtube.com/watch?v=" + video_id

    if start is None or end is None:
      self.opts["outtmpl"] = (
        path
        + "/"
        + f"{video_id_with_chunk_id if video_id_with_chunk_id is not None else video_id}.%(ext)s"
      )
      with yt_dlp.YoutubeDL(self.opts) as ydl:
        ydl.download(url_yt)
      return

    if isinstance(start, str):
      start = time.strptime(start.split(".")[0], "%H:%M:%S")
      start = datetime.timedelta(
        hours=start.tm_hour, minutes=start.tm_min, seconds=start.tm_sec
      ).total_seconds()

    if isinstance(end, str):
      end = time.strptime(end.split(".")[0], "%H:%M:%S")
      end = datetime.timedelta(
        hours=end.tm_hour, minutes=end.tm_min, seconds=end.tm_sec
      ).total_seconds()

    self.opts["download_ranges"] = download_range_func(
      chapters=None, ranges=[(start, end)]
    )
    self.opts["outtmpl"] = (
      path
      + "/"
      + f"{video_id_with_chunk_id if video_id_with_chunk_id is not None else video_id}.%(ext)s"
    )

    with yt_dlp.YoutubeDL(self.opts) as ydl:
      ydl.download(url_yt)
