import os
import yt_dlp
import datetime
import time
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
            "format": "best[ext=mp4]",
            "quiet": True,
            "fixup": "never",
            "no_warnings": True,
            "force_keyframes_at_cuts": True,
        }

    def process(self, url_id: str, start: str, end: str, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        url_yt = "https://www.youtube.com/watch?v=" + url_id

        start = time.strptime(start.split(".")[0], "%H:%M:%S")
        start = datetime.timedelta(
            hours=start.tm_hour, minutes=start.tm_min, seconds=start.tm_sec
        ).total_seconds()
        end = time.strptime(end.split(".")[0], "%H:%M:%S")
        end = datetime.timedelta(
            hours=end.tm_hour, minutes=end.tm_min, seconds=end.tm_sec
        ).total_seconds()

        self.opts["download_ranges"] = download_range_func(None, [(start, end)])
        self.opts["outtmpl"] = path + "/" + f"{url_id}.%(ext)s"

        with yt_dlp.YoutubeDL(self.opts) as ydl:
            ydl.download(url_yt)
