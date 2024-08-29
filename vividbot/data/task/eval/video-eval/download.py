from vividbot.data.processor.download import YoutubeDownloader


VIDEO_URLS = [
  "https://www.youtube.com/watch?v=KkYLA_hXWVY",
  "https://www.youtube.com/watch?v=foQv8Vgi7Og",
  "https://www.youtube.com/watch?v=8SQnjwcR700",
  "https://www.youtube.com/watch?v=CdXMu55FYTg",
  "https://www.youtube.com/watch?v=OUKGsb8CpF8",
  "https://www.youtube.com/watch?v=MlYobdy0p1s",
  "https://www.youtube.com/watch?v=EunEUh07C48",
  "https://www.youtube.com/watch?v=eKPm1pCO9cM",
  "https://www.youtube.com/watch?v=DiTNn-COqFQ",
  "https://www.youtube.com/watch?v=gzcAQHzfgTc",
]

yt_downloader = YoutubeDownloader()

for video_url in VIDEO_URLS:
  print(f"Downloading video {video_url}...")
  yt_downloader.process(
    video_id=video_url.split("=")[-1],
    path="/home/dminhvu/data/eval/video-eval/videos",
    start=15,
    end=30,
  )
