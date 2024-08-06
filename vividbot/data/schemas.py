from typing import List, TypedDict

Conversation = TypedDict(
  "Conversation",
  {
    "from": str,
    "value": str,
  },
)


class Metadata(TypedDict):
  id: str
  conversations: List[Conversation]


class VideoMetadata(Metadata):
  video: str


class ImageMetadata(Metadata):
  image: str
