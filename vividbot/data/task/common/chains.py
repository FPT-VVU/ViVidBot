import os
from pathlib import Path

from langchain.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from vividbot.data.task.vivid_instruct_65k.utils.llms import FALLBACK_LLM, LLM
from vividbot.data.task.vivid_instruct_65k.utils.prompts import get_generate_qa_prompt

os.makedirs(f"{Path.home()}/.cache", exist_ok=True)
set_llm_cache(SQLiteCache(database_path=f"{Path.home()}/.cache/.langchain.db"))


def get_generate_qa_pairs_chain():
  GENERATE_QA_PROMPT = get_generate_qa_prompt()

  return (
    ChatPromptTemplate.from_messages(
      [
        ("system", GENERATE_QA_PROMPT),
        (
          "human",
          "Generate QA pairs as instructed from this description:\n\n{message}\n\nQA pairs:",
        ),
      ]
    )
    | LLM
    | JsonOutputParser()
  ).with_fallbacks(
    [
      ChatPromptTemplate.from_messages(
        [
          ("system", GENERATE_QA_PROMPT),
          (
            "human",
            "Generate QA pairs as instructed from this description:\n\n{message}\n\nQA pairs:",
          ),
        ]
      )
      | FALLBACK_LLM
      | JsonOutputParser()
    ]
  )


def get_dedup_description_chain():
  DEDUP_DESCRIPTION_PROMPT = """Rewrite the text below by removing the duplicated or repeated parts (usually at the end) of the following text in Vietnamese language.
Only keep the unique part and the first occurence of the duplicated/repeated parts of the text. If there is no duplicated or repeated parts, please leave the text *AS IS*.
Also, remove the incomplete sentences or phrases at the end of the text.
Return only the deduplicated text (or the original text if no duplicated or repeated part was found) without any additional information.
If the text is not in Vietnamese, please translate it to Vietnamese after deduplication."""

  return (
    ChatPromptTemplate.from_messages(
      [
        ("system", DEDUP_DESCRIPTION_PROMPT),
        (
          "human",
          "Text:\n\n{message}\n\nDeduplicated text in Vietnamese:",
        ),
      ]
    )
    | LLM
    | StrOutputParser()
  ).with_fallbacks(
    [
      ChatPromptTemplate.from_messages(
        [
          ("system", DEDUP_DESCRIPTION_PROMPT),
          (
            "human",
            "Text:\n\n{message}\n\nDeduplicated text in Vietnamese:",
          ),
        ]
      )
      | FALLBACK_LLM
      | StrOutputParser()
    ]
  )


def get_translate_chain():
  TRANSLATE_PROMPT = """Translate the following text to Vietnamese language.
FYI, those sentences are originally used to describe the content of a video.
Make it natural and fluent in Vietnamese and correct any grammatical errors if there are any.
Return only the translated text without any additional information."""

  return (
    ChatPromptTemplate.from_messages(
      [
        ("system", TRANSLATE_PROMPT),
        (
          "human",
          "Text:\n\n{message}\n\nTranslated text in Vietnamese:",
        ),
      ]
    )
    | LLM
    | StrOutputParser()
  ).with_fallbacks(
    [
      ChatPromptTemplate.from_messages(
        [
          ("system", TRANSLATE_PROMPT),
          (
            "human",
            "Text:\n\n{message}\n\nTranslated text in Vietnamese:",
          ),
        ]
      )
      | FALLBACK_LLM
      | StrOutputParser()
    ]
  )


def get_rewrite_vast_caps_chain():
  REWRITE_VAST_CAPS_PROMPT = """You're given:
1. A sequence of text captions of a video by frames in English.
2. A question in Vietnamese.
Your task is to construct a new description in Vietnamese language by combining/merging/mixing the given sequence of captions so that it answers the question but also makes sense.
Return only the new description without any additional information.

Example 1:
Captions: ["a man talks about some new product","a man is explaining about an app","a man with an fluffy hair talking about a new product","a man is talking about a business he just made","man sitting at his desk talking about a website while he is wearing a black shirt and a black t - shirt that says i shop it on it."]
Question: Bạn có thể mô tả ngắn gọn về nội dung của video không?
Description: Một người đàn ông tóc xù, mặc áo đen có in chữ "I shop it", đang ngồi tại bàn làm việc giải thích về một ứng dụng và trang web mới mà anh ấy vừa tạo ra cho doanh nghiệp của mình.

Example 2:
Captions: ["the three men are having the conversation in spanish.","a split photo is being shown with two guys talking.","a man with a suit is giving a speech about soccer.","a three men are being interviewed about a soccer game.","two men are sitting down on the news and one of them talks."]
Question: Những diễn biến nào được thể hiện trong nội dung video?
Description: Ba người đàn ông đang được phỏng vấn về một trận bóng đá trên một chương trình tin tức. Họ nói chuyện bằng tiếng Tây Ban Nha. Hình ảnh được chia thành hai phần, với hai người ngồi xuống và một người mặc vest đang phát biểu về bóng đá."""

  return (
    ChatPromptTemplate.from_messages(
      [
        ("system", REWRITE_VAST_CAPS_PROMPT),
        (
          "human",
          "Captions: {captions}\n\nQuestion: {question}\n\nDescription:",
        ),
      ]
    )
    | LLM
    | StrOutputParser()
  ).with_fallbacks(
    [
      ChatPromptTemplate.from_messages(
        [
          ("system", REWRITE_VAST_CAPS_PROMPT),
          (
            "human",
            "Captions: {captions}\n\nQuestion: {question}\n\nDescription:",
          ),
        ]
      )
      | FALLBACK_LLM
      | StrOutputParser()
    ]
  )


def get_rewrite_image_caps_chain():
  REWRITE_IMAGE_CAPS_PROMPT = """You're given:
1. A caption of an image in English.
2. A question in Vietnamese.
Your task is to construct a new description/caption in Vietnamese language by based on the given caption content so that it answers the question naturally, fluently and also makes sense.
Return only the new description without any additional information.

Example 1:
Captions: select luxury furniture 3 - inch gel memory foam mattress topper
Question: Bạn có thể mô tả ngắn gọn về hình ảnh này không?
Description: Đây là một tấm đệm lót nệm cao cấp bằng gel memory foam dày 3 inch, được thiết kế để nâng cao chất lượng giấc ngủ và tạo cảm giác thoải mái tối đa. Sản phẩm này thuộc dòng nội thất cao cấp, giúp cải thiện đáng kể sự êm ái của bề mặt ngủ.

Example 2:
Captions: a spartan helmet, laurels and laurel wreath, silhouette logo, emblem
Question: Bạn có thể giải thích ý nghĩa của hình ảnh này không?
Description: Hình ảnh này là một biểu tượng hoặc logo có ý nghĩa tượng trưng cho sức mạnh và vinh quang. Nó bao gồm một mũ giáp Sparta - biểu tượng của chiến binh và lòng dũng cảm - được bao quanh bởi vòng nguyệt quế, vốn là biểu tượng truyền thống của chiến thắng và danh dự. Thiết kế dạng bóng đen tạo nên một hình ảnh mạnh mẽ và dễ nhận biết, thích hợp để sử dụng làm biểu tượng hoặc huy hiệu cho các tổ chức muốn thể hiện tinh thần chiến đấu và thành công.
"""

  return (
    ChatPromptTemplate.from_messages(
      [
        ("system", REWRITE_IMAGE_CAPS_PROMPT),
        (
          "human",
          "Captions: {captions}\n\nQuestion: {question}\n\nDescription:",
        ),
      ]
    )
    | LLM
    | StrOutputParser()
  ).with_fallbacks(
    [
      ChatPromptTemplate.from_messages(
        [
          ("system", REWRITE_IMAGE_CAPS_PROMPT),
          (
            "human",
            "Captions: {captions}\n\nQuestion: {question}\n\nDescription:",
          ),
        ]
      )
      | FALLBACK_LLM
      | StrOutputParser()
    ]
  )
