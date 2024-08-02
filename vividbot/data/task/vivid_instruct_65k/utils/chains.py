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
