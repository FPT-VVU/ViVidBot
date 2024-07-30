from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from vividbot.data.task.vivid_instruct_65k.utils.llms import FALLBACK_LLM, LLM
from vividbot.data.task.vivid_instruct_65k.utils.prompts import get_generate_qa_prompt


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
  DEDUP_DESCRIPTION_PROMPT = """Identify and remove the duplicated part (usually at the end) of the following description in Vietnamese language.
Only keep the unique part of the description. If there is no duplicated part, please leave the description as is.
Return only the deduplicated description (or the original description if no deduplicated part was found) without any additional information."""

  return (
    ChatPromptTemplate.from_messages(
      [
        ("system", DEDUP_DESCRIPTION_PROMPT),
        (
          "human",
          "Description:\n\n{message}\n\nDeduplicated description:",
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
            "Description:\n\n{message}\n\nDeduplicated description:",
          ),
        ]
      )
      | FALLBACK_LLM
      | StrOutputParser()
    ]
  )
