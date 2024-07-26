from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
