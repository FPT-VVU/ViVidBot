from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from vividbot.data.task.vivid_instruct_65k.utils.llms import FALLBACK_LLM, LLM
from vividbot.data.task.vivid_instruct_65k.utils.prompts import GENERATE_QA_PROMPT

GENERATE_QA_PAIRS_CHAIN = (
  ChatPromptTemplate.from_messages(
    [
      ("system", GENERATE_QA_PROMPT),
      (
        "human",
        "Generate QA pairs as instructed from this description:\n\n{message}\n\n5 QA pairs:",
      ),
    ]
  )
  | LLM
  | JsonOutputParser()
).with_fallbacks(
  fallbacks=(
    ChatPromptTemplate.from_messages(
      [
        ("system", GENERATE_QA_PROMPT),
        (
          "human",
          "Generate QA pairs as instructed from this description:\n\n{message}\n\n5 QA pairs:",
        ),
      ]
    )
    | FALLBACK_LLM
    | JsonOutputParser()
  )
)
