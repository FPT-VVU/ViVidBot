from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

GROQ_LLAMA3_1_70B = ChatGroq(
  model="llama-3.1-70b-versatile",
  max_retries=0,
  temperature=1,
  streaming=False,
)
TOGETHER_LLAMA3_1_70B = ChatTogether(
  model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
  max_retries=0,
  temperature=1,
  streaming=False,
)
CLAUDE_3_HAIKU = ChatAnthropic(
  model="claude-3-haiku-20240307",
  max_retries=0,
  temperature=1,
  streaming=False,
)
GPT_4O_MINI = ChatOpenAI(
  model="gpt-4o-mini",
  max_retries=0,
  temperature=1,
  streaming=False,
)

LLM = GROQ_LLAMA3_1_70B.with_fallbacks(
  [TOGETHER_LLAMA3_1_70B, GPT_4O_MINI, CLAUDE_3_HAIKU]
)
