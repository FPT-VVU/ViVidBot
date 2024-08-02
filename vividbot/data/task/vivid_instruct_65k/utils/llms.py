import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatDeepInfra
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

load_dotenv()
FIREWORKS_LLAMA3_1_405B = ChatFireworks(
  model="accounts/fireworks/models/llama-v3p1-405b-instruct",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
LEPTON_LLAMA3_1_405B = ChatOpenAI(
  base_url="https://llama3-1-405b.lepton.run/api/v1/",
  api_key=os.getenv("LEPTON_API_KEY"),
  model="llama3-1-405b",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
FIREWORKS_LLAMA3_1_70B = ChatFireworks(
  model="accounts/fireworks/models/llama-v3p1-70b-instruct",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
GROQ_LLAMA3_1_70B = ChatGroq(
  model="llama-3.1-70b-versatile",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
TOGETHER_LLAMA3_1_70B = ChatTogether(
  model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
DEEPINFRA_LLAMA3_1_70B = ChatDeepInfra(
  model="meta-llama/Meta-Llama-3.1-70B-Instruct",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
DEEPINFRA_LLAMA3_1_405B = ChatDeepInfra(
  model="meta-llama/Meta-Llama-3.1-405B-Instruct",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
CLAUDE_3_HAIKU = ChatAnthropic(
  model="claude-3-haiku-20240307",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
GPT_4O_MINI = ChatOpenAI(
  model="gpt-4o-mini",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
GEMINI_1_5_FLASH = ChatGoogleGenerativeAI(
  model="gemini-1.5-flash",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)
GEMINI_1_5_PRO = ChatGoogleGenerativeAI(
  model="gemini-1.5-pro",
  max_retries=0,
  temperature=0,
  streaming=False,
  max_tokens=2048,
  top_p=0.7,
)

LLM = FIREWORKS_LLAMA3_1_405B.with_fallbacks(
  [
    DEEPINFRA_LLAMA3_1_405B,
    GEMINI_1_5_FLASH,
    FIREWORKS_LLAMA3_1_70B,
    DEEPINFRA_LLAMA3_1_70B,
    TOGETHER_LLAMA3_1_70B,
    GROQ_LLAMA3_1_70B,
    # LEPTON_LLAMA3_1_405B,
    # FIREWORKS_LLAMA3_1_405B,
    GPT_4O_MINI,
    CLAUDE_3_HAIKU,
  ]
)

FALLBACK_LLM = GPT_4O_MINI.with_fallbacks([CLAUDE_3_HAIKU])
