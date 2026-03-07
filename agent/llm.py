from langchain_openai import OpenAI, ChatOpenAI

from agent.env_utils import CLOSEAI_BASE_URL, CLOSEAI_API_KEY, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

llm_chat_deepseek = ChatOpenAI(
    model="deepseek-r1-0528",
    temperature=0.5,
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)

llm_gpt = OpenAI(
    model="gpt-4o-mini",
    temperature=0.8,
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)