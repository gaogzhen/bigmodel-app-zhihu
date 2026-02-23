from langchain_openai import  OpenAI

from agent.env_utils import CLOSEAI_BASE_URL, CLOSEAI_API_KEY

# llm = ChatOpenAI(
#     model="deepseek-r1-0528",
#     temperature=0.8,
#     api_key=DASHSCOPE_API_KEY,
#     base_url=DASHSCOPE_BASE_URL
# )

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.8,
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)