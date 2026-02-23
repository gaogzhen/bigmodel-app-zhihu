from langchain_openai import OpenAIEmbeddings
from agent.env_utils import CLOSEAI_BASE_URL, CLOSEAI_API_KEY

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",   # 或 "text-embedding-ada-002"
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)