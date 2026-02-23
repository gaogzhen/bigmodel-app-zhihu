from openai import OpenAI
from agent.env_utils import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

# 初始化嵌入模型
embeddings = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)