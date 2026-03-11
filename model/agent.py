from langchain.agents import create_agent
from composio import Composio

from model.llm import llm_chat_deepseek as llm
from env_utils import COMPOSIO_API_KEY,COMPOSIO_EXTERNAL_USER_ID



# Initialize Composio
composio = Composio(api_key=COMPOSIO_API_KEY)

external_user_id = COMPOSIO_EXTERNAL_USER_ID

# Create a tool router session
session = composio.create(
    user_id=external_user_id,
)

# Get tools from the session (native)
tools = session.tools()

# Create model
mcp_agent = create_agent(
    tools=tools,
    model=llm,
    system_prompt="你是一个智能助手，可以使用各种工具回答问题。"
)


def send_email(to: str, subject: str, body: str):
    """发送邮件"""
    email = {
        "to": to,
        "subject": subject,
        "body": body,
    }

    # 邮件发送逻辑 todo

agent = create_agent(
    llm,
    tools=[send_email],
    system_prompt="你是一个邮件助手，请始终使用 send_email 工具。"
)