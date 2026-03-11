import os

from dotenv import load_dotenv

load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL')

CLOSEAI_API_KEY = os.getenv('CLOSEAI_API_KEY')
CLOSEAI_BASE_URL = os.getenv('CLOSEAI_BASE_URL')
COMPOSIO_API_KEY = os.getenv('COMPOSIO_API_KEY')
COMPOSIO_EXTERNAL_USER_ID=os.getenv('COMPOSIO_EXTERNAL_USER_ID')
