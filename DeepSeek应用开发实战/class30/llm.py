import os

from langchain_openai import ChatOpenAI

def DeepSeek():
    return ChatOpenAI(
        model= "deepseek-chat",
        api_key= os.environ.get("deepseek"),
        base_url="https://api.deepseek.com",
    )