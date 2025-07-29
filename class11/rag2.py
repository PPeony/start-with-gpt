import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

embedding_function = DashScopeEmbeddings(
    client=openai_client,
    model="text-embedding-v4",
)

vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=embedding_function,
    persist_directory="vectordb"
)
documents = vectorstore.similarity_search("高桥李依上小学之前想要做的职业是什么")
print(documents)