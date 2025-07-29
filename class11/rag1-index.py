import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings, TextEmbedEmbeddings, DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

embedding_function = DashScopeEmbeddings(
    client=openai_client,
    model="text-embedding-v4",
)

loader = TextLoader("source.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=embedding_function,
    persist_directory="vectordb"
)
vectorstore.add_documents(splits)

