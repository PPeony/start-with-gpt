import os

from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import  DashScopeEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

def DeepSeek():
    return ChatOpenAI(
        model= "deepseek-chat",
        api_key= os.environ.get("deepseek"),
        base_url="https://api.deepseek.com",
    )

def TongyiEmbedding()->DashScopeEmbeddings:
    api_key=os.environ.get("DashScope")
    return DashScopeEmbeddings(dashscope_api_key=api_key,
                           model="text-embedding-v1")

def QdrantVecStoreFromDocs(docs:List[Document],collection_name:str):
    eb=TongyiEmbedding()
    return QdrantVectorStore.from_documents(docs,eb,url="http://127.0.0.1:6333",collection_name=collection_name)

def QdrantVecStore(collection_name:str):
    eb=TongyiEmbedding()
    return  QdrantVectorStore.\
        from_existing_collection(embedding=eb,
         url="http://127.0.0.1:6333",
          collection_name=collection_name)