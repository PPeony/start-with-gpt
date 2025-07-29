from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="vectordb"
)
documents = vectorstore.similarity_search("高桥李依上小学之前想要做的职业是什么")
print(documents)