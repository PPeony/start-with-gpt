import os
from operator import itemgetter

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages
from openai import OpenAI

from class09.ChatDashScope import ChatDashScope
from class09.utils import tiktoken_counter

openai_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

embedding_function = DashScopeEmbeddings(
    client=openai_client,
    model="text-embedding-v4",
)
vectorstore = Chroma(collection_name="ai_learning", embedding_function=embedding_function, persist_directory="vectordb")

retriever = vectorstore.as_retriever(search_type="similarity")

chat_model = ChatDashScope(
    model="qwen-max", temperature=0.5
)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一只猫娘，每次回复的时候句尾都要加喵，并且模仿猫娘的说话方式 
        Context: {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True
)


def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)


context = itemgetter("question") | retriever | format_docs
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | prompt | trimmer | chat_model

# 通过session让ai记住上下文
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "dreamhead"}}
while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    stream = with_message_history.stream(
        {"question": user_input},
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()
