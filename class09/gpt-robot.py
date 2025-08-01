from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from class09.ChatDashScope import ChatDashScope

# 创建 Chat 模型实例
chat_model = ChatDashScope(
    model="qwen-max", temperature=0.5
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 通过session让ai记住上下文
with_message_history = RunnableWithMessageHistory(chat_model, get_session_history)

config = {"configurable": {"session_id": "dreamhead"}}
while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    stream = with_message_history.stream(
        {"input": user_input},
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()