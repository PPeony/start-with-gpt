from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

chat_model = ChatOpenAI(model="gpt-4o-mini")

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一只猫娘，每次回复的时候句尾都要加喵，并且模仿猫娘的说话方式"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# 通过session让ai记住上下文
with_message_history = RunnableWithMessageHistory(prompt | chat_model, get_session_history)

config = {"configurable": {"session_id": "dreamhead"}}
while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    stream = with_message_history.stream(
        [HumanMessage(content=user_input)],
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()
