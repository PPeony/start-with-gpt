from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个优秀的翻译专家，非常擅长把中文翻译为{language}"),
        ("user", "{text}")
    ]
)

prompt = prompt_template.invoke({"language": "英文", "text": "欢迎使用 Langchain 进行大模型开发"})

print(prompt, "\n")

# prompt.to_messages() 返回符合上一节预期的 message
print(prompt.to_messages(), "\n")

model_name = "gpt-4o-mini"
chat = ChatOpenAI(model_name=model_name)

# 用于上一节相同的方式进行问答
res = chat.invoke(prompt.to_messages())
print(res.content, "\n")

# 简化写法
res = chat.invoke(prompt)
print(res.content, "\n")