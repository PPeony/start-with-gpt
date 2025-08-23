from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from DeepSeek应用开发实战.class27.llm import QdrantVecStore, DeepSeek


def clearstr(s):
    filter_chars = ['\n', '\r', '\t', '\u3000','  ']
    for char in filter_chars:
        s=s.replace(char,'')
    return s

def format_docs(docs):
    return "\n\n".join(clearstr(doc.page_content) for doc in docs)

def models_search(query:str):
    vec_store=QdrantVecStore(collection_name="data")
    prompt="""
SYSTEM
你是一个 go 语言编程专家，擅长根据问题生成模型实体类代码。
使用上下文来创建实体struct。你只需输出golang代码，无需任何解释和说明。不要将代码放到 ```go ``` 中。

上下文：
{context}

模型名称例子：UserModel

HUMAN
模型或数据表信息：{question}
"""

    retriver=vec_store.as_retriever(search_kwargs={"k":10})
    llm=DeepSeek()
    prompt=ChatPromptTemplate.from_template(prompt)
    chain = {"context": retriver | format_docs,
             "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    ret=chain.invoke(query)
    return ret

@tool
def modelsTool(model_name: str):
    """该工具可用于生成实体类代码"""

    ret=models_search(model_name)
    return ret

def code_search(query:str):
    vec_store=QdrantVecStore(collection_name="code")
    prompt="""
SYSTEM
你是一个 go 语言编程专家，擅长根据问题以及代码库的代码进行代码生成。
使用上下文来生成代码。你只需输出golang代码，无需任何解释和说明。不要将代码放到 ```go ``` 中。

上下文：
{context}

HUMAN
问题：{question}
"""

    retriver=vec_store.as_retriever(search_kwargs={"k":3})
    llm=DeepSeek()
    prompt=ChatPromptTemplate.from_template(prompt)
    chain = {"context": retriver | format_docs,
             "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    ret=chain.invoke(query)
    return ret

@tool
def middlewareTool(question: str):
    """该工具可用于生成中间件函数，参数需传入具体的生成代码的需求，例如：跨域中间件"""

    ret=code_search(question)
    return ret

if __name__ == "__main__":
    ret=code_search("跨域中间件")
    print(ret)
