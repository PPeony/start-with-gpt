from functools import wraps

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from class09.ChatDashScope import ChatDashScope


# @tool 可以提取函数名变成工具名，提取参数变成工具的参数，
# 还有一点就是，它可以提取函数的 Docstring 作为工具的描述。这样一来，calculate 就从一个普通的函数变成了一个工具。
@tool
def calculate(what: str) -> float:
    """Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary"""
    return eval(what)


@tool
def ask_fruit_unit_price(fruit: str) -> str:
    """Asks the user for the price of a fruit"""
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Banana unit price is 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)


prompt = PromptTemplate.from_template('''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important:
- You must use the tools provided.
- After "Action Input", STOP GENERATING. Do not write "Observation" or "Final Answer".
- Wait for the system to provide the result.
- Only after seeing an Observation, you may proceed to Final Answer.
- Do not make up tool results.

Begin!

Question: {input}
Thought:{agent_scratchpad}''')
# agent_scratchpad 是在 Agent 的执行过程中，存放中间过程的，你可以把它理解成我们上一讲的聊天历史部分。
# 要加这部分，不然报错
# Important:
# - You must use the tools provided.
# - After "Action Input", STOP GENERATING. Do not write "Observation" or "Final Answer".
# - Wait for the system to provide the result.
# - Only after seeing an Observation, you may proceed to Final Answer.
# - Do not make up tool results.

tools = [calculate, ask_fruit_unit_price]
model = ChatDashScope(model="qwen-max")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({
    "input": "What is the total price of 3 kg of apple and 2 kg of banana?"
})
print(result)
