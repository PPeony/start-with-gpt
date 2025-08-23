from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from typing_extensions import Literal
from IPython.display import Image, display
from tools import tools_by_name, llm_with_tools


# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    print("------------------")
    print(state["messages"])
    print("------------------")
    # 创建消息列表
    messages = [
                   SystemMessage(
                       content="""
            你是一个股票助手，具备以下技能：
            1. 可以使用 get_stock_info 工具查询股票信息，该工具需要两个参数：
               - code: 股票代码（如果不知道可以传空字符串）
               - name: 股票名称（如果不知道可以传空字符串）

            规则：
            1. 请给出精简的回答，不要做任何的解释和说明
            2. 如果没有匹配到工具，则只会回复"对不起，我无法回答这个问题"
            """
                   )
               ] + state["messages"]

    # 调用 LLM
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


def tool_node(state: dict):
    """Performs the tool call"""
    # print("tool_node")
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        # 将观察结果转换为字符串格式
        if isinstance(observation, list):
            # 如果是列表，将其转换为字符串表示
            observation = str(observation)

        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", "END"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    # print("should_continue")
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return "END"


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        "END": END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# 保存代理工作流程图到文件
graph_png = agent.get_graph(xray=True).draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(graph_png)

# Invoke
messages = [HumanMessage(content="请按以下计划执行：1.300750 是哪只股票的代码？2.宁德时代的股票代码是？")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()