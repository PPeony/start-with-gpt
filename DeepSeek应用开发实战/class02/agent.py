import json
import re

from DeepSeek应用开发实战.class02.prompt import REACT_PROMPT
from llm import client

tools = [
    {
        "name": "get_closing_price",
        "description": "使用该工具获取指定股票的收盘价",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "股票名称",
                }
            },
            "required": ["name"]
        },
    },
]

def get_closing_price(name):
    if name == "青岛啤酒":
        return "67.92"
    elif name == "贵州茅台":
        return "1488.21"
    else:
        return "未搜到该股票"


instructions = "你是一个股票助手，可以回答股票相关的问题"

query = "青岛啤酒和贵州茅台的收盘价哪个贵？"

prompt = REACT_PROMPT.format(instructions=instructions,tools=tools,tool_names="get_closing_price",input=query)

messages = [{"role": "user", "content": prompt}]



def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=messages,
    )
    return response


while True:
    response = send_messages(messages)
    response_text = response.choices[0].message.content

    print("大模型的回复：")
    print(response_text)

    final_answer_match = re.search(r'Final Answer:\s*(.*)', response_text)
    if final_answer_match:
        print("all_messages: ", messages)
        final_answer = final_answer_match.group(1)
        print("最终答案:", final_answer)
        break

    messages.append(response.choices[0].message)

    action_match = re.search(r'Action:\s*(\w+)', response_text)
    action_input_match = re.search(r'Action Input:\s*({.*?}|".*?")', response_text, re.DOTALL)

    if action_match and action_input_match:
        tool_name = action_match.group(1)
        params = json.loads(action_input_match.group(1))

        if tool_name == "get_closing_price":
            observation = get_closing_price(params['name'])
            print("人类的回复：Observation:", observation)
            messages.append({"role": "user", "content": f"Observation: {observation}"})