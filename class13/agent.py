import os
import re

from openai import OpenAI

from class13.prompt import prompt

DEFAULT_MODEL = "qwen-max"
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def invoke(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=self.messages,
            temperature=0
        )
        return completion.choices[0].message.content


def calculate(what):
    return eval(what)


def ask_fruit_unit_price(fruit):
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Banana unit price is 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)


action_re = re.compile(r'^Action: (\w+): (.*)$')

known_actions = {
    "calculate": calculate,
    "ask_fruit_unit_price": ask_fruit_unit_price
}


def query(question, max_turns=5):
    i = 0
    agent = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = agent.invoke(next_prompt)
        print("result: ", result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        print("actions: ", actions)
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


query("What is the total price of 3kg of apple and 2kg of banana?")