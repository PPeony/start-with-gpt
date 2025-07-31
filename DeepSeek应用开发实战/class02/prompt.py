REACT_PROMPT = """
{instructions}

TOOLS:
------

You have access to the following tools:

{tools}

You must use tool to get answer util you get all the tools answers.
To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
```

Then wait for Human will response to you the result of action by use Observation.
... (this Thought/Action/Action Input/Observation can repeat N times)
You need to wait until you get them all.
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {input}

"""

# You need to wait until you get them all.
# 这一行是自己加的，因为一次只能返回一个值，大模型自己猜出来了第二个tool的值，没有调用tool执行
# 这说明改prompt才是关键