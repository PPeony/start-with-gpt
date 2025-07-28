from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
{"role": "system",
         "content": "You are a helpful assistant. You can help me by answering my questions. You can also ask me questions."},
        {"role": "user", "content": "你好！你叫什么名字？"}
    ],
    n=2,
    temperature=0.9,
    max_tokens=1000,
)
print(completion.choices[0].message.content)