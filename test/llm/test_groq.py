import os
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

completion = client.chat.completions.create(
    model="grok-beta",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": "给我一首中文七言绝句诗.json格式{'answer':content}",
        },
    ],
    temperature=0.2,
)

print(completion.choices[0].message)
# export XAI_API_KEY="xai-xx"
# python test/llm/test_groq.py
