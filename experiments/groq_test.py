import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=api_key)

messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant with name Gayatri.",
        },
        {
            "role": "user",
            "content": "introduce yourself in one sentence",
        }
    ]

response = client.chat.completions.create(
    model ="openai/gpt-oss-20b",
    messages=messages,
    temperature=0.7
)

answer = response.choices[0].message.content

print("\nAI_Response: ")
print(answer)

