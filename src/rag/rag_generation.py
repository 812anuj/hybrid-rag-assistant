
import os
from dotenv import load_dotenv
from groq import Groq

from .rag_prompt import create_prompt


# Load environment variables
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

# Create Groq client
client = Groq(api_key=api_key)


# User question
question = "Who published this book?"

# Retrieved PDF context
context = """
This edition can be exported from India only by the publishers,
McGraw Hill Education (India) Private Limited.
"""

# Create RAG prompt
prompt = create_prompt(question, context)


# Send prompt to Groq
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# Extract generated answer
answer = response.choices[0].message.content

print("\nQuestion:", question)
print("\nAI Answer:", answer)
