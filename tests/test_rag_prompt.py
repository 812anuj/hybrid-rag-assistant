
from src.rag.rag_prompt import create_prompt

question = "Who published this book?"

context = """
This edition can be exported from India only by the publishers,
McGraw Hill Education (India) Private Limited.
"""

prompt = create_prompt(question, context)

print(prompt)
