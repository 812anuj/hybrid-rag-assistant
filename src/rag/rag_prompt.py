
def create_prompt(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

If the answer is not available in the context,
say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt
