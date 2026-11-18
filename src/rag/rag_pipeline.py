
import os
import chromadb

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from .rag_prompt import create_prompt


# 1. Connect to ChromaDB
client_db = chromadb.PersistentClient(path="./data/chroma_db")

collection = client_db.get_collection(name="cat_prep")


# 2. Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Connect to Groq
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

groq_client = Groq(api_key=api_key)


# 4. Get user question
question = "Who published this book?"


# 5. Convert question into embedding
query_embedding = embedding_model.encode(question)


# 6. Search ChromaDB
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=10,
    include=["documents", "metadatas"]
)


# 7. Extract retrieved documents
retrieved_documents = results["documents"][0]


# 8. Combine documents into context
context = "\n\n".join(retrieved_documents)


# 9. Create RAG prompt
prompt = create_prompt(question, context)


# 10. Send prompt to Groq
response = groq_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# 11. Extract AI answer
answer = response.choices[0].message.content


# 12. Display results
print("\nQuestion:", question)

print("\nRetrieved Documents:")

retrieved_metadatas = results["metadatas"][0]

for i, document in enumerate(retrieved_documents):

    print(f"\nDocument {i + 1}:")
    print("Chunk ID:", retrieved_metadatas[i]["chunk_id"])
    print("Source:", retrieved_metadatas[i]["source"])
    print("Page:", retrieved_metadatas[i]["page"])
    print("Content:")
    print(document)

print("\nAI Answer:")
print(answer)
