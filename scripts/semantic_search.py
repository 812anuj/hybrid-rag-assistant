
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Connect to persistent database
client = chromadb.PersistentClient(path="./data/chroma_db")

# 2. Access existing collection
collection = client.get_collection(name="cat_prep")

print("Total stored records:", collection.count())

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Enter your question
question = "Who published this book?"

# 5. Convert question into embedding
query_embedding = model.encode(question)

print("Question:", question)
print("Embedding dimensions:", len(query_embedding))

# 6. Search ChromaDB
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

# 7. Display results
for i in range(len(results["ids"][0])):

    print("\n------------------------------")
    print("Result:", i + 1)

    print("ID:", results["ids"][0][i])
    print("Distance:", results["distances"][0][i])

    print("Source:", results["metadatas"][0][i]["source"])
    print("Page:", results["metadatas"][0][i]["page"])

    print("Content:")
    print(results["documents"][0][i])
