import chromadb
from sentence_transformers import SentenceTransformer

from .evaluation_metrics import (
    calculate_precision,
    calculate_recall,
    calculate_mrr
)

from .evaluation_dataset import evaluation_dataset


# 1. Connect to ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="cat_prep")

# 2. Load our embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Store evaluation scores
all_precision = []
all_recall = []
all_mrr = []

# 4. Evaluate every question
for item in evaluation_dataset:

    question = item["question"]
    relevant_ids = item["relevant_chunk_ids"]

    # Convert question into an embedding
    query_embedding = model.encode(question)

    # Retrieve top 5 chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
        include=["documents", "metadatas"]
    )

    # Extract retrieved chunk IDs
    retrieved_ids = results["ids"][0]

    # Calculate evaluation metrics
    precision = calculate_precision(retrieved_ids, relevant_ids)
    recall = calculate_recall(retrieved_ids, relevant_ids)
    mrr = calculate_mrr(retrieved_ids, relevant_ids)

    # Save scores
    all_precision.append(precision)
    all_recall.append(recall)
    all_mrr.append(mrr)

    # Display results for this question
    print("\n--------------------------------")
    print("Question:", question)
    print("Retrieved IDs:", retrieved_ids)
    print("Relevant IDs:", relevant_ids)
    print("Precision@10:", precision)
    print("Recall@10:", recall)
    print("MRR:", mrr)

# 5. Calculate average scores
print("\n================================")
print("SEMANTIC SEARCH EVALUATION")
print("================================")

print("Average Precision@5:", sum(all_precision) / len(all_precision))
print("Average Recall@5:", sum(all_recall) / len(all_recall))
print("Average MRR:", sum(all_mrr) / len(all_mrr))