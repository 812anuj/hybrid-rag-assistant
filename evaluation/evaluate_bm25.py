import chromadb
import re

from rank_bm25 import BM25Okapi

from .evaluation_metrics import (
    calculate_precision,
    calculate_recall,
    calculate_mrr
)

from .evaluation_dataset import evaluation_dataset

# 1. Connect to ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="cat_prep")

# 2. Get all stored documents
all_data = collection.get(
    include=["documents", "metadatas"]
)

documents = all_data["documents"]
ids = all_data["ids"]

# 3. Create tokenizer
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# 4. Tokenize all documents
tokenized_documents = [
    tokenize(document)
    for document in documents
]

# 5. Build BM25 model
bm25 = BM25Okapi(tokenized_documents)

# 6. Store evaluation scores
all_precision = []
all_recall = []
all_mrr = []

# 7. Evaluate every question
for item in evaluation_dataset:

    question = item["question"]
    relevant_ids = item["relevant_chunk_ids"]

    # Tokenize the question
    query_tokens = tokenize(question)

    # Calculate BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Get indexes of top 10 documents
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:10]

    # Convert indexes into document IDs
    retrieved_ids = [
        ids[index]
        for index in top_indices
    ]

    # Calculate evaluation metrics
    precision = calculate_precision(
        retrieved_ids,
        relevant_ids
    )

    recall = calculate_recall(
        retrieved_ids,
        relevant_ids
    )

    mrr = calculate_mrr(
        retrieved_ids,
        relevant_ids
    )

    # Store scores
    all_precision.append(precision)
    all_recall.append(recall)
    all_mrr.append(mrr)

    # Display results
    print("\n--------------------------------")
    print("Question:", question)
    print("Retrieved IDs:", retrieved_ids)
    print("Relevant IDs:", relevant_ids)
    print("Precision@10:", precision)
    print("Recall@10:", recall)
    print("MRR:", mrr)

# 8. Calculate average scores
print("\n================================")
print("BM25 SEARCH EVALUATION")
print("================================")

print(
    "Average Precision@10:",
    sum(all_precision) / len(all_precision)
)

print(
    "Average Recall@10:",
    sum(all_recall) / len(all_recall)
)

print(
    "Average MRR:",
    sum(all_mrr) / len(all_mrr)
)

