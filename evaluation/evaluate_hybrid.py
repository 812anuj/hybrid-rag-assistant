import chromadb
import re

from sentence_transformers import SentenceTransformer
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

# 2. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load all stored documents
all_data = collection.get(
    include=["documents", "metadatas"]
)

documents = all_data["documents"]
metadatas = all_data["metadatas"]
ids = all_data["ids"]

# 4. Prepare BM25
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_documents = [
    tokenize(document)
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

# 5. Prepare evaluation lists
all_precision = []
all_recall = []
all_mrr = []

k = 60

# 6. Evaluate every question
for item in evaluation_dataset:

    question = item["question"]
    relevant_ids = item["relevant_chunk_ids"]

    # Semantic search
    query_embedding = model.encode(question)

    semantic_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
        include=["documents", "metadatas"]
    )

    semantic_ids = semantic_results["ids"][0]


    # BM25 search
    query_tokens = tokenize(question)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    bm25_ids = [
        ids[index]
        for index in bm25_indices
    ]


    # Reciprocal Rank Fusion
    rrf_scores = {}

    for rank, document_id in enumerate(semantic_ids, start=1):

        score = 1 / (k + rank)

        if document_id not in rrf_scores:
            rrf_scores[document_id] = 0

        rrf_scores[document_id] += score


    for rank, document_id in enumerate(bm25_ids, start=1):

        score = 1 / (k + rank)

        if document_id not in rrf_scores:
            rrf_scores[document_id] = 0

        rrf_scores[document_id] += score


    # Sort by RRF score
    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    # Get final top 10 document IDs
    retrieved_ids = [
        document_id
        for document_id, score in ranked_results[:10]
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


    all_precision.append(precision)
    all_recall.append(recall)
    all_mrr.append(mrr)


    # Display question results
    print("\n--------------------------------")
    print("Question:", question)
    print("Retrieved IDs:", retrieved_ids)
    print("Relevant IDs:", relevant_ids)
    print("Precision@10:", precision)
    print("Recall@10:", recall)
    print("MRR:", mrr)


# 7. Display average results
print("\n================================")
print("HYBRID SEARCH EVALUATION")
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






