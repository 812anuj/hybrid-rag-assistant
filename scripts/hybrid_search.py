
import chromadb
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# 1. Connect to ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="cat_prep")

# 2. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Retrieve all documents from ChromaDB
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
    tokenize(document) for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

# 5. Enter your question
query = "What is quantitative aptitude?"

# 6. Semantic Search
query_embedding = model.encode(query)

semantic_results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# 7. BM25 Search
query_tokens = tokenize(query)
bm25_scores = bm25.get_scores(query_tokens)

bm25_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:5]

# 8. Reciprocal Rank Fusion (RRF)
rrf_scores = {}   # this dictionary will hold the RRF scores for each document in tuple format (document_id, score)
k = 60

# Add Semantic Search ranking
for rank, document_id in enumerate(
    semantic_results["ids"][0],
    start=1
):
    score = 1 / (k + rank)

    if document_id not in rrf_scores:
        rrf_scores[document_id] = 0

    rrf_scores[document_id] += score

# Add BM25 ranking
for rank, index in enumerate(bm25_indices, start=1):
    document_id = ids[index]

    score = 1 / (k + rank)

    if document_id not in rrf_scores:
        rrf_scores[document_id] = 0

    rrf_scores[document_id] += score

# 9. Sort documents by RRF score
ranked_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)

# 11. Select top 8 RRF results
top_candidates = ranked_results[:8]

# 12. Prepare question-document pairs
pairs = []

for document_id, rrf_score in top_candidates:
    index = ids.index(document_id)

    document_text = documents[index]

    pairs.append((query, document_text))

# 13. Calculate relevance scores
cross_scores = cross_encoder.predict(pairs)

# 14. Combine document IDs with Cross-Encoder scores
reranked_results = list(zip(
    [document_id for document_id, score in top_candidates],
    cross_scores
))

# 15. Sort by Cross-Encoder score
reranked_results = sorted(
    reranked_results,
    key=lambda item: item[1],
    reverse=True
)

# 16. Display final re-ranked results
print("\nCross-Encoder Re-ranked Results:")

for rank, (document_id, score) in enumerate(
    reranked_results[:5],
    start=1
):
    index = ids.index(document_id)

    print("\n------------------------------")
    print("Rank:", rank)
    print("ID:", document_id)
    print("Cross-Encoder Score:", score)
    print("Source:", metadatas[index]["source"])
    print("Page:", metadatas[index]["page"])
    print("Content:")
    print(documents[index][:500])



# 10. Display final hybrid results
# print("\nTotal hybrid results:", len(ranked_results))

# for rank, (document_id, score) in enumerate(
#     ranked_results[:5],
#     start=1
# ):
#     # Find the document's original position
#     index = ids.index(document_id)

#     print("\n------------------------------")
#     print("Result:", rank)
#     print("ID:", document_id)
#     print("RRF Score:", score)
#     print("Source:", metadatas[index]["source"])
#     print("Page:", metadatas[index]["page"])
#     print("Content:")
#     print(documents[index][:500])
