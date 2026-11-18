import os
import re
import chromadb

from dotenv import load_dotenv
from groq import Groq

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from .rag_prompt import create_prompt

# 1. Connect to ChromaDB
client_db = chromadb.PersistentClient(path="./data/chroma_db")
collection = client_db.get_collection(name="cat_prep")
print("Total stored chunks:", collection.count())

# 2. Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# 3. Load Cross-Encoder model
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# 4. Connect to Groq
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

groq_client = Groq(api_key=api_key)

# 8. Get all documents from ChromaDB
all_data = collection.get(
    include=["documents", "metadatas"]
)

documents = all_data["documents"]
metadatas = all_data["metadatas"]
ids = all_data["ids"]

# 9. Tokenize all documents
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_documents = [
    tokenize(document)
    for document in documents
]

# 10. Create BM25 search engine
bm25 = BM25Okapi(tokenized_documents)

# 5. Create reusable RAG function
def get_rag_response(question):
    # 6. Convert question into embedding
    query_embedding = embedding_model.encode(question)

    # 7. Perform semantic search
    semantic_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
        include=["documents", "metadatas"]
    )

    # Extract semantic result IDs
    semantic_ids = semantic_results["ids"][0]

    # 11. Search using the question
    query_tokens = tokenize(question)

    bm25_scores = bm25.get_scores(query_tokens)

    # 12. Get top 10 BM25 results
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    # Extract BM25 result IDs
    bm25_ids = [
        ids[index]
        for index in bm25_indices
    ]

    # 13. Combine results using Reciprocal Rank Fusion

    rrf_scores = {}

    k = 60

    # Process semantic search results
    for rank, document_id in enumerate(semantic_ids, start=1):

        score = 1 / (k + rank)

        if document_id not in rrf_scores:
            rrf_scores[document_id] = 0

        rrf_scores[document_id] += score


    # Process BM25 search results
    for rank, document_id in enumerate(bm25_ids, start=1):

        score = 1 / (k + rank)

        if document_id not in rrf_scores:
            rrf_scores[document_id] = 0

        rrf_scores[document_id] += score


    # Sort documents by RRF score
    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # 14. Select top candidates for Cross-Encoder
    top_candidates = ranked_results[:16]

    # Prepare question-document pairs
    pairs = []

    for document_id, rrf_score in top_candidates:

        index = ids.index(document_id)

        document_text = documents[index]

        pairs.append((question, document_text))


    # 15. Calculate Cross-Encoder scores
    cross_scores = cross_encoder.predict(pairs)

    # 16. Combine document IDs and Cross-Encoder scores
    reranked_results = list(zip(
        [document_id for document_id, score in top_candidates],
        cross_scores
    ))

    # 17. Sort by Cross-Encoder scores
    reranked_results = sorted(
        reranked_results,
        key=lambda item: item[1],
        reverse=True
    )

    # 19. Select final documents
    final_results = reranked_results[:10]

    # 20. Extract final document IDs
    final_ids = [
        document_id
        for document_id, score in final_results
    ]

    # 21. Get final document texts
    final_documents = []

    for document_id in final_ids:

        index = ids.index(document_id)

        final_documents.append(documents[index])

    # 22. Combine final documents into context
    context = "\n\n".join(final_documents)

    # 23. Create RAG prompt
    prompt = create_prompt(question, context)

    # 24. Send prompt to Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # 25. Extract answer
    answer = response.choices[0].message.content

    return answer