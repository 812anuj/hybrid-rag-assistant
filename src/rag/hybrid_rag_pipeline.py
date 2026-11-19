
import os
import re
import chromadb

from dotenv import load_dotenv
from groq import Groq

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from .ingestion import ingest_uploaded_pdfs
from .rag_prompt import create_prompt


# 1. Connect to ChromaDB
client_db = chromadb.PersistentClient(path="./data/chroma_db")


# 2. Load AI models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# 3. Connect to Groq
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

groq_client = Groq(api_key=api_key)


# 4. Create a separate collection for each session
def get_session_collection(session_id):

    collection_name = f"session_{session_id}"

    return client_db.get_or_create_collection(
        name=collection_name
    )


# 5. Process uploaded PDFs
def process_uploaded_pdfs(uploaded_files, collection):

    # Remove previously uploaded documents from this session
    existing_ids = collection.get()["ids"]

    if existing_ids:
        collection.delete(ids=existing_ids)

    return ingest_uploaded_pdfs(
        uploaded_files,
        collection,
        embedding_model
    )


# 6. Tokenize text for BM25
def tokenize(text):

    return re.findall(r"\b\w+\b", text.lower())


# 7. Hybrid RAG response
def get_rag_response(question, collection):

    # Get documents from this session's collection
    all_data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = all_data["documents"]
    ids = all_data["ids"]

    if not documents:
        return "Please upload and process PDF documents first."

    # Create document ID lookup
    document_lookup = dict(zip(ids, documents))

    # 8. Semantic search
    query_embedding = embedding_model.encode(question)

    semantic_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(10, len(documents)),
        include=["documents", "metadatas"]
    )

    semantic_ids = semantic_results["ids"][0]

    # 9. BM25 search
    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

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

    # 10. Reciprocal Rank Fusion
    rrf_scores = {}

    k = 60

    for rank, document_id in enumerate(semantic_ids, start=1):

        rrf_scores[document_id] = (
            rrf_scores.get(document_id, 0) + 1 / (k + rank)
        )

    for rank, document_id in enumerate(bm25_ids, start=1):

        rrf_scores[document_id] = (
            rrf_scores.get(document_id, 0) + 1 / (k + rank)
        )

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # 11. CrossEncoder reranking
    top_candidates = ranked_results[:16]

    pairs = [
        (question, document_lookup[document_id])
        for document_id, score in top_candidates
    ]

    cross_scores = cross_encoder.predict(pairs)

    reranked_results = sorted(
        zip(
            [document_id for document_id, score in top_candidates],
            cross_scores
        ),
        key=lambda item: item[1],
        reverse=True
    )

    # 12. Select final documents
    final_results = reranked_results[:10]

    final_documents = [
        document_lookup[document_id]
        for document_id, score in final_results
    ]

    # 13. Create context
    context = "\n\n".join(final_documents)

    # 14. Create prompt
    prompt = create_prompt(question, context)

    # 15. Generate answer using Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
