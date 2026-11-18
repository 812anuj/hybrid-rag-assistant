import chromadb
import re
from rank_bm25 import BM25Okapi

# 1. Connect to our existing ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="cat_prep")

# 2. Retrieve all stored documents and metadata
all_data = collection.get(
    include=["documents", "metadatas"]
)

documents = all_data["documents"]
metadatas = all_data["metadatas"]

print("Total documents:", len(documents))

# 3. Create a tokenizer function
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# 4. Tokenize every document
tokenized_documents = [
    tokenize(document) for document in documents
]

# 5. Create the BM25 search engine
bm25 = BM25Okapi(tokenized_documents)

# 6. Enter your question
query = "How should students prepare for CAT?"

# 7. Tokenize the question
query_tokens = tokenize(query)

# 8. Calculate BM25 scores
scores = bm25.get_scores(query_tokens)

# 9. Get the indexes of the top 3 results
top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:3]

# 10. Display the results
for rank, index in enumerate(top_indices, start=1):
    print("\n------------------------------")
    print("Result:", rank)
    print("BM25 Score:", scores[index])
    print("ID:", all_data["ids"][index])
    print("Source:", metadatas[index]["source"])
    print("Page:", metadatas[index]["page"])
    print("Content:")
    print(documents[index][:500])
