from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# STEP 1: Configure chunking
chunk_size = 500
chunk_overlap = 100

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# STEP 2: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3: Find all PDF files
documents_folder = Path("data/documents")
pdf_files = list(documents_folder.rglob("*.pdf"))

print("Total PDFs:", len(pdf_files))

# STEP 4: Create an empty list
all_chunks = []

# STEP 5: Read and chunk every PDF
for pdf_path in pdf_files:

    print(f"\nProcessing: {pdf_path.name}")

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages[:30], start=1):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        chunks = text_splitter.split_text(text)

        for chunk_number, chunk in enumerate(chunks, start=1):

            chunk_data = {
                "content": chunk,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                    "doc_type": "pdf",
                    "chunk_id": (
                        f"{pdf_path.stem}_p{page_number}_c{chunk_number}"
                    )
                }
            }

            all_chunks.append(chunk_data)

# STEP 6: Extract text from all chunks
chunk_texts = [
    item["content"] for item in all_chunks
]

# STEP 7: Generate embeddings
embeddings = model.encode(
    chunk_texts,
    show_progress_bar=True
)

# Attach embeddings to chunks
for item, embedding in zip(all_chunks, embeddings):
    item["embedding"] = embedding.tolist()

# Connect to persistent ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")

# Create collection if it doesn't exist
collection = client.get_or_create_collection(
    name="cat_prep"
)

print("\nChromaDB collection:", collection.name)

# Prepare IDs
ids = [
    item["metadata"]["chunk_id"]
    for item in all_chunks
]

# Prepare document text
documents = [
    item["content"]
    for item in all_chunks
]

# Prepare metadata
metadatas = [
    item["metadata"]
    for item in all_chunks
]

# Prepare embeddings
vectors = [
    item["embedding"]
    for item in all_chunks
]

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=vectors,
)
print("\nData successfully stored in ChromaDB!")

print("Total stored records:", collection.count())

result = collection.get(
    limit=1,
    include=["documents", "metadatas"]
)

print("\nFirst stored ID:")
print(result["ids"][0])

print("\nFirst stored metadata:")
print(result["metadatas"][0])

print("\nFirst stored document:")
print(result["documents"][0][:300])

# Display results
# print("\nTotal chunks:", len(all_chunks))
# print("Total embeddings:", len(embeddings))
# print("Vector dimensions:", len(embeddings[0]))

# print("\nFirst chunk metadata:")
# print(all_chunks[0]["metadata"])

# print("\nFirst embedding:")
# print(all_chunks[0]["embedding"][:10])

