
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_documents(collection, embedding_model):

    # 1. Check whether documents are already stored
    if collection.count() > 0:
        print("Documents already exist in ChromaDB.")
        return

    # 2. Find PDF files
    documents_folder = Path("data/documents")
    pdf_files = list(documents_folder.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF documents found!")

    print("Total PDFs:", len(pdf_files))

    # 3. Configure text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    # 4. Read and split PDFs
    for pdf_path in pdf_files:

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(
            reader.pages[:30], start=1
        ):

            text = page.extract_text() or ""

            if not text.strip():
                continue

            chunks = text_splitter.split_text(text)

            for chunk_number, chunk in enumerate(chunks, start=1):

                all_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                        "doc_type": "pdf",
                        "chunk_id": (
                            f"{pdf_path.stem}_p{page_number}_c{chunk_number}"
                        )
                    }
                })

    if not all_chunks:
        raise ValueError("No text chunks extracted from PDFs!")

    # 5. Generate embeddings
    chunk_texts = [
        item["content"] for item in all_chunks
    ]

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=True
    )

    # 6. Prepare ChromaDB data
    ids = [
        item["metadata"]["chunk_id"]
        for item in all_chunks
    ]

    documents = chunk_texts

    metadatas = [
        item["metadata"]
        for item in all_chunks
    ]

    vectors = embeddings.tolist()

    # 7. Store everything in ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors
    )

    print("Successfully stored documents!")
    print("Total stored chunks:", collection.count())
