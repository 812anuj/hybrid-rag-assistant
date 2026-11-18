# can be deleted 
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size= 500
chunk_overlap=100

text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

document_folder= Path("data/documents")
pdf_files= list(document_folder.glob("*.pdf"))

print("Total PDFs : ",len(pdf_files))

all_chunks=[]

for pdf_path in pdf_files:
    print("\nProcessing : ",pdf_path)
    reader = PdfReader(pdf_path)

    for page_number , page in enumerate(reader.pages[:30], start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue
        
        chunks=text_splitter.split_text(text)
        for chunk_number, chunk in enumerate(chunks, start=1):
            chunk_data={
                "content":chunk,
                "metadata":{
                    "source":pdf_path.name,
                    "page":page_number,
                    "doc_type":"pdf",
                    "chunk_id":(
                        f'{pdf_path.stem}_p{page_number}_c{chunk_number}'          
                    )
                }
            }
            all_chunks.append(chunk_data)

print("\nTotal number of chunks : ",len(all_chunks))

for item in all_chunks[:3]:
    print("\n------------------------------")
    print("Chunk ID:", item["metadata"]["chunk_id"])
    print("Source:", item["metadata"]["source"])
    print("Page:", item["metadata"]["page"])
    print("Content:")
    print(item["content"][:300])