from pathlib import Path
from pypdf import PdfReader

documents_folder = Path("data/documents") 

pdf_files = list(documents_folder.glob("*.pdf"))
print("Total PDFs : ",len(pdf_files))

all_documents=[]

for pdf_file in pdf_files:
    print("\nReading file : ",pdf_file)
    reader = PdfReader(pdf_file)

    for page_number , page in enumerate(reader.pages[:5], start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        document={
            "content":text,
            "metadata":{
                "source":pdf_file.name,
                "page_number":page_number,
                "doc_type":"pdf"
            }
        }

        all_documents.append(document)  
for document in all_documents:
    print("\nContent : ")
    print(document["content"][:100])

    print("\nMetadata : ")
    print(document["metadata"])        