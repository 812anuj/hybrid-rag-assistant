
import chromadb

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="cat_prep")

results = collection.get(
    ids=[
        "Quants_p4_c2",
        "Quants_p4_c3",
        "Quants_p4_c4"
    ],
    include=["documents", "metadatas"]
)

for i, document_id in enumerate(results["ids"]):
    print("\n------------------------------")
    print("ID:", document_id)
    print("Content:")
    print(results["documents"][i])
