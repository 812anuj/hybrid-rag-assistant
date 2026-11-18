from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love programming.",
    "I enjoy writing code.",
    "The weather is sunny today."
]

embeddings = model.encode(sentences)

print("no. of embeddings : ",len(embeddings))
print("vector dimension : ",len(embeddings[0]))
print("\nfirst embedding : ",embeddings[0])