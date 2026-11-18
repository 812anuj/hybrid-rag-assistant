from sentence_transformers import SentenceTransformer, util

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Three sentences
sentences = [
    "I love programming.",
    "I enjoy writing code.",
    "The weather is sunny today."
]

# Compute sentence embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

similarity_1= util.cos_sim(embeddings[0], embeddings[1])
similarity_2= util.cos_sim(embeddings[0], embeddings[2])

print("Similarity between sentence 1 and 2:")
print(similarity_1.item())

print("\nSimilarity between sentence 1 and 3:")
print(similarity_2.item())