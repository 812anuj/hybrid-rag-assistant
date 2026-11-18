import chromadb

client = chromadb.Client()

collection = client.create_collection("cat_prep")

print('collection created')
print(f'collection name : {collection.name}')