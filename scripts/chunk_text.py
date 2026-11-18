from langchain_text_splitters import RecursiveCharacterTextSplitter

text=""" 
Artificial Intelligence is a branch of computer science.

Machine Learning is a subset of Artificial Intelligence.
It allows computers to learn from data.

Deep Learning is a subset of Machine Learning.
It uses neural networks to learn complex patterns.

Natural Language Processing helps computers understand
and process human language.
"""
chunkSize=150
overlap=30
text_spliter=RecursiveCharacterTextSplitter(
    chunk_size=chunkSize,
    chunk_overlap=overlap,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

chunks=text_spliter.split_text(text)
print("Total chunks : ",len(chunks))

for(i,chunk) in enumerate(chunks):
    print(f"Chunk {i} : {chunk}")