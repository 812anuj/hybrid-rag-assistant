
evaluation_dataset = [
    {
    "question": "Who published this book?",
    "relevant_chunk_ids": [
        "Quants_p4_c2"
    ]
    },
    {
        "question": "What topics are included in Quantitative Aptitude?",
        "relevant_chunk_ids": [
            "Quants_p10_c5",
            "Quants_p10_c9"
        ]
    }
]

print("Total evaluation questions:", len(evaluation_dataset))

for item in evaluation_dataset:
    print("\nQuestion:", item["question"])
    print("Relevant chunks:", item["relevant_chunk_ids"])
