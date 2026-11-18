
def calculate_precision(retrieved_ids, relevant_ids):

    # Convert relevant IDs into a set
    relevant_set = set(relevant_ids)

    # Find matching IDs
    matched_ids = [
        document_id
        for document_id in retrieved_ids
        if document_id in relevant_set
    ]

    # Calculate precision
    if len(retrieved_ids) == 0:
        return 0.0

    precision = len(matched_ids) / len(retrieved_ids)

    return precision


def calculate_recall(retrieved_ids, relevant_ids):

    # Convert relevant IDs into a set
    relevant_set = set(relevant_ids)

    # Find matching IDs
    matched_ids = [
        document_id
        for document_id in retrieved_ids
        if document_id in relevant_set
    ]

    # Avoid division by zero
    if len(relevant_ids) == 0:
        return 0.0

    # Calculate recall
    recall = len(matched_ids) / len(relevant_ids)

    return recall


def calculate_mrr(retrieved_ids, relevant_ids):

    # Convert relevant IDs into a set
    relevant_set = set(relevant_ids)

    # Check each retrieved document in ranking order
    for rank, document_id in enumerate(retrieved_ids, start=1):

        # Check whether this document is relevant
        if document_id in relevant_set:

            # Return reciprocal rank of first relevant result
            return 1 / rank

    # Return 0 if no relevant result was found
    return 0.0


# Test our function
if __name__ == "__main__":
    retrieved_ids = ["A", "B", "C", "D", "E"]
    relevant_ids = ["C", "D", "E"]

    precision = calculate_precision(retrieved_ids, relevant_ids)
    recall = calculate_recall(retrieved_ids, relevant_ids)
    mrr = calculate_mrr(retrieved_ids, relevant_ids)

    print("Precision:", precision)
    print("Recall:", recall)
    print("MRR:", mrr)
