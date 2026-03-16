def retrieve_documents(vector_store, question: str, top_k: int = 5):
    """Retrieve and deduplicate top documents by source id."""
    retrieved_docs = vector_store.similarity_search(question, k=top_k)

    unique = {}
    for doc in retrieved_docs:
        doc_id = doc.metadata.get("id")
        if doc_id not in unique:
            unique[doc_id] = doc

    unique_docs = list(unique.values())
    retrieved_ids = [doc.metadata.get("id") for doc in unique_docs]
    return unique_docs, retrieved_ids
