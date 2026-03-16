from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pubmedqa(data_path: str):
    """Load and normalize PubMedQA-style data into documents/questions dataframes."""
    tmp_data = pd.read_json(data_path).T
    tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

    documents = pd.DataFrame(
        {
            "abstract": tmp_data.apply(
                lambda row: " ".join(row.CONTEXTS + [row.LONG_ANSWER]), axis=1
            ),
            "year": tmp_data.YEAR,
        }
    )
    questions = pd.DataFrame(
        {
            "question": tmp_data.QUESTION,
            "year": tmp_data.YEAR,
            "gold_label": tmp_data.final_decision,
            "gold_context": tmp_data.LONG_ANSWER,
            "gold_document_id": documents.index,
        }
    )
    return documents, questions


def build_vector_store(documents_df, config: dict):
    """Build or update a persistent vector store from document dataframe."""
    chunk_cfg = config.get("chunking", {})
    embedding_cfg = config.get("embedding", {})
    store_cfg = config.get("vector_store", {})
    path_cfg = config.get("paths", {})
    ingest_cfg = config.get("ingest", {})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.get("chunk_size", 1000),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 200),
        add_start_index=True,
    )

    metadatas = [{"id": int(idx)} for idx in documents_df.index]
    docs = splitter.create_documents(
        texts=documents_df.abstract.tolist(),
        metadatas=metadatas,
    )
    splits = splitter.split_documents(docs)

    for i, split in enumerate(splits):
        split.metadata["chunk_index"] = i

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2")
    )

    chroma_dir = ensure_path(path_cfg.get("chroma_dir", "./chroma_db"))
    collection_name = store_cfg.get("collection_name", "pubmedqa_rag")
    force_rebuild = ingest_cfg.get("force_rebuild", False)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    current_count = vector_store._collection.count()
    if current_count > 0 and not force_rebuild:
        return vector_store

    if current_count > 0 and force_rebuild:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )

    ids = []
    for split in splits:
        doc_id = split.metadata.get("id")
        chunk_index = split.metadata.get("chunk_index")
        start_index = split.metadata.get("start_index", 0)
        ids.append(f"{doc_id}-{chunk_index}-{start_index}")

    vector_store.add_documents(documents=splits, ids=ids)
    return vector_store


def ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
