import json
import logging
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.generate import generate_answer
from rag.ingest import build_vector_store, ensure_path, load_pubmedqa
from rag.retrieve import retrieve_documents


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _get_logger() -> logging.Logger:
    config = _load_config()
    logs_dir = ensure_path(config.get("paths", {}).get("logs_dir", "./logs"))
    log_file = config.get("paths", {}).get("query_log_file", str(logs_dir / "queries.log"))

    logger = logging.getLogger("rag_query_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

    return logger


@lru_cache(maxsize=1)
def _get_vector_store():
    config = _load_config()
    path_cfg = config.get("paths", {})
    store_cfg = config.get("vector_store", {})
    embedding_cfg = config.get("embedding", {})

    chroma_dir = ensure_path(path_cfg.get("chroma_dir", "./chroma_db"))
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2")
    )
    vector_store = Chroma(
        collection_name=store_cfg.get("collection_name", "pubmedqa_rag"),
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    if vector_store._collection.count() == 0:
        dataset_path = path_cfg.get("default_dataset", "./ori_pqal.json")
        if not Path(dataset_path).exists():
            raise FileNotFoundError(
                "Vector store is empty and default dataset is missing. "
                "Upload a dataset through /upload or set paths.default_dataset."
            )
        documents_df, _ = load_pubmedqa(dataset_path)
        vector_store = build_vector_store(documents_df, config)

    return vector_store


def reindex_from_dataset(dataset_path: str):
    config = _load_config()
    reindex_config = dict(config)
    ingest_cfg = dict(reindex_config.get("ingest", {}))
    ingest_cfg["force_rebuild"] = True
    reindex_config["ingest"] = ingest_cfg

    documents_df, _ = load_pubmedqa(dataset_path)
    vector_store = build_vector_store(documents_df, reindex_config)
    _get_vector_store.cache_clear()
    return vector_store._collection.count()


def get_service_status() -> dict:
    config = _load_config()
    path_cfg = config.get("paths", {})

    default_dataset = path_cfg.get("default_dataset", "./ori_pqal.json")
    chroma_dir = path_cfg.get("chroma_dir", "./chroma_db")
    query_log_file = path_cfg.get("query_log_file", "./logs/queries.log")

    vector_store = _get_vector_store()
    indexed_chunks = vector_store._collection.count()

    return {
        "status": "ok",
        "vector_store_ready": indexed_chunks > 0,
        "indexed_chunks": indexed_chunks,
        "default_dataset_path": default_dataset,
        "default_dataset_exists": Path(default_dataset).exists(),
        "chroma_dir": chroma_dir,
        "query_log_file": query_log_file,
        "llm_provider": config.get("llm", {}).get("provider", "groq"),
        "llm_model": config.get("llm", {}).get("model_name", "llama-3.1-8b-instant"),
    }


def rag_pipeline(question: str):
    """End-to-end RAG pipeline entrypoint."""
    config = _load_config()
    logger = _get_logger()
    vector_store = _get_vector_store()

    top_k = config.get("retrieval", {}).get("top_k", 5)
    docs, retrieved_ids = retrieve_documents(vector_store, question, top_k=top_k)
    context = "\n\n".join(doc.page_content for doc in docs)

    prediction, raw_answer = generate_answer(question, context, config)

    logger.info(
        json.dumps(
            {
                "question": question,
                "prediction": prediction,
                "raw_answer": raw_answer,
                "retrieved_doc_ids": retrieved_ids,
                "retrieved_documents": [
                    {
                        "id": doc.metadata.get("id"),
                        "snippet": doc.page_content[:300],
                    }
                    for doc in docs
                ],
            },
            ensure_ascii=False,
        )
    )

    return {
        "question": question,
        "prediction": prediction,
        "answer": raw_answer,
        "retrieved_doc_ids": retrieved_ids,
    }
