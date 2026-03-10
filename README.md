# Overview

This project implements a simple RAG system for answering biomedical yes/no questions using the PubMedQA dataset. The system retrieves relevant research paper abstracts and uses a language model to determine whether the answer to a question is yes or no.

The goal of the project is to explore RAG pipelines using LangChain and how RAG affects the performance of a language model.

# Setup

- The dataset used is the PubMedQA.
- Document abstracts are constructed from provided context and long answers.
- Document abstracts are split into smaller overlapping chunks for embedding.
- Embeddings are stored in a Chroma vector database.
- For each question, the system retrieves the top-5 most similar document chunks.
- The retrieved context and the question is given to a language model.
- The model outputs either a yes or no.

# Evaluation and Results
The RAG system evaluates:
- Accuracy
- F1 score
- Retrieval Recall@k

A baseline model without RAG is also evaluated for comparison.

RAG system results:
- Accuracy: 0.78
- F1 (yes): 0.83
- Recall@1: 0.982
- Recall@3: 0.982
- Recall@5: 0.982

Baseline model results:
- Accuracy: 0.58
- F1 (yes): 0.68