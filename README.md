# Overview RAG QA 

This project implements a RAG question answering (yes/no) system for biomedical research papers.

The system use a FastAPI backend that retrieves relevant document chunks from a vector database and uses a large language model via the Groq API to generate answers.

The goal of the project is to simulate a realistic backend architecture for an AI-powered QA system, including API endpoints, persistent storage, and external model inference.

## Project Structure

```
rag/
	ingest.py
	retrieve.py
	generate.py
	pipeline.py

api/
	main.py
	routes.py

config/
	config.yaml

data/
logs/

Dockerfile
docker-compose.yml
requirements.txt
README.md
```

## Pipeline 

1. Documents are indexed and stored in a local vector database.
2. User sends a question to the API.
3. Relevant document chunks are retrieved from the database.
4. The question and context are sent to the LLM.
5. The model returns the final answer.

## Local Run

1. Install dependencies (requirements.txt)

2. Set Groq API key

3. Start server:

   - uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

   - API docs will be available at `http://localhost:8000/docs`.

## API Endpoints

### `GET /status`

Returns service and index status, including:
- whether vector store is ready
- number of indexed chunks
- dataset and Chroma paths in use

### `POST /ask`

Request body:

```json
{
	"question": "Is programmed cell death the regulated death of cells?"
}
```

Response:

```json
{
	"question": "...",
	"prediction": "yes",
	"answer": "yes",
	"retrieved_doc_ids": [123, 456]
}
```

### `POST /upload`

Upload a `.json` dataset file (multipart form-data, field name: `file`).

Behavior:
- Saves file in `data/`
- Rebuilds persistent vector index from uploaded file

## Configuration

Edit `config/config.yaml`:
- `llm.provider`: must be `groq`
- `llm.model_name`: Groq model name
- `llm.api_key_env`: environment variable name for Groq key (`GROQ_API_KEY`)
- `llm.api_base`: Groq API base URL
- `embedding.model_name`: embedding model for retrieval
- `paths.chroma_dir`: persistent vector DB location
- `retrieval.top_k`: number of docs to retrieve
- `chunking.chunk_size`, `chunking.chunk_overlap`

## Logging

Each `/ask` request logs:
- question
- prediction
- raw answer
- retrieved document IDs

Log file path is configured by `paths.query_log_file` (default `./logs/queries.log`).


## Offline Evaluation

Offline evaluation comparing RAG vs LM-only (no API endpoint calls). Using RAG significantly improved performance compared to using the LLM without retrieval, improving accuracy by $\sim 25\%$ and F1 (yes) by $\sim 57\%$


## LM-only results (no retrieval):
- Valid predictions: 162
- Accuracy: 0.4506
- F1: 0.2261

## RAG results:
- Valid predictions: 195
- Accuracy: 0.6974
- F1: 0.7958

Retrieval metrics:
- Gold retrieved anywhere: 194 / 195 (0.995)
- Recall@1: 0.979
- Recall@3: 0.995
- Recall@5: 0.995
