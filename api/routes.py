from pathlib import Path

import yaml
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag.pipeline import get_service_status, rag_pipeline, reindex_from_dataset


router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.get("/status")
def get_status():
    try:
        return get_service_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {exc}") from exc


@router.post("/ask")
def ask_question(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        return rag_pipeline(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported")

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    output_path = data_dir / safe_name

    content = await file.read()
    output_path.write_bytes(content)

    try:
        chunks_count = reindex_from_dataset(str(output_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {exc}") from exc

    return {
        "status": "ok",
        "dataset_path": str(output_path),
        "indexed_chunks": chunks_count,
    }
