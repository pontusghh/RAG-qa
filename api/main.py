from fastapi import FastAPI

from api.routes import router


app = FastAPI(title="RAG QA Backend", version="1.0.0")
app.include_router(router)
