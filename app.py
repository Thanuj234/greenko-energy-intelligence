from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rag_engine import RAGEngine

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Greenko Energy Intelligence",
    description="Document-grounded RAG assistant for renewable-energy operations.",
    version="1.0.0",
)
engine = RAGEngine(DATA_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)

@app.on_event("startup")
def startup_event():
    engine.load_knowledge_base()

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "Greenko Energy Intelligence", **engine.status()}

@app.get("/api/documents")
def documents():
    return {"documents": engine.list_documents()}

@app.post("/api/run")
def run_query(request: QueryRequest):
    return engine.answer(request.text, request.top_k)

@app.get("/api/evaluate")
def evaluate():
    return engine.evaluate()
