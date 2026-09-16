"""
main.py — RAG Document Q&A Chatbot Backend
FastAPI + LangChain + FAISS + Gemini/Groq

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

rag: Optional[RAGEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGEngine()
    print("[Startup] RAG engine ready.")
    yield

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Q&A Chatbot",
    description="Upload documents and ask questions. Powered by LangChain + FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int

class StatusResponse(BaseModel):
    status: str
    documents_loaded: int
    total_chunks: int

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        status="ready" if rag and rag.is_ready() else "no documents loaded",
        documents_loaded=rag.document_count() if rag else 0,
        total_chunks=rag.chunk_count() if rag else 0,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    The document is chunked and indexed into the FAISS vector store.
    """
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Use PDF, TXT, or MD."
        )

    # Save to disk
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    content = await file.read()
    save_path.write_bytes(content)

    # Index into vector store
    try:
        chunks = rag.ingest_document(str(save_path), file.filename)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        filename=file.filename,
        chunks_indexed=chunks,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question about the uploaded documents.
    Returns an answer grounded in the document context plus source references.
    """
    if not rag or not rag.is_ready():
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload a document first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or uuid.uuid4().hex

    try:
        answer, sources = rag.answer(request.question, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session_id,
    )


@app.delete("/reset")
def reset():
    """Clear all indexed documents and conversation history."""
    rag.reset()
    return {"message": "Vector store and conversation history cleared."}

from fastapi.responses import FileResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")