"""
rag_engine.py
Core RAG (Retrieval-Augmented Generation) logic.

Pipeline:
  1. Document ingestion → chunking → embedding → FAISS index
  2. At query time → retrieve top-k chunks → inject into LLM prompt
  3. LLM generates grounded answer with source references

LLM: Gemini 1.5 Flash (free) with Groq Llama3 as fallback
Embeddings: sentence-transformers all-MiniLM-L6-v2 (free, local)
Vector Store: FAISS (in-memory, no server needed)
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import faiss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
TOP_K = 5               # number of chunks to retrieve per query
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    if gemini_key:
        from google import genai

        client = genai.Client(api_key=gemini_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        return response.text.strip()

    elif groq_key:
        import httpx

        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 800,
        }

        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=25,
        )

        resp.raise_for_status()

        return resp.json()["choices"][0]["message"]["content"].strip()

    else:
        raise EnvironmentError(
            "No LLM API key found. Set GEMINI_API_KEY or GROQ_API_KEY in .env"
        )


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        # Fallback: try pdfminer
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            raise ImportError(
                "PDF support requires 'pypdf' or 'pdfminer.six'. "
                "Run: pip install pypdf"
            )


def load_document(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    return _load_text(path)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split document text into overlapping chunks while trying
    to preserve paragraph and sentence boundaries.
    """

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # If we're not at the end, try to find a natural boundary.
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)

            if boundary <= start:
                boundary = text.rfind(". ", start, end)

            if boundary > start + int(CHUNK_SIZE * 0.6):
                end = boundary + 1

        chunk = text[start:end].strip()

        if len(chunk) > 50:
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": f"{source}::chunk_{idx}",
            })
            idx += 1

        # Move forward while maintaining overlap.
        next_start = end - CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    def __init__(self):
        self._chunks: list[dict] = []
        self._doc_names: list[str] = []
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._faiss_index = None

    # --- Ingestion ---

    def ingest_document(self, path: str, display_name: str) -> int:
        """Load, chunk, embed, and index a document. Returns chunk count."""
        if display_name in self._doc_names:
            raise ValueError(f"Document '{display_name}' has already been uploaded.")

        text = load_document(path)
        if not text.strip():
            raise ValueError("Document appears to be empty or unreadable.")

        new_chunks = chunk_text(text, display_name)
        if not new_chunks:
            raise ValueError("No usable text extracted from document.")

        model = _get_embed_model()
        new_texts = [c["text"] for c in new_chunks]
        new_embs = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)

        # Convert embeddings to float32 for FAISS
        new_embs = np.asarray(new_embs, dtype=np.float32)

        # Normalize embeddings so inner product = cosine similarity
        faiss.normalize_L2(new_embs)

        # Create FAISS index on first document
        if self._faiss_index is None:
            embedding_dimension = new_embs.shape[1]
            self._faiss_index = faiss.IndexFlatIP(embedding_dimension)

        # Add embeddings to FAISS
        self._faiss_index.add(new_embs)

        # Store chunk metadata
        self._chunks.extend(new_chunks)

        if display_name not in self._doc_names:
            self._doc_names.append(display_name)

        return len(new_chunks)

    # --- Retrieval ---

    def _retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        if self._faiss_index is None or len(self._chunks) == 0:
            return []

        model = _get_embed_model()

        # Create query embedding
        q_emb = model.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)

        # Normalize query embedding
        faiss.normalize_L2(q_emb)

        # Don't request more results than we have chunks
        k = min(k, len(self._chunks))

        # Search FAISS index
        scores, indices = self._faiss_index.search(q_emb, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            # Ignore weak matches
            if float(score) <= 0.2:
                continue

            chunk = self._chunks[int(idx)].copy()
            chunk["score"] = float(score)

            results.append(chunk)

        return results
    # --- Answer generation ---

    def answer(self, question: str, session_id: str) -> tuple[str, list[str]]:
    # Get conversation history before retrieval.
        history = self._history[session_id]

    # Build a retrieval query using recent conversation context.
        retrieval_query = question

        if history:
            recent_history = history[-4:]

            history_context = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in recent_history
            )

            retrieval_query = (
                f"Previous conversation:\n{history_context}\n\n"
                f"Current question: {question}"
            )

        # Retrieve using both the current question and conversation context.
        relevant_chunks = self._retrieve(retrieval_query)

        if not relevant_chunks:
            return (
                "I couldn't find relevant information in the uploaded documents "
                "to answer that question.",
                [],
            )

    # Build context block
        # Build context block
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}"
            for c in relevant_chunks
        )

        # Build conversation history for multi-turn awareness
        #history = self._history[session_id]
        history_text = ""
        if history:
            history_text = "Previous conversation:\n" + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in history[-4:]  # last 2 turns
            ) + "\n\n"

        # Prompt
        prompt = f"""You are a helpful document assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the context,
say "I don't have enough information in the uploaded documents to answer that."

Do not make up facts. Keep the answer clear and concise.

{history_text}Context from documents:
{context}

User question: {question}

Answer:"""

        answer_text = _call_llm(prompt)

        # Store turn in history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer_text})

        # Extract unique sources
        sources = list(dict.fromkeys(c["source"] for c in relevant_chunks))

        return answer_text, sources

    # --- Utility ---

    def is_ready(self) -> bool:
        return len(self._chunks) > 0

    def document_count(self) -> int:
        return len(self._doc_names)

    def chunk_count(self) -> int:
        return len(self._chunks)

    def reset(self):
        self._chunks.clear()
        self._faiss_index = None
        self._doc_names.clear()
        self._history.clear()
