# DocMind — RAG Document Q&A Chatbot

A full-stack RAG (Retrieval-Augmented Generation) chatbot that lets you
upload documents and ask questions. Every answer is grounded in your
documents — no hallucination.

Built with: FastAPI · LangChain · FAISS · Sentence Transformers · Gemini

---

## Architecture

```
User uploads PDF/TXT
        ↓
  Text extraction (pypdf)
        ↓
  Chunking (500 chars, 100 overlap)
        ↓
  Embedding (all-MiniLM-L6-v2, local)
        ↓
  FAISS in-memory vector store
        ↓
User asks question
        ↓
  Query embedded → cosine similarity → top-5 chunks retrieved
        ↓
  Chunks + question injected into Gemini prompt
        ↓
  Grounded answer + source citations returned
```

---

## Local Setup

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Add API key
cp .env.example .env
# Edit .env → add GEMINI_API_KEY

# 3. Start backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open frontend
# Just open frontend/index.html in your browser
```

---

## Deploy to HuggingFace Spaces (Free)

1. Create a new Space on huggingface.co → select "Docker" template
2. Push this project to the Space repo
3. Add `GEMINI_API_KEY` as a Space secret
4. HuggingFace builds and deploys automatically

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Readiness check |
| `/status` | GET | Docs loaded + chunk count |
| `/upload` | POST | Upload and index a document |
| `/chat` | POST | Ask a question |
| `/reset` | DELETE | Clear all documents |

---

## What This Demonstrates (for your Resume)

- **RAG pipeline** built from scratch (not just calling an API)
- **Vector similarity search** with embeddings
- **FastAPI** with proper schema validation, error handling
- **Multi-turn conversation** with session history
- **Full-stack**: working frontend + backend
- **Deployed**: live URL on HuggingFace Spaces
