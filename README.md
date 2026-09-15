# 🤖 RAG Document Q&A Chatbot

A full-stack **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload documents and ask questions about their content.

The application retrieves the most relevant document chunks using **FAISS vector search** and uses **Google Gemini** to generate grounded, context-aware answers.

---

## ✨ Features

- 📄 Upload **PDF, TXT, and Markdown** documents
- ✂️ Intelligent document chunking with overlap
- 🧠 Local text embeddings using **Sentence Transformers**
- 🔎 Semantic similarity search using **FAISS**
- 🤖 Grounded responses using **Google Gemini**
- 💬 Multi-turn conversational context
- 📚 Source references with relevance scores
- 🚫 Prevents duplicate document indexing
- 🔄 Reset vector store and conversation history
- ⚡ FastAPI backend with REST APIs
- 🌐 Simple HTML/CSS/JavaScript frontend

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      Frontend       │
                    │    HTML / CSS / JS  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Document Upload │        │ User Question   │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Text Extraction │        │ Query Embedding │
        │     (PyPDF)     │        └────────┬────────┘
        └────────┬────────┘                 │
                 ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐
        │    Chunking     │        │      FAISS      │
        │ + Overlapping   │◄───────│  Vector Search  │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐
        │   Embeddings    │        │ Top-K Relevant  │
        │ Sentence        │        │     Chunks      │
        │ Transformers    │        └────────┬────────┘
        └────────┬────────┘                 │
                 │                          │
                 └────────────┬─────────────┘
                              ▼
                    ┌─────────────────────┐
                    │    Gemini LLM       │
                    │  Context + Query    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Grounded Answer +   │
                    │ Source References   │
                    └─────────────────────┘