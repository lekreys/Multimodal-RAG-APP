# RAG_MULTIMODAL

RAG_MULTIMODAL is a Retrieval-Augmented Generation (RAG) system that combines:

- Unstructured for PDF and document parsing
- ChromaDB as a local vector store
- LangChain + OpenAI for LLM-based reasoning and generation
- Supabase for file/metadata storage (optional)
- FastAPI as the backend API
- Dash as an interactive dashboard

This repository can be used as a template for building RAG-based document applications or as a starting point for research and production systems.

---

## Features

- PDF extraction with Unstructured  
  Extracts structured text (and optionally images) from PDF files for downstream processing.

- Semantic retrieval with ChromaDB  
  Indexes document chunks as embeddings and retrieves top-k relevant context for a given query.

- RAG answer generation  
  Combines retrieved context with an LLM (via LangChain + OpenAI) to generate grounded answers.

- FastAPI backend  
  Provides API endpoints for:
  - Ingesting and processing uploaded PDF files
  - Storing embeddings in Chroma
  - Querying the RAG pipeline

- Dash dashboard  
  Frontend dashboard for:
  - Uploading files
  - Sending questions
  - Viewing answers and context

- Modular architecture  
  Clear separation between:
  - Core logic (extraction, retrieval, generation, store)
  - Infrastructure clients (Supabase)
  - Configuration (logging)
  - Application layer (FastAPI and Dash)

---

## Project Structure

```text
RAG_MULTIMODAL/
├─ app/
│  ├─ __init__.py
│  ├─ app_dash.py          # Dash dashboard entrypoint
│  └─ main.py              # FastAPI application entrypoint
│
├─ clients/
│  ├─ __init__.py
│  └─ supabase_client.py   # Supabase client configuration
│
├─ config/
│  ├─ __init__.py
│  └─ logger_config.py     # Logging setup and helpers
│
├─ core/
│  ├─ __init__.py
│  ├─ extraction.py        # PDF parsing with Unstructured
│  ├─ retrieval.py         # Chroma retrieval functions
│  ├─ generation.py        # LLM-based answer generation
│  └─ store.py             # Chroma configuration and helpers
│
├─ data/
│  ├─ chromaa_db/       # ChromaDB instance (example)
│        # Another ChromaDB namespace (example)
│
├─ logs/                   # Runtime logs
├─ venv/                   # Virtual environment (git-ignored)
├─ .env                    # Environment variables (git-ignored)
├─ .gitignore
├─ requirements.txt
└─ README.md
