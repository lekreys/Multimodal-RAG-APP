# RAG_MULTIMODAL

Sistem **Retrieval-Augmented Generation (RAG) multimodal** yang memanfaatkan:

- **Unstructured** untuk ekstraksi konten PDF (dan dokumen lain),
- **ChromaDB** untuk vector store,
- **LangChain + OpenAI** untuk LLM,
- **Supabase** untuk penyimpanan metadata / file,
- **FastAPI** sebagai backend API,
- **Dash** sebagai dashboard interaktif.

Repo ini bisa dipakai sebagai template untuk project RAG berbasis dokumen yang ingin dikembangkan jadi produk beneran.

---

## ✨ Fitur Utama

- 📄 **Ekstraksi PDF Multimodal dengan Unstructured**
  - Ekstraksi teks + struktur + (opsional) gambar dari PDF.
  - Bisa diintegrasikan dengan Supabase untuk upload / download file.

- 🔎 **Semantic Retrieval dengan ChromaDB**
  - Menyimpan embedding dokumen ke vector store lokal (`./data/...`).
  - Pencarian top-k chunk relevan berdasarkan pertanyaan user.

- 🧠 **LLM Answer Generation (RAG)**
  - Menggabungkan hasil retrieval ke dalam prompt.
  - Menghasilkan jawaban + (opsional) sumber referensi.

- 🌐 **FastAPI Backend**
  - Endpoint untuk:
    - Upload / ingest PDF,
    - Menyimpan embedding ke Chroma,
    - Query RAG (tanya jawab berbasis dokumen).

- 📊 **Dash Dashboard**
  - UI sederhana untuk:
    - Upload file,
    - Kirim pertanyaan,
    - Melihat jawaban + konteks.

- 🧱 **Arsitektur Modular**
  - `core/` untuk logic utama (extraction, store, retrieval, generation).
  - `clients/` untuk integrasi eksternal (Supabase).
  - `config/` untuk logging.
  - `app/` untuk FastAPI + Dash.

---

## 📁 Struktur Project

```text
RAG_MULTIMODAL/
├─ app/
│  ├─ __init__.py
│  ├─ app_dash.py          # Dashboard (Dash)
│  └─ main.py              # FastAPI app / entrypoint backend
│
├─ clients/
│  ├─ __init__.py
│  └─ supabase_client.py   # Koneksi ke Supabase
│
├─ config/
│  ├─ __init__.py
│  └─ logger_config.py     # Setup & helper logging
│
├─ core/
│  ├─ __init__.py
│  ├─ extraction.py        # Ekstraksi PDF dengan Unstructured
│  ├─ retrieval.py         # Fungsi retrieval ke Chroma
│  ├─ generation.py        # Prompting & LLM answer generation
│  └─ store.py             # Konfigurasi & helper Chroma
│
├─ data/
│  ├─ chromaa_hendb/       # Folder instance ChromaDB
│  └─ chromaa_lawak/       # Instance lain (namespace lain)
│
├─ logs/                   # File log runtime
├─ venv/                   # Virtual environment (ignored)
├─ .env                    # Environment variables (ignored)
├─ .gitignore
├─ requirements.txt
└─ README.md
