# 📄 AI Document Search — RAG Chatbot

Chat with your PDFs and text files using semantic search + an LLM. Upload a document,
ask a question in plain English, and get an answer grounded in your document's actual
content — with sources cited, not hallucinated.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

## What it does

1. **Upload** a PDF, TXT, or MD file.
2. The backend extracts the text, splits it into overlapping chunks, and embeds
   each chunk with OpenAI's embedding model.
3. Embeddings are stored in a local vector database ([Chroma](https://www.trychroma.com/)).
4. **Ask a question.** The question is embedded, the most semantically similar
   chunks are retrieved, and they're passed to an LLM as context.
5. The LLM answers **using only the retrieved context** and the response cites
   which source chunks it used.

This is Retrieval-Augmented Generation (RAG): instead of hoping the model
"knows" the answer from training data, you hand it the relevant excerpts at
query time. That means answers stay accurate and up to date with whatever you
upload, without any fine-tuning.

## Why this project instead of a toy demo

- Chunking respects sentence boundaries instead of cutting mid-sentence.
- Ingestion runs as a background task so uploads don't block on embedding calls.
- Failed ingestion is recorded on the document (`status: failed` +
  `error_message`), not swallowed silently.
- Chat history is persisted per session and fed back to the LLM for
  context-aware follow-up questions.
- The vector store is isolated behind one module (`services/vectorstore.py`),
  so swapping Chroma for Pinecone or Weaviate at scale touches one file.
- Comes with a real test suite (`pytest`) and a working Docker setup.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI | Async, auto-generated OpenAPI docs, production-ready |
| Vector store | Chroma (local, embedded) | Zero external accounts needed to run this locally — swap for Pinecone/Weaviate to scale |
| LLM + embeddings | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) | Fast, cheap, easy to swap for Claude or another provider |
| Relational DB | SQLite (default) / Postgres | Tracks documents, chunks metadata, and chat history |
| Frontend | Single-file HTML/CSS/JS | No build step — open it and it works |

## Quick start

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...

uvicorn app.main:app --reload
```

The API is now running at `http://localhost:8000`. Interactive docs (Swagger)
are auto-generated at `http://localhost:8000/docs`.

### 2. Frontend

No build step — just open `frontend/index.html` in a browser, or serve it:

```bash
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500`. Upload a file, wait for it to say
**Ready**, and start asking questions.

### 3. Run tests

```bash
cd backend
pytest -v
```

### 4. Run with Docker

```bash
cd backend
docker build -t rag-chatbot-backend .
docker run -p 8000:8000 --env-file .env rag-chatbot-backend
```

## Project structure

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + CORS + startup
│   │   ├── config.py            # Environment-driven settings
│   │   ├── database.py          # SQLAlchemy engine/session
│   │   ├── models.py            # Document, DocumentChunk, ChatSession, ChatMessage
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── routes/
│   │   │   ├── documents.py     # Upload / list / delete documents
│   │   │   └── chat.py          # Chat endpoint + history
│   │   └── services/
│   │       ├── extraction.py    # PDF/text extraction
│   │       ├── chunking.py      # Sentence-aware text splitting
│   │       ├── embeddings.py    # OpenAI embeddings wrapper
│   │       ├── vectorstore.py   # Chroma wrapper (swap point for scale)
│   │       ├── llm.py           # OpenAI chat completion + system prompt
│   │       └── rag.py           # Orchestrates ingest + retrieve + answer
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   └── index.html                # Chat UI, upload widget, source citations
├── .github/workflows/test.yml    # CI: install + pytest on every push
└── README.md
```

## How RAG works here (short version)

```
 Upload                          Ask a question
 ───────                         ───────────────
 PDF/TXT file                    "What does the contract say about renewal?"
     │                                   │
     ▼                                   ▼
 Extract text                    Embed the question
     │                                   │
     ▼                                   ▼
 Split into chunks                Vector similarity search
 (800 chars, 150 overlap)         against stored chunk embeddings
     │                                   │
     ▼                                   ▼
 Embed each chunk                 Top-5 most relevant chunks
     │                                   │
     ▼                                   ▼
 Store in Chroma  ──────────────▶ Build prompt: "Using this context,
 + save chunk text in DB           answer the question" → LLM
                                          │
                                          ▼
                                  Answer + cited sources
```

## Configuration

All settings live in `backend/.env` (copy from `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI key. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Model used to generate answers |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model used to embed text |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Characters per chunk |
| `TOP_K` | `5` | How many chunks to retrieve per query |
| `DATABASE_URL` | `sqlite:///./rag_chatbot.db` | Swap for a Postgres URL in production |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |

## Scaling this beyond a portfolio project

- Swap SQLite → Postgres (`DATABASE_URL`) for concurrent writers.
- Swap Chroma → Pinecone/Weaviate by reimplementing `services/vectorstore.py` —
  nothing else in the codebase needs to change.
- Add Redis caching for repeated queries.
- Add JWT auth (`routes/auth.py`) and scope documents per user.
- Put the FastAPI app behind a reverse proxy and add rate limiting.
- Move ingestion to a real task queue (Celery/RQ) instead of `BackgroundTasks`
  once volume grows past what a single worker thread can handle.

## License

MIT — see [LICENSE](LICENSE).
