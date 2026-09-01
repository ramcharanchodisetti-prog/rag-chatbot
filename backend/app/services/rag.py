"""
Orchestrates the two core RAG pipelines:
  ingest_document()  -> extract -> chunk -> embed -> store
  answer_question()  -> embed query -> vector search -> build prompt -> LLM

Keeping this logic in one place (rather than spread across route handlers)
makes both pipelines independently testable.
"""
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts, embed_query
from app.services.extraction import extract_text
from app.services.vectorstore import upsert_chunks, query as vector_query, delete_document
from app.services.llm import generate_answer

settings = get_settings()


def ingest_document(db: Session, document_id: str, file_path: str):
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        return

    try:
        raw_text = extract_text(file_path)
        chunks = chunk_text(raw_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        if not chunks:
            raise ValueError("No extractable text found in document")

        embeddings = embed_texts(chunks)

        chunk_rows = []
        metadatas = []
        for i, content in enumerate(chunks):
            row = DocumentChunk(document_id=document.id, chunk_index=i, content=content)
            db.add(row)
            chunk_rows.append(row)

        db.flush()  # assign IDs before reading row.id below

        chunk_ids = [row.id for row in chunk_rows]
        for i in range(len(chunks)):
            metadatas.append({
                "document_id": document.id,
                "filename": document.filename,
                "chunk_index": i,
            })

        upsert_chunks(chunk_ids, embeddings, chunks, metadatas)

        document.status = DocumentStatus.ready
        document.chunk_count = len(chunks)
        db.commit()

    except Exception as exc:  # noqa: BLE001 - surface any failure on the record
        db.rollback()
        document.status = DocumentStatus.failed
        document.error_message = str(exc)
        db.commit()
        raise


def remove_document(db: Session, document_id: str):
    delete_document(document_id)
    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        db.delete(document)
        db.commit()


def retrieve_chunks(question: str, top_k: int | None = None) -> list[dict]:
    query_embedding = embed_query(question)
    results = vector_query(query_embedding, top_k=top_k or settings.TOP_K)

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, distance in zip(docs, metas, distances):
        chunks.append({
            "document_id": meta.get("document_id"),
            "filename": meta.get("filename"),
            "chunk_index": meta.get("chunk_index"),
            "excerpt": text,
            # Chroma returns a distance; convert to a rough 0-1 similarity score.
            "score": max(0.0, 1 - distance),
        })
    return chunks


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    chunks = retrieve_chunks(question)

    if not chunks:
        return {
            "answer": "I don't have any documents to search yet. Upload one first.",
            "sources": [],
        }

    answer = generate_answer(question, chunks, history)
    return {"answer": answer, "sources": chunks}
