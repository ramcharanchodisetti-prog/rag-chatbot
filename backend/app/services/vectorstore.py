"""
Vector store wrapper, backed by Chroma (local, embedded, free).

This is intentionally the *only* file that knows about Chroma's API.
To swap in Pinecone or Weaviate for a larger-scale deployment, reimplement
these three functions against that SDK and nothing else in the codebase
needs to change.
"""
try:
    # GitHub Actions' default Python build ships an older SQLite than
    # chromadb requires (>=3.35.0). pysqlite3-binary bundles a modern
    # SQLite and this swap makes chromadb use it transparently, without
    # needing to touch the system package. No-op on systems where the
    # system SQLite is already new enough (e.g. most local dev machines).
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb

from app.config import get_settings

settings = get_settings()
_client = None


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _client.get_or_create_collection(settings.CHROMA_COLLECTION)


def upsert_chunks(
    chunk_ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
):
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(embedding: list[float], top_k: int = 5, where: dict | None = None):
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
    )
    return results


def delete_document(document_id: str):
    collection = get_collection()
    collection.delete(where={"document_id": document_id})
