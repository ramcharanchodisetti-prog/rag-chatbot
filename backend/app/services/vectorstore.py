"""
Vector store wrapper, backed by Chroma (local, embedded, free).

This is intentionally the *only* file that knows about Chroma's API.
To swap in Pinecone or Weaviate for a larger-scale deployment, reimplement
these three functions against that SDK and nothing else in the codebase
needs to change.
"""
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
