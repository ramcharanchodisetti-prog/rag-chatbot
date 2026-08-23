"""
Thin wrapper around the OpenAI embeddings endpoint.

Batches requests (the API accepts a list of strings per call) to cut
latency and cost versus embedding one chunk at a time.
"""
from openai import OpenAI

from app.config import get_settings

settings = get_settings()
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_client()
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    # response.data is returned in the same order as the input list.
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
