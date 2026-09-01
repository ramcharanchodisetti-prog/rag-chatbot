"""
Thin wrapper around the Gemini embeddings endpoint (Google AI Studio).

Batches requests where possible to cut latency and cost versus embedding
one chunk at a time.
"""
from google import genai
from google.genai import types

from app.config import get_settings

settings = get_settings()
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


GEMINI_EMBED_BATCH_LIMIT = 100  # Gemini's batch embedding API caps requests at 100 per call


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_client()
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), GEMINI_EMBED_BATCH_LIMIT):
        batch = texts[i : i + GEMINI_EMBED_BATCH_LIMIT]
        response = client.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        # response.embeddings is returned in the same order as the input batch.
        embeddings.extend(e.values for e in response.embeddings)
    return embeddings


def embed_query(text: str) -> list[float]:
    client = get_client()
    response = client.models.embed_content(
        model=settings.GEMINI_EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values
