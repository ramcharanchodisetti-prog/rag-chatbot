"""
Thin wrapper around the OpenAI chat completions endpoint, with the RAG
system prompt centralized here so it's easy to tune in one place.
"""
from app.config import get_settings
from app.services.embeddings import get_client

settings = get_settings()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using \
ONLY the provided document excerpts (context). Rules:
1. If the answer is not contained in the context, say you don't know based \
on the uploaded documents. Do not use outside knowledge.
2. Cite which excerpt(s) you used, e.g. "(Source 2)".
3. Be concise and direct.
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Source {i} - {c['filename']}]\n{c['excerpt']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    context = build_context(chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-10:])  # keep recent turns only

    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    client = get_client()
    response = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content
