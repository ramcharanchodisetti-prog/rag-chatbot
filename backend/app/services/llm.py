"""
Thin wrapper around the Gemini chat endpoint, with the RAG system prompt
centralized here so it's easy to tune in one place.
"""
from google.genai import types

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

    # Gemini's history format uses "user"/"model" roles and a "parts" list,
    # so recent OpenAI-style {"role", "content"} turns are translated here.
    contents = []
    if history:
        for turn in history[-10:]:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    contents.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

    client = get_client()
    response = client.models.generate_content(
        model=settings.GEMINI_CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text
