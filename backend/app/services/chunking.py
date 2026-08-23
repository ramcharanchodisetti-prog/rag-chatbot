"""
Simple, dependency-free text chunker with overlap.

Character-based (not token-based) chunking keeps this project free of an
extra tokenizer dependency. For production, swap in a token-aware splitter
(e.g. tiktoken) so chunk sizes map precisely to the embedding model's
context window.
"""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to break on a sentence/paragraph boundary near `end` so we
        # don't cut a sentence in half, which hurts retrieval quality.
        if end < length:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break
        start = end - overlap

    return chunks
