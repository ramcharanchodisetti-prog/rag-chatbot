from app.services.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    result = chunk_text("Hello world.", chunk_size=800, overlap=150)
    assert result == ["Hello world."]


def test_long_text_is_split_into_multiple_chunks():
    text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 800 + 50  # allow small slack for sentence-boundary snapping


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=150)
