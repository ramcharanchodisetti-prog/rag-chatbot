from datetime import datetime
from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class SourceSnippet(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceSnippet]
