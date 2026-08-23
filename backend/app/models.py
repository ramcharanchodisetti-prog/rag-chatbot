"""
ORM models.

Note: chunk embeddings live in Chroma (the vector store), not in this
relational DB. Postgres only tracks documents, chunk metadata, and chat
history. This keeps the vector index swappable (Chroma -> Pinecone ->
pgvector) without touching this schema.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, Enum
from sqlalchemy.orm import relationship
import enum

from app.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class DocumentStatus(str, enum.Enum):
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=gen_uuid)
    filename = Column(String, nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.processing, nullable=False)
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Metadata row per chunk. The actual embedding vector lives in Chroma;
    this row lets us map a vector-store hit back to readable text + source."""
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=gen_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=gen_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON-encoded list of source snippets
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")
