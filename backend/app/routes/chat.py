import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ChatSession, ChatMessage
from app.schemas import ChatRequest, ChatResponse, SourceSnippet
from app.services.rag import answer_question

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    session = None
    if payload.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    if session is None:
        session = ChatSession()
        db.add(session)
        db.commit()
        db.refresh(session)

    # Build short conversation history for context-aware follow-ups.
    history = [
        {"role": m.role, "content": m.content}
        for m in db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    ]

    db.add(ChatMessage(session_id=session.id, role="user", content=payload.message))
    db.commit()

    result = answer_question(payload.message, history)

    db.add(ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        sources=json.dumps(result["sources"]),
    ))
    db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=result["answer"],
        sources=[SourceSnippet(**s) for s in result["sources"]],
    )


@router.get("/{session_id}/history")
def get_history(session_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [
        {
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources) if m.sources else [],
            "created_at": m.created_at,
        }
        for m in messages
    ]
