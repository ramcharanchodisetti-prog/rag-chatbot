import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Document, DocumentStatus
from app.schemas import DocumentOut
from app.services.rag import ingest_document, remove_document

router = APIRouter(prefix="/api/documents", tags=["documents"])
settings = get_settings()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}")

    document = Document(filename=file.filename, status=DocumentStatus.processing)
    db.add(document)
    db.commit()
    db.refresh(document)

    dest_path = UPLOAD_DIR / f"{document.id}{suffix}"
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    with dest_path.open("wb") as out_file:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out_file.close()
                dest_path.unlink(missing_ok=True)
                db.delete(document)
                db.commit()
                raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")
            out_file.write(chunk)

    # Ingestion (extract -> chunk -> embed -> store) runs after the response
    # so the client isn't stuck waiting on a slow upload request.
    background_tasks.add_task(_ingest_in_background, document.id, str(dest_path))

    return document


def _ingest_in_background(document_id: str, file_path: str):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        ingest_document(db, document_id, file_path)
    finally:
        db.close()


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.created_at.desc()).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(404, "Document not found")
    return document


@router.delete("/{document_id}", status_code=204)
def delete_document_route(document_id: str, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(404, "Document not found")
    remove_document(db, document_id)
    for f in UPLOAD_DIR.glob(f"{document_id}.*"):
        f.unlink(missing_ok=True)
