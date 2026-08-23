"""
Extracts raw text from uploaded files. Supports PDF and plain text/markdown.
Add more branches here (docx, html, etc.) as the project grows.
"""
from pathlib import Path

from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if suffix in {".txt", ".md"}:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")
