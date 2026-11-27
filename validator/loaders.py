"""Чтение документа требований в текст."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = ".md, .txt, .pdf, .docx"


@dataclass(frozen=True)
class Document:
    path: Path
    title: str
    text: str


def load_document(path: Path) -> Document:
    if not path.is_file():
        raise FileNotFoundError(f"файл не найден: {path}")
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8-sig")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".docx":
        import docx

        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    else:
        raise ValueError(f"формат {suffix or 'без расширения'} не поддерживается, нужен {SUPPORTED}")
    if not text.strip():
        raise ValueError(f"в документе нет текста: {path}")
    return Document(path=path, title=_title(text, path), text=text)


def _title(text: str, path: Path) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:200]
    return path.stem
