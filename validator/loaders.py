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
        text = _docx_to_markdown(path)
    else:
        raise ValueError(f"формат {suffix or 'без расширения'} не поддерживается, нужен {SUPPORTED}")
    if not text.strip():
        raise ValueError(f"в документе нет текста: {path}")
    return Document(path=path, title=_title(text, path), text=text)


def _docx_to_markdown(path: Path) -> str:
    """Paragraphs and tables in document order; headings become '#' lines, tables become pipe tables."""
    import docx
    from docx.table import Table

    out: list[str] = []
    for block in docx.Document(str(path)).iter_inner_content():
        if isinstance(block, Table):
            rows = [[c.text.strip().replace("\n", " ").replace("|", "/") for c in r.cells] for r in block.rows]
            if not rows:
                continue
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * len(rows[0]))
            out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
            out.append("")
            continue
        text = block.text.strip()
        if not text:
            continue
        style = (block.style.name if block.style is not None else "") or ""
        if style.startswith("Heading") and style[-1:].isdigit():
            out.append("#" * int(style[-1]) + " " + text)
        else:
            out.append(text)
    return "\n".join(out)


def _title(text: str, path: Path) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:200]
    return path.stem
