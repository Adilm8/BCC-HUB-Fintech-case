"""Minimal private fallbacks used only until loaders.py returns RawDocument and structure.py exists."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from validator.models import Block, BlockKind, Document, RawDocument, Section

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST = re.compile(r"^\s*(?:[*\-+•]|\d+[.)])\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_DOC_ID = re.compile(
    r"^\s*(?:\*\*)?((?:[A-Za-zА-Яа-яЁё]{1,8}[-_]\d+(?:[.\-]\d+)*)|(?:\d+\.\d+(?:\.\d+)*))(?:\*\*)?[.:)\s]"
)
_INLINE = re.compile(r"\*\*|__|`")


def to_raw(loaded, path: Path | None = None) -> RawDocument:
    """Accept either a RawDocument or the older loader result (path/title/text)."""
    if isinstance(loaded, RawDocument):
        return loaded
    p = Path(getattr(loaded, "path", path or "document.md"))
    suffix = p.suffix.lower().lstrip(".")
    fmt = {"markdown": "md"}.get(suffix, suffix)
    if fmt not in ("md", "txt", "pdf", "docx"):
        fmt = "txt"
    return RawDocument(path=str(p), name=p.name, title=getattr(loaded, "title", p.stem), format=fmt,
                       text=loaded.text, warnings=list(getattr(loaded, "warnings", []) or []))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _INLINE.sub("", text)).strip()


def _cells(line: str) -> list[str]:
    s = line.strip()
    s = s.removeprefix("|")
    s = s.removesuffix("|")
    return [_clean(c) for c in s.split("|")]


def parse_structure(raw: RawDocument) -> Document:
    lines = raw.text.splitlines()
    sections: list[Section] = []
    blocks: list[Block] = []
    stack: list[Section] = []
    level1 = sum(1 for ln in lines if re.match(r"^#\s+", ln))
    part_level = 2 if level1 <= 1 else 1
    current_part = ""

    def section() -> Section:
        if not sections:
            sections.append(Section(id="S01", title=raw.title, level=0, path=[], parent_id=None, part="",
                                    block_ids=[]))
        return stack[-1] if stack else sections[0]

    def add(kind: BlockKind, text: str, raw_text: str, start: int, end: int, cells=None) -> None:
        sec = section()
        text = _clean(text)
        if not text:
            return
        bid = f"B{len(blocks) + 1:03d}"
        m = _DOC_ID.match(raw_text.lstrip("*-+• ").strip()) if kind != BlockKind.CODE else None
        blocks.append(Block(id=bid, kind=kind, text=text, raw=raw_text, section_id=sec.id,
                            section_path=list(sec.path), part=sec.part, line_start=start, line_end=end,
                            doc_id=m.group(1) if m else None, cells=cells))
        sec.block_ids.append(bid)

    i = 0
    para: list[str] = []
    para_start = 0

    def flush_para(end: int) -> None:
        nonlocal para
        if para:
            add(BlockKind.PARAGRAPH, " ".join(p.strip() for p in para), "\n".join(para), para_start, end)
            para = []

    table_header_done = False
    while i < len(lines):
        line = lines[i]
        n = i + 1
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para(n - 1)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            body = "\n".join(lines[i + 1: j])
            add(BlockKind.CODE, body, "\n".join(lines[i: j + 1]), n, j + 1)
            i = j + 1
            continue
        h = _HEADING.match(line)
        if h:
            flush_para(n - 1)
            level, title = len(h.group(1)), _clean(h.group(2))
            while stack and stack[-1].level >= level:
                stack.pop()
            if level == part_level or (level < part_level and level1 > 1):
                current_part = title
            elif level < part_level:
                current_part = ""
            parent = stack[-1] if stack else None
            path = (parent.path if parent else []) + [title]
            sec = Section(id=f"S{len(sections) + 1:02d}", title=title, level=level, path=path,
                          parent_id=parent.id if parent else None, part=current_part, block_ids=[])
            if not sections:
                sections.append(sec)
            else:
                sections.append(sec)
            stack.append(sec)
            add(BlockKind.HEADING, title, line, n, n)
            table_header_done = False
            i += 1
            continue
        if not stripped or re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            flush_para(n - 1)
            table_header_done = False
            i += 1
            continue
        if stripped.startswith("|"):
            flush_para(n - 1)
            if _TABLE_SEP.match(stripped):
                i += 1
                continue
            cells = _cells(stripped)
            is_header = not table_header_done and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip())
            kind = BlockKind.TABLE_HEADER if is_header else BlockKind.TABLE_ROW
            table_header_done = True
            if any(cells):
                add(kind, " | ".join(cells), line, n, n, cells=cells)
            i += 1
            continue
        table_header_done = False
        m = _LIST.match(line)
        if m:
            flush_para(n - 1)
            add(BlockKind.LIST_ITEM, m.group(1), line, n, n)
            i += 1
            continue
        if not para:
            para_start = n
        para.append(line)
        i += 1
    flush_para(len(lines))
    if not sections:
        section()
    parts: list[str] = []
    for b in blocks:
        if b.part and b.part not in parts:
            parts.append(b.part)
    return Document(path=raw.path, name=raw.name, title=raw.title, format=raw.format,
                    sha256=hashlib.sha256(raw.text.encode("utf-8")).hexdigest(), char_count=len(raw.text),
                    text=raw.text, sections=sections, blocks=blocks, parts=parts, warnings=list(raw.warnings))
