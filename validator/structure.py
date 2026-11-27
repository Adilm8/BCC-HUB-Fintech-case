"""Parse markdown-like RawDocument text into a section tree and indexed blocks."""

from __future__ import annotations

import hashlib
import re

from validator.models import Block, BlockKind, Document, RawDocument, Section

_ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})(?:\s+(.*?))?\s*$")
_TRAILING_HASHES_RE = re.compile(r"\s+#+\s*$")
_SETEXT_H1_RE = re.compile(r"^\s*={3,}\s*$")
_SETEXT_H2_RE = re.compile(r"^\s*-{3,}\s*$")
_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+•]|\d+[.)])\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_TABLE_SEP_CELL_RE = re.compile(r"^\s*:?-{1,}:?\s*$")

_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*")
_BOLD_UNDER_RE = re.compile(r"(?<!\w)__(.+?)__(?!\w)")
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_FMT_TAG_RE = re.compile(r"</?(?:b|strong|i|em|code|u|s|sub|sup)\b[^>]*>", re.IGNORECASE)
_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")

# Alphanumeric requirement IDs at block start (e.g. FR-12, REQ_001, SYS-REQ-1.2, or Cyrillic prefix).
_ALPHA_ID_RE = re.compile(
    r"^\s*[\[\(]?"
    r"(?P<id>[A-ZА-ЯЁ][A-Za-zА-ЯЁа-яё]{0,9}(?:[-_][A-ZА-ЯЁ][A-Za-zА-ЯЁа-яё]{0,9})?[-_]\d+(?:\.\d+)*)"
    r"[\]\)]?"
    r"(?=(?:[.:;)\]\s—–-]|$))"
)
# Hierarchical dotted numeric IDs at block start (e.g. 3.2.1, 1.2.).
_NUM_ID_RE = re.compile(
    r"^\s*[\[\(]?"
    r"(?P<id>[1-9]\d{0,2}(?:\.\d{1,3}){1,4})"
    r"(?P<punct>[.:)\]]?)"
    r"(?P<rest>\s+.*|$)"
)
# Common standards, protocols, and encodings that look like PREFIX-NUMBER but are not requirement IDs.
_NON_ID_PREFIXES = frozenset(
    {
        "UTF",
        "SHA",
        "MD",
        "ISO",
        "IEC",
        "IEEE",
        "RFC",
        "TLS",
        "SSL",
        "HTTP",
        "HTTPS",
        "TCP",
        "UDP",
        "IP",
        "SQL",
        "HTML",
        "CSS",
        "JSON",
        "XML",
        "AES",
        "RSA",
        "HMAC",
        "JWT",
        "OAUTH",
        "BASE",
    }
)


def parse_structure(raw: RawDocument) -> Document:
    """Build a section tree and ordered blocks B001.. from markdown-like RawDocument text.

    Never raises on malformed or heading-less markup: a document without headings produces
    a single root section.
    """
    text = raw.text or ""
    lines = text.splitlines()
    n = len(lines)

    sections: list[Section] = []
    real_section_ids: set[str] = set()
    heading_stack: list[Section] = []
    current_section: Section | None = None
    blocks: list[Block] = []
    fallback_title = (raw.title or raw.name or "Документ").strip() or "Документ"

    def next_section_id() -> str:
        return f"S{len(sections) + 1:02d}"

    def next_block_id() -> str:
        return f"B{len(blocks) + 1:03d}"

    def ensure_section() -> Section:
        nonlocal current_section
        if current_section is None:
            sec = Section(
                id=next_section_id(),
                title=fallback_title,
                level=1,
                path=[fallback_title],
                parent_id=None,
                part=fallback_title,
                block_ids=[],
            )
            sections.append(sec)
            current_section = sec
        return current_section

    def add_heading(level: int, raw_title: str) -> None:
        nonlocal current_section
        title = _strip_inline_markdown(raw_title).strip()
        if not title:
            return
        while heading_stack and heading_stack[-1].level >= level:
            heading_stack.pop()
        parent = heading_stack[-1] if heading_stack else None
        path = [*parent.path, title] if parent is not None else [title]
        sec = Section(
            id=next_section_id(),
            title=title,
            level=level,
            path=path,
            parent_id=parent.id if parent is not None else None,
            part=title,
            block_ids=[],
        )
        sections.append(sec)
        real_section_ids.add(sec.id)
        heading_stack.append(sec)
        current_section = sec

    def add_block(
        kind: BlockKind,
        block_text: str,
        raw_chunk: str,
        line_start: int,
        line_end: int,
        doc_id: str | None = None,
        cells: list[str] | None = None,
    ) -> None:
        sec = ensure_section()
        blk = Block(
            id=next_block_id(),
            kind=kind,
            text=block_text,
            raw=raw_chunk,
            section_id=sec.id,
            section_path=list(sec.path),
            part=sec.part,
            line_start=line_start,
            line_end=line_end,
            doc_id=doc_id,
            cells=cells,
        )
        blocks.append(blk)
        sec.block_ids.append(blk.id)

    i = 0
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # 1. Fenced code block (``` or ~~~).
        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            fence = fence_match.group(1)
            fence_char = fence[0]
            fence_len = len(fence)
            close_re = re.compile(rf"^\s*{re.escape(fence_char)}{{{fence_len},}}\s*$")
            start = i
            i += 1
            inner: list[str] = []
            while i < n:
                if close_re.match(lines[i]):
                    break
                inner.append(lines[i])
                i += 1
            end = i if i < n else n - 1
            raw_chunk = "\n".join(lines[start : end + 1])
            add_block(
                kind=BlockKind.CODE,
                block_text="\n".join(inner).strip("\n"),
                raw_chunk=raw_chunk,
                line_start=start + 1,
                line_end=end + 1,
            )
            i = end + 1
            continue

        # 2. ATX markdown heading (# .. ######).
        atx_match = _ATX_HEADING_RE.match(line)
        if atx_match:
            level = len(atx_match.group(1))
            raw_title = _TRAILING_HASHES_RE.sub("", atx_match.group(2) or "")
            add_heading(level, raw_title)
            i += 1
            continue

        # 3. Horizontal rule (---, ***, ___).
        if _HR_RE.match(line):
            i += 1
            continue

        # 4. Markdown pipe table with header + separator row.
        if i + 1 < n and "|" in line and _is_table_separator(lines[i + 1]):
            header_cells = [_strip_inline_markdown(c) for c in _split_table_cells(line)]
            add_block(
                kind=BlockKind.TABLE_HEADER,
                block_text=" | ".join(header_cells),
                raw_chunk=line,
                line_start=i + 1,
                line_end=i + 1,
                cells=header_cells,
            )
            i += 2
            while i < n and _is_table_row_line(lines[i]):
                if _is_table_separator(lines[i]):
                    i += 1
                    continue
                row_cells = [_strip_inline_markdown(c) for c in _split_table_cells(lines[i])]
                row_doc_id = _extract_doc_id(row_cells[0]) if row_cells else None
                add_block(
                    kind=BlockKind.TABLE_ROW,
                    block_text=" | ".join(row_cells),
                    raw_chunk=lines[i],
                    line_start=i + 1,
                    line_end=i + 1,
                    doc_id=row_doc_id,
                    cells=row_cells,
                )
                i += 1
            continue

        # 5. Headerless pipe table rows (| a | b | without separator line).
        if _is_strict_pipe_row(line) and not _is_table_separator(line):
            while i < n and _is_strict_pipe_row(lines[i]):
                if _is_table_separator(lines[i]):
                    i += 1
                    continue
                row_cells = [_strip_inline_markdown(c) for c in _split_table_cells(lines[i])]
                row_doc_id = _extract_doc_id(row_cells[0]) if row_cells else None
                add_block(
                    kind=BlockKind.TABLE_ROW,
                    block_text=" | ".join(row_cells),
                    raw_chunk=lines[i],
                    line_start=i + 1,
                    line_end=i + 1,
                    doc_id=row_doc_id,
                    cells=row_cells,
                )
                i += 1
            continue

        # Stray pipe separator line on odd markup: skip cleanly.
        if _is_table_separator(line):
            i += 1
            continue

        # 6. List item (unordered or ordered) with optional indented continuation lines.
        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            start = i
            base_indent = len(list_match.group(1).expandtabs(4))
            parts_text = [list_match.group(2).strip()]
            i += 1
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    break
                if _starts_structural_block(lines, i):
                    break
                nxt_indent = len(nxt.expandtabs(4)) - len(nxt.expandtabs(4).lstrip(" "))
                if nxt_indent <= base_indent:
                    break
                if _extract_doc_id(_strip_inline_markdown(nxt.strip())):
                    break
                parts_text.append(nxt.strip())
                i += 1
            end = i - 1
            cleaned = _strip_inline_markdown(" ".join(p for p in parts_text if p))
            add_block(
                kind=BlockKind.LIST_ITEM,
                block_text=cleaned,
                raw_chunk="\n".join(lines[start : end + 1]),
                line_start=start + 1,
                line_end=end + 1,
                doc_id=_extract_doc_id(cleaned),
            )
            continue

        # 7. Blockquote (> ...).
        quote_match = _BLOCKQUOTE_RE.match(line)
        if quote_match:
            start = i
            quote_lines: list[str] = []
            while i < n:
                m = _BLOCKQUOTE_RE.match(lines[i])
                if not m:
                    break
                payload = m.group(1).strip()
                if payload:
                    quote_lines.append(payload)
                i += 1
            end = i - 1
            cleaned = _strip_inline_markdown(" ".join(quote_lines))
            if cleaned:
                add_block(
                    kind=BlockKind.PARAGRAPH,
                    block_text=cleaned,
                    raw_chunk="\n".join(lines[start : end + 1]),
                    line_start=start + 1,
                    line_end=end + 1,
                    doc_id=_extract_doc_id(cleaned),
                )
            continue

        # 8. Setext heading (title followed directly by === or ---).
        setext_level = _match_setext_heading(lines, i)
        if setext_level is not None:
            add_heading(setext_level, line.strip())
            i += 2
            continue

        # 9. Paragraph (merge wrapped lines until blank line, structural boundary, or new doc_id).
        start = i
        para_lines = [line.strip()]
        i += 1
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break
            if _starts_structural_block(lines, i):
                break
            if _match_setext_heading(lines, i) is not None:
                break
            if _extract_doc_id(_strip_inline_markdown(nxt.strip())):
                break
            para_lines.append(nxt.strip())
            i += 1
        end = i - 1
        cleaned = _strip_inline_markdown(" ".join(para_lines))
        if cleaned:
            add_block(
                kind=BlockKind.PARAGRAPH,
                block_text=cleaned,
                raw_chunk="\n".join(lines[start : end + 1]),
                line_start=start + 1,
                line_end=end + 1,
                doc_id=_extract_doc_id(cleaned),
            )

    if not sections:
        ensure_section()

    parts = _assign_parts(sections, real_section_ids, blocks)

    return Document(
        path=raw.path,
        name=raw.name,
        title=raw.title,
        format=raw.format,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        char_count=len(text),
        text=text,
        sections=sections,
        blocks=blocks,
        parts=parts,
        warnings=list(raw.warnings),
    )


def _assign_parts(
    sections: list[Section],
    real_section_ids: set[str],
    blocks: list[Block],
) -> list[str]:
    """Assign top-level document parts per spec:
    - Single top-level heading with subheadings -> subheadings at the next level are parts.
    - Multiple top-level headings -> top-level headings themselves are parts.
    - No headings or single heading without subheadings -> single part.
    """
    by_id = {s.id: s for s in sections}
    real_sections = [s for s in sections if s.id in real_section_ids]

    if not real_sections:
        root_part = sections[0].title
        for sec in sections:
            sec.part = root_part
        for blk in blocks:
            blk.part = root_part
        return [root_part]

    top_level = min(s.level for s in real_sections)
    top_sections = [s for s in real_sections if s.level == top_level]

    if len(top_sections) >= 2:
        part_level = top_level
        part_sections = top_sections
        default_part: str | None = None
    else:
        sub_sections = [s for s in real_sections if s.level > top_level]
        if sub_sections:
            part_level = min(s.level for s in sub_sections)
            part_sections = [s for s in real_sections if s.level == part_level]
            default_part = top_sections[0].title
        else:
            part_level = top_level
            part_sections = top_sections
            default_part = top_sections[0].title

    for sec in sections:
        found: str | None = None
        cur: Section | None = sec
        while cur is not None:
            if cur.id in real_section_ids and cur.level == part_level:
                found = cur.title
                break
            cur = by_id.get(cur.parent_id) if cur.parent_id else None
        sec.part = found if found is not None else (default_part or sec.title)

    for blk in blocks:
        blk.part = by_id[blk.section_id].part

    parts: list[str] = []
    for sec in part_sections:
        if sec.title not in parts:
            parts.append(sec.title)
    return parts


def _starts_structural_block(lines: list[str], idx: int) -> bool:
    line = lines[idx]
    if _FENCE_OPEN_RE.match(line):
        return True
    if _ATX_HEADING_RE.match(line):
        return True
    if _HR_RE.match(line):
        return True
    if _LIST_ITEM_RE.match(line):
        return True
    if _BLOCKQUOTE_RE.match(line):
        return True
    if idx + 1 < len(lines) and "|" in line and _is_table_separator(lines[idx + 1]):
        return True
    return _is_strict_pipe_row(line)


def _match_setext_heading(lines: list[str], idx: int) -> int | None:
    if idx + 1 >= len(lines):
        return None
    title_line = lines[idx].strip()
    underline = lines[idx + 1]
    if not title_line or len(title_line) > 120:
        return None
    if _SETEXT_H1_RE.match(underline):
        return 1
    if _SETEXT_H2_RE.match(underline) and not title_line.endswith((".", ";", ",")):
        return 2
    return None


def _is_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_cells(line)
    if not cells:
        return False
    return all(bool(_TABLE_SEP_CELL_RE.match(cell)) for cell in cells)


def _is_table_row_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or "|" not in stripped:
        return False
    return not (_ATX_HEADING_RE.match(line) or _FENCE_OPEN_RE.match(line))


def _is_strict_pipe_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _split_table_cells(line: str) -> list[str]:
    """Split a markdown pipe table line into cells, preserving escaped pipes and empty cells."""
    stripped = line.strip().removeprefix("|")
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in stripped:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            buf.append(ch)
            continue
        if ch == "|":
            cells.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


def _strip_inline_markdown(text: str) -> str:
    """Remove inline markdown formatting while keeping identifier underscores and code text intact."""
    if not text:
        return ""

    code_spans: list[str] = []

    def _hold_code(match: re.Match[str]) -> str:
        idx = len(code_spans)
        code_spans.append(match.group(1))
        return f"\x00CODE{idx}\x00"

    cleaned = _INLINE_CODE_RE.sub(_hold_code, text)
    cleaned = _IMAGE_RE.sub(lambda m: m.group(1).strip() or m.group(2).strip(), cleaned)
    cleaned = _LINK_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_STAR_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_UNDER_RE.sub(r"\1", cleaned)
    cleaned = _ITALIC_STAR_RE.sub(r"\1", cleaned)
    cleaned = _ITALIC_UNDER_RE.sub(r"\1", cleaned)
    cleaned = _STRIKE_RE.sub(r"\1", cleaned)
    cleaned = _BR_TAG_RE.sub(" ", cleaned)
    cleaned = _HTML_FMT_TAG_RE.sub("", cleaned)
    cleaned = _ESCAPE_RE.sub(r"\1", cleaned)

    for idx, span in enumerate(code_spans):
        cleaned = cleaned.replace(f"\x00CODE{idx}\x00", span)

    cleaned = cleaned.replace("\u00a0", " ")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _extract_doc_id(text: str) -> str | None:
    """Extract a document requirement identifier located at the start of a block."""
    if not text:
        return None

    alpha_match = _ALPHA_ID_RE.match(text)
    if alpha_match:
        candidate = alpha_match.group("id")
        prefix = re.split(r"[-_]", candidate, maxsplit=1)[0].upper()
        if prefix not in _NON_ID_PREFIXES:
            return candidate

    num_match = _NUM_ID_RE.match(text)
    if num_match:
        candidate = num_match.group("id")
        punct = num_match.group("punct")
        rest = num_match.group("rest").lstrip()
        segments = candidate.split(".")
        # Two-segment numbers like "1.5 seconds" are decimals unless punctuated or followed by title/ID end.
        if (
            len(segments) == 2
            and not punct
            and rest
            and not (rest[0].isupper() or rest[0] in ("—", "–", "-"))
        ):
            return None
        return candidate

    return None
