"""Generate DOCX and PDF fixtures from docs/requirements.md using installed libraries only."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MD = ROOT / "docs" / "requirements.md"
OUT_DIR = Path(__file__).resolve().parent / "docs"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{1,}:?\s*\|?\s*$")
_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$")
_FENCE_RE = re.compile(r"^\s*```")

_CHAR_WIDTH_RATIO = 0.52
_PAGE_W = 595.0
_PAGE_H = 842.0
_MARGIN_X = 36.0
_CONTENT_W = _PAGE_W - 2 * _MARGIN_X


@dataclass(frozen=True)
class _TextEntry:
    size: float
    text: str


@dataclass(frozen=True)
class _TableEntry:
    rows: list[list[str]]
    font_size: float = 8.5


_PdfEntry = _TextEntry | _TableEntry


def _clean_md_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", text)
    return text.strip()


def _split_md_cells(line: str) -> list[str]:
    stripped = line.strip().removeprefix("|").removesuffix("|")
    return [_clean_md_inline(cell) for cell in stripped.split("|")]


def build_docx_from_markdown(md_text: str, out_path: Path) -> None:
    """Convert a markdown document into a DOCX file preserving headings, lists, and tables."""
    doc = docx.Document()
    lines = md_text.replace("\r\n", "\n").splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if not line.strip() or _HR_RE.match(line):
            i += 1
            continue

        if _FENCE_RE.match(line):
            i += 1
            code_lines: list[str] = []
            while i < n and not _FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            i += 1
            if code_lines:
                doc.add_paragraph("\n".join(code_lines))
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = min(len(heading_match.group(1)), 4)
            doc.add_heading(_clean_md_inline(heading_match.group(2)), level=level)
            i += 1
            continue

        if i + 1 < n and "|" in line and _TABLE_SEP_RE.match(lines[i + 1]):
            rows = [_split_md_cells(line)]
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_split_md_cells(lines[i]))
                i += 1
            cols = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=cols)
            for r_idx, row in enumerate(rows):
                for c_idx in range(cols):
                    table.cell(r_idx, c_idx).text = row[c_idx] if c_idx < len(row) else ""
            continue

        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            doc.add_paragraph(_clean_md_inline(bullet_match.group(1)), style="List Bullet")
            i += 1
            continue

        ordered_match = _ORDERED_RE.match(line)
        if ordered_match:
            doc.add_paragraph(_clean_md_inline(ordered_match.group(1)), style="List Number")
            i += 1
            continue

        para_lines = [_clean_md_inline(line)]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and not _HEADING_RE.match(lines[i])
            and not _BULLET_RE.match(lines[i])
            and not _ORDERED_RE.match(lines[i])
            and not _HR_RE.match(lines[i])
            and not _FENCE_RE.match(lines[i])
            and not (i + 1 < n and "|" in lines[i] and _TABLE_SEP_RE.match(lines[i + 1]))
        ):
            para_lines.append(_clean_md_inline(lines[i]))
            i += 1
        doc.add_paragraph(" ".join(para_lines))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))


def _encode_pdf_hex(text: str) -> str:
    return "".join(f"{ord(ch):04X}" for ch in text)


def _assemble_pdf_bytes(
    page_streams: list[bytes],
    chars: list[str],
    *,
    include_image_xobject: bool = False,
) -> bytes:
    """Assemble a valid PDF 1.4 with a Type0 Identity-H font, FontDescriptor, and ToUnicode CMap."""
    objs: list[bytes] = []

    def add_obj(body: bytes) -> int:
        objs.append(body)
        return len(objs)

    cmap_lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]
    for start in range(0, len(chars), 100):
        chunk = chars[start : start + 100]
        cmap_lines.append(f"{len(chunk)} beginbfchar")
        for ch in chunk:
            code = f"{ord(ch):04X}"
            cmap_lines.append(f"<{code}> <{code}>")
        cmap_lines.append("endbfchar")
    cmap_lines.extend(
        ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    )
    cmap_bytes = "\n".join(cmap_lines).encode("ascii")
    cmap_id = add_obj(
        b"<< /Length " + str(len(cmap_bytes)).encode("ascii") + b" >>\nstream\n" + cmap_bytes + b"\nendstream"
    )

    font_desc_id = add_obj(
        b"<< /Type /FontDescriptor /FontName /Helvetica /Flags 32 "
        b"/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 >>"
    )
    dw = int(_CHAR_WIDTH_RATIO * 1000)
    cid_font_id = add_obj(
        (
            f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Helvetica "
            f"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
            f"/FontDescriptor {font_desc_id} 0 R /DW {dw} /W [32 [280]] >>"
        ).encode("ascii")
    )
    font_id = add_obj(
        (
            f"<< /Type /Font /Subtype /Type0 /BaseFont /Helvetica /Encoding /Identity-H "
            f"/DescendantFonts [{cid_font_id} 0 R] /ToUnicode {cmap_id} 0 R >>"
        ).encode("ascii")
    )

    image_id: int | None = None
    if include_image_xobject:
        img_data = b"\x80"
        image_id = add_obj(
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\n"
            b"stream\n" + img_data + b"\nendstream"
        )

    pages_obj_id = add_obj(b"")
    page_ids: list[int] = []
    xobj_part = f" /XObject << /Im1 {image_id} 0 R >>" if image_id is not None else ""

    for stream in page_streams:
        content_id = add_obj(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = add_obj(
            (
                f"<< /Type /Page /Parent {pages_obj_id} 0 R /MediaBox [0 0 {_PAGE_W:g} {_PAGE_H:g}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >>{xobj_part} >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs[pages_obj_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    ).encode("ascii")
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_obj_id} 0 R >>".encode("ascii"))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objs) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


def build_minimal_pdf_bytes(
    pages_lines: list[list[tuple[float, str]]],
    *,
    include_image_xobject: bool = False,
) -> bytes:
    """Build a valid multi-page PDF 1.4 with Type0 Identity-H Unicode text and font sizes."""
    chars = sorted({ch for page in pages_lines for _, txt in page for ch in txt})
    streams: list[bytes] = []
    for page in pages_lines:
        ops: list[str] = []
        if page:
            ops.append("BT")
            y = _PAGE_H - 36.0
            for size, text in page:
                y -= max(size * 1.15, 11.0)
                if text:
                    ops.append(f"/F1 {size:g} Tf")
                    ops.append(f"1 0 0 1 42 {y:.1f} Tm")
                    ops.append(f"<{_encode_pdf_hex(text)}> Tj")
                y -= max(size * 0.35, 3.0)
            ops.append("ET")
        if include_image_xobject:
            ops.append("q 100 0 0 100 42 500 cm /Im1 Do Q")
        streams.append("\n".join(ops).encode("ascii"))
    return _assemble_pdf_bytes(streams, chars, include_image_xobject=include_image_xobject)


def _wrap_text(text: str, max_width: float, font_size: float) -> list[str]:
    if not text:
        return [""]
    char_w = font_size * _CHAR_WIDTH_RATIO
    max_chars = max(int(max_width / max(char_w, 1.0)), 6)
    if len(text) <= max_chars:
        return [text]

    lines: list[str] = []
    words = text.split(" ")
    current = ""
    for word in words:
        if not current:
            if len(word) <= max_chars:
                current = word
            else:
                for pos in range(0, len(word), max_chars):
                    chunk = word[pos : pos + max_chars]
                    if len(chunk) == max_chars:
                        lines.append(chunk)
                    else:
                        current = chunk
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            if len(word) <= max_chars:
                current = word
            else:
                current = ""
                for pos in range(0, len(word), max_chars):
                    chunk = word[pos : pos + max_chars]
                    if len(chunk) == max_chars:
                        lines.append(chunk)
                    else:
                        current = chunk
    if current:
        lines.append(current)
    return lines or [""]


def _compute_col_widths(rows: list[list[str]], total_width: float) -> list[float]:
    num_cols = max(len(r) for r in rows)
    raw_weights = [1.0] * num_cols
    for row in rows:
        for idx, cell in enumerate(row):
            # Cap weight per column so one very long column does not starve short ones.
            raw_weights[idx] = max(raw_weights[idx], min(float(len(cell) or 4), 40.0))

    min_col = min(46.0, total_width / num_cols)
    weight_sum = sum(raw_weights)
    widths = [max(min_col, total_width * (w / weight_sum)) for w in raw_weights]
    scale = total_width / sum(widths)
    return [w * scale for w in widths]


def build_vector_tables_pdf_bytes(
    entries: list[_PdfEntry],
    *,
    header_text: str | None = None,
    footer_template: str | None = None,
) -> bytes:
    """Render text entries and real vector-bordered tables (no literal pipes) to PDF 1.4 bytes."""
    pad_x = 4.0
    pad_y = 3.5
    top_y = _PAGE_H - 58.0
    bottom_y = 54.0

    # Collect all characters for the ToUnicode CMap.
    char_set: set[str] = set()
    if header_text:
        char_set.update(header_text)
    if footer_template:
        char_set.update(footer_template)
        char_set.update("0123456789")
    for entry in entries:
        if isinstance(entry, _TextEntry):
            char_set.update(entry.text)
        else:
            for row in entry.rows:
                for cell in row:
                    char_set.update(cell)

    pages_ops: list[list[str]] = [[]]
    y = top_y

    def new_page() -> None:
        nonlocal y
        pages_ops.append([])
        y = top_y

    for entry in entries:
        if isinstance(entry, _TextEntry):
            if not entry.text:
                y -= 8.0
                continue
            wrapped = _wrap_text(entry.text, _CONTENT_W, entry.size)
            line_h = max(entry.size * 1.45, 13.0)
            needed = len(wrapped) * line_h + (4.0 if entry.size >= 11.5 else 0.0)
            if y - needed < bottom_y and pages_ops[-1]:
                new_page()
            if entry.size >= 11.5 and pages_ops[-1]:
                y -= 4.0
            ops = pages_ops[-1]
            ops.append("BT")
            for wline in wrapped:
                y -= line_h * 0.78
                ops.append(f"/F1 {entry.size:g} Tf")
                ops.append(f"1 0 0 1 {_MARGIN_X:.1f} {y:.1f} Tm")
                ops.append(f"<{_encode_pdf_hex(wline)}> Tj")
                y -= line_h * 0.22
            ops.append("ET")
            continue

        # Real vector-bordered table: stroke cell grid rectangles and draw cell text inside.
        rows = entry.rows
        if not rows:
            continue
        col_widths = _compute_col_widths(rows, _CONTENT_W)
        num_cols = len(col_widths)
        font_size = entry.font_size
        line_h = font_size * 1.35

        # Leave a slight gap before the table so it never touches preceding text.
        y -= 4.0
        for row in rows:
            padded_row = [row[c] if c < len(row) else "" for c in range(num_cols)]
            wrapped_cells = [
                _wrap_text(cell, max(col_widths[c] - 2 * pad_x, 12.0), font_size)
                for c, cell in enumerate(padded_row)
            ]
            max_lines = max(len(lines) for lines in wrapped_cells)
            row_h = max_lines * line_h + 2 * pad_y

            if y - row_h < bottom_y and pages_ops[-1]:
                new_page()

            ops = pages_ops[-1]
            row_bottom = y - row_h
            ops.append("0.5 w 0 G")
            x = _MARGIN_X
            for c_idx, col_w in enumerate(col_widths):
                # Draw cell border rectangle.
                ops.append(f"{x:.2f} {row_bottom:.2f} {col_w:.2f} {row_h:.2f} re S")
                cell_lines = wrapped_cells[c_idx]
                if any(cell_lines):
                    ops.append("BT")
                    text_y = y - pad_y - font_size * 0.85
                    for cl in cell_lines:
                        if cl:
                            ops.append(f"/F1 {font_size:g} Tf")
                            ops.append(f"1 0 0 1 {x + pad_x:.2f} {text_y:.2f} Tm")
                            ops.append(f"<{_encode_pdf_hex(cl)}> Tj")
                        text_y -= line_h
                    ops.append("ET")
                x += col_w
            y = row_bottom
        y -= 6.0

    # Add optional running headers and footers to every page.
    total_pages = len(pages_ops)
    streams: list[bytes] = []
    for page_num, body_ops in enumerate(pages_ops, start=1):
        full_ops: list[str] = []
        if header_text:
            full_ops.extend(
                [
                    "BT",
                    "/F1 8.5 Tf",
                    f"1 0 0 1 {_MARGIN_X:.1f} {_PAGE_H - 28.0:.1f} Tm",
                    f"<{_encode_pdf_hex(header_text)}> Tj",
                    "ET",
                ]
            )
        full_ops.extend(body_ops)
        if footer_template:
            footer = footer_template.format(page=page_num, total=total_pages)
            full_ops.extend(
                [
                    "BT",
                    "/F1 8.5 Tf",
                    f"1 0 0 1 {_MARGIN_X:.1f} 24.0 Tm",
                    f"<{_encode_pdf_hex(footer)}> Tj",
                    "ET",
                ]
            )
        streams.append("\n".join(full_ops).encode("ascii"))

    return _assemble_pdf_bytes(streams, sorted(char_set))


def build_pdf_from_markdown(md_text: str, out_path: Path) -> None:
    """Render a markdown document to a multi-page PDF with heading font sizes and running headers."""
    heading_sizes = {1: 18.0, 2: 15.0, 3: 13.0, 4: 11.5, 5: 11.0, 6: 10.5}
    body_size = 10.0
    entries: list[tuple[float, str]] = []
    in_code = False

    for raw_line in md_text.replace("\r\n", "\n").splitlines():
        if _FENCE_RE.match(raw_line):
            in_code = not in_code
            entries.append((body_size, raw_line.strip()))
            continue
        if in_code:
            entries.append((body_size, raw_line))
            continue
        if not raw_line.strip():
            entries.append((body_size, ""))
            continue
        if _HR_RE.match(raw_line):
            continue

        heading_match = _HEADING_RE.match(raw_line)
        if heading_match:
            level = len(heading_match.group(1))
            entries.append((heading_sizes.get(level, 11.0), _clean_md_inline(heading_match.group(2))))
            continue

        bullet_match = _BULLET_RE.match(raw_line)
        if bullet_match:
            entries.append((body_size, "• " + _clean_md_inline(bullet_match.group(1))))
            continue

        entries.append((body_size, _clean_md_inline(raw_line)))

    pages: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = []
    used_height = 0.0
    max_height = 680.0

    for item in entries:
        step = max(item[0] * 1.5, 14.0)
        if current and used_height + step > max_height:
            pages.append(current)
            current = []
            used_height = 0.0
        current.append(item)
        used_height += step
    if current:
        pages.append(current)

    total_pages = len(pages)
    decorated: list[list[tuple[float, str]]] = []
    for page_num, page_entries in enumerate(pages, start=1):
        decorated.append(
            [
                (8.5, "Спецификация требований — копия для проверки"),
                *page_entries,
                (8.5, f"Страница {page_num} из {total_pages}"),
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(build_minimal_pdf_bytes(decorated))


def build_real_tables_pdf_from_markdown(md_text: str, out_path: Path) -> None:
    """Render markdown to a real PDF with vector-drawn table grids and NO literal pipe characters."""
    heading_sizes = {1: 18.0, 2: 15.0, 3: 13.0, 4: 11.5, 5: 11.0, 6: 10.5}
    body_size = 10.0
    entries: list[_PdfEntry] = []
    lines = md_text.replace("\r\n", "\n").splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if not line.strip() or _HR_RE.match(line):
            i += 1
            continue

        if _FENCE_RE.match(line):
            entries.append(_TextEntry(body_size, line.strip()))
            i += 1
            while i < n and not _FENCE_RE.match(lines[i]):
                entries.append(_TextEntry(body_size, lines[i].replace("|", "/")))
                i += 1
            if i < n:
                entries.append(_TextEntry(body_size, lines[i].strip()))
                i += 1
            entries.append(_TextEntry(body_size, ""))
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = _clean_md_inline(heading_match.group(2)).replace("|", "/")
            entries.append(_TextEntry(heading_sizes.get(level, 11.0), title))
            i += 1
            continue

        if i + 1 < n and "|" in line and _TABLE_SEP_RE.match(lines[i + 1]):
            rows = [_split_md_cells(line)]
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_split_md_cells(lines[i]))
                i += 1
            entries.append(_TableEntry(rows=rows))
            continue

        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            entries.append(
                _TextEntry(body_size, "• " + _clean_md_inline(bullet_match.group(1)).replace("|", "/"))
            )
            i += 1
            continue

        ordered_match = _ORDERED_RE.match(line)
        if ordered_match:
            entries.append(_TextEntry(body_size, _clean_md_inline(line).replace("|", "/")))
            i += 1
            continue

        para_lines = [_clean_md_inline(line).replace("|", "/")]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and not _HEADING_RE.match(lines[i])
            and not _BULLET_RE.match(lines[i])
            and not _ORDERED_RE.match(lines[i])
            and not _HR_RE.match(lines[i])
            and not _FENCE_RE.match(lines[i])
            and not (i + 1 < n and "|" in lines[i] and _TABLE_SEP_RE.match(lines[i + 1]))
        ):
            para_lines.append(_clean_md_inline(lines[i]).replace("|", "/"))
            i += 1
        entries.append(_TextEntry(body_size, " ".join(para_lines)))

    pdf_bytes = build_vector_tables_pdf_bytes(
        entries,
        header_text="Спецификация требований — копия для проверки",
        footer_template="Страница {page} из {total}",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)


def generate_fixtures(source_md: Path = SOURCE_MD, out_dir: Path = OUT_DIR) -> tuple[Path, Path, Path]:
    """Generate DOCX, simple PDF, and real-vector-table PDF versions of the source markdown."""
    md_text = source_md.read_text(encoding="utf-8-sig")
    docx_path = out_dir / "requirements.docx"
    pdf_path = out_dir / "requirements.pdf"
    tables_pdf_path = out_dir / "requirements_tables.pdf"
    build_docx_from_markdown(md_text, docx_path)
    build_pdf_from_markdown(md_text, pdf_path)
    build_real_tables_pdf_from_markdown(md_text, tables_pdf_path)
    return docx_path, pdf_path, tables_pdf_path


if __name__ == "__main__":
    docx_out, pdf_out, tables_pdf_out = generate_fixtures()
    print(f"Generated: {docx_out}")
    print(f"Generated: {pdf_out}")
    print(f"Generated: {tables_pdf_out}")
