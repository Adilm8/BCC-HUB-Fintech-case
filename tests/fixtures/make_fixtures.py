"""Generate DOCX and PDF fixtures from docs/requirements.md using installed libraries only."""

from __future__ import annotations

import re
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


def build_minimal_pdf_bytes(
    pages_lines: list[list[tuple[float, str]]],
    *,
    include_image_xobject: bool = False,
) -> bytes:
    """Build a valid multi-page PDF 1.4 with Type0 Identity-H Unicode text and font sizes."""
    objs: list[bytes] = []

    def add_obj(body: bytes) -> int:
        objs.append(body)
        return len(objs)

    chars = sorted({ch for page in pages_lines for _, txt in page for ch in txt})
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

    cid_font_id = add_obj(
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Helvetica "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 600 >>"
    )
    font_id = add_obj(
        (
            f"<< /Type /Font /Subtype /Type0 /BaseFont /Helvetica /Encoding /Identity-H "
            f"/DescendantFonts [{cid_font_id} 0 R] /ToUnicode {cmap_id} 0 R >>"
        ).encode("ascii")
    )

    image_id: int | None = None
    if include_image_xobject:
        # Minimal 1x1 grayscale image XObject.
        img_data = b"\x80"
        image_id = add_obj(
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\n"
            b"stream\n" + img_data + b"\nendstream"
        )

    pages_obj_id = add_obj(b"")
    page_ids: list[int] = []

    for page in pages_lines:
        ops: list[str] = []
        if page:
            ops.append("BT")
            y = 805.0
            for size, text in page:
                if text:
                    hex_str = "".join(f"{ord(ch):04X}" for ch in text)
                    ops.append(f"/F1 {size:g} Tf")
                    ops.append(f"1 0 0 1 42 {y:.1f} Tm")
                    ops.append(f"<{hex_str}> Tj")
                y -= max(size * 1.4, 12.0)
            ops.append("ET")
        if image_id is not None:
            ops.append("q 100 0 0 100 42 500 cm /Im1 Do Q")
        stream = "\n".join(ops).encode("ascii")
        content_id = add_obj(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        xobj_part = f" /XObject << /Im1 {image_id} 0 R >>" if image_id is not None else ""
        page_id = add_obj(
            (
                f"<< /Type /Page /Parent {pages_obj_id} 0 R /MediaBox [0 0 595 842] "
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

        if "|" in raw_line:
            entries.append((body_size, _clean_md_inline(raw_line)))
            continue

        entries.append((body_size, _clean_md_inline(raw_line)))

    # Paginate entries onto A4 pages with a running header and page footer.
    pages: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = []
    used_height = 0.0
    max_height = 700.0

    for item in entries:
        step = max(item[0] * 1.4, 12.0)
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


def generate_fixtures(source_md: Path = SOURCE_MD, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """Generate DOCX and PDF versions of the source markdown document."""
    md_text = source_md.read_text(encoding="utf-8-sig")
    docx_path = out_dir / "requirements.docx"
    pdf_path = out_dir / "requirements.pdf"
    build_docx_from_markdown(md_text, docx_path)
    build_pdf_from_markdown(md_text, pdf_path)
    return docx_path, pdf_path


if __name__ == "__main__":
    docx_out, pdf_out = generate_fixtures()
    print(f"Generated: {docx_out}")
    print(f"Generated: {pdf_out}")
