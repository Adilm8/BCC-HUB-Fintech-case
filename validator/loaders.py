"""Load requirements documents (.md, .txt, .pdf, .docx) into markdown-like RawDocument text."""

from __future__ import annotations

import codecs
import hashlib
import io
import logging
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from validator.models import RawDocument

# Alias: a loaded document is a RawDocument.
Document = RawDocument

SUPPORTED = ".md, .txt, .pdf, .docx"
_FORMAT_BY_SUFFIX: dict[str, Literal["md", "txt", "pdf", "docx"]] = {
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
}

_HR_LINE_RE = re.compile(r"^[-=*_|:\s]+$")
_BULLET_LINE_RE = re.compile(r"^(\s*)[•●▪◦‣⁃]\s+")
_SOFT_HYPHEN = "\u00ad"
_DEHYPHEN_RE = re.compile(
    r"([A-Za-zА-ЯЁа-яё]{2,})[-\u2010\u2011][ \t]*\n[ \t]*([a-zа-яё][A-Za-zА-ЯЁа-яё]*)"
)
_PAGE_NUM_RE = re.compile(
    r"^\s*(?:"
    r"(?:[-–—]\s*)?\d{1,4}(?:\s*(?:/|из|of)\s*\d{1,4})?(?:\s*[-–—])?"
    r"|(?:стр\.?|страница|с\.|page|p\.)\s*\d{1,4}(?:\s*(?:/|из|of)\s*\d{1,4})?"
    r")\s*$",
    re.IGNORECASE,
)
_HEADING_KEYWORD_RE = re.compile(
    r"^(?:Часть|ЧАСТЬ|Раздел|РАЗДЕЛ|Глава|ГЛАВА|Приложение|ПРИЛОЖЕНИЕ"
    r"|Part|PART|Section|SECTION|Chapter|CHAPTER|Appendix|APPENDIX)"
    r"\s+(?:\d+(?:\.\d+)*|[A-ZА-ЯЁ]|[IVX]{1,4})(?:[.\s:–—-].*)?$"
)
_MULTILEVEL_NUM_HEADING_RE = re.compile(r"^([1-9]\d*(?:\.\d+)+)\.?\s+([А-ЯЁA-Z].*)$")
_SINGLE_NUM_HEADING_RE = re.compile(r"^([1-9]\d*)\.\s+([А-ЯЁA-Z].*)$")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_TOC_DOTS_RE = re.compile(r"\.{3,}\s*\d+\s*$")
_LIST_START_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_DOCX_HEADING_RE = re.compile(r"^(?:Heading|Заголовок)\s*([1-6])$", re.IGNORECASE)
_DOCX_BULLET_STYLES = ("list bullet", "маркированный список")
_DOCX_NUMBER_STYLES = ("list number", "нумерованный список")
_CP1251_VALID_SYMBOLS = frozenset("\u00a0«»—–№…“”‘’„•§©®°±€₽™·")


@dataclass(frozen=True)
class _Span:
    text: str
    font_size: float
    bold: bool


@dataclass(frozen=True)
class _PageLine:
    text: str
    font_size: float | None = None
    bold: bool = False


@dataclass
class _PageTable:
    rows: list[list[str]]


_PageItem = _PageLine | _PageTable


def load_document(path: Path) -> RawDocument:
    """Read a supported requirements file and normalize it to markdown-like text."""
    file_path = Path(path)
    if file_path.is_dir():
        raise ValueError(f"Указан каталог вместо файла: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    suffix = file_path.suffix.lower()
    fmt = _FORMAT_BY_SUFFIX.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Формат {suffix or 'без расширения'} не поддерживается, нужен {SUPPORTED}"
        )

    warnings: list[str] = []
    if fmt in ("md", "txt"):
        text, warnings = _read_text_file(file_path)
    elif fmt == "docx":
        text, warnings = _docx_to_markdown(file_path)
    else:
        text, warnings = _pdf_to_markdown(file_path)

    text = _normalize_newlines(text)
    if not text.strip():
        if fmt == "pdf":
            raise ValueError(
                f"В PDF-документе нет текстового слоя (возможно, это скан): {file_path}"
            )
        raise ValueError(f"В документе нет текста: {file_path}")

    return RawDocument(
        path=str(file_path),
        name=file_path.name,
        title=_extract_title(text, file_path),
        format=fmt,
        text=text,
        warnings=warnings,
        sha256=hashlib.sha256(file_path.read_bytes()).hexdigest(),
    )


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text_file(path: Path) -> tuple[str, list[str]]:
    raw_bytes = path.read_bytes()

    # 1. Default UTF-8 (with optional UTF-8 BOM), no warning.
    try:
        return raw_bytes.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        pass

    # 2. UTF-16 with BOM (LE or BE, excluding UTF-32 LE which shares the first two bytes).
    if raw_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) and not raw_bytes.startswith(
        codecs.BOM_UTF32_LE
    ):
        try:
            text = raw_bytes.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Не удалось прочитать файл {path.name}: повреждённая кодировка UTF-16."
            ) from exc
        if "\x00" not in text:
            return text, [f"Файл {path.name} прочитан в кодировке UTF-16 (вместо UTF-8)."]

    # 3. Fallback to Windows-1251 (cp1251) when bytes look like valid text.
    if b"\x00" not in raw_bytes:
        try:
            text = raw_bytes.decode("cp1251")
        except UnicodeDecodeError:
            text = ""
        if text and _is_plausible_cp1251(text):
            return text, [f"Файл {path.name} прочитан в кодировке cp1251 (вместо UTF-8)."]

    raise ValueError(
        f"Не удалось прочитать файл {path.name}: поддерживаются кодировки UTF-8, UTF-16 (с BOM) и cp1251."
    )


def _is_plausible_cp1251(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (code < 32 and ch not in "\t\n\r\f\v") or code == 127:
            return False
    non_ascii = [ch for ch in text if ord(ch) >= 128]
    if not non_ascii:
        return True
    valid = sum(
        1 for ch in non_ascii if "\u0400" <= ch <= "\u04ff" or ch in _CP1251_VALID_SYMBOLS
    )
    return (valid / len(non_ascii)) >= 0.7


def _docx_to_markdown(path: Path) -> tuple[str, list[str]]:
    """Extract DOCX paragraphs and tables in document order as markdown-like text."""
    import docx
    from docx.opc.exceptions import PackageNotFoundError
    from docx.table import Table

    try:
        doc = docx.Document(str(path))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ValueError(
            f"Не удалось прочитать DOCX-файл {path.name}: файл повреждён или не является DOCX."
        ) from exc

    warnings: list[str] = []
    if _docx_has_images(doc):
        warnings.append(
            "В документе DOCX есть изображения или диаграммы — они не анализировались."
        )
    if _docx_has_body_nodes(doc, ("txbxContent",)):
        warnings.append(
            "В документе DOCX есть текстовые поля (надписи) — часть текста могла не извлечься."
        )
    if _docx_has_body_nodes(doc, ("sdt",)):
        warnings.append(
            "В документе DOCX есть элементы управления содержимым (SDT) — часть текста могла не извлечься."
        )

    out: list[str] = []
    prev_kind = ""
    num_counter = 0
    has_nested_tables = False

    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            num_counter = 0
            rows, nested = _extract_docx_table_rows(block)
            if nested:
                has_nested_tables = True
            if not rows:
                continue
            if out and out[-1] != "":
                out.append("")
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * len(rows[0]))
            out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
            out.append("")
            prev_kind = "table"
            continue

        text = re.sub(r"[ \t]+", " ", block.text.strip())
        if not text:
            continue

        style = ((block.style.name if block.style is not None else "") or "").strip()
        style_lower = style.lower()
        heading_match = _DOCX_HEADING_RE.match(style)

        if style_lower in ("title", "название"):
            num_counter = 0
            if out and out[-1] != "":
                out.append("")
            out.append("# " + text)
            out.append("")
            prev_kind = "heading"
        elif heading_match:
            num_counter = 0
            level = int(heading_match.group(1))
            if out and out[-1] != "":
                out.append("")
            out.append("#" * level + " " + text)
            out.append("")
            prev_kind = "heading"
        elif any(style_lower.startswith(prefix) for prefix in _DOCX_BULLET_STYLES):
            num_counter = 0
            if out and prev_kind not in ("", "list", "heading", "table") and out[-1] != "":
                out.append("")
            if not _LIST_START_RE.match(text):
                text = "- " + text
            out.append(text)
            prev_kind = "list"
        elif any(style_lower.startswith(prefix) for prefix in _DOCX_NUMBER_STYLES):
            if prev_kind != "num_list":
                num_counter = 0
                if out and prev_kind not in ("", "heading", "table") and out[-1] != "":
                    out.append("")
            num_counter += 1
            if not _LIST_START_RE.match(text):
                text = f"{num_counter}. " + text
            out.append(text)
            prev_kind = "num_list"
        else:
            num_counter = 0
            if out and out[-1] != "":
                out.append("")
            out.append(text)
            prev_kind = "paragraph"

    if has_nested_tables:
        warnings.append(
            "В документе DOCX есть вложенные таблицы — их строки включены в общую таблицу."
        )

    return "\n".join(out).strip(), warnings


def _docx_cell_content(cell: object) -> tuple[str, list[list[str]], bool]:
    """Extract paragraph text and recursively flattened nested table rows from a DOCX cell."""
    from docx.table import Table

    text_parts: list[str] = []
    nested_rows: list[list[str]] = []
    has_nested = False

    for item in cell.iter_inner_content():  # type: ignore[attr-defined]
        if isinstance(item, Table):
            has_nested = True
            child_rows, _ = _extract_docx_table_rows(item)
            nested_rows.extend(child_rows)
        else:
            cleaned = re.sub(r"\s+", " ", item.text.strip()).replace("|", "/")
            if cleaned:
                text_parts.append(cleaned)

    return " ".join(text_parts), nested_rows, has_nested


def _extract_docx_table_rows(table: object) -> tuple[list[list[str]], bool]:
    """Convert a DOCX table to pipe-table rows, collapsing merged cells and recursing into nested tables."""
    rows: list[list[str]] = []
    has_nested = False

    for row in table.rows:  # type: ignore[attr-defined]
        cells_text: list[str] = []
        nested_after_row: list[list[str]] = []
        prev_tc = None

        for cell in row.cells:
            # Skip repeated references to the same underlying <w:tc> from horizontal cell merges.
            tc = getattr(cell, "_tc", None)
            if tc is not None and tc is prev_tc:
                continue
            prev_tc = tc

            cell_text, child_rows, cell_has_nested = _docx_cell_content(cell)
            cells_text.append(cell_text)
            if cell_has_nested:
                has_nested = True
                nested_after_row.extend(child_rows)

        if cells_text and (any(cells_text) or not nested_after_row):
            rows.append(cells_text)
        rows.extend(nested_after_row)

    return rows, has_nested


def _docx_has_images(doc: object) -> bool:
    inline = getattr(doc, "inline_shapes", None)
    if inline is not None and len(inline) > 0:
        return True
    return _docx_has_body_nodes(doc, ("drawing", "pict"))


def _docx_has_body_nodes(doc: object, local_names: tuple[str, ...]) -> bool:
    element = getattr(doc, "element", None)
    body = getattr(element, "body", None)
    if body is None or not hasattr(body, "xpath"):
        return False
    cond = " or ".join(f"local-name()='{name}'" for name in local_names)
    try:
        return bool(body.xpath(f".//*[{cond}]"))
    except (AttributeError, TypeError, ValueError):
        return False


def _pdf_to_markdown(path: Path) -> tuple[str, list[str]]:
    """Extract PDF text and tables per page, strip repeated headers/footers, de-hyphenate, detect headings."""
    for logger_name in ("pypdf", "pdfminer", "pdfplumber"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    pdf_bytes = path.read_bytes()
    raw_pages, has_images = _extract_pdf_pages(pdf_bytes, path)

    warnings: list[str] = []
    if has_images:
        warnings.append(
            "В документе PDF есть изображения или диаграммы — они не анализировались."
        )

    cleaned_pages = _strip_repeated_headers_footers(raw_pages)
    merged_pages = _merge_split_pdf_tables(cleaned_pages)

    has_any_text = any(
        (isinstance(item, _PageLine) and bool(item.text.strip()))
        or (isinstance(item, _PageTable) and any(any(c.strip() for c in r) for r in item.rows))
        for page in merged_pages
        for item in page
    )
    if not has_any_text:
        raise ValueError(
            f"В PDF-документе нет текстового слоя (возможно, это скан): {path}"
        )

    headed_pages = _apply_pdf_heading_heuristics(merged_pages)
    page_texts: list[str] = []
    for page in headed_pages:
        rendered = _render_pdf_page(page)
        if rendered.strip():
            page_texts.append(rendered.strip())

    # Join pages and run de-hyphenation across page boundaries too.
    combined = "\n\n".join(page_texts)
    combined = _dehyphenate(combined)

    if not combined.strip():
        raise ValueError(
            f"В PDF-документе нет текстового слоя (возможно, это скан): {path}"
        )

    if not any(_MD_HEADING_RE.match(line.strip()) for line in combined.splitlines()):
        warnings.append(
            "PDF без заголовков: структуру разделов не удалось определить автоматически."
        )

    return combined, warnings


def _extract_pdf_pages(pdf_bytes: bytes, path: Path) -> tuple[list[list[_PageItem]], bool]:
    """Prefer pdfplumber for table-aware extraction; fall back to pypdf if unavailable or failing."""
    try:
        return _extract_pdf_pages_pdfplumber(pdf_bytes, path)
    except ValueError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, AttributeError, KeyError, IndexError):
        return _extract_pdf_pages_pypdf(pdf_bytes, path)


def _is_pdf_password_error(exc: BaseException, password_exc_cls: type[BaseException]) -> bool:
    inner = exc.args[0] if exc.args else None
    return isinstance(inner, password_exc_cls) or isinstance(exc.__cause__, password_exc_cls)


def _extract_pdf_pages_pdfplumber(pdf_bytes: bytes, path: Path) -> tuple[list[list[_PageItem]], bool]:
    import pdfplumber
    from pdfminer.pdfdocument import PDFPasswordIncorrect
    from pdfminer.pdfparser import PDFSyntaxError
    from pdfplumber.utils.exceptions import PdfminerException

    try:
        pdf_ctx = pdfplumber.open(io.BytesIO(pdf_bytes))
    except PdfminerException as exc:
        if _is_pdf_password_error(exc, PDFPasswordIncorrect):
            raise ValueError(
                f"PDF-файл защищён паролем и не может быть прочитан: {path}"
            ) from exc
        raise RuntimeError(str(exc)) from exc
    except PDFPasswordIncorrect as exc:
        raise ValueError(
            f"PDF-файл защищён паролем и не может быть прочитан: {path}"
        ) from exc
    except PDFSyntaxError as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        with pdf_ctx as pdf:
            if len(pdf.pages) == 0:
                raise ValueError(f"В PDF-документе нет страниц: {path}")

            has_images = False
            pages_out: list[list[_PageItem]] = []
            for page in pdf.pages:
                if not has_images and bool(getattr(page, "images", None)):
                    has_images = True
                pages_out.append(_extract_pdfplumber_page(page))
            return pages_out, has_images
    except (PdfminerException, PDFSyntaxError) as exc:
        raise RuntimeError(str(exc)) from exc


def _extract_pdfplumber_page(page: object) -> list[_PageItem]:
    page_w = float(getattr(page, "width", 0.0) or 0.0)
    page_h = float(getattr(page, "height", 0.0) or 0.0)
    page_area = page_w * page_h

    raw_tables = page.find_tables()  # type: ignore[attr-defined]
    accepted_tables: list[tuple[tuple[float, float, float, float], _PageTable]] = []

    for tbl in raw_tables:
        bbox = tuple(float(v) for v in tbl.bbox)
        extracted = tbl.extract()
        if not extracted:
            continue
        rows = _clean_pdfplumber_table(extracted)
        if not rows:
            continue
        max_cols = max(len(r) for r in rows)
        # Skip 1-column decorative frames or boxed code snippets.
        if max_cols < 2:
            continue
        tbl_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if page_area > 0 and tbl_area / page_area > 0.92 and len(rows) <= 1:
            continue
        accepted_tables.append((bbox, _PageTable(rows=rows)))

    # Exclude characters inside accepted table bounding boxes so table text is not duplicated.
    chars = list(getattr(page, "chars", ()) or ())
    outside_chars = [
        ch for ch in chars if not _char_in_any_bbox(ch, [bbox for bbox, _ in accepted_tables])
    ]
    text_entries = _group_chars_into_lines(outside_chars)

    # Interleave text lines and tables in top-to-bottom reading order.
    ordered: list[tuple[float, int, _PageItem]] = []
    for top, page_line in text_entries:
        ordered.append((top, 0, page_line))
    for bbox, page_table in accepted_tables:
        ordered.append((bbox[1], 1, page_table))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return [item for _, _, item in ordered]


def _clean_pdfplumber_table(extracted: list[list[str | None]]) -> list[list[str]]:
    cleaned_rows: list[list[str]] = []
    for raw_row in extracted:
        cells: list[str] = []
        for raw_cell in raw_row:
            if raw_cell is None:
                cells.append("")
                continue
            cell_text = _dehyphenate(_normalize_newlines(raw_cell))
            cells.append(re.sub(r"\s+", " ", cell_text.strip()).replace("|", "/"))
        if cells:
            cleaned_rows.append(cells)
    return cleaned_rows


def _char_in_any_bbox(ch: dict[str, object], bboxes: list[tuple[float, float, float, float]]) -> bool:
    if not bboxes:
        return False
    cx = (float(ch.get("x0", 0.0)) + float(ch.get("x1", 0.0))) / 2.0  # type: ignore[arg-type]
    cy = (float(ch.get("top", 0.0)) + float(ch.get("bottom", 0.0))) / 2.0  # type: ignore[arg-type]
    margin = 1.0
    for x0, top, x1, bottom in bboxes:
        if x0 - margin <= cx <= x1 + margin and top - margin <= cy <= bottom + margin:
            return True
    return False


def _group_chars_into_lines(chars: list[dict[str, object]]) -> list[tuple[float, _PageLine]]:
    """Cluster pdfplumber chars into lines by baseline (bottom) to avoid cross-size interleaving."""
    if not chars:
        return []

    # Sort by baseline first, then left-to-right.
    sorted_chars = sorted(
        chars,
        key=lambda c: (float(c.get("bottom", 0.0)), float(c.get("x0", 0.0))),  # type: ignore[arg-type]
    )

    clusters: list[tuple[float, list[dict[str, object]]]] = []
    for ch in sorted_chars:
        bottom = float(ch.get("bottom", 0.0))  # type: ignore[arg-type]
        size = abs(float(ch.get("size", 10.0) or 10.0))  # type: ignore[arg-type]
        tol = max(2.5, min(size * 0.3, 4.5))
        if not clusters or abs(bottom - clusters[-1][0]) > tol:
            clusters.append((bottom, [ch]))
        else:
            clusters[-1][1].append(ch)

    result: list[tuple[float, _PageLine]] = []
    prev_bottom: float | None = None
    prev_size = 10.0

    for baseline, line_chars in clusters:
        line_chars.sort(key=lambda c: float(c.get("x0", 0.0)))  # type: ignore[arg-type]
        parts: list[str] = []
        size_weights: Counter[float] = Counter()
        bold_chars = 0
        total_chars = 0
        prev_ch: dict[str, object] | None = None

        for ch in line_chars:
            txt = str(ch.get("text", "") or "")
            if not txt:
                continue
            x0 = float(ch.get("x0", 0.0))  # type: ignore[arg-type]
            size = round(abs(float(ch.get("size", 0.0) or 0.0)), 1)  # type: ignore[arg-type]

            if prev_ch is not None:
                prev_txt = str(prev_ch.get("text", "") or "")
                prev_x0 = float(prev_ch.get("x0", 0.0))  # type: ignore[arg-type]
                prev_x1 = float(prev_ch.get("x1", 0.0))  # type: ignore[arg-type]
                prev_b = float(prev_ch.get("bottom", 0.0))  # type: ignore[arg-type]
                # Skip duplicate fake-bold glyph overlays.
                if txt == prev_txt and abs(x0 - prev_x0) < 0.8 and abs(baseline - prev_b) < 0.8:
                    continue
                gap = x0 - prev_x1
                ref_size = min(size, abs(float(prev_ch.get("size", size) or size)))  # type: ignore[arg-type]
                space_gap = max(1.5, ref_size * 0.22)
                if not txt.isspace() and not prev_txt.isspace() and gap > space_gap:
                    parts.append(" ")

            parts.append(" " if txt.isspace() else txt)
            if not txt.isspace():
                if size > 0:
                    size_weights[size] += len(txt)
                fontname = str(ch.get("fontname", "") or "").lower()
                if any(tok in fontname for tok in ("bold", "black", "heavy", "demi", "semibold")):
                    bold_chars += len(txt)
                total_chars += len(txt)
            prev_ch = ch

        line_text = re.sub(r"[ \t]{2,}", " ", "".join(parts)).strip()
        if not line_text:
            continue

        top = min(float(c.get("top", baseline)) for c in line_chars)  # type: ignore[arg-type]
        dom_size = size_weights.most_common(1)[0][0] if size_weights else None
        is_bold = total_chars > 0 and (bold_chars / total_chars) >= 0.7

        if prev_bottom is not None and (baseline - prev_bottom) > max(prev_size * 1.9, 18.0):
            result.append((prev_bottom + 0.1, _PageLine(text="")))

        result.append((top, _PageLine(text=line_text, font_size=dom_size, bold=is_bold)))
        prev_bottom = baseline
        if dom_size:
            prev_size = dom_size

    return result


def _extract_pdf_pages_pypdf(pdf_bytes: bytes, path: Path) -> tuple[list[list[_PageItem]], bool]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать PDF-файл {path.name}: {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise ValueError(
                f"PDF-файл защищён паролем и не может быть прочитан: {path}"
            ) from exc
        if not decrypted:
            raise ValueError(f"PDF-файл защищён паролем и не может быть прочитан: {path}")

    if len(reader.pages) == 0:
        raise ValueError(f"В PDF-документе нет страниц: {path}")

    has_images = False
    raw_pages: list[list[_PageItem]] = []
    for page in reader.pages:
        if not has_images and _pdf_page_has_images(page):
            has_images = True
        raw_pages.append(list(_extract_pdf_page_lines(page)))
    return raw_pages, has_images


def _merge_split_pdf_tables(pages: list[list[_PageItem]]) -> list[list[_PageItem]]:
    """Merge a table split across consecutive PDF pages into a single table."""
    if len(pages) < 2:
        return pages

    merged: list[list[_PageItem]] = [list(page) for page in pages]
    for idx in range(len(merged) - 1):
        prev_page = merged[idx]
        next_page = merged[idx + 1]

        prev_pos = _last_non_empty_index(prev_page)
        next_pos = _first_non_empty_index(next_page)
        if prev_pos is None or next_pos is None:
            continue

        prev_item = prev_page[prev_pos]
        next_item = next_page[next_pos]
        if not (isinstance(prev_item, _PageTable) and isinstance(next_item, _PageTable)):
            continue
        if not prev_item.rows or not next_item.rows:
            continue
        if len(prev_item.rows[0]) != len(next_item.rows[0]):
            continue

        continuation = next_item.rows[1:] if next_item.rows[0] == prev_item.rows[0] else next_item.rows
        prev_item.rows.extend(continuation)
        del next_page[next_pos]

    return merged


def _first_non_empty_index(items: list[_PageItem]) -> int | None:
    for idx, item in enumerate(items):
        if isinstance(item, _PageTable) or item.text.strip():
            return idx
    return None


def _last_non_empty_index(items: list[_PageItem]) -> int | None:
    for idx in range(len(items) - 1, -1, -1):
        item = items[idx]
        if isinstance(item, _PageTable) or item.text.strip():
            return idx
    return None


def _render_pdf_page(items: list[_PageItem]) -> str:
    """Render interleaved PDF text lines and tables into markdown text."""
    chunks: list[str] = []
    line_buf: list[str] = []

    def flush_lines() -> None:
        if not line_buf:
            return
        text = _dehyphenate("\n".join(line_buf)).strip()
        if text:
            chunks.append(text)
        line_buf.clear()

    for item in items:
        if isinstance(item, _PageTable):
            flush_lines()
            if not item.rows:
                continue
            table_lines = [
                "| " + " | ".join(item.rows[0]) + " |",
                "|" + "---|" * len(item.rows[0]),
                *("| " + " | ".join(row) + " |" for row in item.rows[1:]),
            ]
            chunks.append("\n".join(table_lines))
        else:
            line_buf.append(_normalize_pdf_line(item.text))

    flush_lines()
    return "\n\n".join(chunks)


def _pdf_page_has_images(page: object) -> bool:
    from pypdf.errors import PdfReadError

    try:
        resources = page.get("/Resources")  # type: ignore[attr-defined]
        if resources is None:
            return False
        if hasattr(resources, "get_object"):
            resources = resources.get_object()
        xobjects = resources.get("/XObject") if hasattr(resources, "get") else None
        if xobjects is None:
            return False
        if hasattr(xobjects, "get_object"):
            xobjects = xobjects.get_object()
        if not hasattr(xobjects, "values"):
            return False
        for obj in xobjects.values():
            target = obj.get_object() if hasattr(obj, "get_object") else obj
            if hasattr(target, "get") and target.get("/Subtype") == "/Image":
                return True
    except (PdfReadError, AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return False
    return False


def _extract_pdf_page_lines(page: object) -> list[_PageLine]:
    from pypdf.errors import PdfReadError

    spans: list[_Span] = []

    def visitor(text: str, cm: list[float], tm: list[float], font_dict: object, font_size: float) -> None:
        if not text:
            return
        cleaned = text.strip()
        if not cleaned:
            return
        eff_size = _effective_font_size(cm, tm, font_size)
        is_bold = _is_bold_font(font_dict)
        spans.append(_Span(text=cleaned, font_size=eff_size, bold=is_bold))

    extract_errors = (PdfReadError, AttributeError, KeyError, TypeError, ValueError, IndexError, RuntimeError)
    try:
        raw_text = page.extract_text(visitor_text=visitor) or ""  # type: ignore[attr-defined]
    except extract_errors:
        try:
            raw_text = page.extract_text() or ""  # type: ignore[attr-defined]
        except extract_errors:
            raw_text = ""
        spans.clear()

    raw_text = _normalize_newlines(raw_text)
    raw_lines = raw_text.splitlines()
    return _align_lines_with_spans(raw_lines, spans)


def _effective_font_size(cm: list[float], tm: list[float], font_size: float) -> float:
    base = abs(float(font_size)) if font_size else 0.0
    if base <= 0:
        return 0.0
    tm_scale = 1.0
    if tm and len(tm) >= 4:
        s = math.hypot(float(tm[2]), float(tm[3]))
        if s > 0:
            tm_scale = s
    cm_scale = 1.0
    if cm and len(cm) >= 4:
        s = math.hypot(float(cm[2]), float(cm[3]))
        if s > 0:
            cm_scale = s
    return round(base * tm_scale * cm_scale, 1)


def _is_bold_font(font_dict: object) -> bool:
    if font_dict is None or not hasattr(font_dict, "get"):
        return False
    try:
        base_font = str(font_dict.get("/BaseFont", "") or "")
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    lower = base_font.lower()
    return any(tok in lower for tok in ("bold", "black", "heavy", "demi", "semibold"))


def _align_lines_with_spans(raw_lines: list[str], spans: list[_Span]) -> list[_PageLine]:
    if not spans:
        return [_PageLine(text=line) for line in raw_lines]

    result: list[_PageLine] = []
    span_idx = 0
    total_spans = len(spans)

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            result.append(_PageLine(text=""))
            continue

        matched_sizes: Counter[float] = Counter()
        bold_chars = 0
        total_chars = 0

        # Greedily consume spans belonging to this extracted line.
        probe = span_idx
        compact_line = re.sub(r"\s+", "", stripped)
        pos = 0
        while probe < total_spans and pos < len(compact_line):
            sp = spans[probe]
            compact_span = re.sub(r"\s+", "", sp.text)
            if not compact_span:
                probe += 1
                continue
            found = compact_line.find(compact_span, pos)
            if found == -1:
                # Allow a small lookahead skip if pypdf reordered a span.
                if pos == 0 and probe - span_idx < 3:
                    probe += 1
                    continue
                break
            weight = len(compact_span)
            if sp.font_size > 0:
                matched_sizes[sp.font_size] += weight
            if sp.bold:
                bold_chars += weight
            total_chars += weight
            pos = found + weight
            probe += 1
            span_idx = probe

        dom_size = matched_sizes.most_common(1)[0][0] if matched_sizes else None
        is_bold = total_chars > 0 and (bold_chars / total_chars) >= 0.7
        result.append(_PageLine(text=line, font_size=dom_size, bold=is_bold))

    return result


def _normalize_header_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip()).lower()
    return re.sub(r"\d+", "#", collapsed)


def _strip_repeated_headers_footers(pages: list[list[_PageItem]]) -> list[list[_PageItem]]:
    """Remove repeated headers/footers and standalone page numbers across multi-page PDFs."""
    num_pages = len(pages)
    if num_pages < 2:
        return pages

    edge_window = 3
    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()
    header_exact: Counter[str] = Counter()
    footer_exact: Counter[str] = Counter()

    for page in pages:
        non_empty = [
            item.text.strip()
            for item in page
            if isinstance(item, _PageLine) and item.text.strip()
        ]
        if not non_empty:
            continue
        top = non_empty[:edge_window]
        bottom = non_empty[-edge_window:]

        seen_top_keys: set[str] = set()
        seen_top_exact: set[str] = set()
        for item in top:
            if len(item) <= 120:
                seen_top_keys.add(_normalize_header_key(item))
                seen_top_exact.add(re.sub(r"\s+", " ", item).lower())
        header_counts.update(seen_top_keys)
        header_exact.update(seen_top_exact)

        seen_bot_keys: set[str] = set()
        seen_bot_exact: set[str] = set()
        for item in bottom:
            if len(item) <= 120:
                seen_bot_keys.add(_normalize_header_key(item))
                seen_bot_exact.add(re.sub(r"\s+", " ", item).lower())
        footer_counts.update(seen_bot_keys)
        footer_exact.update(seen_bot_exact)

    threshold = 2 if num_pages <= 3 else max(2, math.ceil(num_pages * 0.5))

    def is_repeated(item: str, counts: Counter[str], exact_counts: Counter[str]) -> bool:
        if len(item) > 120:
            return False
        if _PAGE_NUM_RE.match(item):
            return True
        exact = re.sub(r"\s+", " ", item).lower()
        key = _normalize_header_key(item)
        if num_pages == 2:
            # On 2-page PDFs require either exact repetition or a page-number-like template.
            return exact_counts[exact] >= 2 or (
                counts[key] >= 2 and ("#" in key and any(w in key for w in ("стр", "page", "из", "/")))
            )
        return counts[key] >= threshold

    cleaned_pages: list[list[_PageItem]] = []
    for page in pages:
        non_empty_indices = [
            idx
            for idx, item in enumerate(page)
            if isinstance(item, _PageLine) and item.text.strip()
        ]
        if not non_empty_indices:
            cleaned_pages.append(page)
            continue

        drop_indices: set[int] = set()
        for idx in non_empty_indices[:edge_window]:
            line_item = page[idx]
            if isinstance(line_item, _PageLine) and is_repeated(
                line_item.text.strip(), header_counts, header_exact
            ):
                drop_indices.add(idx)
        for idx in non_empty_indices[-edge_window:]:
            line_item = page[idx]
            if isinstance(line_item, _PageLine) and is_repeated(
                line_item.text.strip(), footer_counts, footer_exact
            ):
                drop_indices.add(idx)

        cleaned_pages.append([item for idx, item in enumerate(page) if idx not in drop_indices])

    return cleaned_pages


def _dehyphenate(text: str) -> str:
    """Join words split across line breaks with hyphens while keeping uppercase prefixes intact."""
    # Always join soft hyphens across line breaks and remove stray soft hyphens.
    text = re.sub(rf"{_SOFT_HYPHEN}[ \t]*\n[ \t]*", "", text)
    text = text.replace(_SOFT_HYPHEN, "")

    def _replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        # Preserve hyphen for all-uppercase technical prefixes like HTTP-код or API-метод.
        if left.isupper() and len(left) >= 2:
            return f"{left}-{right}"
        return f"{left}{right}"

    return _DEHYPHEN_RE.sub(_replace, text)


def _normalize_pdf_line(line: str) -> str:
    stripped = line.rstrip()
    if not stripped:
        return ""
    return _BULLET_LINE_RE.sub(r"\1- ", stripped)


def _apply_pdf_heading_heuristics(pages: list[list[_PageItem]]) -> list[list[_PageItem]]:
    """Promote likely PDF headings to markdown '#' lines using font sizes and text patterns."""
    size_weights: Counter[float] = Counter()
    for page in pages:
        for item in page:
            if not isinstance(item, _PageLine):
                continue
            stripped = item.text.strip()
            if stripped and item.font_size and item.font_size > 0:
                size_weights[item.font_size] += len(stripped)

    body_size = _weighted_median(size_weights) if size_weights else None
    heading_size_to_level: dict[float, int] = {}
    if body_size is not None and body_size > 0:
        larger_sizes = sorted(
            {s for s in size_weights if s >= body_size * 1.12},
            reverse=True,
        )
        # Cluster nearly identical font sizes within 0.4 pt into the same heading level.
        clusters: list[list[float]] = []
        for s in larger_sizes:
            if not clusters or (clusters[-1][-1] - s) > 0.4:
                clusters.append([s])
            else:
                clusters[-1].append(s)
        for level, cluster in enumerate(clusters[:6], start=1):
            for s in cluster:
                heading_size_to_level[s] = level

    has_h1 = False
    out_pages: list[list[_PageItem]] = []
    for page in pages:
        out_page: list[_PageItem] = []
        non_empty = [
            (idx, item.text.strip())
            for idx, item in enumerate(page)
            if isinstance(item, _PageLine) and item.text.strip()
        ]
        pos_in_non_empty = {idx: pos for pos, (idx, _) in enumerate(non_empty)}

        for idx, item in enumerate(page):
            if not isinstance(item, _PageLine):
                out_page.append(item)
                continue
            stripped = item.text.strip()
            if not stripped or _MD_HEADING_RE.match(stripped):
                if _MD_HEADING_RE.match(stripped) and stripped.startswith("# "):
                    has_h1 = True
                out_page.append(item)
                continue

            level = _detect_heading_level(
                item,
                stripped,
                heading_size_to_level,
                non_empty,
                pos_in_non_empty.get(idx, -1),
                has_h1,
            )
            if level is not None:
                if level == 1:
                    has_h1 = True
                out_page.append(
                    _PageLine(
                        text=f"{'#' * level} {stripped}",
                        font_size=item.font_size,
                        bold=item.bold,
                    )
                )
            else:
                out_page.append(item)
        out_pages.append(out_page)

    return out_pages


def _weighted_median(weights: Counter[float]) -> float | None:
    total = sum(weights.values())
    if total <= 0:
        return None
    half = total / 2
    acc = 0
    for value in sorted(weights):
        acc += weights[value]
        if acc >= half:
            return value
    return None


def _detect_heading_level(
    ln: _PageLine,
    stripped: str,
    heading_size_to_level: dict[float, int],
    non_empty: list[tuple[int, str]],
    non_empty_pos: int,
    has_h1: bool,
) -> int | None:
    # Skip pipe table rows, code/JSON snippets, TOC lines, and bullet items.
    if len(stripped) > 140 or "|" in stripped or _TOC_DOTS_RE.search(stripped):
        return None
    if stripped.startswith(("{", "}", "[", "]", "```", "~~~", "-", "*", "•")):
        return None

    # 1. Font-size-based heading detection (most reliable for exported PDFs).
    if (
        ln.font_size is not None
        and ln.font_size in heading_size_to_level
        and (not stripped.endswith((".", ";", ",")) or _HEADING_KEYWORD_RE.match(stripped))
    ):
        return heading_size_to_level[ln.font_size]

    # 2. Text-based heading heuristics (works even when font size is uniform).
    if stripped.endswith((".", ";", ",")):
        return None

    if _HEADING_KEYWORD_RE.match(stripped) and len(stripped) <= 100:
        return 2 if has_h1 else 1

    multi_match = _MULTILEVEL_NUM_HEADING_RE.match(stripped)
    if multi_match and len(stripped) <= 100 and len(stripped.split()) <= 12:
        depth = multi_match.group(1).count(".") + 1
        return min(depth, 6)

    single_match = _SINGLE_NUM_HEADING_RE.match(stripped)
    if single_match and len(stripped) <= 90 and len(stripped.split()) <= 10:
        # Avoid promoting consecutive numbered list items to headings.
        prev_txt = non_empty[non_empty_pos - 1][1] if non_empty_pos > 0 else ""
        next_txt = non_empty[non_empty_pos + 1][1] if 0 <= non_empty_pos < len(non_empty) - 1 else ""
        if not _SINGLE_NUM_HEADING_RE.match(prev_txt) and not _SINGLE_NUM_HEADING_RE.match(next_txt):
            return 2 if has_h1 else 1

    # 3. Short ALL-CAPS headings (e.g. "ОБЩИЕ СВЕДЕНИЯ").
    if _is_all_caps_heading(stripped):
        return 2 if has_h1 else 1

    return None


def _is_all_caps_heading(text: str) -> bool:
    if len(text) > 90 or any(ch in text for ch in ("=", "{", "}", "/", "\\", "`", ":")):
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 4:
        return False
    if not all(ch.isupper() for ch in letters):
        return False
    words = [w for w in re.split(r"\s+", text) if any(ch.isalpha() for ch in w)]
    return len(words) >= 2 or len(letters) >= 6


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned and not _HR_LINE_RE.match(cleaned):
            return cleaned[:200]
    return path.stem
