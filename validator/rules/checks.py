"""Deterministic prechecks: lexical hints, rule-based findings, and value inventory."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from validator.models import (
    Block,
    BlockKind,
    Criterion,
    Document,
    Hint,
    InventoryItem,
    PrecheckResult,
    RuleFinding,
    Section,
    SetCheck,
    Severity,
)
from validator.rules.lexicon import (
    CRITICAL_SECTION_HEADING_RE,
    DECLARED_TYPE_RE,
    HINT_CATEGORIES,
    NORMATIVE_STATEMENT_RE,
    PLACEHOLDER_INLINE_RE,
    PLACEHOLDER_LABELED_RE,
    PLACEHOLDER_VALUE_RE,
    PRIORITY_CODE_RE,
    PRIORITY_MARKER_RE,
    REQUIREMENT_SECTION_HEADING_RE,
    TEMPLATE_OPTION_CELL_RE,
)

_MAX_QUOTE_LEN = 200

# Standard / protocol prefixes that look like "PREFIX-NUMBER" but are not requirement IDs.
_NON_REQ_ID_PREFIXES = frozenset(
    {
        "UTF",
        "ISO",
        "IEC",
        "IEEE",
        "SHA",
        "MD",
        "RFC",
        "TLS",
        "SSL",
        "HTTP",
        "HTTPS",
        "TCP",
        "UDP",
        "JSON",
        "XML",
        "HTML",
        "SQL",
        "REST",
        "API",
        "URL",
        "URI",
        "UUID",
        "HMAC",
        "JWT",
        "AES",
        "RSA",
        "CRC",
        "BASE",
        "OAUTH",
        "ASCII",
        "MIME",
    }
)

_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_PLACEHOLDER_ACRONYMS = frozenset({"TBD", "TBA", "TBC", "TODO", "FIXME", "WIP"})

_HTTP_STATUS_CODES = frozenset(
    {
        "100",
        "101",
        "200",
        "201",
        "202",
        "204",
        "206",
        "301",
        "302",
        "304",
        "307",
        "308",
        "400",
        "401",
        "402",
        "403",
        "404",
        "405",
        "408",
        "409",
        "410",
        "413",
        "415",
        "422",
        "429",
        "500",
        "501",
        "502",
        "503",
        "504",
    }
)

_PREFIXED_ID_START_RE = re.compile(
    r"^(?:[\*•\-–—]\s*)?(?:\*\*)?([A-Za-zА-Яа-яЁё]{1,10}(?:[-_][A-Za-zА-Яа-яЁё]{1,10})?)([-_])(\d{1,5})(?:\*\*)?(?=[\.\)\:\s]|$)"
)
_PREFIXED_ID_EXACT_RE = re.compile(
    r"^([A-Za-zА-Яа-яЁё]{1,10}(?:[-_][A-Za-zА-Яа-яЁё]{1,10})?)([-_])(\d{1,5})$"
)
_DOTTED_ID_EXACT_RE = re.compile(r"^(\d+(?:\.\d+)+)$")
_HEADING_NUM_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")

_STANDARD_CONTEXT_RE = re.compile(
    r"(?:ISO|IEC|IEEE|RFC|ГОСТ|СНиП|СанПиН|ФЗ|Закон|ст\.)\b",
    re.IGNORECASE,
)

_REF_ID_EXPLICIT_RE = re.compile(
    r"(?:"
    r"(?:см\.?|смотри|согласно|по|в|из|к)\s+"
    r"(?:требовани(?:ю|ям|я|е|и)|пункт(?:у|е|а)?|п\.?)\s+"
    r"([A-Za-zА-Яа-яЁё]{1,6}[-_]\d{1,5})"
    r"|(?:см\.?|смотри|согласно)\s+([A-Za-zА-Яа-яЁё]{1,6}[-_]\d{1,5})"
    r"|(?:see|per|according\s+to|refer(?:ence|ring)?\s+to)\s+"
    r"(?:requirement|req\.?|item)?\s*([A-Za-z]{1,6}[-_]\d{1,5})"
    r")(?![\w-])",
    re.IGNORECASE,
)

_REF_SECTION_NUM_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:раздел(?:а|у|е|ом)?|подраздел(?:а|у|е)?|глав(?:а|е|у|ы)|пункт(?:а|у|е)?|п\.)\s+(\d+(?:\.\d+)*)"
    r"|(?:section|sec\.?|chapter|clause)\s+(\d+(?:\.\d+)*)"
    r")(?![\w.])",
    re.IGNORECASE,
)

_REF_SECTION_TITLE_RE = re.compile(
    r"(?<!\w)(?:раздел(?:а|у|е|ом)?|подраздел(?:а|у|е)?|глав(?:а|е|у|ы)|section|chapter)\s+"
    r"[«\"']([^«»\"']{2,100})[»\"']",
    re.IGNORECASE,
)

_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)|!\[[^\]]*\]\[[^\]]+\]|<img\b[^>]*>", re.IGNORECASE)

_UNIT_WORDS = (
    r"₸|₽|€|\$"
    r"|тенге|тиын(?:а|ов|ах|ам|ами|у|е)?"
    r"|KZT|USD|EUR|RUB|GBP"
    r"|руб(?:\.|лей|ля|ль)?|копе(?:ек|йки|йка|йку|йкой)|доллар(?:ов|а)?|евро|cent(?:s)?"
    r"|%|процент(?:ов|а|у|ом|е|ам|ами|ах)?|percent|pct"
    r"|миллисекунд(?:а|ы|у|ой|е|ам|ами|ах)?|мсек|мс|millisecond(?:s)?|msec|ms"
    r"|секунд(?:а|ы|у|ой|е|ам|ами|ах)?|сек\.?|second(?:s)?|sec\.?"
    r"|минут(?:а|ы|у|ой|е|ам|ами|ах)?|мин\.?|minute(?:s)?|min(?:s)?"
    r"|час(?:а|ов|у|ом|е|ам|ами|ах)?|hour(?:s)?|hr(?:s)?"
    r"|дн(?:ей|я|ю|[её]м|е|ям|ями|ях)|день|сут(?:ок|ки|кам|ками|ках)|day(?:s)?"
    r"|недел(?:ь|я|и|ю|ей|е|ям|ями|ях)|week(?:s)?"
    r"|месяц(?:ев|а|у|ем|е|ам|ами|ах)?|мес\.?|month(?:s)?"
    r"|лет|год(?:а|ов|у|ом|е|ам|ами|ах)?|year(?:s)?"
    r"|KiB|MiB|GiB|КБ|МБ|ГБ|ТБ|KB|MB|GB|TB"
    r"|байт(?:а|ов)?|byte(?:s)?|бит(?:а|ов)?|bit(?:s)?"
    r"|знак(?:ов|а)?|символ(?:ов|а)?|character(?:s)?|char(?:s)?"
    r"|RPS|TPS|QPS|req/s|ops/s"
    r"|с(?=\s*(?:[.,;)\]]|$))|ч(?=\s*(?:[.,;)\]]|$))"
)

_NUM_TOKEN = r"\d{1,3}(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_RANGE_JOIN = r"(?:\s*[–—-]\s*|\s+до\s+|\s+to\s+)"

_QUANTITY_SUFFIX_RE = re.compile(
    rf"(?<!\d)(?<!\w)((?:{_NUM_TOKEN})(?:{_RANGE_JOIN}(?:{_NUM_TOKEN}))?)\s*({_UNIT_WORDS})(?!\w)",
    re.IGNORECASE,
)

_QUANTITY_PREFIX_CURRENCY_RE = re.compile(
    rf"(?<!\w)([₸₽€$])\s*((?:{_NUM_TOKEN})(?:{_RANGE_JOIN}(?:{_NUM_TOKEN}))?)(?!\w)"
)

_WORD_DURATION_RE = re.compile(
    r"(?<!\w)(в\s+течение\s+(секунды|минуты|часа|дня|суток|недели|месяца)|"
    r"within\s+(?:a|one)\s+(second|minute|hour|day|week|month))(?!\w)",
    re.IGNORECASE,
)

_HTTP_METHOD_RE = re.compile(r"(?<!\w)(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)(?!\w)")

_HTTP_CONTEXT_RE = re.compile(
    r"(?:HTTP|REST|API|статус|код|ответ|ошибк|результат|возвраща|метод|запрос|"
    r"status|code|response|error|unauthorized|forbidden|bad\s+request|not\s+found|"
    r"created|internal\s+server|successful|unsuccessful|endpoint)",
    re.IGNORECASE,
)

_HTTP_CODE_RE = re.compile(r"(?<![\dA-Za-zА-Яа-яЁё_])([1-5]\d{2})(?![\dA-Za-zА-Яа-яЁё_])")
_FOLLOWED_BY_UNIT_RE = re.compile(rf"^\s*(?:{_UNIT_WORDS})(?!\w)", re.IGNORECASE)

_IDENT_CAMEL_RE = re.compile(
    r"(?<!\w)(?:[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)(?!\w)"
)
_IDENT_SNAKE_RE = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?!\w)")
_IDENT_ENDPOINT_RE = re.compile(
    r"(?<![:\w/])(/[A-Za-z][A-Za-z0-9_{}-]*(?:/[A-Za-z0-9_{}-]+)*)(?![\w/])"
)
_IDENT_HEADER_RE = re.compile(r"(?<!\w)([A-Z][a-z0-9]*(?:-[A-Z][a-z0-9]+)+)(?!\w)")
_IDENT_BACKTICK_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.-]{1,63})`")
_IDENT_ALLCAPS_RE = re.compile(r"(?<!\w)([A-Z][A-Z0-9_]{1,23})(?!\w)")

_HORIZONTAL_RULE_RE = re.compile(r"^[\s\-_*=—–]+$")
_DATE_LIKE_RE = re.compile(r"^\d{1,4}\s*/\s*\d{1,4}(?:\s*/\s*\d{1,4})?$")

_PARAM_NAME_HEADER_RE = re.compile(
    r"^(?:parameter|param|field|name|key|attribute|property|column|"
    r"параметр|поле|имя|название|атрибут|ключ|реквизит)$",
    re.IGNORECASE,
)
_PARAM_TYPE_HEADER_RE = re.compile(
    r"^(?:type|format|data\s*type|тип|тип\s+данных|формат)$",
    re.IGNORECASE,
)
_PARAM_FIELD_NAME_RE = re.compile(r"^[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_.\[\]-]{0,63}$")
_META_REVIEW_SECTION_RE = re.compile(r"ревью|review|согласован|approv", re.IGNORECASE)


@dataclass(frozen=True)
class _TableGroup:
    section_id: str
    header: Block | None
    rows: list[Block]


@dataclass(frozen=True)
class _ReqIdOccurrence:
    raw_id: str
    kind: str  # "prefixed" or "dotted"
    prefix: str
    sep: str
    num: int
    width: int
    canonical: str
    dotted_parts: tuple[int, ...]
    block: Block


def _section_label(block: Block) -> str:
    if block.section_path:
        return " › ".join(block.section_path)
    return block.section_id or "документ"


def _section_display(section: Section) -> str:
    if section.path:
        return " › ".join(section.path)
    return section.title


def _get_cells(block: Block) -> list[str]:
    if block.cells is not None:
        return list(block.cells)
    raw = block.text
    if "|" not in raw:
        return [raw.strip()]
    parts = raw.split("|")
    if parts and parts[0].strip() == "" and raw.lstrip().startswith("|"):
        parts = parts[1:]
    if parts and parts[-1].strip() == "" and raw.rstrip().endswith("|"):
        parts = parts[:-1]
    return [part.strip() for part in parts]


def _quote_for_span(text: str, start: int = 0, end: int | None = None) -> str:
    """Return an exact substring of ``text`` covering ``start:end`` within ``_MAX_QUOTE_LEN``."""
    if len(text) <= _MAX_QUOTE_LEN:
        return text
    if end is None:
        end = min(len(text), _MAX_QUOTE_LEN)
    match_len = max(0, end - start)
    if match_len >= _MAX_QUOTE_LEN:
        return text[start : start + _MAX_QUOTE_LEN]
    pad = (_MAX_QUOTE_LEN - match_len) // 2
    lo = max(0, start - pad)
    hi = min(len(text), lo + _MAX_QUOTE_LEN)
    lo = max(0, hi - _MAX_QUOTE_LEN)
    # Snap window edges to whitespace boundaries when possible without losing the span.
    if lo > 0:
        space = text.find(" ", lo, start)
        if space != -1 and space + 1 < start:
            lo = space + 1
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        if space != -1 and space > end:
            hi = space
    return text[lo:hi]


def _quote_containing(text: str, needle: str) -> str:
    if not text:
        return ""
    if len(text) <= _MAX_QUOTE_LEN:
        return text
    idx = text.find(needle)
    if idx != -1:
        return _quote_for_span(text, idx, idx + len(needle))
    idx_ci = text.lower().find(needle.lower())
    if idx_ci != -1:
        return _quote_for_span(text, idx_ci, idx_ci + len(needle))
    return _quote_for_span(text, 0, min(len(text), _MAX_QUOTE_LEN))


def _group_tables(blocks: list[Block]) -> list[_TableGroup]:
    tables: list[_TableGroup] = []
    current_sec: str | None = None
    current_header: Block | None = None
    current_rows: list[Block] = []
    last_line_end = 0

    def flush() -> None:
        nonlocal current_sec, current_header, current_rows
        if current_sec is not None and (current_header is not None or current_rows):
            tables.append(_TableGroup(section_id=current_sec, header=current_header, rows=current_rows))
        current_sec = None
        current_header = None
        current_rows = []

    for block in blocks:
        if block.kind == BlockKind.TABLE_HEADER:
            flush()
            current_sec = block.section_id
            current_header = block
            current_rows = []
            last_line_end = block.line_end
        elif block.kind == BlockKind.TABLE_ROW:
            line_gap = (
                block.line_start > 0
                and last_line_end > 0
                and block.line_start - last_line_end > 2
            )
            if current_sec != block.section_id or line_gap:
                flush()
                current_sec = block.section_id
            current_rows.append(block)
            last_line_end = block.line_end
        else:
            flush()
    flush()
    return tables


def _normalize_option_cell(cell: str) -> str | None:
    stripped = cell.strip()
    if "/" not in stripped:
        return None
    if not TEMPLATE_OPTION_CELL_RE.match(stripped):
        return None
    if _DATE_LIKE_RE.match(stripped):
        return None
    parts = [p.strip() for p in stripped.split("/")]
    if len(parts) < 2 or any(not p for p in parts):
        return None
    if not all(any(ch.isalpha() for ch in p) for p in parts):
        return None
    return " / ".join(re.sub(r"\s+", " ", p).lower() for p in parts)


def _check_unfilled_template_tables(
    tables: list[_TableGroup],
) -> tuple[list[RuleFinding], set[str]]:
    findings: list[RuleFinding] = []
    handled_block_ids: set[str] = set()

    for table in tables:
        if len(table.rows) < 3:
            continue
        col_option_rows: dict[tuple[int, str], list[tuple[Block, str]]] = defaultdict(list)
        for row in table.rows:
            cells = _get_cells(row)
            for col_idx, cell in enumerate(cells):
                norm = _normalize_option_cell(cell)
                if norm is not None:
                    col_option_rows[(col_idx, norm)].append((row, cell.strip()))

        qualifying = [entries for entries in col_option_rows.values() if len(entries) >= 3]
        if not qualifying:
            continue

        # Sort deterministically by count descending, then first row order.
        row_order = {row.id: idx for idx, row in enumerate(table.rows)}
        qualifying.sort(key=lambda entries: (-len(entries), row_order[entries[0][0].id]))

        primary_entries = qualifying[0]
        matched_rows_by_id: dict[str, Block] = {}
        for entries in qualifying:
            for row, _ in entries:
                matched_rows_by_id[row.id] = row

        ordered_rows = [row for row in table.rows if row.id in matched_rows_by_id]
        first_row, sample_option = primary_entries[0]
        sec_label = _section_label(first_row)
        quote = _quote_containing(first_row.text, sample_option)

        findings.append(
            RuleFinding(
                rule="unfilled_template_table",
                block_ids=[row.id for row in ordered_rows],
                criterion=Criterion.SET_COMPLETE,
                severity=Severity.MAJOR,
                set_check=SetCheck.EMPTY_SECTION,
                quote=quote,
                explanation=(
                    f"В таблице раздела «{sec_label}» обнаружен незаполненный шаблон: "
                    f"варианты «{sample_option}» повторяются без выбора значения "
                    f"в {len(ordered_rows)} строках."
                ),
                recommendation=(
                    "Заполнить строки таблицы конкретными значениями или явно пометить "
                    "раздел как неприменимый."
                ),
                clarifying_question=(
                    f"Какие фактические значения должны быть указаны в таблице раздела "
                    f"«{sec_label}» вместо шаблонных вариантов «{sample_option}»?"
                ),
            )
        )
        # Mark all rows of this unfilled template table so per-row placeholder/empty checks do not spam.
        for row in table.rows:
            handled_block_ids.add(row.id)

    return findings, handled_block_ids


def _check_empty_sections(doc: Document) -> tuple[list[RuleFinding], set[str]]:
    findings: list[RuleFinding] = []
    handled_block_ids: set[str] = set()

    block_by_id = {b.id: b for b in doc.blocks}
    blocks_by_section: dict[str, list[Block]] = defaultdict(list)
    for b in doc.blocks:
        blocks_by_section[b.section_id].append(b)

    # Identify sections that have child subsections (by parent_id or path prefix).
    parent_ids: set[str] = {s.parent_id for s in doc.sections if s.parent_id}
    for sec in doc.sections:
        if sec.id in parent_ids:
            continue
        for other in doc.sections:
            if (
                other.id != sec.id
                and len(other.path) > len(sec.path)
                and other.path[: len(sec.path)] == sec.path
            ):
                parent_ids.add(sec.id)
                break

    for sec in doc.sections:
        if sec.id in parent_ids:
            continue

        sec_blocks: list[Block] = []
        seen_ids: set[str] = set()
        for bid in sec.block_ids:
            b = block_by_id.get(bid)
            if b is not None and b.id not in seen_ids:
                sec_blocks.append(b)
                seen_ids.add(b.id)
        for b in blocks_by_section.get(sec.id, []):
            if b.id not in seen_ids:
                sec_blocks.append(b)
                seen_ids.add(b.id)

        heading_block = next((b for b in sec_blocks if b.kind == BlockKind.HEADING), None)
        content_blocks = [
            b for b in sec_blocks if b.kind not in (BlockKind.HEADING, BlockKind.TABLE_HEADER)
        ]

        has_meaningful_content = False
        empty_row_ids: list[str] = []
        for b in content_blocks:
            if b.kind == BlockKind.TABLE_ROW:
                cells = _get_cells(b)
                if any(c.strip() for c in cells):
                    has_meaningful_content = True
                    break
                empty_row_ids.append(b.id)
            else:
                stripped = b.text.strip()
                if stripped and not _HORIZONTAL_RULE_RE.match(stripped):
                    has_meaningful_content = True
                    break

        if has_meaningful_content:
            continue

        sec_label = _section_display(sec)
        severity = (
            Severity.CRITICAL
            if CRITICAL_SECTION_HEADING_RE.search(sec.title)
            and not _META_REVIEW_SECTION_RE.search(sec.title)
            else Severity.MAJOR
        )
        if heading_block is not None:
            block_ids = [heading_block.id, *empty_row_ids]
            quote = _quote_for_span(heading_block.text)
        elif sec_blocks:
            first_b = sec_blocks[0]
            block_ids = [first_b.id, *[r for r in empty_row_ids if r != first_b.id]]
            quote = _quote_for_span(first_b.text) or sec.title
        else:
            block_ids = []
            quote = sec.title

        findings.append(
            RuleFinding(
                rule="empty_section",
                block_ids=block_ids,
                criterion=Criterion.SET_COMPLETE,
                severity=severity,
                set_check=SetCheck.EMPTY_SECTION,
                quote=quote,
                explanation=(
                    f"Раздел «{sec_label}» не содержит заполненных требований или данных."
                ),
                recommendation=(
                    "Заполнить раздел содержательными требованиями либо явно указать, "
                    "что раздел не применим к данной задаче."
                ),
                clarifying_question=(
                    f"Какие требования должны быть указаны в пустом разделе «{sec_label}» "
                    "или этот раздел следует исключить?"
                ),
            )
        )
        handled_block_ids.update(empty_row_ids)

    return findings, handled_block_ids


def _check_placeholders_and_empty_cells(
    doc: Document,
    tables: list[_TableGroup],
    skip_block_ids: set[str],
) -> list[RuleFinding]:
    findings: list[RuleFinding] = []

    # Map each table row to its header cell names for clearer error messages.
    row_headers: dict[str, list[str]] = {}
    row_header_block: dict[str, Block | None] = {}
    for table in tables:
        headers = _get_cells(table.header) if table.header is not None else []
        for row in table.rows:
            row_headers[row.id] = headers
            row_header_block[row.id] = table.header

    # Map sections to their titles and detect leaf sections consisting solely of placeholders (stub sections).
    sec_title_by_id = {s.id: s.title for s in doc.sections}
    content_blocks_by_section: dict[str, list[Block]] = defaultdict(list)
    for b in doc.blocks:
        if b.kind not in (BlockKind.HEADING, BlockKind.TABLE_HEADER):
            content_blocks_by_section[b.section_id].append(b)

    def _block_is_placeholder_or_empty(blk: Block) -> bool:
        if blk.id in skip_block_ids:
            return True
        if blk.kind == BlockKind.TABLE_ROW:
            row_cells = _get_cells(blk)
            return all(
                not c.strip()
                or bool(PLACEHOLDER_VALUE_RE.match(c.strip()))
                or bool(PLACEHOLDER_INLINE_RE.search(c.strip()))
                for c in row_cells
            )
        txt = blk.text.strip()
        if not txt or _HORIZONTAL_RULE_RE.match(txt):
            return True
        return bool(
            PLACEHOLDER_VALUE_RE.match(txt)
            or PLACEHOLDER_INLINE_RE.search(txt)
            or PLACEHOLDER_LABELED_RE.search(txt)
        )

    stub_section_ids: set[str] = {
        sec_id
        for sec_id, sec_blks in content_blocks_by_section.items()
        if sec_blks and all(_block_is_placeholder_or_empty(b) for b in sec_blks)
    }

    for block in doc.blocks:
        if block.id in skip_block_ids:
            continue
        if block.kind in (BlockKind.CODE, BlockKind.TABLE_HEADER):
            continue

        sec_label = _section_label(block)

        if block.kind == BlockKind.TABLE_ROW:
            cells = _get_cells(block)
            headers = row_headers.get(block.id, [])
            placeholder_hit: str | None = None
            empty_cols: list[str] = []

            for idx, cell in enumerate(cells):
                stripped = cell.strip()
                if not stripped:
                    col_name = (
                        headers[idx]
                        if idx < len(headers) and headers[idx].strip()
                        else f"колонка {idx + 1}"
                    )
                    empty_cols.append(col_name)
                    continue
                if placeholder_hit is None:
                    if PLACEHOLDER_VALUE_RE.match(stripped):
                        placeholder_hit = stripped
                    else:
                        m = PLACEHOLDER_INLINE_RE.search(stripped) or PLACEHOLDER_LABELED_RE.search(
                            stripped
                        )
                        if m is not None:
                            placeholder_hit = m.group(0)

            if placeholder_hit is not None:
                quote = _quote_containing(block.text, placeholder_hit)
                findings.append(
                    RuleFinding(
                        rule="placeholder",
                        block_ids=[block.id],
                        criterion=Criterion.SET_COMPLETE,
                        severity=Severity.MAJOR,
                        set_check=SetCheck.MISSING_REQUIREMENT,
                        quote=quote,
                        explanation=(
                            f"В строке таблицы раздела «{sec_label}» указано значение-заглушка "
                            f"«{placeholder_hit}» вместо конкретного параметра."
                        ),
                        recommendation=(
                            "Заменить заглушку фактическим значением или явно зафиксировать, "
                            "что параметр не применим."
                        ),
                        clarifying_question=(
                            f"Какое фактическое значение должно быть указано вместо заглушки "
                            f"«{placeholder_hit}» в разделе «{sec_label}»?"
                        ),
                    )
                )

            if empty_cols:
                cols_text = ", ".join(f"«{c}»" for c in empty_cols)
                if block.text.strip():
                    bid_list = [block.id]
                    quote = _quote_for_span(block.text)
                else:
                    hdr = row_header_block.get(block.id)
                    if hdr is not None and hdr.text.strip():
                        bid_list = [hdr.id, block.id]
                        quote = _quote_for_span(hdr.text)
                    else:
                        bid_list = [block.id]
                        quote = block.text
                findings.append(
                    RuleFinding(
                        rule="empty_cell",
                        block_ids=bid_list,
                        criterion=Criterion.SET_COMPLETE,
                        severity=Severity.MAJOR,
                        set_check=SetCheck.MISSING_REQUIREMENT,
                        quote=quote,
                        explanation=(
                            f"В строке таблицы раздела «{sec_label}» не заполнены ячейки: "
                            f"{cols_text}."
                        ),
                        recommendation=(
                            "Заполнить пустые ячейки таблицы или удалить лишнюю строку."
                        ),
                        clarifying_question=(
                            f"Какие значения должны быть указаны в пустых полях ({cols_text}) "
                            f"таблицы раздела «{sec_label}»?"
                        ),
                    )
                )
            continue

        # Prose, list item, heading, or other block.
        stripped = block.text.strip()
        if not stripped or _HORIZONTAL_RULE_RE.match(stripped):
            continue

        match_text: str | None = None
        if PLACEHOLDER_VALUE_RE.match(stripped):
            match_text = stripped
        else:
            m = PLACEHOLDER_INLINE_RE.search(block.text) or PLACEHOLDER_LABELED_RE.search(block.text)
            if m is not None:
                match_text = m.group(0)

        if match_text is None:
            continue

        is_section_placeholder = (
            block.kind != BlockKind.HEADING and block.section_id in stub_section_ids
        )
        sec_title = sec_title_by_id.get(
            block.section_id,
            block.section_path[-1] if block.section_path else "",
        )
        severity = (
            Severity.CRITICAL
            if is_section_placeholder
            and CRITICAL_SECTION_HEADING_RE.search(sec_title)
            and not _META_REVIEW_SECTION_RE.search(sec_title)
            else Severity.MAJOR
        )
        quote = _quote_containing(block.text, match_text)
        findings.append(
            RuleFinding(
                rule="placeholder",
                block_ids=[block.id],
                criterion=Criterion.SET_COMPLETE,
                severity=severity,
                set_check=SetCheck.EMPTY_SECTION if is_section_placeholder else SetCheck.MISSING_REQUIREMENT,
                quote=quote,
                explanation=(
                    f"В разделе «{sec_label}» присутствует заглушка «{match_text}»: "
                    "требования не зафиксированы окончательно."
                ),
                recommendation=(
                    "Заменить заглушку окончательной формулировкой либо указать ответственного "
                    "и срок заполнения."
                ),
                clarifying_question=(
                    f"Кто, к какому сроку и каким содержанием должен заменить заглушку "
                    f"«{match_text}» в разделе «{sec_label}»?"
                ),
            )
        )

    return findings


def _check_lexicon_hints(doc: Document) -> list[Hint]:
    hints: list[Hint] = []

    for block in doc.blocks:
        if block.kind in (BlockKind.CODE, BlockKind.TABLE_HEADER, BlockKind.HEADING):
            continue
        if not block.text.strip():
            continue

        # Collect candidate matches across categories and resolve overlapping spans
        # in favor of the longest match (breaking ties by category priority).
        candidates: list[tuple[int, int, int, str, str, str]] = []
        for cat_idx, category in enumerate(HINT_CATEGORIES):
            for match in category.pattern.finditer(block.text):
                start, end = match.span()
                candidates.append(
                    (start, end, cat_idx, category.rule, category.label_ru, match.group(0))
                )

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        accepted: list[tuple[int, int, str, str, str]] = []
        for start, end, _cat_idx, rule, label_ru, matched in candidates:
            if any(start < a_end and end > a_start for a_start, a_end, *_ in accepted):
                continue
            accepted.append((start, end, rule, label_ru, matched))

        for _start, _end, rule, label_ru, matched in accepted:
            hints.append(
                Hint(
                    rule=rule,
                    block_id=block.id,
                    match=matched,
                    message=f"{label_ru}: «{matched}».",
                )
            )

    return hints


def _extract_req_id(block: Block) -> _ReqIdOccurrence | None:
    if block.kind in (BlockKind.CODE, BlockKind.TABLE_HEADER, BlockKind.HEADING):
        return None

    candidate: str | None = block.doc_id.strip() if block.doc_id else None
    if candidate:
        m_pref = _PREFIXED_ID_EXACT_RE.match(candidate)
        if m_pref:
            prefix, sep, num_str = m_pref.groups()
            if prefix.upper() not in _NON_REQ_ID_PREFIXES:
                num = int(num_str)
                return _ReqIdOccurrence(
                    raw_id=candidate,
                    kind="prefixed",
                    prefix=prefix,
                    sep=sep,
                    num=num,
                    width=len(num_str),
                    canonical=f"{prefix.upper()}-{num}",
                    dotted_parts=(),
                    block=block,
                )
        m_dot = _DOTTED_ID_EXACT_RE.match(candidate)
        if m_dot:
            parts = tuple(int(p) for p in candidate.split("."))
            return _ReqIdOccurrence(
                raw_id=candidate,
                kind="dotted",
                prefix="",
                sep=".",
                num=parts[-1],
                width=0,
                canonical=".".join(str(p) for p in parts),
                dotted_parts=parts,
                block=block,
            )
        return None

    # Fallback when structure parser did not populate doc_id.
    text_probe = _get_cells(block)[0] if block.kind == BlockKind.TABLE_ROW else block.text.strip()
    m = _PREFIXED_ID_START_RE.match(text_probe)
    if not m:
        return None
    prefix, sep, num_str = m.groups()
    if prefix.upper() in _NON_REQ_ID_PREFIXES:
        return None
    raw_id = f"{prefix}{sep}{num_str}"
    num = int(num_str)
    return _ReqIdOccurrence(
        raw_id=raw_id,
        kind="prefixed",
        prefix=prefix,
        sep=sep,
        num=num,
        width=len(num_str),
        canonical=f"{prefix.upper()}-{num}",
        dotted_parts=(),
        block=block,
    )


def _check_ids_and_gaps(doc: Document) -> tuple[list[RuleFinding], list[_ReqIdOccurrence]]:
    findings: list[RuleFinding] = []
    occurrences: list[_ReqIdOccurrence] = []

    for block in doc.blocks:
        occ = _extract_req_id(block)
        if occ is not None:
            occurrences.append(occ)

    # 1. Duplicate IDs.
    by_canonical: dict[str, list[_ReqIdOccurrence]] = defaultdict(list)
    for occ in occurrences:
        by_canonical[occ.canonical].append(occ)

    for occs in by_canonical.values():
        if len(occs) < 2:
            continue
        first = occs[0]
        display_id = first.raw_id
        sections_desc = ", ".join(dict.fromkeys(f"«{_section_label(o.block)}»" for o in occs))
        findings.append(
            RuleFinding(
                rule="duplicate_id",
                block_ids=[o.block.id for o in occs],
                criterion=Criterion.TRACEABLE,
                severity=Severity.MAJOR,
                set_check=SetCheck.TRACEABILITY,
                quote=_quote_containing(first.block.text, display_id),
                explanation=(
                    f"Идентификатор «{display_id}» повторяется {len(occs)} раз(а) "
                    f"(разделы: {sections_desc}), что нарушает уникальность ссылок и трассируемость."
                ),
                recommendation=(
                    "Присвоить каждому требованию уникальный идентификатор и обновить "
                    "перекрёстные ссылки."
                ),
                clarifying_question=(
                    f"Какие уникальные идентификаторы следует присвоить повторяющимся "
                    f"записям «{display_id}»?"
                ),
            )
        )

    # 2. Numbering gaps in prefixed ID series.
    prefixed_series: dict[str, list[_ReqIdOccurrence]] = defaultdict(list)
    for occ in occurrences:
        if occ.kind == "prefixed":
            prefixed_series[occ.prefix.upper()].append(occ)

    for occs in prefixed_series.values():
        nums_map: dict[int, _ReqIdOccurrence] = {}
        for o in occs:
            nums_map.setdefault(o.num, o)
        sorted_nums = sorted(nums_map)
        if len(sorted_nums) < 2:
            continue
        display_prefix = occs[0].prefix
        display_sep = occs[0].sep
        pad_width = occs[0].width if all(o.width == occs[0].width and o.width > 1 for o in occs) else 0

        def fmt_num(n: int, _p: str = display_prefix, _s: str = display_sep, _w: int = pad_width) -> str:
            return f"{_p}{_s}{n:0{_w}d}" if _w else f"{_p}{_s}{n}"

        missing: list[int] = []
        if sorted_nums[0] == 2:
            missing.append(1)
        for prev_n, next_n in pairwise(sorted_nums):
            if next_n - prev_n > 1:
                missing.extend(range(prev_n + 1, next_n))

        if not missing:
            continue

        # Anchor on the first block immediately following a gap (or first block if 1 is missing).
        anchor_num = sorted_nums[0]
        for prev_n, next_n in pairwise(sorted_nums):
            if next_n - prev_n > 1:
                anchor_num = next_n
                break
        anchor_occ = nums_map[anchor_num]

        if len(missing) <= 5:
            missing_str = ", ".join(fmt_num(n) for n in missing)
        else:
            missing_str = f"{fmt_num(missing[0])}–{fmt_num(missing[-1])} (всего {len(missing)})"

        findings.append(
            RuleFinding(
                rule="numbering_gap",
                block_ids=[anchor_occ.block.id],
                criterion=Criterion.TRACEABLE,
                severity=Severity.MINOR,
                set_check=SetCheck.TRACEABILITY,
                quote=_quote_containing(anchor_occ.block.text, anchor_occ.raw_id),
                explanation=(
                    f"В нумерации серии «{display_prefix}{display_sep}*» обнаружен пропуск: "
                    f"отсутствуют {missing_str}."
                ),
                recommendation=(
                    "Восстановить пропущенные требования либо исправить сквозную нумерацию "
                    "и перекрёстные ссылки."
                ),
                clarifying_question=(
                    f"Требования {missing_str} были удалены намеренно или пропущены "
                    "при подготовке документа?"
                ),
            )
        )

    # 3. Numbering gaps in dotted hierarchical IDs (e.g. 1.1, 1.2, 1.4).
    dotted_groups: dict[tuple[int, ...], dict[int, _ReqIdOccurrence]] = defaultdict(dict)
    for occ in occurrences:
        if occ.kind == "dotted" and len(occ.dotted_parts) >= 2:
            parent_tuple = occ.dotted_parts[:-1]
            dotted_groups[parent_tuple].setdefault(occ.dotted_parts[-1], occ)

    for parent_tuple, child_map in dotted_groups.items():
        sorted_children = sorted(child_map)
        if len(sorted_children) < 2:
            continue
        missing_children: list[int] = []
        if sorted_children[0] == 2:
            missing_children.append(1)
        for prev_n, next_n in pairwise(sorted_children):
            if next_n - prev_n > 1:
                missing_children.extend(range(prev_n + 1, next_n))
        if not missing_children:
            continue

        prefix_str = ".".join(str(p) for p in parent_tuple)
        anchor_num = sorted_children[0]
        for prev_n, next_n in pairwise(sorted_children):
            if next_n - prev_n > 1:
                anchor_num = next_n
                break
        anchor_occ = child_map[anchor_num]
        if len(missing_children) <= 5:
            missing_str = ", ".join(f"{prefix_str}.{n}" for n in missing_children)
        else:
            missing_str = (
                f"{prefix_str}.{missing_children[0]}–{prefix_str}.{missing_children[-1]} "
                f"(всего {len(missing_children)})"
            )
        findings.append(
            RuleFinding(
                rule="numbering_gap",
                block_ids=[anchor_occ.block.id],
                criterion=Criterion.TRACEABLE,
                severity=Severity.MINOR,
                set_check=SetCheck.TRACEABILITY,
                quote=_quote_containing(anchor_occ.block.text, anchor_occ.raw_id),
                explanation=(
                    f"В иерархической нумерации «{prefix_str}.*» обнаружен пропуск: "
                    f"отсутствуют {missing_str}."
                ),
                recommendation=(
                    "Проверить, не потеряны ли пункты, и привести нумерацию в порядок."
                ),
                clarifying_question=(
                    f"Пункты {missing_str} были удалены намеренно или пропущены в документе?"
                ),
            )
        )

    return findings, occurrences


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title.strip())
    cleaned = cleaned.replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", cleaned).lower()


def _check_dangling_references(
    doc: Document,
    occurrences: list[_ReqIdOccurrence],
) -> list[RuleFinding]:
    findings: list[RuleFinding] = []

    defined_canonical_ids: set[str] = {o.canonical for o in occurrences}
    defined_prefixes: set[str] = {o.prefix.upper() for o in occurrences if o.kind == "prefixed"}

    # Collect known section numbers and normalized titles.
    section_numbers: set[str] = set()
    section_titles: set[str] = set()
    for sec in doc.sections:
        raw_title = sec.title.strip()
        if raw_title:
            section_titles.add(_normalize_title(raw_title))
            section_titles.add(re.sub(r"\s+", " ", raw_title.replace("ё", "е").replace("Ё", "Е")).lower())
            m_num = _HEADING_NUM_PREFIX_RE.match(raw_title)
            if m_num:
                section_numbers.add(m_num.group(1))
    # Also allow references to defined dotted block IDs.
    for occ in occurrences:
        if occ.kind == "dotted":
            section_numbers.add(occ.canonical)

    # Build regex for known prefixed ID series in this document.
    known_prefix_re: re.Pattern[str] | None = None
    if defined_prefixes:
        alt = "|".join(re.escape(p) for p in sorted(defined_prefixes, key=len, reverse=True))
        known_prefix_re = re.compile(rf"(?<!\w)({alt})([-_])(\d{{1,5}})(?![\w-])", re.IGNORECASE)

    reported: set[tuple[str, str]] = set()

    def add_dangling(block: Block, target: str, match_span: tuple[int, int], kind_ru: str) -> None:
        key = (block.id, target.lower())
        if key in reported:
            return
        reported.add(key)
        sec_label = _section_label(block)
        quote = _quote_for_span(block.text, match_span[0], match_span[1])
        findings.append(
            RuleFinding(
                rule="dangling_reference",
                block_ids=[block.id],
                criterion=Criterion.TRACEABLE,
                severity=Severity.MAJOR,
                set_check=SetCheck.TRACEABILITY,
                quote=quote,
                explanation=(
                    f"В разделе «{sec_label}» найдена ссылка на несуществующий {kind_ru} "
                    f"«{target}»."
                ),
                recommendation=(
                    "Исправить ссылку на существующий идентификатор или раздел либо добавить "
                    "отсутствующий элемент в документ."
                ),
                clarifying_question=(
                    f"На какой существующий {kind_ru} должна указывать ссылка «{target}» "
                    f"в разделе «{sec_label}»?"
                ),
            )
        )

    for block in doc.blocks:
        if block.kind in (BlockKind.CODE, BlockKind.TABLE_HEADER, BlockKind.HEADING):
            continue
        text = block.text
        if not text.strip():
            continue

        # 1. Mentions of IDs belonging to a known defined prefix series.
        if known_prefix_re is not None:
            for m in known_prefix_re.finditer(text):
                prefix, _sep, num_str = m.groups()
                canonical = f"{prefix.upper()}-{int(num_str)}"
                if canonical not in defined_canonical_ids:
                    add_dangling(block, m.group(0), m.span(), "идентификатор")

        # 2. Explicit requirement ID references (even if prefix is not defined anywhere).
        for m in _REF_ID_EXPLICIT_RE.finditer(text):
            ref_id = next(g for g in m.groups() if g is not None)
            m_pref = _PREFIXED_ID_EXACT_RE.match(ref_id)
            if not m_pref:
                continue
            prefix, _sep, num_str = m_pref.groups()
            if prefix.upper() in _NON_REQ_ID_PREFIXES:
                continue
            canonical = f"{prefix.upper()}-{int(num_str)}"
            if canonical not in defined_canonical_ids:
                add_dangling(block, ref_id, m.span(), "идентификатор")

        # 3. Explicit numeric section references (skip external standards like ISO/IEEE/ГОСТ).
        for m in _REF_SECTION_NUM_RE.finditer(text):
            ref_num = next(g for g in m.groups() if g is not None)
            lookbehind = text[max(0, m.start() - 45) : m.start()]
            if _STANDARD_CONTEXT_RE.search(lookbehind):
                continue
            if ref_num not in section_numbers:
                add_dangling(block, ref_num, m.span(), "раздел")

        # 4. Explicit quoted section title references.
        for m in _REF_SECTION_TITLE_RE.finditer(text):
            raw_title = m.group(1).strip()
            norm_title = _normalize_title(raw_title)
            if norm_title not in section_titles:
                add_dangling(block, raw_title, m.span(), "раздел")

    return findings


def _check_missing_req_ids(
    doc: Document,
    occurrences: list[_ReqIdOccurrence],
    skip_block_ids: set[str],
) -> list[RuleFinding]:
    """Flag documents that use explicit IDs inconsistently, leaving most normative statements unnumbered."""
    blocks_with_id = {occ.block.id for occ in occurrences}
    if not blocks_with_id:
        return []

    normative_without_id: list[Block] = []
    for block in doc.blocks:
        if block.id in blocks_with_id or block.id in skip_block_ids:
            continue
        if block.kind not in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM, BlockKind.TABLE_ROW):
            continue
        text = block.text.strip()
        if not text or _HORIZONTAL_RULE_RE.match(text):
            continue
        if PLACEHOLDER_VALUE_RE.match(text) or PLACEHOLDER_INLINE_RE.search(text):
            continue

        sec_heading = block.section_path[-1] if block.section_path else ""
        in_req_section = bool(
            REQUIREMENT_SECTION_HEADING_RE.search(sec_heading)
            and not _META_REVIEW_SECTION_RE.search(sec_heading)
        )
        has_normative_verb = bool(NORMATIVE_STATEMENT_RE.search(text))

        if has_normative_verb or in_req_section:
            normative_without_id.append(block)

    if len(normative_without_id) <= len(blocks_with_id):
        return []

    total = len(blocks_with_id) + len(normative_without_id)
    first = normative_without_id[0]
    return [
        RuleFinding(
            rule="missing_req_ids",
            block_ids=[b.id for b in normative_without_id],
            criterion=Criterion.TRACEABLE,
            severity=Severity.MINOR,
            set_check=SetCheck.TRACEABILITY,
            quote=_quote_for_span(first.text),
            explanation=(
                f"Большинство нормативных формулировок не имеет идентификаторов: "
                f"{len(normative_without_id)} из {total} без ID при наличии "
                f"{len(blocks_with_id)} пронумерованных требований."
            ),
            recommendation=(
                "Присвоить уникальные идентификаторы всем требованиям документа для "
                "сквозной трассируемости с критериями приёмки и тестами."
            ),
            clarifying_question=(
                "Планируется ли ввести единую нумерацию идентификаторов для всех "
                "требований документа?"
            ),
        )
    ]


def _check_no_ranking(doc: Document) -> list[RuleFinding]:
    """Flag documents that contain no priority or importance markers anywhere."""
    anchor_block: Block | None = None
    for sec in doc.sections:
        if PRIORITY_MARKER_RE.search(sec.title) or PRIORITY_CODE_RE.search(sec.title):
            return []

    for block in doc.blocks:
        if block.kind == BlockKind.CODE:
            continue
        text = block.text.strip()
        if not text:
            continue
        if anchor_block is None:
            anchor_block = block
        if PRIORITY_MARKER_RE.search(text) or PRIORITY_CODE_RE.search(text):
            return []

    if anchor_block is None:
        return []

    return [
        RuleFinding(
            rule="no_ranking",
            block_ids=[anchor_block.id],
            criterion=Criterion.RANKED,
            severity=Severity.MINOR,
            set_check=SetCheck.MISSING_REQUIREMENT,
            quote=_quote_for_span(anchor_block.text),
            explanation=(
                "В документе не указаны приоритеты или уровни важности требований "
                "(например, Must/Should/Could, P1–P3, высокий/средний/низкий)."
            ),
            recommendation=(
                "Добавить схему приоритизации и указать приоритет для каждого требования."
            ),
            clarifying_question=(
                "Какие требования являются обязательными для первого релиза, а какие "
                "имеют более низкий приоритет?"
            ),
        )
    ]


def _extract_param_defs(tables: list[_TableGroup]) -> dict[str, tuple[str, str, str]]:
    """Map parameter-definition table row block_ids to ``(field_name, declared_type, context_text)``."""
    param_defs: dict[str, tuple[str, str, str]] = {}

    for table in tables:
        name_col: int | None = None
        type_col: int | None = None
        if table.header is not None:
            header_cells = _get_cells(table.header)
            for idx, hdr in enumerate(header_cells):
                cleaned_hdr = hdr.strip()
                if name_col is None and _PARAM_NAME_HEADER_RE.match(cleaned_hdr):
                    name_col = idx
                elif type_col is None and _PARAM_TYPE_HEADER_RE.match(cleaned_hdr):
                    type_col = idx

        for row in table.rows:
            cells = _get_cells(row)
            if len(cells) < 2:
                continue

            field_name: str | None = None
            declared_type: str | None = None

            if name_col is not None and type_col is not None:
                if name_col < len(cells) and type_col < len(cells):
                    cand_name = cells[name_col].strip().strip("`*")
                    cand_type = cells[type_col].strip().strip("`*")
                    if _PARAM_FIELD_NAME_RE.match(cand_name) and DECLARED_TYPE_RE.match(cand_type):
                        field_name = cand_name
                        declared_type = cand_type
            else:
                # Headerless fallback: col 0 looks like a field name and col 1 (or col 2) is a declared type.
                cand_name = cells[0].strip().strip("`*")
                if _PARAM_FIELD_NAME_RE.match(cand_name):
                    for type_idx in (1, 2):
                        if type_idx < len(cells):
                            cand_type = cells[type_idx].strip().strip("`*")
                            if DECLARED_TYPE_RE.match(cand_type):
                                field_name = cand_name
                                declared_type = cand_type
                                break

            if field_name is not None and declared_type is not None:
                sec_tail = row.section_path[-1] if row.section_path else ""
                context = f"{field_name}: {declared_type} ({sec_tail})" if sec_tail else f"{field_name}: {declared_type}"
                param_defs[row.id] = (field_name, declared_type, context)

    return param_defs


def _collect_inventory(doc: Document, tables: list[_TableGroup]) -> list[InventoryItem]:
    items: list[InventoryItem] = []
    param_defs = _extract_param_defs(tables)

    for block in doc.blocks:
        if block.kind == BlockKind.TABLE_HEADER:
            continue
        text = block.text
        if not text.strip():
            continue

        seen: set[tuple[str, str, str]] = set()
        typed_ident_names: set[str] = set()

        def add_item(
            kind: str,
            value: str,
            unit: str,
            snippet: str,
            _bid: str = block.id,
            _seen: set[tuple[str, str, str]] = seen,
        ) -> None:
            key = (kind, value, unit)
            if key in _seen:
                return
            _seen.add(key)
            items.append(
                InventoryItem(
                    block_id=_bid,
                    kind=kind,  # type: ignore[arg-type]
                    value=value,
                    unit=unit,
                    text=snippet[:120],
                )
            )

        # 0. Parameter definitions from table rows (field name + declared type).
        param_def = param_defs.get(block.id)
        if param_def is not None:
            field_name, declared_type, context = param_def
            add_item("identifier", field_name, declared_type, context)
            typed_ident_names.add(field_name)

        # 1. Quantities with units (skip headings and code blocks to avoid noise).
        if block.kind not in (BlockKind.CODE, BlockKind.HEADING):
            for m in _QUANTITY_SUFFIX_RE.finditer(text):
                value = re.sub(r"\s+", " ", m.group(1)).strip()
                unit = m.group(2).strip().rstrip(".")
                add_item("quantity", value, unit, m.group(0).strip())

            for m in _QUANTITY_PREFIX_CURRENCY_RE.finditer(text):
                unit = m.group(1).strip()
                value = re.sub(r"\s+", " ", m.group(2)).strip()
                add_item("quantity", value, unit, m.group(0).strip())

            for m in _WORD_DURATION_RE.finditer(text):
                unit = (m.group(2) or m.group(3) or "").strip()
                add_item("quantity", "1", unit, m.group(0).strip())

        # 2. HTTP methods (uppercase only) and HTTP status codes.
        for m in _HTTP_METHOD_RE.finditer(text):
            method = m.group(1)
            add_item("code", method, "HTTP method", _quote_for_span(text, m.start(), m.end()))

        sec_context = " ".join(block.section_path)
        has_http_context = bool(_HTTP_CONTEXT_RE.search(text) or _HTTP_CONTEXT_RE.search(sec_context))
        for m in _HTTP_CODE_RE.finditer(text):
            code = m.group(1)
            if code not in _HTTP_STATUS_CODES:
                continue
            # Ignore numbers immediately followed by a measurement/currency unit (e.g. "500 ₸").
            trailing = text[m.end() :]
            if _FOLLOWED_BY_UNIT_RE.match(trailing):
                continue
            # Ignore numbers preceded by thousands-separator digits (e.g. "1 500").
            prefix_slice = text[max(0, m.start() - 3) : m.start()]
            if re.search(r"\d[\s\u00a0]+$", prefix_slice):
                continue
            at_start = m.start() <= 3
            if has_http_context or at_start:
                add_item("code", code, "HTTP status", _quote_for_span(text, m.start(), m.end()))

        # 3. Latin identifiers (camelCase, PascalCase, snake_case, API paths, HTTP headers,
        # backticked tokens, ALL_CAPS constants).
        if block.kind != BlockKind.HEADING:
            for m in _IDENT_ENDPOINT_RE.finditer(text):
                add_item("identifier", m.group(1), "", m.group(1))
            for m in _IDENT_CAMEL_RE.finditer(text):
                if m.group(0) not in typed_ident_names:
                    add_item("identifier", m.group(0), "", m.group(0))
            for m in _IDENT_SNAKE_RE.finditer(text):
                if m.group(0) not in typed_ident_names:
                    add_item("identifier", m.group(0), "", m.group(0))
            for m in _IDENT_HEADER_RE.finditer(text):
                if m.group(1) not in typed_ident_names:
                    add_item("identifier", m.group(1), "", m.group(1))
            for m in _IDENT_BACKTICK_RE.finditer(block.raw):
                token = m.group(1)
                if (
                    token not in typed_ident_names
                    and token not in _HTTP_METHODS
                    and token.upper() not in _PLACEHOLDER_ACRONYMS
                ):
                    add_item("identifier", token, "", token)
            for m in _IDENT_ALLCAPS_RE.finditer(text):
                token = m.group(1)
                if token in typed_ident_names or token in _HTTP_METHODS or token in _PLACEHOLDER_ACRONYMS:
                    continue
                add_item("identifier", token, "", token)

    return items


def extract_image_warnings(doc: Document) -> list[str]:
    """Return Russian passport warning strings when the document contains image links."""
    image_blocks: list[str] = []
    for block in doc.blocks:
        if _IMAGE_LINK_RE.search(block.raw) or _IMAGE_LINK_RE.search(block.text):
            image_blocks.append(block.id)
    if not image_blocks:
        return []
    blocks_preview = ", ".join(image_blocks[:5])
    if len(image_blocks) > 5:
        blocks_preview += f" и ещё {len(image_blocks) - 5}"
    return [
        (
            f"В документе обнаружены ссылки на изображения (блоки: {blocks_preview}): "
            "содержимое изображений и диаграмм не анализировалось."
        )
    ]


def _check_image_link_hints(doc: Document) -> list[Hint]:
    hints: list[Hint] = []
    for block in doc.blocks:
        source = block.raw if _IMAGE_LINK_RE.search(block.raw) else block.text
        for m in _IMAGE_LINK_RE.finditer(source):
            hints.append(
                Hint(
                    rule="image_link",
                    block_id=block.id,
                    match=m.group(0),
                    message="Обнаружена ссылка на изображение: содержимое изображений не анализируется.",
                )
            )
    return hints


def run_prechecks(doc: Document) -> PrecheckResult:
    """Run all deterministic prechecks on a structured document without mutating it."""
    tables = _group_tables(doc.blocks)

    template_findings, template_block_ids = _check_unfilled_template_tables(tables)
    empty_sec_findings, empty_sec_block_ids = _check_empty_sections(doc)
    skip_blocks = template_block_ids | empty_sec_block_ids
    placeholder_findings = _check_placeholders_and_empty_cells(
        doc,
        tables,
        skip_block_ids=skip_blocks,
    )
    id_findings, id_occurrences = _check_ids_and_gaps(doc)
    dangling_findings = _check_dangling_references(doc, id_occurrences)
    missing_id_findings = _check_missing_req_ids(doc, id_occurrences, skip_blocks)
    ranking_findings = _check_no_ranking(doc)

    hints = _check_lexicon_hints(doc)
    hints.extend(_check_image_link_hints(doc))
    inventory = _collect_inventory(doc, tables)

    findings = [
        *template_findings,
        *empty_sec_findings,
        *placeholder_findings,
        *id_findings,
        *dangling_findings,
        *missing_id_findings,
        *ranking_findings,
    ]
    return PrecheckResult(hints=hints, findings=findings, inventory=inventory)
