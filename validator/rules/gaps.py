"""Domain-neutral deterministic gap rules: units mismatch, idempotency, toggle flags, and one-line diagrams."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from validator.models import (
    Block,
    BlockKind,
    Criterion,
    Document,
    InventoryItem,
    RuleFinding,
    SetCheck,
    Severity,
)

_MAX_QUOTE_LEN = 200
_HAS_ADDRESSEE = "addressee" in RuleFinding.model_fields
_HAS_ANSWER_OPTIONS = "answer_options" in RuleFinding.model_fields


def _make_finding(
    *,
    rule: str,
    block_ids: list[str],
    criterion: Criterion,
    severity: Severity,
    set_check: SetCheck | None,
    quote: str,
    explanation: str,
    recommendation: str,
    clarifying_question: str,
    answer_options: list[str],
) -> RuleFinding:
    kwargs: dict[str, Any] = {
        "rule": rule,
        "block_ids": block_ids,
        "criterion": criterion,
        "severity": severity,
        "set_check": set_check,
        "quote": quote,
        "explanation": explanation,
        "recommendation": recommendation,
        "clarifying_question": clarifying_question,
    }
    if _HAS_ADDRESSEE:
        kwargs["addressee"] = "analyst"
    if _HAS_ANSWER_OPTIONS:
        kwargs["answer_options"] = answer_options
    return RuleFinding(**kwargs)


def _quote_for_span(text: str, start: int = 0, end: int | None = None, *, limit: int = _MAX_QUOTE_LEN) -> str:
    """Return an exact substring of ``text`` covering ``start:end`` within ``limit``."""
    if len(text) <= limit:
        return text
    if end is None:
        end = min(len(text), limit)
    match_len = max(0, end - start)
    if match_len >= limit:
        return text[start : start + limit]
    pad = (limit - match_len) // 2
    lo = max(0, start - pad)
    hi = min(len(text), lo + limit)
    lo = max(0, hi - limit)
    if lo > 0:
        space = text.find(" ", lo, start)
        if space != -1 and space + 1 < start:
            lo = space + 1
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        if space != -1 and space > end:
            hi = space
    return text[lo:hi]


def _quote_containing(text: str, needle: str, *, limit: int = _MAX_QUOTE_LEN) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    idx = text.lower().find(needle.lower())
    if idx != -1:
        return _quote_for_span(text, idx, idx + len(needle), limit=limit)
    return _quote_for_span(text, 0, min(len(text), limit), limit=limit)


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
    return [p.strip() for p in parts]


def _section_label(block: Block) -> str:
    return " › ".join(block.section_path) if block.section_path else (block.section_id or "документ")


@dataclass(frozen=True)
class _TableGroup:
    section_id: str
    header: Block | None
    rows: list[Block]


def _group_tables(blocks: list[Block]) -> list[_TableGroup]:
    tables: list[_TableGroup] = []
    cur_sec: str | None = None
    cur_header: Block | None = None
    cur_rows: list[Block] = []
    last_line_end = 0

    def flush() -> None:
        nonlocal cur_sec, cur_header, cur_rows
        if cur_sec is not None and (cur_header is not None or cur_rows):
            tables.append(_TableGroup(section_id=cur_sec, header=cur_header, rows=cur_rows))
        cur_sec = None
        cur_header = None
        cur_rows = []

    for blk in blocks:
        if blk.kind == BlockKind.TABLE_HEADER:
            flush()
            cur_sec = blk.section_id
            cur_header = blk
            cur_rows = []
            last_line_end = blk.line_end
        elif blk.kind == BlockKind.TABLE_ROW:
            line_gap = blk.line_start > 0 and last_line_end > 0 and blk.line_start - last_line_end > 2
            if cur_sec != blk.section_id or line_gap:
                flush()
                cur_sec = blk.section_id
            cur_rows.append(blk)
            last_line_end = blk.line_end
        else:
            flush()
    flush()
    return tables


# ---------------------------------------------------------------- rule 1: units_mismatch

# Prose unit declarations without a leading digit (e.g. "в тиынах", "в тенге", "in cents", "in KB").
_PROSE_UNIT_RE = re.compile(
    r"(?<!\w)(?:в|in)\s+("
    r"тиын(?:ах|ам|ами|а|ов|у|е)?"
    r"|тенге|теңге"
    r"|копейк(?:ах|ам|ами|и|е|у)?|копеек"
    r"|рубл(?:ях|ям|ями|ей|я|ю|е)?"
    r"|цент(?:ах|ам|ами|ов|а|у|е)?|cents?"
    r"|доллар(?:ах|ам|ами|ов|а|у|е)?|dollars?"
    r"|евроцент(?:ах|ам|ами|ов|а)?|eurocents?"
    r"|евро|euros?"
    r"|миллисекунд(?:ах|ам|ами|ы|е|у)?|milliseconds?|ms"
    r"|секунд(?:ах|ам|ами|ы|е|у)?|seconds?|sec"
    r"|минут(?:ах|ам|ами|ы|е|у)?|minutes?|min"
    r"|час(?:ах|ам|ами|ов|а|у|е)?|hours?"
    r"|байт(?:ах|ам|ами|ов|а|у|е)?|bytes?"
    r"|килобайт(?:ах|ам|ами|ов|а)?|kilobytes?|КБ|KB|KiB"
    r"|мегабайт(?:ах|ам|ами|ов|а)?|megabytes?|МБ|MB|MiB"
    r"|гигабайт(?:ах|ам|ами|ов|а)?|gigabytes?|ГБ|GB|GiB"
    r")(?!\w)",
    re.IGNORECASE,
)

# Precision / granularity phrases ("с точностью до 1 копейки") describe decimal precision, not unit storage.
_PRECISION_PREFIX_RE = re.compile(
    r"(?:с\s+точностью\s+до(?:\s+\d+)?|до\s+точности|to\s+the\s+nearest|precision\s+of(?:\s+\d+)?)\s*$",
    re.IGNORECASE,
)

# Parenthetical equivalence examples ("например, 50000 соответствует 500,00 руб.") are illustrations, not conflicts.
_EXAMPLE_EQUIV_RE = re.compile(
    r"\((?:например|e\.g\.|i\.e\.)[^)]*(?:соответств\w*|равн\w*|=|equals?)[^)]*\)",
    re.IGNORECASE,
)

# Explicit conversion or rounding statements that resolve unit differences.
_CURRENCY_CONVERSION_RE = re.compile(
    r"(?:"
    r"1\s*(?:тенге|теңге|руб\w*|доллар\w*|евро|kzt|rub|usd|eur|₸|₽|\$|€)\s*=\s*100"
    r"|100\s*(?:тиын\w*|коп\w*|цент\w*|cents?)"
    r"|(?:умнож\w*|дел\w*|multiply|divide|dividing)\s+(?:на\s+|by\s+)?100"
    r"|(?:×|\*|/)\s*100\b"
    r"|округл\w*|round(?:ing|ed)?\s+(?:up|down|half|to)"
    r"|правил\w*\s+(?:пересч\w*|конверт\w*|перевод\w*|округл\w*)"
    r"|(?:conversion|rounding)\s+rule"
    r"|пересчитыва\w*\s+(?:в|из)\s+(?:тиын|тенге|коп|руб|цент|доллар)"
    r")",
    re.IGNORECASE,
)

_TIME_SIZE_CONVERSION_RE = re.compile(
    r"(?:"
    r"1024\s*(?:байт|byte|кб|kb|мб|mb|гб|gb)"
    r"|1000\s*(?:мс|ms|байт|byte|кб|kb|мб|mb)"
    r"|60\s*(?:сек\w*|sec\w*|мин\w*|min\w*)"
    r"|округл\w*|конверт\w*|пересч\w*"
    r"|(?:conversion|rounding)\s+rule"
    r")",
    re.IGNORECASE,
)

# Adjacent convertible time pairs (ignore unrelated scales like ms vs days/months).
_ADJACENT_TIME_PAIRS = frozenset(
    {
        frozenset({"ms", "s"}),
        frozenset({"s", "min"}),
        frozenset({"min", "h"}),
        frozenset({"h", "d"}),
    }
)

_UNIT_FAMILY_LABELS = {
    "currency_kzt": "денежных сумм (тенге / тиыны)",
    "currency_rub": "денежных сумм (рубли / копейки)",
    "currency_usd": "денежных сумм (доллары / центы)",
    "currency_eur": "денежных сумм (евро / евроценты)",
    "time": "интервалов времени",
    "size": "объёмов данных",
}

# Bilingual canonical quantity nouns partitioned by unit family (currency / time / size).
_CANONICAL_NOUNS_BY_GROUP: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "currency": (
        ("payment", re.compile(r"плат[её]ж\w*|оплат\w*|оплач\w*|списан\w*|payment\w*", re.IGNORECASE)),
        ("transfer", re.compile(r"перевод\w*|transfer\w*", re.IGNORECASE)),
        ("purchase", re.compile(r"покупк\w*|purchase\w*", re.IGNORECASE)),
        ("subscription", re.compile(r"подписк\w*|subscription\w*", re.IGNORECASE)),
        ("balance", re.compile(r"баланс\w*|остат\w*|balance\w*", re.IGNORECASE)),
        ("limit", re.compile(r"лимит\w*|limit\w*", re.IGNORECASE)),
        ("fee", re.compile(r"комисс\w*|fee\b|commission\w*", re.IGNORECASE)),
    ),
    "time": (
        ("timeout", re.compile(r"таймаут\w*|timeout\w*", re.IGNORECASE)),
        ("latency", re.compile(r"задержк\w*|delay\w*|latency\b", re.IGNORECASE)),
        ("session", re.compile(r"сесси\w*|токен\w*|session\w*|token\w*", re.IGNORECASE)),
        ("cache", re.compile(r"кэш\w*|кеш\w*|cache\w*|ttl\b", re.IGNORECASE)),
        ("retention", re.compile(r"архив\w*|retention\b", re.IGNORECASE)),
    ),
    "size": (
        ("size", re.compile(r"размер\w*|объ[её]м\w*|size\b", re.IGNORECASE)),
        ("file", re.compile(r"файл\w*|вложен\w*|file\w*|attachment\w*", re.IGNORECASE)),
        ("payload", re.compile(r"пакет\w*|тело\s+запроса|payload\w*", re.IGNORECASE)),
        ("quota", re.compile(r"квот\w*|quota\w*", re.IGNORECASE)),
        ("memory", re.compile(r"памят\w*|диск\w*|memory\b|storage\b|disk\b", re.IGNORECASE)),
    ),
}

_CANONICAL_NOUN_LABELS_RU: dict[str, str] = {
    "timeout": "таймаута",
    "payment": "суммы платежа",
    "transfer": "суммы перевода",
    "purchase": "суммы покупки",
    "subscription": "стоимости подписки",
    "balance": "баланса",
    "limit": "лимита",
    "fee": "комиссии",
    "size": "размера",
    "file": "размера файла",
    "payload": "размера сообщения",
    "latency": "задержки",
    "session": "времени жизни сессии",
    "cache": "времени кэширования",
    "retention": "срока хранения",
    "quota": "квоты",
    "memory": "объёма памяти",
}

_FIELD_PAREN_UNIT_RE = re.compile(
    r"(?<!\w)([A-Za-z_][A-Za-z0-9_]{1,40})\s*\(([^)]{0,100})\)"
)
_FIELD_LIST_PAREN_RE = re.compile(
    r"\(([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)+)\)"
)
_TYPE_NAMES_LOW = frozenset(
    {
        "string",
        "integer",
        "int",
        "number",
        "decimal",
        "float",
        "double",
        "boolean",
        "bool",
        "object",
        "array",
        "uuid",
        "enum",
        "timestamp",
        "date",
        "datetime",
        "json",
        "http",
        "https",
        "utc",
        "iso",
        "rest",
        "api",
        "ok",
        "error",
        "get",
        "post",
        "put",
        "patch",
        "delete",
    }
)


def _classify_unit(raw_unit: str) -> tuple[str, str] | None:
    """Return ``(family, canonical_subunit)`` for currency, time, and byte-size units."""
    u = raw_unit.strip().lower().rstrip(".")
    if not u:
        return None

    # Currency major / minor pairs.
    if u in {"₸", "kzt", "тенге", "теңге"}:
        return ("currency_kzt", "major")
    if u.startswith("тиын"):
        return ("currency_kzt", "minor")
    if u in {"₽", "rub", "руб"} or u.startswith("рубл"):
        return ("currency_rub", "major")
    if u == "коп" or u.startswith("копе"):
        return ("currency_rub", "minor")
    if u in {"$", "usd"} or u.startswith(("доллар", "dollar")):
        return ("currency_usd", "major")
    if u == "¢" or u.startswith(("цент", "cent")):
        return ("currency_usd", "minor")
    if u in {"€", "eur", "евро"} or u.startswith("euro"):
        if u.startswith("eurocent"):
            return ("currency_eur", "minor")
        return ("currency_eur", "major")
    if u.startswith("евроцент"):
        return ("currency_eur", "minor")

    # Time units.
    if u in {"мс", "мсек", "ms", "msec"} or u.startswith(("миллисекунд", "millisecond")):
        return ("time", "ms")
    if u in {"с", "сек", "sec"} or u.startswith(("секунд", "second")):
        return ("time", "s")
    if u in {"мин", "min", "mins"} or u.startswith(("минут", "minute")):
        return ("time", "min")
    if u in {"ч", "hr", "hrs"} or u.startswith(("час", "hour")):
        return ("time", "h")
    if u in {"день"} or u.startswith(("дн", "сут", "day")):
        return ("time", "d")

    # Byte-size units (excludes bits / characters).
    if u in {"б", "b"} or u.startswith(("байт", "byte")):
        return ("size", "B")
    if u in {"кб", "kb", "kib"} or u.startswith(("килобайт", "kilobyte")):
        return ("size", "KB")
    if u in {"мб", "mb", "mib"} or u.startswith(("мегабайт", "megabyte")):
        return ("size", "MB")
    if u in {"гб", "gb", "gib"} or u.startswith(("гигабайт", "gigabyte")):
        return ("size", "GB")
    if u in {"тб", "tb", "tib"}:
        return ("size", "TB")

    return None


@dataclass(frozen=True)
class _UnitHit:
    family: str
    subunit: str
    raw_unit: str
    block: Block
    span_start: int
    span_end: int
    fields: frozenset[str]
    nouns: frozenset[str]
    table_col_key: tuple[str, int] | None


def _is_inside_example_equiv(text: str, pos: int) -> bool:
    return any(m.start() <= pos < m.end() for m in _EXAMPLE_EQUIV_RE.finditer(text))


def _extract_clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Return sentence/clause slice boundaries around ``start:end``."""
    left = max(
        text.rfind(".", 0, start),
        text.rfind(";", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind("\n", 0, start),
    )
    lo = left + 1 if left != -1 else 0
    right_candidates = [
        idx
        for sep in (".", ";", "!", "?", "\n")
        if (idx := text.find(sep, end)) != -1
    ]
    hi = min(right_candidates) if right_candidates else len(text)
    return lo, hi


def _extract_local_anchors(
    block: Block,
    family: str,
    span_start: int,
    span_end: int,
    table_col_key: tuple[str, int] | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Extract field identifiers and family-specific quantity nouns associated with a unit hit."""
    text = block.text
    clause_lo, clause_hi = _extract_clause_bounds(text, span_start, span_end)
    clause = text[clause_lo:clause_hi]
    rel_start = span_start - clause_lo
    rel_end = span_end - clause_lo

    fields: set[str] = set()

    # 1. Exact enclosing field(type, unit) pattern such as "amount (number, ... в рублях)".
    enclosing_field: str | None = None
    for m in _FIELD_PAREN_UNIT_RE.finditer(clause):
        if m.start(2) <= rel_start and rel_end <= m.end(2):
            cand = m.group(1)
            if cand.lower() not in _TYPE_NAMES_LOW:
                enclosing_field = cand
                fields.add(cand)
                break

    if enclosing_field is None:
        # 2. Parenthesized list of field names earlier in the same clause (e.g. "(amount, fee) ... в копейках").
        prefix_clause = clause[:rel_start]
        for m in _FIELD_LIST_PAREN_RE.finditer(prefix_clause):
            for tok in re.split(r"\s*,\s*", m.group(1)):
                if tok and tok.lower() not in _TYPE_NAMES_LOW:
                    fields.add(tok)
        # 3. Single snake_case or camelCase identifier immediately in the same clause.
        for m in re.finditer(
            r"(?<!\w)(?:[a-z][a-z0-9]*_[a-z0-9_]+|[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*)(?!\w)",
            clause,
        ):
            tok = m.group(0)
            if tok.lower() not in _TYPE_NAMES_LOW:
                fields.add(tok)

    # 4. Table row primary field identifier (column 0).
    if block.kind == BlockKind.TABLE_ROW:
        cells = _get_cells(block)
        if cells:
            col0 = cells[0].strip().strip("`*")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{1,40}", col0) and col0.lower() not in _TYPE_NAMES_LOW:
                fields.add(col0)

    # 5. Family-matched canonical quantity nouns in the local context of this hit (and inside its field names).
    noun_group = "currency" if family.startswith("currency_") else family
    patterns = _CANONICAL_NOUNS_BY_GROUP.get(noun_group, ())
    win_lo = max(clause_lo, span_start - 150)
    win_hi = min(clause_hi, span_end + 25)
    local_window = text[win_lo:win_hi]

    nouns: set[str] = set()
    for canon_name, pat in patterns:
        if pat.search(local_window):
            nouns.add(canon_name)
    for fld in fields:
        for canon_name, pat in patterns:
            if pat.search(fld):
                nouns.add(canon_name)

    _ = table_col_key
    return frozenset(fields), frozenset(nouns)


def _find_table_col_for_pos(block: Block, pos: int, table_id: str) -> tuple[str, int] | None:
    if block.kind != BlockKind.TABLE_ROW:
        return None
    col_idx = block.text[:pos].count("|")
    return (table_id, col_idx)


def _check_units_mismatch(doc: Document, inventory: list[InventoryItem]) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    block_by_id = {b.id: b for b in doc.blocks}
    full_prose = "\n".join(b.text for b in doc.blocks if b.kind != BlockKind.CODE)

    tables = _group_tables(doc.blocks)
    table_id_by_row: dict[str, str] = {}
    for t_idx, table in enumerate(tables):
        t_key = f"T{t_idx}"
        for row in table.rows:
            table_id_by_row[row.id] = t_key

    hits_by_family: dict[str, list[_UnitHit]] = defaultdict(list)
    seen_hit: set[tuple[str, str, str, int]] = set()

    def record(family: str, subunit: str, raw_unit: str, blk: Block, start: int, end: int) -> None:
        # Ignore precision/granularity phrases ("с точностью до 1 копейки") and parenthetical examples.
        if _PRECISION_PREFIX_RE.search(blk.text[max(0, start - 40) : start]):
            return
        if _is_inside_example_equiv(blk.text, start):
            return
        key = (blk.id, family, subunit, start)
        if key in seen_hit:
            return
        seen_hit.add(key)
        t_id = table_id_by_row.get(blk.id)
        t_col = _find_table_col_for_pos(blk, start, t_id) if t_id is not None else None
        fields, nouns = _extract_local_anchors(blk, family, start, end, t_col)
        hits_by_family[family].append(
            _UnitHit(family, subunit, raw_unit, blk, start, end, fields, nouns, t_col)
        )

    # 1. Collect from existing quantity inventory, locating exact span in block text.
    for item in inventory:
        if item.kind != "quantity" or not item.unit:
            continue
        classified = _classify_unit(item.unit)
        if classified is None:
            continue
        blk = block_by_id.get(item.block_id)
        if blk is None:
            continue
        needle = item.text.strip() or item.unit
        idx = blk.text.find(needle)
        if idx == -1:
            idx = blk.text.lower().find(item.unit.lower())
        if idx == -1:
            continue
        record(classified[0], classified[1], item.unit, blk, idx, idx + len(needle))

    # 2. Collect explicit prose unit declarations ("в тиынах", "в тенге", "в копейках", "in cents").
    for blk in doc.blocks:
        if blk.kind in (BlockKind.CODE, BlockKind.TABLE_HEADER, BlockKind.HEADING):
            continue
        for m in _PROSE_UNIT_RE.finditer(blk.text):
            raw_unit = m.group(1).strip()
            classified = _classify_unit(raw_unit)
            if classified is not None:
                record(classified[0], classified[1], raw_unit, blk, m.start(), m.end())

    has_currency_conv = bool(_CURRENCY_CONVERSION_RE.search(full_prose))
    has_time_size_conv = bool(_TIME_SIZE_CONVERSION_RE.search(full_prose))

    for family, hits in hits_by_family.items():
        subunits = {h.subunit for h in hits}
        if len(subunits) < 2:
            continue
        if family.startswith("currency_") and has_currency_conv:
            continue
        if family in {"time", "size"} and has_time_size_conv:
            continue

        # Score candidate conflicting pairs that refer to the SAME quantity:
        # Priority 3: shared explicit field name (e.g. amount in request vs response),
        # Priority 2: same table column,
        # Priority 1: shared canonical quantity noun (e.g. timeout, purchase, file, size).
        best_pair: tuple[int, int, _UnitHit, _UnitHit, str, str | None] | None = None
        for idx, h1 in enumerate(hits):
            for h2 in hits[idx + 1 :]:
                if h1.subunit == h2.subunit:
                    continue
                if family == "time" and frozenset({h1.subunit, h2.subunit}) not in _ADJACENT_TIME_PAIRS:
                    continue

                shared_fields = h1.fields & h2.fields
                same_table_col = h1.table_col_key is not None and h1.table_col_key == h2.table_col_key
                shared_nouns = h1.nouns & h2.nouns

                matched_field: str | None = None
                if shared_fields:
                    priority = 3
                    matched_field = min(shared_fields)
                    shared_desc = f"поля «{matched_field}»"
                elif same_table_col:
                    priority = 2
                    shared_desc = "одной колонки таблицы"
                elif shared_nouns:
                    priority = 1
                    noun_key = min(shared_nouns)
                    shared_desc = _CANONICAL_NOUN_LABELS_RU.get(noun_key, f"величины «{noun_key}»")
                else:
                    continue

                same_part_bonus = 1 if h1.block.part == h2.block.part else 0
                cand = (priority, same_part_bonus, h1, h2, shared_desc, matched_field)
                if best_pair is None or (cand[0], cand[1]) > (best_pair[0], best_pair[1]):
                    best_pair = cand

        if best_pair is None:
            continue

        _prio, _sp, hit_a, hit_b, shared_desc, matched_field = best_pair
        family_ru = _UNIT_FAMILY_LABELS.get(family, "величин")

        def _hit_quote(hit: _UnitHit, field: str | None, *, limit: int = _MAX_QUOTE_LEN) -> str:
            lo, hi = hit.span_start, hit.span_end
            if field is not None:
                f_idx = hit.block.text.rfind(field, max(0, lo - 180), lo)
                if f_idx != -1 and hi - f_idx <= limit:
                    lo = f_idx
            return _quote_for_span(hit.block.text, lo, hi, limit=limit)

        if hit_a.block.id == hit_b.block.id:
            lo = min(hit_a.span_start, hit_b.span_start)
            hi = max(hit_a.span_end, hit_b.span_end)
            primary_quote = _quote_for_span(hit_a.block.text, lo, hi)
            q_a = _hit_quote(hit_a, matched_field, limit=95)
            q_b = _hit_quote(hit_b, matched_field, limit=95)
            block_ids = [hit_a.block.id]
            sec_a = _section_label(hit_a.block)
            places_desc = (
                f"в разделе «{sec_a}» одновременно указаны «{hit_a.raw_unit}» («{q_a}») "
                f"и «{hit_b.raw_unit}» («{q_b}»)"
            )
        else:
            primary_quote = _hit_quote(hit_a, matched_field)
            secondary_quote = _hit_quote(hit_b, matched_field, limit=150)
            block_ids = [hit_a.block.id, hit_b.block.id]
            sec_a = _section_label(hit_a.block)
            sec_b = _section_label(hit_b.block)
            places_desc = (
                f"в разделе «{sec_a}» указана единица «{hit_a.raw_unit}» («{primary_quote}»), "
                f"а в разделе «{sec_b}» — «{hit_b.raw_unit}» («{secondary_quote}»)"
            )

        findings.append(
            _make_finding(
                rule="units_mismatch",
                block_ids=block_ids,
                criterion=Criterion.SET_CONSISTENT,
                severity=Severity.MAJOR,
                set_check=SetCheck.CONTRADICTION,
                quote=primary_quote,
                explanation=(
                    f"Для {shared_desc} ({family_ru}) используются разные единицы "
                    f"измерения: {places_desc}, при этом не указано правило пересчёта или округления."
                ),
                recommendation=(
                    "Зафиксировать единую единицу измерения во всех разделах спецификации "
                    "либо явно описать коэффициент пересчёта и правило округления."
                ),
                clarifying_question=(
                    f"Следует ли привести единицы {shared_desc} («{hit_a.raw_unit}» и «{hit_b.raw_unit}») "
                    "к единой единице измерения или явно зафиксировать правило пересчёта и округления?"
                ),
                answer_options=[
                    "Привести все значения к одной единице измерения во всех слоях спецификации",
                    "Оставить разные единицы и зафиксировать явное правило пересчёта и округления",
                ],
            )
        )

    return findings


# ---------------------------------------------------------------- rule 2: idempotency_without_key

_GLOSSARY_SECTION_RE = re.compile(
    r"глоссар|термин|сокращен|определен|glossary|definition|terms|abbreviat",
    re.IGNORECASE,
)
_GLOSSARY_TERM_HEADER_RE = re.compile(r"^(?:термин|term|сокращен|abbreviat|понятие)", re.IGNORECASE)
_GLOSSARY_DEF_HEADER_RE = re.compile(r"^(?:определен|definition|расшифровк|описание\s+термина)", re.IGNORECASE)

# Normative idempotency / deduplication claims (excludes plain business retry attempts and glossary definitions).
_IDEMPOTENCY_CLAIM_RE = re.compile(
    r"(?<!\w)(?:"
    r"идемпотент\w*"
    r"|дедупликац\w*"
    r"|повторн(?:ая|ой|ую|ое|ого|ые|ых)\s+(?:доставк\w*|отправк\w*|обработк\w*)[^.?!;\n]{0,70}"
    r"(?:сохран\w*\s+(?:исходн|прежн|тот\s+же)|не\s+(?:меня|измен|созда|привод|вызыв|дублир|увелич|спис)|"
    r"игнорир\w*|отбрасыв\w*|пропуск\w*|возвраща\w*\s+(?:прежн|исходн|тот\s+же)|исключ\w*)"
    r"|повторн(?:ый|ого|ом|ому|ые|ых)\s+(?:запрос\w*|вызов\w*|событ\w*|сообщен\w*)[^.?!;\n]{0,60}"
    r"(?:не\s+(?:меня|измен|созда|привод|вызыв|дублир|увелич|спис)|сохран\w*\s+(?:исходн|прежн|тот\s+же)|"
    r"игнорир\w*|пропуск\w*|возвраща\w*\s+(?:прежн|исходн|тот\s+же))"
    r"|(?:дубликат\w*|дублирующ\w*|дубл[ией]\w*)[^.?!;\n]{0,50}"
    r"(?:игнорир\w*|отбрасыв\w*|отклон\w*|пропуск\w*|не\s+обрабатыва\w*|не\s+созда\w*|не\s+привод\w*|исключ\w*)"
    r"|(?:игнорир\w*|отбрасыв\w*|исключ\w*|защит\w*\s+от)\s+[^.?!;\n]{0,30}(?:дубликат\w*|дубл\w*|повторн\w*\s+(?:доставк|запрос|списан|проводк|создан))"
    r"|idempoten\w*"
    r"|deduplicat\w*"
    r"|(?:duplicate|repeated)\s+(?:delivery|request|event|message|call|submission)s?\b"
    r"[^.?!;\n]{0,60}(?:must\s+not|shall\s+not|does\s+not|do\s+not|ignored?|discarded?|skipped?|"
    r"preserves?|keeps?|safe|no\s+effect)"
    r"|(?:ignore|discard|reject|skip|prevent)\w*\s+(?:duplicate|repeated)\s+"
    r"(?:delivery|request|event|message|call)s?\b"
    r")",
    re.IGNORECASE,
)

# Normative statement marker: modal obligation or present-tense system requirement predicate.
_NORMATIVE_CLAIM_PREDICATE_RE = re.compile(
    r"(?<!\w)(?:"
    r"долж(?:ен|на|но|ны)|обязан(?:а|о|ы)?|необходимо|требует(?:ся)?"
    r"|сохраняет|игнориру(?:ет|ется|ются)|отбрасыва(?:ет|ется|ются)|отклоня(?:ет|ется|ются)"
    r"|пропуска(?:ет|ется|ются)|исключа(?:ет|я)|предотвраща(?:ет|ется)|обеспечива(?:ет|ть)"
    r"|поддержива(?:ет|ют|ть)|не\s+(?:меняет|изменяет|созда[её]т|приводит|вызывает|дублирует|списывает|выполняется|обрабатывается)"
    r"|shall|must|requires?|preserves?|keeps?|ignores?|discards?|rejects?|prevents?|supports?|does\s+not|do\s+not"
    r")(?!\w)",
    re.IGNORECASE,
)

_DEDUP_KEY_RE = re.compile(
    r"(?:"
    r"(?<!\w)(?:X-)?(?:Idempotency-Key|Request-Id|Correlation-Id)(?!\w)"
    r"|(?<!\w)(?:idempotency_?key|dedup(?:lication)?_?key|event_?id|message_?id|"
    r"operation_?id|transaction_?id|request_?id|payment_?id|transfer_?id|order_?id)(?!\w)"
    r"|ключ\w*\s+(?:идемпотентност\w*|дедупликац\w*)"
    r"|(?:по|через|с\s+помощью|на\s+основе|используя|за\s+сч[её]т(?:\s+передачи)?)\s+"
    r"(?:уникальн\w*\s+)?(?:идентификатор\w*|ключ\w*|ID|полю|полей|заголовк\w*)"
    r"|уникальн\w*\s+(?:идентификатор\w*|ключ\w*)"
    r"|(?:using|by|via|keyed\s+by|based\s+on)\s+(?:a\s+|the\s+)?(?:unique\s+)?"
    r"(?:idempotency\s+)?(?:key|identifier|id|header|field)"
    r")",
    re.IGNORECASE,
)

_DEDUP_WINDOW_RE = re.compile(
    r"(?:"
    r"(?:в\s+течение|окн\w*|период\w*|срок\w*|хранен\w*|храним\w*|TTL|window|within|for|retention|expir\w*)"
    r"[^.?!;\n]{0,35}\d+\s*"
    r"(?:мс|ms|сек\w*|sec\w*|мин\w*|min\w*|час\w*|hour\w*|ч\.|дн\w*|день|сут\w*|day\w*|недел\w*|week\w*|мес\w*|month\w*)"
    r"|окн\w*\s+дедупликац\w*"
    r"|(?:dedup\w*|idempotency)\s+(?:window|ttl|period|retention)"
    r")",
    re.IGNORECASE,
)


def _glossary_table_row_ids(doc: Document) -> set[str]:
    row_ids: set[str] = set()
    for table in _group_tables(doc.blocks):
        if table.header is None:
            continue
        headers = [h.strip() for h in _get_cells(table.header)]
        if (
            len(headers) >= 2
            and _GLOSSARY_TERM_HEADER_RE.match(headers[0])
            and _GLOSSARY_DEF_HEADER_RE.match(headers[1])
        ):
            for row in table.rows:
                row_ids.add(row.id)
    return row_ids


def _is_glossary_block(block: Block, glossary_rows: set[str]) -> bool:
    return block.id in glossary_rows or any(
        _GLOSSARY_SECTION_RE.search(sec_title) for sec_title in block.section_path
    )


def _check_idempotency_without_key(doc: Document) -> list[RuleFinding]:
    glossary_rows = _glossary_table_row_ids(doc)
    claim_blocks: list[tuple[Block, re.Match[str]]] = []

    for blk in doc.blocks:
        if blk.kind in (BlockKind.CODE, BlockKind.TABLE_HEADER, BlockKind.HEADING):
            continue
        if _is_glossary_block(blk, glossary_rows):
            continue
        m = _IDEMPOTENCY_CLAIM_RE.search(blk.text)
        if m is None:
            continue
        if not _NORMATIVE_CLAIM_PREDICATE_RE.search(blk.text):
            continue
        claim_blocks.append((blk, m))

    if not claim_blocks:
        return []

    has_key = any(_DEDUP_KEY_RE.search(blk.text) for blk, _ in claim_blocks)
    has_window = any(_DEDUP_WINDOW_RE.search(blk.text) for blk, _ in claim_blocks)

    if not has_key:
        explicit_header_re = re.compile(
            r"(?<!\w)(?:X-)?Idempotency-Key(?!\w)|idempotency_?key|dedup(?:lication)?_?key|"
            r"ключ\w*\s+(?:идемпотентност\w*|дедупликац\w*)",
            re.IGNORECASE,
        )
        has_key = any(
            explicit_header_re.search(b.text)
            for b in doc.blocks
            if b.kind != BlockKind.HEADING and not _is_glossary_block(b, glossary_rows)
        )
    if not has_window:
        explicit_window_re = re.compile(
            r"окн\w*\s+дедупликац\w*|(?:dedup\w*|idempotency)\s+(?:window|ttl|period|retention)",
            re.IGNORECASE,
        )
        has_window = any(
            explicit_window_re.search(b.text)
            for b in doc.blocks
            if b.kind != BlockKind.HEADING and not _is_glossary_block(b, glossary_rows)
        )

    if has_key or has_window:
        return []

    first_blk, first_match = claim_blocks[0]
    sec_label = _section_label(first_blk)
    return [
        _make_finding(
            rule="idempotency_without_key",
            block_ids=[blk.id for blk, _ in claim_blocks],
            criterion=Criterion.COMPLETE,
            severity=Severity.MAJOR,
            set_check=None,
            quote=_quote_for_span(first_blk.text, first_match.start(), first_match.end()),
            explanation=(
                f"В разделе «{sec_label}» заявлена идемпотентность или подавление дублей "
                f"(«{first_match.group(0)}»), но нигде рядом не указаны ключ/идентификатор "
                "дедупликации и окно хранения обработанных ключей."
            ),
            recommendation=(
                "Явно указать поле или заголовок-ключ дедупликации, срок хранения ключей "
                "(окно дедупликации) и ожидаемый ответ при повторной доставке."
            ),
            clarifying_question=(
                "По какому ключу (идентификатору события/запроса) и в течение какого окна "
                "времени должна выполняться дедупликация при повторной доставке?"
            ),
            answer_options=[
                "Дедупликация по уникальному ключу операции/события с фиксированным окном (например, 24–72 часа)",
                "Хранение идентификаторов всех обработанных операций за весь расчётный период",
            ],
        )
    ]


# ---------------------------------------------------------------- rule 3: flag_with_mandatory_params

_PARAM_NAME_COL_RE = re.compile(
    r"^(?:parameter|param|field|name|key|attribute|property|параметр|поле|имя|название|атрибут)$",
    re.IGNORECASE,
)
_MANDATORY_COL_RE = re.compile(r"^(?:mandatory|required|обязательн\w*|обяз\.?)$", re.IGNORECASE)
_TYPE_COL_RE = re.compile(r"^(?:type|format|data\s*type|тип|тип\s+данных|формат)$", re.IGNORECASE)
_LOGIC_COL_RE = re.compile(
    r"^(?:logic|rules?|constraints?|validation|range|values?|логик\w*|правил\w*|огранич\w*|значен\w*)$",
    re.IGNORECASE,
)
_DESC_COL_RE = re.compile(r"^(?:description|desc|comment|note|описан\w*|комментар\w*)$", re.IGNORECASE)

_MANDATORY_YES_RE = re.compile(
    r"^(?:да|yes|true|required|mandatory|обязательн\w*|\+|1|\*)$",
    re.IGNORECASE,
)
_FLAG_NAME_RE = re.compile(
    r"(?:^|_)(?:enabled?|active|is_?enabled|is_?active|toggle|switch|on_?off|flag|opt_?in|включ\w*|актив\w*)(?:$|_)"
    r"|(?:Enabled|Active|Flag|Toggle)$",
    re.IGNORECASE,
)
_FLAG_VALUES_RE = re.compile(
    r"(?<!\w)(?:[«\"']?0[»\"']?\s*(?:или|or|/|,)\s*[«\"']?1[»\"']?|"
    r"true\s*(?:или|or|/|,)\s*false|enabled?\s*/\s*disabled?|вкл\.?\s*/\s*выкл\.?)(?!\w)",
    re.IGNORECASE,
)
_FLAG_DESC_RE = re.compile(
    r"включ|отключ|выключ|переключател|признак\s+актив|enable|disable|toggle|switch|on/off",
    re.IGNORECASE,
)
_BOOL_TYPE_RE = re.compile(r"^(?:bool(?:ean)?|булев\w*|логическ\w*)$", re.IGNORECASE)

_VALUE_RANGE_RE = re.compile(
    r"(?:"
    r"\d[\d\s\u00a0.,]*\s*(?:[–—-]|\.\.|\s+до\s+|\s+to\s+)\s*\d[\d\s\u00a0.,]*"
    r"|(?:>=|<=|>|<|от\s+\d|не\s+менее\s+\d|не\s+более\s+\d|min\s*[:=]\s*\d|max\s*[:=]\s*\d)"
    r")",
    re.IGNORECASE,
)

_FLAG_OFF_BEHAVIOR_RE = re.compile(
    r"(?:"
    r"(?:при|если|когда)\s+(?:отключ\w*|выключ\w*|деактив\w*|снят\w*\s+флаг\w*|"
    r"значени\w*\s+[«\"']?(?:0|false)[»\"']?|"
    r"(?:флаг\w*|переключател\w*|\w*enabled?\w*|\w*active\w*)\s*(?:=|==|равен|равн\w*)\s*[«\"']?(?:0|false)[»\"']?)"
    r"|(?:при|если)\s+[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_]*\s*={1,2}\s*[«\"']?(?:0|false)[»\"']?"
    r"|(?:when|if)\s+(?:the\s+)?(?:flag\s+is\s+)?(?:disabled|off|turned\s+off|inactive|false|0|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*(?:=|==|is)\s*(?:false|0))(?!\w)"
    r"|(?:ignored|optional|omitted|nullable|not\s+validated|skipped|cleared|retained)\s+"
    r"(?:when|if)\s+(?:disabled|off|false|0|[A-Za-z_][A-Za-z0-9_]*\s*(?:=|==|is)\s*(?:false|0))"
    r")",
    re.IGNORECASE,
)


def _is_flag_row(
    cells: list[str],
    *,
    name_col: int,
    type_col: int | None,
    logic_col: int | None,
    desc_col: int | None,
) -> tuple[bool, str]:
    name = cells[name_col].strip().strip("`*") if name_col < len(cells) else ""
    if not name:
        return False, ""
    type_val = cells[type_col].strip() if type_col is not None and type_col < len(cells) else ""
    logic_val = cells[logic_col].strip() if logic_col is not None and logic_col < len(cells) else ""
    desc_val = cells[desc_col].strip() if desc_col is not None and desc_col < len(cells) else ""
    row_text = " ".join(cells)

    has_flag_name = bool(_FLAG_NAME_RE.search(name))
    has_bool_type = bool(type_val and _BOOL_TYPE_RE.match(type_val))
    has_flag_vals = bool(_FLAG_VALUES_RE.search(logic_val or row_text))
    has_flag_desc = bool(_FLAG_DESC_RE.search(desc_val or row_text))

    is_flag = (
        (has_flag_name and (has_bool_type or has_flag_vals or has_flag_desc))
        or (has_bool_type and has_flag_desc)
        or (has_flag_vals and has_flag_desc)
    )
    return is_flag, name


def _check_flag_with_mandatory_params(doc: Document) -> list[RuleFinding]:
    tables = _group_tables(doc.blocks)
    findings: list[RuleFinding] = []

    full_prose = "\n".join(b.text for b in doc.blocks if b.kind != BlockKind.CODE)
    if _FLAG_OFF_BEHAVIOR_RE.search(full_prose):
        return []

    for table in tables:
        if table.header is None or len(table.rows) < 2:
            continue
        headers = [h.strip() for h in _get_cells(table.header)]

        name_col = next((i for i, h in enumerate(headers) if _PARAM_NAME_COL_RE.match(h)), 0)
        mandatory_col = next((i for i, h in enumerate(headers) if _MANDATORY_COL_RE.match(h)), None)
        type_col = next((i for i, h in enumerate(headers) if _TYPE_COL_RE.match(h)), None)
        logic_col = next((i for i, h in enumerate(headers) if _LOGIC_COL_RE.match(h)), None)
        desc_col = next((i for i, h in enumerate(headers) if _DESC_COL_RE.match(h)), None)

        if mandatory_col is None:
            continue

        flag_row: Block | None = None
        flag_name = ""
        ranged_mandatory_rows: list[tuple[Block, str, bool]] = []

        for row in table.rows:
            cells = _get_cells(row)
            is_flag, cand_flag_name = _is_flag_row(
                cells,
                name_col=name_col,
                type_col=type_col,
                logic_col=logic_col,
                desc_col=desc_col,
            )
            if is_flag and flag_row is None:
                flag_row = row
                flag_name = cand_flag_name
                continue

            if mandatory_col >= len(cells) or not _MANDATORY_YES_RE.match(cells[mandatory_col].strip()):
                continue

            param_name = cells[name_col].strip().strip("`*") if name_col < len(cells) else ""
            logic_val = cells[logic_col].strip() if logic_col is not None and logic_col < len(cells) else ""
            desc_val = cells[desc_col].strip() if desc_col is not None and desc_col < len(cells) else ""
            other_cells_text = " ".join(
                c for i, c in enumerate(cells) if i not in {name_col, mandatory_col, type_col}
            )
            in_logic = bool(_VALUE_RANGE_RE.search(logic_val) or _VALUE_RANGE_RE.search(desc_val))
            in_any = in_logic or bool(_VALUE_RANGE_RE.search(other_cells_text))
            if in_any and param_name:
                ranged_mandatory_rows.append((row, param_name, in_logic))

        if flag_row is None or not ranged_mandatory_rows:
            continue

        preferred = [entry for entry in ranged_mandatory_rows if entry[2]] or ranged_mandatory_rows
        param_names = ", ".join(f"«{name}»" for _, name, _ in preferred[:3])
        sec_label = _section_label(flag_row)

        findings.append(
            _make_finding(
                rule="flag_with_mandatory_params",
                block_ids=[flag_row.id, *[row.id for row, _, _ in preferred[:3]]],
                criterion=Criterion.SET_COMPLETE,
                severity=Severity.MAJOR,
                set_check=SetCheck.MISSING_SCENARIO,
                quote=_quote_for_span(flag_row.text),
                explanation=(
                    f"В таблице параметров раздела «{sec_label}» присутствует флаг включения/отключения "
                    f"«{flag_name}» вместе с обязательными параметрами с проверкой диапазона ({param_names}), "
                    "но не описано, должны ли эти параметры передаваться и валидироваться при выключенном флаге."
                ),
                recommendation=(
                    f"Зафиксировать поведение параметров {param_names} при выключенном флаге «{flag_name}»: "
                    "обязательность передачи, пропуск или выполнение валидации диапазона и сохранение прежнего значения."
                ),
                clarifying_question=(
                    f"Как должны обрабатываться обязательные параметры ({param_names}) при выключенном "
                    f"флаге «{flag_name}»: игнорироваться (стать необязательными) или валидироваться по диапазону?"
                ),
                answer_options=[
                    "При выключенном флаге параметры становятся необязательными и не валидируются по диапазону",
                    "Параметры остаются обязательными с проверкой диапазона независимо от значения флага",
                ],
            )
        )

    return findings


# ---------------------------------------------------------------- rule 4: one_line_process (aggregated)

_PROCESS_HEADING_RE = re.compile(
    r"(?<!\w)(?:bpmn|sequence|процесс\w*|process\w*|диаграмм\w*|diagram\w*|схем\w*|workflow|flow(?:chart)?)(?!\w)",
    re.IGNORECASE,
)
_ARROW_RE = re.compile(r"→|-->|->|=>")
_FAILURE_BRANCH_RE = re.compile(
    r"(?<!\w)(?:"
    r"ошибк\w*|иначе|если\s+не\b|отказ\w*|неуспеш\w*|невалид\w*|некорректн\w*|"
    r"исключен\w*|сбо[йяюем]\w*|таймаут\w*|отклон\w*|"
    r"error\w*|else\b|otherwise\b|if\s+not\b|fail(?:ure|ed|s)?\b|reject(?:ed|s)?\b|"
    r"exception\w*|timeout\w*|fallback\b|invalid\b"
    r")",
    re.IGNORECASE,
)


def _check_one_line_process(doc: Document) -> list[RuleFinding]:
    matches: list[tuple[str, Block]] = []

    parent_ids = {s.parent_id for s in doc.sections if s.parent_id}
    blocks_by_sec: dict[str, list[Block]] = defaultdict(list)
    for blk in doc.blocks:
        blocks_by_sec[blk.section_id].append(blk)

    for sec in doc.sections:
        if sec.id in parent_ids:
            continue
        if not _PROCESS_HEADING_RE.search(sec.title):
            continue

        content_blks = [b for b in blocks_by_sec.get(sec.id, []) if b.kind != BlockKind.HEADING]
        if len(content_blks) != 1:
            continue

        blk = content_blks[0]
        if blk.kind not in (BlockKind.PARAGRAPH, BlockKind.CODE, BlockKind.LIST_ITEM, BlockKind.OTHER):
            continue

        non_empty_lines = [ln.strip() for ln in blk.text.splitlines() if ln.strip()]
        if len(non_empty_lines) != 1:
            continue

        line = non_empty_lines[0]
        if not _ARROW_RE.search(line):
            continue
        if _FAILURE_BRANCH_RE.search(line):
            continue

        sec_label = " › ".join(sec.path) if sec.path else sec.title
        matches.append((sec_label, blk))

    if not matches:
        return []

    first_label, first_blk = matches[0]
    if len(matches) == 1:
        explanation = (
            f"Раздел «{first_label}» содержит только однострочную цепочку шагов со стрелками "
            "без веток обработки ошибок, отказов и альтернативных исходов."
        )
        question = (
            f"Какие альтернативные и ошибочные ветки должны быть отражены в процессе "
            f"раздела «{first_label}»?"
        )
    else:
        sections_list = ", ".join(f"«{label}»" for label, _ in matches)
        explanation = (
            f"В {len(matches)} разделах со схемами процессов ({sections_list}) приведены только "
            "однострочные цепочки шагов со стрелками без веток обработки ошибок, отказов и альтернативных исходов."
        )
        question = (
            f"Какие альтернативные и ошибочные ветки должны быть отражены в схемах "
            f"разделов {sections_list}?"
        )

    return [
        _make_finding(
            rule="one_line_process",
            block_ids=[blk.id for _, blk in matches],
            criterion=Criterion.COMPLETE,
            severity=Severity.MINOR,
            set_check=SetCheck.MISSING_SCENARIO,
            quote=_quote_for_span(first_blk.text),
            explanation=explanation,
            recommendation=(
                "Дополнить схемы процессов ветками ошибок, таймаутов и альтернативных "
                "сценариев либо приложить полноценные диаграммы."
            ),
            clarifying_question=question,
            answer_options=[
                "Добавить ветки отказов, ошибок валидации и недоступности смежных сервисов в схемы",
                "Оставить схемы обзорными и указать ссылку на раздел с обработкой всех ошибок",
            ],
        )
    ]


def run_gap_checks(doc: Document, inventory: list[InventoryItem]) -> list[RuleFinding]:
    """Run conservative deterministic gap rules for units, idempotency, toggle flags, and linear processes."""
    findings: list[RuleFinding] = []
    findings.extend(_check_units_mismatch(doc, inventory))
    findings.extend(_check_idempotency_without_key(doc))
    findings.extend(_check_flag_with_mandatory_params(doc))
    findings.extend(_check_one_line_process(doc))
    return findings
