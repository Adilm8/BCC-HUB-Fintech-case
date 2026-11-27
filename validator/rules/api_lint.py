"""Domain-neutral REST/API specification linter producing deterministic RuleFinding items."""

from __future__ import annotations

import json
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
    Section,
    SetCheck,
    Severity,
)

_MAX_QUOTE_LEN = 200
_HAS_ADDRESSEE = "addressee" in RuleFinding.model_fields

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

# Canonical RFC 9110 status phrases mapped to their standard status codes.
# Longer phrases are ordered before shorter ones to prevent substring clashes.
_STATUS_REASON_PHRASES: tuple[tuple[str, str], ...] = (
    ("internal server error", "500"),
    ("unprocessable content", "422"),
    ("unprocessable entity", "422"),
    ("unsupported media type", "415"),
    ("service unavailable", "503"),
    ("method not allowed", "405"),
    ("moved permanently", "301"),
    ("too many requests", "429"),
    ("payload too large", "413"),
    ("content too large", "413"),
    ("gateway timeout", "504"),
    ("bad request", "400"),
    ("bad gateway", "502"),
    ("not modified", "304"),
    ("unauthorized", "401"),
    ("no content", "204"),
    ("not found", "404"),
    ("forbidden", "403"),
    ("accepted", "202"),
    ("conflict", "409"),
    ("created", "201"),
    ("gone", "410"),
    ("ok", "200"),
)

_METHOD_KEY_RE = re.compile(
    r"^(?:http\s+)?(?:rest\s+)?(?:method|type|verb|метод|тип\s+запроса|http[-\s]*метод)$",
    re.IGNORECASE,
)
_DESC_KEY_RE = re.compile(
    r"^(?:description|summary|purpose|operation|action|endpoint|url|request\s+url|path|"
    r"описание|назначение|действие|операция|путь|адрес)$",
    re.IGNORECASE,
)
_SAFE_METHOD_EXACT_RE = re.compile(r"^(GET|HEAD)$")
_ANY_METHOD_EXACT_RE = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)$")
_INLINE_METHOD_DECL_RE = re.compile(
    r"(?:(?:http\s+)?(?:rest\s+)?(?:method|type|verb|метод|тип\s+запроса|http[-\s]*метод)\s*:\s*"
    r"(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)(?!\w))"
    r"|(?:^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/[A-Za-z0-9_{}/.-]+)",
    re.IGNORECASE,
)
_HEADING_METHOD_RE = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/[A-Za-z0-9_{}/.-]+")

_URL_PATH_RE = re.compile(r"(?<![:\w/])(/[A-Za-z][A-Za-z0-9_{}-]*(?:/[A-Za-z0-9_{}-]+)*)")

_PATH_WRITE_TOKENS = frozenset(
    {
        "save",
        "update",
        "create",
        "delete",
        "remove",
        "set",
        "change",
        "edit",
        "modify",
        "insert",
        "add",
        "replace",
        "upsert",
        "write",
        "reset",
        "cancel",
        "revoke",
        "approve",
        "reject",
        "submit",
        "register",
    }
)

_TEXT_WRITE_RE = re.compile(
    r"(?<!\w)(?:"
    r"сохранен\w*|сохрани\w*|сохраня\w*"
    r"|изменен\w*|измени\w*|изменя\w*"
    r"|создани\w*|создат\w*|созда[её]\w*"
    r"|удален\w*|удали\w*|удаля\w*"
    r"|установк\w*|установи\w*|устанавлива\w*"
    r"|замен\w*"
    r"|обновлен\w*|обнови\w*|обновля\w*"
    r"|добавлен\w*|добави\w*|добавля\w*"
    r"|редактирован\w*|редактир\w*"
    r"|запис[аы]\w*"
    r"|save|saving|update|updating|create|creating|delete|deleting|remove|removing"
    r"|set|setting|change|changing|modify|modifying|replace|replacing|insert|inserting|upsert|write|writing"
    r")(?!\w)",
    re.IGNORECASE,
)

_QUERY_STRING_RE = re.compile(
    r"(?:"
    r"query[-\s]*(?:параметр\w*|param\w*|string|аргумент\w*)"
    r"|в\s+строке\s+запроса|в\s+query|in\s+(?:the\s+)?query(?:\s+string)?"
    r"|curl\s+[^\n]*(?:-G|--get)\b[^\n]*(?:--data(?:-urlencode)?|-d)\b"
    r"|\?[A-Za-z_][A-Za-z0-9_]*="
    r")",
    re.IGNORECASE,
)

_STATUS_SECTION_RE = re.compile(r"статус|status|код\w*\s+ответ|response\s+code|http\s+code", re.IGNORECASE)
_STATUS_HEADING_START_RE = re.compile(r"^([1-5]\d{2})(?:[\.\s:—–-]|$)")
_STATUS_DECL_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s+)?(?:\*\*)?([1-5]\d{2})(?:\.)?(?:\*\*)?"
    r"(?:\.\s+|\s*[—–:-]\s+)\S"
)
_STATUS_TABLE_HEADER_RE = re.compile(r"^(?:code|status|http\s*code|код|статус)$", re.IGNORECASE)
_STATUS_CODE_EXACT_RE = re.compile(r"^([1-5]\d{2})$")

_STATUS_REF_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:возвраща(?:ет(?:ся)?|ются)|верн[её]т(?:ся)?|вернуть|отда[её]т(?:ся)?|"
    r"ответ(?:а|ом|е)?|статус(?:а|ом|е)?|код(?:а|ом|е|ы)?|результат(?:а|е)?)\s+"
    r"(?:HTTP\s+)?([1-5]\d{2}(?:\s*(?:,|или|и|/)\s*[1-5]\d{2})*)"
    r"|(?:returns?|returning|responds?\s+with|status(?:\s+code)?|http(?:\s+code)?|code|result)\s+"
    r"(?:is\s+)?([1-5]\d{2}(?:\s*(?:,|or|and|/)\s*[1-5]\d{2})*)"
    r"|([1-5]\d{2})\s+(?:возвращается|is\s+returned)"
    r")(?!\w)",
    re.IGNORECASE,
)

_UNIT_AFTER_NUM_RE = re.compile(
    r"^\s*(?:₸|₽|€|\$|тенге|тиын|KZT|USD|EUR|RUB|%|мс|ms|сек|sec|мин|min|час|дн|day|байт|byte|KB|MB|GB|КБ|МБ|ГБ)",
    re.IGNORECASE,
)

_2XX_RE = re.compile(r"(?<!\w)(20[0-6])(?!\w)")
_4XX_BAD_INPUT_RE = re.compile(
    r"неверн|некорректн|ошибк|валидац|параметр|недопустим|"
    r"bad\s+request|invalid|validation|malformed|unprocessable|wrong|incorrect",
    re.IGNORECASE,
)

_ERROR_MARKER_VALUES = frozenset(
    {"error", "err", "fail", "failed", "failure", "invalid", "rejected", "denied"}
)
_ERROR_MESSAGE_RE = re.compile(
    r"ошибк|неверн|некорректн|проверьте|недопустим|не\s+найден|"
    r"invalid|error|fail|bad\s+request|wrong|missing|required|rejected|denied",
    re.IGNORECASE,
)

_REQ_SECTION_ROLE_RE = re.compile(
    r"запрос|входн|request|query|path[-\s]*парам|параметры\s+запроса|input",
    re.IGNORECASE,
)
_RESP_SECTION_ROLE_RE = re.compile(
    r"ответ|выходн|response|output|результат",
    re.IGNORECASE,
)

_BODY_SECTION_RE = re.compile(
    r"request\s*body|тело\s+запроса|\bbody\b|тело\s+ответа|payload",
    re.IGNORECASE,
)
_REQ_BODY_SECTION_RE = re.compile(r"request\s*body|тело\s+запроса", re.IGNORECASE)
_CONTENT_TYPE_RE = re.compile(r"(?<!\w)content-type\b", re.IGNORECASE)
_BODY_UNUSED_RE = re.compile(
    r"(?:(?:тело(?:\s+запроса)?\s+)?(?:не\s+используется|не\s+переда[её]тся|отсутствует)|"
    r"пустое\s+тело|без\s+тела|"
    r"(?:body\s+is\s+)?(?:not\s+used|unused|empty|no\s+body|omitted|not\s+applicable))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _MethodDecl:
    method: str
    block: Block
    scope_blocks: list[Block]
    scope_sections: list[Section]
    is_row_scoped: bool


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
    return RuleFinding(**kwargs)


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
    if lo > 0:
        space = text.find(" ", lo, start)
        if space != -1 and space + 1 < start:
            lo = space + 1
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        if space != -1 and space > end:
            hi = space
    return text[lo:hi]


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


def _split_identifier_tokens(segment: str) -> list[str]:
    """Split a URL path segment along camelCase, underscore, and hyphen boundaries."""
    pieces = re.split(r"[_-]+", segment)
    tokens: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        for tok in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+", piece):
            tokens.append(tok.lower())
    return tokens


def _path_write_verb(text: str) -> str | None:
    for m in _URL_PATH_RE.finditer(text):
        path = m.group(1)
        for seg in path.strip("/").split("/"):
            if seg.startswith("{") and seg.endswith("}"):
                continue
            for tok in _split_identifier_tokens(seg):
                if tok in _PATH_WRITE_TOKENS:
                    return f"{path} ({tok})"
    return None


def _collect_method_declarations(doc: Document) -> list[_MethodDecl]:
    """Locate explicit HTTP method declarations and resolve each declaration's owning scope."""
    # Map table rows to their header columns.
    row_header_cells: dict[str, list[str]] = {}
    current_headers: list[str] = []
    current_sec: str | None = None
    for blk in doc.blocks:
        if blk.kind == BlockKind.TABLE_HEADER:
            current_headers = _get_cells(blk)
            current_sec = blk.section_id
        elif blk.kind == BlockKind.TABLE_ROW:
            if blk.section_id != current_sec:
                current_headers = []
                current_sec = blk.section_id
            row_header_cells[blk.id] = current_headers
        else:
            current_headers = []
            current_sec = None

    raw_decls: list[tuple[str, Block, bool]] = []  # (method, block, is_row_scoped)
    seen_blocks: set[str] = set()

    for blk in doc.blocks:
        if blk.kind == BlockKind.CODE:
            continue
        if blk.kind == BlockKind.TABLE_ROW:
            cells = _get_cells(blk)
            headers = row_header_cells.get(blk.id, [])
            # Pattern A: key-value metadata row like "| REST type | GET |".
            if len(cells) >= 2 and _METHOD_KEY_RE.match(cells[0].strip()):
                val = cells[1].strip()
                if _ANY_METHOD_EXACT_RE.match(val):
                    raw_decls.append((val, blk, False))
                    seen_blocks.add(blk.id)
                    continue
            # Pattern B: multi-endpoint summary table with a dedicated Method column.
            method_col = next(
                (i for i, h in enumerate(headers) if _METHOD_KEY_RE.match(h.strip())),
                None,
            )
            if method_col is not None and method_col < len(cells):
                val = cells[method_col].strip()
                if _ANY_METHOD_EXACT_RE.match(val):
                    raw_decls.append((val, blk, True))
                    seen_blocks.add(blk.id)
                    continue

        if blk.id in seen_blocks:
            continue
        if blk.kind in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM, BlockKind.HEADING):
            m = _INLINE_METHOD_DECL_RE.search(blk.text.strip())
            if m:
                method = (m.group(1) or m.group(2) or "").upper()
                if method:
                    raw_decls.append((method, blk, False))
                    seen_blocks.add(blk.id)

    # Also detect endpoint headings like "### GET /items/create" when structure parser omitted HEADING blocks.
    blocks_by_sec: dict[str, list[Block]] = defaultdict(list)
    for blk in doc.blocks:
        blocks_by_sec[blk.section_id].append(blk)

    sec_has_decl = {blk.section_id for _, blk, _ in raw_decls}
    for sec in doc.sections:
        if sec.id in sec_has_decl:
            continue
        m = _HEADING_METHOD_RE.match(sec.title.strip())
        if m and blocks_by_sec.get(sec.id):
            raw_decls.append((m.group(1), blocks_by_sec[sec.id][0], False))
            sec_has_decl.add(sec.id)

    # Precompute section tree helpers to assign conservative scopes.
    sec_by_id = {s.id: s for s in doc.sections}
    children_by_parent: dict[str | None, list[Section]] = defaultdict(list)
    for s in doc.sections:
        children_by_parent[s.parent_id].append(s)

    def subtree_section_ids(root_id: str) -> set[str]:
        out = {root_id}
        stack = [root_id]
        while stack:
            cur = stack.pop()
            for child in children_by_parent.get(cur, []):
                if child.id not in out:
                    out.add(child.id)
                    stack.append(child.id)
        return out

    non_row_decls_per_part: dict[str, int] = defaultdict(int)
    for _, blk, is_row in raw_decls:
        if not is_row:
            non_row_decls_per_part[blk.part] += 1
    total_non_row_decls = sum(non_row_decls_per_part.values())

    sections_per_part: dict[str, list[Section]] = defaultdict(list)
    for s in doc.sections:
        sections_per_part[s.part].append(s)

    result: list[_MethodDecl] = []
    for method, blk, is_row in raw_decls:
        if is_row:
            sec = sec_by_id.get(blk.section_id)
            result.append(_MethodDecl(method, blk, [blk], [sec] if sec else [], True))
            continue

        part_secs = sections_per_part.get(blk.part, [])
        if total_non_row_decls == 1 and len(part_secs) <= 1:
            # Standalone single-endpoint specification where top-level sections act as parts.
            scope_secs = list(doc.sections)
            scope_blks = list(doc.blocks)
        elif non_row_decls_per_part.get(blk.part, 0) == 1:
            scope_secs = part_secs
            scope_blks = [b for b in doc.blocks if b.part == blk.part]
        else:
            sec_ids = subtree_section_ids(blk.section_id)
            scope_secs = [s for s in doc.sections if s.id in sec_ids]
            scope_blks = [b for b in doc.blocks if b.section_id in sec_ids]

        result.append(_MethodDecl(method, blk, scope_blks, scope_secs, False))

    return result


# ---------------------------------------------------------------- check 1: GET/HEAD vs write side effect


def _check_get_write_side_effects(decls: list[_MethodDecl]) -> list[RuleFinding]:
    findings: list[RuleFinding] = []

    for decl in decls:
        if not _SAFE_METHOD_EXACT_RE.match(decl.method):
            continue

        write_reason: str | None = None
        write_block: Block | None = None

        # 1. Check section headings in scope (skip root part heading if unrelated, prefer leaf headings).
        for sec in decl.scope_sections:
            path_hit = _path_write_verb(sec.title)
            if path_hit:
                write_reason = f"путь {path_hit}"
                break
            if not decl.is_row_scoped and _HEADING_METHOD_RE.match(sec.title.strip()):
                m_txt = _TEXT_WRITE_RE.search(sec.title)
                if m_txt:
                    write_reason = f"заголовок «{m_txt.group(0)}»"
                    break

        # 2. Check blocks in scope: URL paths and description/summary rows.
        if write_reason is None:
            for blk in decl.scope_blocks:
                if blk.kind == BlockKind.CODE:
                    continue
                path_hit = _path_write_verb(blk.text)
                if path_hit:
                    write_reason = f"путь {path_hit}"
                    write_block = blk
                    break
                if blk.kind == BlockKind.TABLE_ROW:
                    cells = _get_cells(blk)
                    if decl.is_row_scoped:
                        m_txt = _TEXT_WRITE_RE.search(blk.text)
                        if m_txt:
                            write_reason = f"описание операции «{m_txt.group(0)}»"
                            write_block = blk
                            break
                    elif len(cells) >= 2 and _DESC_KEY_RE.match(cells[0].strip()):
                        m_txt = _TEXT_WRITE_RE.search(cells[1])
                        if m_txt:
                            write_reason = f"описание операции «{m_txt.group(0)}»"
                            write_block = blk
                            break
                elif blk.id == decl.block.id:
                    m_txt = _TEXT_WRITE_RE.search(blk.text)
                    if m_txt:
                        write_reason = f"формулировка «{m_txt.group(0)}»"
                        write_block = blk
                        break

        if write_reason is None:
            continue

        # Determine if query-string parameters are passed in the same endpoint scope.
        has_query_data = any(_QUERY_STRING_RE.search(s.title) for s in decl.scope_sections) or any(
            _QUERY_STRING_RE.search(b.text) or _QUERY_STRING_RE.search(b.raw)
            for b in decl.scope_blocks
            if b.kind != BlockKind.CODE
        )

        severity = Severity.CRITICAL if has_query_data else Severity.MAJOR
        block_ids = [decl.block.id]
        if write_block is not None and write_block.id != decl.block.id:
            block_ids.append(write_block.id)

        sec_label = _section_label(decl.block)
        query_note = (
            " При этом изменяемые параметры передаются через query-строку, что нарушает "
            "семантику безопасного метода и ведёт к риску кэширования и утечки данных в логах."
            if has_query_data
            else ""
        )
        findings.append(
            _make_finding(
                rule="api_get_write_side_effect",
                block_ids=block_ids,
                criterion=Criterion.CORRECT,
                severity=severity,
                set_check=None,
                quote=_quote_for_span(decl.block.text),
                explanation=(
                    f"В разделе «{sec_label}» указан безопасный HTTP-метод {decl.method}, "
                    f"однако {write_reason} указывает на операцию изменения данных (RFC 9110).{query_note}"
                ),
                recommendation=(
                    "Заменить метод на POST, PUT или PATCH с передачей изменяемых параметров "
                    "в теле запроса либо разделить операции чтения и записи на разные эндпоинты."
                ),
                clarifying_question=(
                    f"Какой HTTP-метод (POST, PUT или PATCH) должен использоваться вместо "
                    f"{decl.method} для операции изменения данных в разделе «{sec_label}»?"
                ),
            )
        )

    return findings


# ---------------------------------------------------------------- check 2: status-code consistency


def _is_error_json_body(text: str) -> str | None:
    """Return a short marker description when a JSON example represents an error payload."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    data: Any = None
    try:
        data = json.loads(stripped)
    except ValueError:
        # Fallback regex detection for slightly non-strict JSON examples.
        m_kv = re.search(
            r'"(?:result|status|state|outcome)"\s*:\s*"(error|err|fail|failed|failure|invalid|rejected)"',
            stripped,
            re.IGNORECASE,
        )
        if m_kv:
            return m_kv.group(0)
        m_bool = re.search(r'"(?:success|ok|valid)"\s*:\s*false\b', stripped, re.IGNORECASE)
        if m_bool:
            return m_bool.group(0)
        return None

    if not isinstance(data, dict):
        return None

    for key, val in data.items():
        k_low = str(key).lower()
        if (
            k_low in {"result", "status", "state", "outcome"}
            and isinstance(val, str)
            and val.strip().lower() in _ERROR_MARKER_VALUES
        ):
            return f'"{key}": "{val}"'
        if k_low in {"success", "ok", "valid"} and (
            val is False or (isinstance(val, str) and val.strip().lower() in {"false", "0"})
        ):
            return f'"{key}": {val}'
        if k_low in {"error", "errors", "errorcode", "error_code"} and val:
            return f'"{key}"'
        if (
            k_low in {"message", "detail", "errormessage", "error_message"}
            and isinstance(val, str)
            and _ERROR_MESSAGE_RE.search(val)
        ):
            return f'"{key}": "{val}"'
    return None


def _part_has_status_decl(part_sections: list[Section], part_blocks: list[Block]) -> bool:
    for sec in part_sections:
        m = _STATUS_HEADING_START_RE.match(sec.title.strip())
        if m and m.group(1) in _HTTP_STATUS_CODES:
            return True
    for blk in part_blocks:
        if blk.kind in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM):
            for raw_line in blk.raw.splitlines():
                m_line = _STATUS_DECL_LINE_RE.match(raw_line)
                if m_line and m_line.group(1) in _HTTP_STATUS_CODES:
                    return True
    return False


def _check_status_consistency(doc: Document) -> list[RuleFinding]:
    findings: list[RuleFinding] = []

    blocks_by_part: dict[str, list[Block]] = defaultdict(list)
    blocks_by_sec: dict[str, list[Block]] = defaultdict(list)
    for blk in doc.blocks:
        blocks_by_part[blk.part].append(blk)
        blocks_by_sec[blk.section_id].append(blk)

    sections_by_part: dict[str, list[Section]] = defaultdict(list)
    for sec in doc.sections:
        sections_by_part[sec.part].append(sec)

    # Decide grouping: if multiple parts each declare statuses, lint per part;
    # if a standalone API doc puts "## Ответ" and "## Статусы" as sibling parts, lint as one unit.
    status_parts = [
        part
        for part, p_blks in blocks_by_part.items()
        if _part_has_status_decl(sections_by_part.get(part, []), p_blks)
    ]
    if len(status_parts) == 1:
        sole_part = status_parts[0]
        sole_secs = sections_by_part.get(sole_part, [])
        has_non_status_subsection = any(
            not _STATUS_SECTION_RE.search(s.title)
            and not _STATUS_HEADING_START_RE.match(s.title.strip())
            for s in sole_secs
            if s.title != sole_part
        )
        if has_non_status_subsection:
            groups: list[tuple[list[Section], list[Block]]] = [
                (sole_secs, blocks_by_part[sole_part])
            ]
        else:
            groups = [(list(doc.sections), list(doc.blocks))]
    else:
        groups = [
            (sections_by_part.get(part, []), p_blks) for part, p_blks in blocks_by_part.items()
        ]

    for part_sections, part_blocks in groups:
        # Collect declared status codes and track any 4xx invalid-parameter declarations.
        declared_codes: set[str] = set()
        declared_block_by_code: dict[str, Block] = {}
        decl_line_blocks: set[str] = set()
        four_xx_invalid_block: Block | None = None
        four_xx_invalid_code: str | None = None

        # 1) Headings starting with an HTTP status code (e.g. "#### 400. Unsuccessful — Bad request").
        for sec in part_sections:
            m = _STATUS_HEADING_START_RE.match(sec.title.strip())
            if m and m.group(1) in _HTTP_STATUS_CODES:
                code = m.group(1)
                declared_codes.add(code)
                sec_blks = blocks_by_sec.get(sec.id, [])
                if sec_blks and code not in declared_block_by_code:
                    declared_block_by_code[code] = sec_blks[0]
                if (
                    four_xx_invalid_code is None
                    and (code in {"400", "422"} or (code.startswith("4") and _4XX_BAD_INPUT_RE.search(sec.title)))
                ):
                    four_xx_invalid_code = code
                    if sec_blks:
                        four_xx_invalid_block = sec_blks[0]

        # 2) Status list lines and status tables.
        in_status_table = False
        status_table_sec: str | None = None
        for blk in part_blocks:
            if blk.kind == BlockKind.TABLE_HEADER:
                headers = _get_cells(blk)
                sec_ctx = " ".join(blk.section_path)
                in_status_table = bool(
                    headers
                    and (
                        _STATUS_TABLE_HEADER_RE.match(headers[0].strip())
                        or _STATUS_SECTION_RE.search(sec_ctx)
                    )
                )
                status_table_sec = blk.section_id
                continue

            if blk.kind == BlockKind.TABLE_ROW:
                if blk.section_id != status_table_sec:
                    in_status_table = False
                cells = _get_cells(blk)
                if in_status_table and len(cells) >= 2:
                    m_code = _STATUS_CODE_EXACT_RE.match(cells[0].strip())
                    if m_code and m_code.group(1) in _HTTP_STATUS_CODES:
                        code = m_code.group(1)
                        declared_codes.add(code)
                        declared_block_by_code.setdefault(code, blk)
                        decl_line_blocks.add(blk.id)
                        if four_xx_invalid_block is None and (
                            code in {"400", "422"}
                            or (code.startswith("4") and _4XX_BAD_INPUT_RE.search(blk.text))
                        ):
                            four_xx_invalid_code = code
                            four_xx_invalid_block = blk
                continue

            in_status_table = False
            if blk.kind not in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM):
                continue
            for raw_line in blk.raw.splitlines():
                m_line = _STATUS_DECL_LINE_RE.match(raw_line)
                if m_line and m_line.group(1) in _HTTP_STATUS_CODES:
                    code = m_line.group(1)
                    declared_codes.add(code)
                    declared_block_by_code.setdefault(code, blk)
                    decl_line_blocks.add(blk.id)
                    if four_xx_invalid_block is None and (
                        code in {"400", "422"}
                        or (code.startswith("4") and _4XX_BAD_INPUT_RE.search(raw_line))
                    ):
                        four_xx_invalid_code = code
                        four_xx_invalid_block = blk

        # 2a) Undeclared status codes referenced in prose (only when a declared status list exists).
        if len(declared_codes) >= 2:
            reported_missing: set[str] = set()
            for blk in part_blocks:
                if blk.kind in (BlockKind.CODE, BlockKind.TABLE_HEADER, BlockKind.HEADING):
                    continue
                if blk.id in decl_line_blocks:
                    continue
                for m_ref in _STATUS_REF_RE.finditer(blk.text):
                    group_text = next((g for g in m_ref.groups() if g is not None), "")
                    if not group_text:
                        continue
                    for m_num in re.finditer(r"[1-5]\d{2}", group_text):
                        code = m_num.group(0)
                        if code not in _HTTP_STATUS_CODES:
                            continue
                        # Guard against quantities like "500 ₸" or "200 мс".
                        abs_end = m_ref.start() + m_ref.group(0).find(group_text) + m_num.end()
                        if _UNIT_AFTER_NUM_RE.match(blk.text[abs_end:]):
                            continue
                        if code in declared_codes or code in reported_missing:
                            continue
                        reported_missing.add(code)
                        sec_label = _section_label(blk)
                        declared_str = ", ".join(sorted(declared_codes))
                        # the text sits under the heading of a declared code but prescribes another one:
                        # the same case gets two different codes
                        heading = blk.section_path[-1] if blk.section_path else ""
                        own = [c for c in re.findall(r"[1-5]\d{2}", heading) if c in declared_codes and c != code]
                        if own:
                            x = own[0]
                            ids = [blk.id]
                            decl_blk = declared_block_by_code.get(x)
                            if decl_blk is not None and decl_blk.id != blk.id:
                                ids.insert(0, decl_blk.id)
                            findings.append(
                                _make_finding(
                                    rule="api_status_code_conflict",
                                    block_ids=ids,
                                    criterion=Criterion.SET_CONSISTENT,
                                    severity=Severity.CRITICAL,
                                    set_check=SetCheck.CONTRADICTION,
                                    quote=_quote_for_span(blk.text, m_ref.start(), m_ref.end()),
                                    explanation=(
                                        f"Перечень статусов объявляет код {x} для этого случая, но в подразделе "
                                        f"«{heading}» предписано возвращать {code}, которого нет в перечне "
                                        f"({declared_str}). Для одного и того же запроса документ требует два "
                                        f"разных кода: {x} и {code}."
                                    ),
                                    recommendation=(
                                        f"Выбрать один код для этого случая ({x} или {code}) и привести к нему "
                                        "перечень статусов, подраздел и пример ответа."
                                    ),
                                    clarifying_question=(
                                        f"Какой код возвращается в этом случае: {x}, как в перечне статусов, "
                                        f"или {code}, как в тексте подраздела?"
                                    ),
                                )
                            )
                            continue
                        findings.append(
                            _make_finding(
                                rule="api_undeclared_status_code",
                                block_ids=[blk.id],
                                criterion=Criterion.SET_CONSISTENT,
                                severity=Severity.MAJOR,
                                set_check=SetCheck.CONTRADICTION,
                                quote=_quote_for_span(blk.text, m_ref.start(), m_ref.end()),
                                explanation=(
                                    f"В разделе «{sec_label}» упоминается HTTP-статус {code}, "
                                    f"однако в перечне объявленных статусов ({declared_str}) он отсутствует."
                                ),
                                recommendation=(
                                    f"Добавить описание и пример ответа для кода {code} в перечень "
                                    "статусов либо привести текст к одному из объявленных кодов."
                                ),
                                clarifying_question=(
                                    f"Должен ли метод действительно возвращать HTTP-статус {code} "
                                    f"(и в каком формате тела), или следует использовать один из кодов "
                                    f"({declared_str})?"
                                ),
                            )
                        )

        # 2b) Success (2xx) example containing an error payload while a 4xx invalid-input code is declared.
        if four_xx_invalid_code is not None:
            for blk in part_blocks:
                if blk.kind != BlockKind.CODE or not blk.section_path:
                    continue
                heading = blk.section_path[-1]
                m_2xx = _2XX_RE.search(heading)
                if not m_2xx:
                    continue
                err_marker = _is_error_json_body(blk.text)
                if err_marker is None:
                    continue
                code_2xx = m_2xx.group(1)
                block_ids = [blk.id]
                if four_xx_invalid_block is not None and four_xx_invalid_block.id != blk.id:
                    block_ids.append(four_xx_invalid_block.id)
                findings.append(
                    _make_finding(
                        rule="api_success_code_error_body",
                        block_ids=block_ids,
                        criterion=Criterion.SET_CONSISTENT,
                        severity=Severity.CRITICAL,
                        set_check=SetCheck.CONTRADICTION,
                        quote=_quote_for_span(blk.text),
                        explanation=(
                            f"Пример ответа под успешным заголовком «{heading}» ({code_2xx}) "
                            f"содержит признак ошибки ({err_marker}), тогда как для некорректных "
                            f"параметров в спецификации отдельно объявлен код {four_xx_invalid_code}."
                        ),
                        recommendation=(
                            f"Устранить противоречие: возвращать ошибки валидации с кодом "
                            f"{four_xx_invalid_code} и единым форматом ошибки либо чётко разграничить "
                            f"условия для {code_2xx} и {four_xx_invalid_code}."
                        ),
                        clarifying_question=(
                            f"Какой HTTP-статус ({code_2xx} с телом ошибки или {four_xx_invalid_code}) "
                            "должен возвращать сервис при некорректных входных данных?"
                        ),
                    )
                )

        # 2c) Status heading that mixes two different HTTP status codes (e.g. "200. Successful — Created").
        for sec in part_sections:
            m_start = _STATUS_HEADING_START_RE.match(sec.title.strip())
            if not m_start or m_start.group(1) not in _HTTP_STATUS_CODES:
                continue
            declared_code = m_start.group(1)
            tail = sec.title[m_start.end() :]
            conflict_desc: str | None = None

            # Check for a second distinct HTTP code in the same heading.
            for m_other in re.finditer(r"(?<!\w)([1-5]\d{2})(?!\w)", tail):
                other_code = m_other.group(1)
                if other_code in _HTTP_STATUS_CODES and other_code != declared_code:
                    conflict_desc = f"коды {declared_code} и {other_code}"
                    break

            # Check for an RFC 9110 reason phrase belonging to a different status code.
            if conflict_desc is None:
                tail_low = tail.lower()
                for phrase, expected_code in _STATUS_REASON_PHRASES:
                    if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", tail_low):
                        if expected_code != declared_code:
                            conflict_desc = (
                                f"код {declared_code} и формулировка «{phrase}» "
                                f"(соответствует коду {expected_code})"
                            )
                        break

            if conflict_desc is None:
                continue

            sec_blks = blocks_by_sec.get(sec.id, [])
            heading_blk = next((b for b in sec_blks if b.kind == BlockKind.HEADING), None)
            anchor_blk = heading_blk or (sec_blks[0] if sec_blks else declared_block_by_code.get(declared_code))
            if anchor_blk is None:
                continue

            findings.append(
                _make_finding(
                    rule="api_status_heading_mismatch",
                    block_ids=[anchor_blk.id],
                    criterion=Criterion.UNAMBIGUOUS,
                    severity=Severity.MAJOR,
                    set_check=SetCheck.CONTRADICTION,
                    quote=_quote_for_span(anchor_blk.text),
                    explanation=(
                        f"Заголовок «{sec.title}» смешивает два разных HTTP-статуса: {conflict_desc}, "
                        "что создаёт неоднозначность ожидаемого кода ответа."
                    ),
                    recommendation=(
                        "Привести заголовок и описание к одному конкретному HTTP-статусу "
                        "с соответствующей стандартной формулировкой RFC 9110."
                    ),
                    clarifying_question=(
                        f"Какой именно HTTP-статус должен возвращаться в сценарии «{sec.title}»?"
                    ),
                )
            )

    return findings


# ---------------------------------------------------------------- check 3: request vs response type mismatch


def _canonical_type(declared_type: str) -> str:
    t = declared_type.strip().lower()
    t = re.sub(r"\s*\?$|\s*\|\s*null$", "", t).strip()
    base = re.split(r"[\(\[<]", t, maxsplit=1)[0].strip()

    if base in {"string", "str", "text", "varchar", "char", "строка", "строки", "строковый", "строковой", "строковая", "строковое", "текст"}:
        return "string"
    if base in {"integer", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "long", "bigint", "smallint", "short", "целое", "целый", "целое число"}:
        return "integer"
    if base in {"number", "numeric", "decimal", "float", "float32", "float64", "double", "число", "числовой", "числовое", "дробное", "дробный", "вещественное", "вещественный", "десятичное", "десятичный"} or "число" in base:
        return "number"
    if base in {"boolean", "bool", "булево", "булевое", "булевый", "булевая", "логический", "логическое", "логическая"}:
        return "boolean"
    if base in {"date", "дата"}:
        return "date"
    if base in {"datetime", "timestamp", "дата и время"}:
        return "datetime"
    if base in {"time", "время"}:
        return "time"
    if base in {"uuid", "guid"}:
        return "uuid"
    if base in {"array", "list", "массив", "список"} or t.endswith("[]"):
        return "array"
    if base in {"object", "json", "map", "dict", "record", "struct", "объект", "словарь"}:
        return "object"
    if base in {"binary", "bytes", "file", "файл"}:
        return "binary"
    if base in {"enum", "перечисление"}:
        return "enum"
    return base


def _section_role(section_path: list[str]) -> str | None:
    """Classify a block's section path as ``request`` or ``response`` (innermost heading wins)."""
    for title in reversed(section_path):
        is_req = bool(_REQ_SECTION_ROLE_RE.search(title))
        is_resp = bool(_RESP_SECTION_ROLE_RE.search(title))
        if is_req and not is_resp:
            return "request"
        if is_resp and not is_req:
            return "response"
    return None


def _check_request_response_type_mismatch(
    doc: Document,
    inventory: list[InventoryItem],
) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    block_by_id = {b.id: b for b in doc.blocks}

    typed_entries: list[tuple[str, str, str, str, Block]] = []  # (role, field, canon, raw, blk)
    req_parts: set[str] = set()
    resp_parts: set[str] = set()

    for item in inventory:
        if item.kind != "identifier" or not item.unit:
            continue
        blk = block_by_id.get(item.block_id)
        if blk is None:
            continue
        role = _section_role(blk.section_path)
        if role is None:
            continue
        typed_entries.append((role, item.value, _canonical_type(item.unit), item.unit, blk))
        if role == "request":
            req_parts.add(blk.part)
        else:
            resp_parts.add(blk.part)

    # Scope by part when multiple parts each contain both request and response tables;
    # in standalone single-API docs ("## Параметры запроса" vs "## Ответ"), compare document-wide.
    shared_parts = req_parts & resp_parts
    scope_by_part = len(shared_parts) >= 2 or (len(shared_parts) == 1 and len(req_parts | resp_parts) > 1)

    req_fields: dict[tuple[str, str], tuple[str, str, Block]] = {}
    resp_fields: dict[tuple[str, str], tuple[str, str, Block]] = {}
    for role, field_name, canon_type, raw_type, blk in typed_entries:
        scope_key = blk.part if scope_by_part else ""
        key = (scope_key, field_name)
        entry = (canon_type, raw_type, blk)
        if role == "request":
            req_fields.setdefault(key, entry)
        else:
            resp_fields.setdefault(key, entry)

    for key, (req_canon, req_raw, req_blk) in req_fields.items():
        resp_entry = resp_fields.get(key)
        if resp_entry is None:
            continue
        resp_canon, resp_raw, resp_blk = resp_entry
        if req_canon == resp_canon:
            continue

        _scope, field_name = key
        req_sec = _section_label(req_blk)
        resp_sec = _section_label(resp_blk)
        findings.append(
            _make_finding(
                rule="api_req_resp_type_mismatch",
                block_ids=[req_blk.id, resp_blk.id],
                criterion=Criterion.SET_CONSISTENT,
                severity=Severity.MAJOR,
                set_check=SetCheck.CONTRADICTION,
                quote=_quote_for_span(req_blk.text),
                explanation=(
                    f"Поле «{field_name}» имеет разные типы данных в запросе и ответе: "
                    f"в разделе «{req_sec}» указан тип «{req_raw}», а в разделе «{resp_sec}» — «{resp_raw}»."
                ),
                recommendation=(
                    f"Привести тип поля «{field_name}» к единому формату в запросе и ответе "
                    "либо явно обосновать и задокументировать преобразование типа."
                ),
                clarifying_question=(
                    f"Какой единый тип данных ({req_raw} или {resp_raw}) должен использоваться "
                    f"для поля «{field_name}» в запросе и ответе?"
                ),
            )
        )

    return findings


# ---------------------------------------------------------------- check 4: unused body w/ Content-Type or GET body


def _check_body_and_content_type(doc: Document, decls: list[_MethodDecl]) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    blocks_by_sec: dict[str, list[Block]] = defaultdict(list)
    for blk in doc.blocks:
        blocks_by_sec[blk.section_id].append(blk)

    # 4a) Body section declares Content-Type while also stating that the body is not used.
    unused_body_sections: set[str] = set()
    for sec in doc.sections:
        if not _BODY_SECTION_RE.search(sec.title):
            continue
        sec_blks = [b for b in blocks_by_sec.get(sec.id, []) if b.kind != BlockKind.HEADING]
        ct_blk = next((b for b in sec_blks if _CONTENT_TYPE_RE.search(b.text)), None)
        unused_blk = next((b for b in sec_blks if _BODY_UNUSED_RE.search(b.text)), None)
        if unused_blk is not None:
            unused_body_sections.add(sec.id)
        if ct_blk is not None and unused_blk is not None:
            block_ids = [ct_blk.id]
            if unused_blk.id != ct_blk.id:
                block_ids.append(unused_blk.id)
            sec_label = " › ".join(sec.path) if sec.path else sec.title
            findings.append(
                _make_finding(
                    rule="api_body_content_type_unused",
                    block_ids=block_ids,
                    criterion=Criterion.SET_CONSISTENT,
                    severity=Severity.MINOR,
                    set_check=SetCheck.CONTRADICTION,
                    quote=_quote_for_span(ct_blk.text),
                    explanation=(
                        f"В разделе «{sec_label}» одновременно указан заголовок Content-Type "
                        "и отмечено, что тело запроса не используется."
                    ),
                    recommendation=(
                        "Убрать избыточный заголовок Content-Type при отсутствии тела запроса "
                        "либо описать передаваемое тело."
                    ),
                    clarifying_question=(
                        f"Требуется ли передавать заголовок Content-Type в разделе «{sec_label}», "
                        "если тело не используется?"
                    ),
                )
            )

    # 4b) GET/HEAD endpoint declaring an active JSON body (code block or parameter table).
    reported_get_body: set[str] = set()
    for decl in decls:
        if not _SAFE_METHOD_EXACT_RE.match(decl.method):
            continue
        for sec in decl.scope_sections:
            if not _REQ_BODY_SECTION_RE.search(sec.title) or sec.id in unused_body_sections:
                continue
            if sec.id in reported_get_body:
                continue
            sec_blks = [b for b in blocks_by_sec.get(sec.id, []) if b.kind != BlockKind.HEADING]
            body_blk = next(
                (
                    b
                    for b in sec_blks
                    if (b.kind == BlockKind.CODE and b.text.strip().startswith(("{", "[")))
                    or b.kind == BlockKind.TABLE_ROW
                ),
                None,
            )
            if body_blk is None:
                continue
            reported_get_body.add(sec.id)
            sec_label = " › ".join(sec.path) if sec.path else sec.title
            findings.append(
                _make_finding(
                    rule="api_get_with_body",
                    block_ids=[body_blk.id, decl.block.id],
                    criterion=Criterion.CORRECT,
                    severity=Severity.MINOR,
                    set_check=SetCheck.CONTRADICTION,
                    quote=_quote_for_span(body_blk.text),
                    explanation=(
                        f"Для HTTP-метода {decl.method} в разделе «{sec_label}» описано тело запроса, "
                        "что противоречит семантике RFC 9110 и может отбрасываться клиентами и прокси."
                    ),
                    recommendation=(
                        f"Перенести параметры в query/path либо заменить метод {decl.method} на POST/PUT/PATCH."
                    ),
                    clarifying_question=(
                        f"Следует ли заменить метод {decl.method} на POST/PUT или убрать тело запроса "
                        f"в разделе «{sec_label}»?"
                    ),
                )
            )

    return findings


def run_api_lint(doc: Document, inventory: list[InventoryItem]) -> list[RuleFinding]:
    """Run conservative, domain-neutral REST/API checks on a structured document."""
    decls = _collect_method_declarations(doc)
    findings: list[RuleFinding] = []
    findings.extend(_check_get_write_side_effects(decls))
    findings.extend(_check_status_consistency(doc))
    findings.extend(_check_request_response_type_mismatch(doc, inventory))
    findings.extend(_check_body_and_content_type(doc, decls))
    return findings
