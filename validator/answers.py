"""Author answers YAML template generation, YAML parsing, and before/after finding diff."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from validator.models import (
    AuthorAnswer,
    BaseFinding,
    Finding,
    FindingDelta,
    FindingDeltaStatus,
    Question,
    QuestionEffectRow,
    RefinementSummary,
    Report,
    SetCheck,
    SetFinding,
    Severity,
    Verdict,
    defect_class_for,
)
from validator.pipeline.quotes import normalize, quotes_overlap
from validator.report.labels import severity_label

if TYPE_CHECKING:
    from validator.pipeline import postprocess as pp

_META_KEYS = {"document", "document_sha256", "report_sha256", "document_name", "schema_version"}
QUESTION_SIMILARITY_THRESHOLD = 0.6


# ---------------------------------------------------------------- YAML serializer


def _yaml_quote(text: str) -> str:
    """Emit a double-quoted YAML 1.2 scalar string."""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return json.dumps(normalized, ensure_ascii=False)


def _question_quotes(
    q: Question,
    finding_by_id: dict[str, BaseFinding],
    limit: int = 4,
) -> list[str]:
    """Collect distinct quoted fragments from findings linked to `q`."""
    quotes: list[str] = []
    seen: set[str] = set()
    for fid in q.finding_ids:
        finding = finding_by_id.get(fid)
        if finding is None:
            continue
        for ev in finding.evidence:
            cleaned = " ".join((ev.quote or "").split()).strip()
            if not cleaned:
                continue
            key = normalize(cleaned)
            if not key or key in seen:
                continue
            seen.add(key)
            quotes.append(cleaned)
            if len(quotes) >= limit:
                return quotes
    return quotes


def _question_section(q: Question, finding_by_id: dict[str, BaseFinding]) -> str:
    if q.section:
        return q.section
    for fid in q.finding_ids:
        finding = finding_by_id.get(fid)
        if finding is not None and finding.section:
            return finding.section
    return ""


def render_questions_template(report: Report) -> str:
    """Render a human-editable YAML template with every report question and an empty `answer:` field."""
    lines = [
        "# Шаблон ответов автора ТЗ на уточняющие вопросы из отчёта.",
        "# Заполните поле `answer:` нужным текстом (можно выбрать вариант из `options` или написать свой ответ).",
        "# Не удаляйте поле `text:`: при повторном запуске ответы сверяются с вопросами из `--previous` по ID и тексту вопроса.",
        "# Вопросы с пустым полем `answer:` при повторном запуске пропускаются.",
        f"document: {_yaml_quote(report.run.document_title or report.run.document_name)}",
        f"document_sha256: {_yaml_quote(report.run.document_sha256)}",
    ]
    if not report.questions:
        lines.append("questions: []")
        return "\n".join(lines) + "\n"

    finding_by_id: dict[str, BaseFinding] = {
        **{f.id: f for f in report.findings},
        **{sf.id: sf for sf in report.set_findings},
    }

    lines.append("questions:")
    for q in report.questions:
        lines.append(f"  - id: {q.id}")
        lines.append(f"    blocking: {'true' if q.blocking else 'false'}")
        if q.finding_ids:
            lines.append("    finding_ids:")
            for fid in q.finding_ids:
                lines.append(f"      - {fid}")
        else:
            lines.append("    finding_ids: []")
        if q.requirement_ids:
            lines.append("    requirement_ids:")
            for rid in q.requirement_ids:
                lines.append(f"      - {rid}")
        section = _question_section(q, finding_by_id)
        if section:
            lines.append(f"    section: {_yaml_quote(section)}")
        quotes = _question_quotes(q, finding_by_id)
        if quotes:
            lines.append("    quotes:")
            for quote in quotes:
                lines.append(f"      - {_yaml_quote(quote)}")
        else:
            lines.append("    quotes: []")
        lines.append(f"    text: {_yaml_quote(q.text)}")
        if q.options:
            lines.append("    options:")
            for opt in q.options:
                lines.append(f"      - {_yaml_quote(opt)}")
        else:
            lines.append("    options: []")
        if q.option_consequences:
            lines.append("    option_consequences:")
            for cons in q.option_consequences:
                lines.append(f"      - {_yaml_quote(cons)}")
        lines.append('    answer: ""')
    return "\n".join(lines) + "\n"


def load_previous_report(report_path: Path) -> Report:
    """Load and validate a previous JSON report."""
    path = Path(report_path)
    if path.is_dir():
        raise ValueError(f"Указана папка вместо JSON-отчёта: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Файл предыдущего отчёта не найден: {path}")
    raw = path.read_text(encoding="utf-8-sig")
    try:
        return Report.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"Файл {path} не является корректным JSON-отчётом валидатора: {exc.errors()[0]['msg']}") from exc


def write_questions_template(report_path: Path, out_path: Path) -> int:
    """Read `<report.json>` and write `<answers.yaml>` template. Returns number of questions written."""
    report = load_previous_report(report_path)
    out = Path(out_path)
    if out.is_dir():
        raise ValueError(f"{out} — это папка; укажите путь к файлу шаблона .yaml.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_questions_template(report), encoding="utf-8")
    return len(report.questions)


# ---------------------------------------------------------------- pure-Python YAML parser


def _strip_inline_comment(line: str) -> str:
    """Strip an unquoted `# ...` comment from a YAML line."""
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(line):
        if in_double:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
        elif in_single:
            if ch == "'":
                in_single = False
        else:
            if ch == '"':
                in_double = True
            elif ch == "'":
                in_single = True
            elif ch == "#" and (idx == 0 or line[idx - 1] in (" ", "\t")):
                return line[:idx].rstrip()
    return line.rstrip()


def _split_mapping_entry(text: str) -> tuple[str, str] | None:
    """Split `key: value` at the first unquoted colon followed by whitespace or EOL."""
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_double:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
        elif in_single:
            if ch == "'":
                in_single = False
        else:
            if ch == '"':
                in_double = True
            elif ch == "'":
                in_single = True
            elif ch in ("[", "{") and idx == 0:
                return None
            elif ch == ":" and (idx + 1 == len(text) or text[idx + 1] in (" ", "\t")):
                key = _parse_scalar(text[:idx].strip())
                if not isinstance(key, str) or not key:
                    return None
                return key, text[idx + 1 :].strip()
    return None


def _parse_flow_list(raw: str) -> list[Any]:
    inner = raw[1:-1].strip()
    if not inner:
        return []
    items: list[Any] = []
    cur: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    depth = 0
    for ch in inner:
        if in_double:
            cur.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
        elif in_single:
            cur.append(ch)
            if ch == "'":
                in_single = False
        else:
            if ch == '"':
                in_double = True
                cur.append(ch)
            elif ch == "'":
                in_single = True
                cur.append(ch)
            elif ch in ("[", "{"):
                depth += 1
                cur.append(ch)
            elif ch in ("]", "}"):
                depth = max(0, depth - 1)
                cur.append(ch)
            elif ch == "," and depth == 0:
                token = "".join(cur).strip()
                if token:
                    items.append(_parse_scalar(token))
                cur = []
            else:
                cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        items.append(_parse_scalar(tail))
    return items


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s or s in ("null", "Null", "NULL", "~"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        return s[1:-1].replace("''", "'")
    if s.startswith("[") and s.endswith("]"):
        return _parse_flow_list(s)
    if s == "{}":
        return {}
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def _consume_block_scalar(lines: list[str], start_idx: int, parent_indent: int, indicator: str) -> tuple[str, int]:
    style = indicator[0]  # '|' or '>'
    chomp = indicator[1:]  # '', '-', '+'
    collected: list[str] = []
    block_indent: int | None = None
    idx = start_idx
    while idx < len(lines):
        raw_line = lines[idx]
        if not raw_line.strip():
            collected.append("")
            idx += 1
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent <= parent_indent:
            break
        if block_indent is None:
            block_indent = indent
        collected.append(raw_line[min(indent, block_indent) :])
        idx += 1

    while collected and collected[-1] == "" and chomp != "+":
        collected.pop()

    if style == "|":
        result = "\n".join(collected)
    else:
        # Folded scalar: single line breaks become spaces, empty lines preserve paragraphs.
        parts: list[str] = []
        for i, line in enumerate(collected):
            if line == "":
                parts.append("\n")
            elif i > 0 and collected[i - 1] != "":
                parts.append(" " + line)
            else:
                parts.append(line)
        result = "".join(parts)

    if chomp == "-":
        result = result.rstrip("\n")
    return result, idx


def parse_yaml_document(text: str) -> Any:
    """Parse a YAML 1.2 document (mappings, lists, quoted/plain/block scalars, comments) without external deps."""
    cleaned = (text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    stripped = cleaned.strip()
    if not stripped:
        return {}
    if stripped[0] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    raw_lines = cleaned.split("\n")
    tokens: list[tuple[int, int, str]] = []  # (line_no, indent, content)
    idx = 0
    # Keep raw_lines indices mapped so block scalars preserve exact blank lines and comments inside blocks.
    n = len(raw_lines)

    def next_significant(pos: int) -> tuple[int, int, str] | None:
        while pos < n:
            raw = raw_lines[pos]
            if "\t" in raw[: len(raw) - len(raw.lstrip(" \t"))]:
                raise ValueError(f"Строка {pos + 1}: табуляция в отступе YAML не поддерживается")
            content = _strip_inline_comment(raw).lstrip(" ")
            if not content or content in ("---", "..."):
                pos += 1
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            return pos, indent, content
        return None

    def parse_node(pos: int, min_indent: int) -> tuple[Any, int]:
        nxt = next_significant(pos)
        if nxt is None:
            return None, n
        line_idx, indent, content = nxt
        if indent < min_indent:
            return None, line_idx
        if content == "-" or content.startswith("- "):
            return parse_seq(line_idx, indent)
        if _split_mapping_entry(content) is not None:
            return parse_map(line_idx, indent)
        return _parse_scalar(content), line_idx + 1

    def parse_seq(pos: int, seq_indent: int) -> tuple[list[Any], int]:
        out: list[Any] = []
        cur = pos
        while True:
            nxt = next_significant(cur)
            if nxt is None:
                return out, n
            line_idx, indent, content = nxt
            if indent < seq_indent:
                return out, line_idx
            if indent > seq_indent:
                raise ValueError(f"Строка {line_idx + 1}: неожиданный отступ внутри списка YAML")
            if not (content == "-" or content.startswith("- ")):
                return out, line_idx
            rest = content[2:].lstrip(" ") if content.startswith("- ") else ""
            item_indent = indent + 2
            if not rest:
                val, cur = parse_node(line_idx + 1, indent + 1)
                out.append(val)
                continue
            if rest in ("|", "|-", "|+", ">", ">-", ">+"):
                val, cur = _consume_block_scalar(raw_lines, line_idx + 1, indent, rest)
                out.append(val)
                continue
            entry = _split_mapping_entry(rest)
            if entry is not None:
                key, val_str = entry
                mapping: dict[str, Any] = {}
                cur = line_idx + 1
                if val_str in ("|", "|-", "|+", ">", ">-", ">+"):
                    mapping[key], cur = _consume_block_scalar(raw_lines, cur, item_indent, val_str)
                elif val_str:
                    mapping[key] = _parse_scalar(val_str)
                else:
                    sub = next_significant(cur)
                    if sub is not None and sub[1] > item_indent:
                        mapping[key], cur = parse_node(cur, sub[1])
                    elif sub is not None and sub[1] == item_indent and (sub[2] == "-" or sub[2].startswith("- ")):
                        mapping[key], cur = parse_seq(sub[0], sub[1])
                    else:
                        mapping[key] = None
                # Read remaining keys belonging to this mapping item.
                rest_map, cur = parse_map(cur, item_indent)
                mapping.update(rest_map)
                out.append(mapping)
                continue
            out.append(_parse_scalar(rest))
            cur = line_idx + 1

    def parse_map(pos: int, map_indent: int) -> tuple[dict[str, Any], int]:
        out: dict[str, Any] = {}
        cur = pos
        while True:
            nxt = next_significant(cur)
            if nxt is None:
                return out, n
            line_idx, indent, content = nxt
            if indent < map_indent:
                return out, line_idx
            if indent > map_indent:
                raise ValueError(f"Строка {line_idx + 1}: неожиданный отступ внутри словаря YAML")
            if content == "-" or content.startswith("- "):
                return out, line_idx
            entry = _split_mapping_entry(content)
            if entry is None:
                raise ValueError(f"Строка {line_idx + 1}: ожидалась пара «ключ: значение», получено {content!r}")
            key, val_str = entry
            cur = line_idx + 1
            if val_str in ("|", "|-", "|+", ">", ">-", ">+"):
                out[key], cur = _consume_block_scalar(raw_lines, cur, indent, val_str)
            elif val_str:
                out[key] = _parse_scalar(val_str)
            else:
                sub = next_significant(cur)
                if sub is not None and sub[1] > indent:
                    out[key], cur = parse_node(cur, sub[1])
                elif sub is not None and sub[1] == indent and (sub[2] == "-" or sub[2].startswith("- ")):
                    out[key], cur = parse_seq(sub[0], sub[1])
                else:
                    out[key] = None
        return out, cur

    _ = tokens
    root, next_pos = parse_node(idx, 0)
    leftover = next_significant(next_pos)
    if leftover is not None:
        raise ValueError(f"Строка {leftover[0] + 1}: лишние данные после конца YAML-документа")
    return root if root is not None else {}


# ---------------------------------------------------------------- loading answers


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _as_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Да" if value else "Нет"
    if isinstance(value, list):
        return "; ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def normalize_question_text(text: str) -> str:
    """Normalize question text for comparison: lowercase, NFKC, ё->е, punctuation stripped."""
    return " ".join(re.findall(r"[0-9a-zа-я]+", normalize(text or "")))


def question_similarity(a: str, b: str) -> float:
    """Return normalized similarity in [0.0, 1.0] between two clarifying questions."""
    na, nb = normalize_question_text(a), normalize_question_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq_sim = SequenceMatcher(autojunk=False, a=na, b=nb).ratio()
    wa = {w for w in na.split() if len(w) > 1}
    wb = {w for w in nb.split() if len(w) > 1}
    tok_sim = len(wa & wb) / len(wa | wb) if (wa and wb) else 0.0
    return max(seq_sim, tok_sim)


def _short_label(text: str, limit: int = 60) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip(" ,.;:—-") + "…"


def load_answers(answers_path: Path, previous_report: Report | None = None) -> tuple[list[AuthorAnswer], list[str]]:
    """Load non-empty author answers from YAML and match them safely against `previous_report`.

    When `previous_report` is supplied, each answer is matched by ID only if its stored question text
    matches the target question (normalized similarity >= 0.6). If the ID points to a different
    question or does not exist, matching falls back to text similarity across unclaimed questions.
    Unmatched or unverifiable answers are reported in `warnings` and ignored.
    """
    path = Path(answers_path)
    if path.is_dir():
        raise ValueError(f"Указана папка вместо файла ответов: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Файл ответов не найден: {path}")

    raw_text = path.read_text(encoding="utf-8-sig")
    data = parse_yaml_document(raw_text)

    raw_entries: list[dict[str, Any]] | None = None
    if isinstance(data, list):
        raw_entries = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("questions"), list):
            raw_entries = [item for item in data["questions"] if isinstance(item, dict)]
        elif isinstance(data.get("answers"), list):
            raw_entries = [item for item in data["answers"] if isinstance(item, dict)]
        elif isinstance(data.get("answers"), dict):
            raw_entries = [
                {"id": k, **v} if isinstance(v, dict) else {"id": k, "answer": v}
                for k, v in data["answers"].items()
            ]
        else:
            raw_entries = [
                {"id": k, "answer": v}
                for k, v in data.items()
                if k not in _META_KEYS and not isinstance(v, (dict, list))
            ]
    if raw_entries is None:
        raise ValueError(f"Файл ответов {path} должен содержать YAML-словарь или список вопросов.")

    parsed_entries: list[tuple[int, str, str, str, dict[str, Any]]] = []
    for idx, entry in enumerate(raw_entries):
        answer_text = _as_answer_text(entry.get("answer"))
        if not answer_text:
            continue
        qid_raw = str(entry.get("id") or entry.get("question_id") or "").strip()
        stored_text = str(entry.get("text") or entry.get("question") or entry.get("question_text") or "").strip()
        parsed_entries.append((idx, qid_raw, stored_text, answer_text, entry))

    if previous_report is None:
        answers: list[AuthorAnswer] = []
        warnings: list[str] = []
        seen_qids: set[str] = set()
        for seq_idx, (_idx, qid_raw, stored_text, answer_text, entry) in enumerate(parsed_entries, 1):
            if not qid_raw and not stored_text:
                warnings.append(f"Запись #{seq_idx} в файле ответов не содержит ни id, ни текста вопроса и проигнорирована.")
                continue
            qid = qid_raw or f"Q-{seq_idx:03d}"
            if qid.upper() in seen_qids:
                warnings.append(f"Повторный ответ для вопроса {qid} в файле ответов проигнорирован.")
                continue
            seen_qids.add(qid.upper())
            answers.append(
                AuthorAnswer(
                    question_id=qid,
                    question_text=stored_text,
                    answer=answer_text,
                    finding_ids=_as_str_list(entry.get("finding_ids")),
                    requirement_ids=_as_str_list(entry.get("requirement_ids")),
                )
            )
        return answers, warnings

    prev_questions = previous_report.questions
    by_qid = {q.id.upper(): q for q in prev_questions}
    matched: dict[int, Question] = {}
    claimed_qids: set[str] = set()
    duplicate_entries: dict[int, str] = {}

    # Pass 1: exact ID match verified by normalized text similarity >= threshold.
    for idx, qid_raw, stored_text, _answer_text, _entry in parsed_entries:
        if not qid_raw or not stored_text:
            continue
        q_obj = by_qid.get(qid_raw.upper())
        if q_obj is None:
            continue
        if question_similarity(stored_text, q_obj.text) >= QUESTION_SIMILARITY_THRESHOLD:
            key = q_obj.id.upper()
            if key in claimed_qids:
                duplicate_entries[idx] = q_obj.id
            else:
                matched[idx] = q_obj
                claimed_qids.add(key)

    # Pass 2: fallback by text similarity for entries whose ID was reordered, absent, or pointed elsewhere.
    scored: list[tuple[float, int, Question]] = []
    for idx, _qid_raw, stored_text, _answer_text, _entry in parsed_entries:
        if idx in matched or idx in duplicate_entries or not stored_text:
            continue
        for q in prev_questions:
            if q.id.upper() in claimed_qids:
                continue
            sim = question_similarity(stored_text, q.text)
            if sim >= QUESTION_SIMILARITY_THRESHOLD:
                scored.append((sim, idx, q))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].id))
    for _sim, idx, q in scored:
        key = q.id.upper()
        if idx in matched or key in claimed_qids:
            continue
        matched[idx] = q
        claimed_qids.add(key)

    # Pass 3: assemble matched answers in input order; report and ignore unmatched entries.
    answers = []
    warnings = []
    for idx, qid_raw, stored_text, answer_text, _entry in parsed_entries:
        q_obj = matched.get(idx)
        if q_obj is not None:
            answers.append(
                AuthorAnswer(
                    question_id=q_obj.id,
                    question_text=q_obj.text,
                    answer=answer_text,
                    finding_ids=list(q_obj.finding_ids),
                    requirement_ids=list(q_obj.requirement_ids),
                )
            )
            continue

        if idx in duplicate_entries:
            warnings.append(f"Повторный ответ для вопроса {duplicate_entries[idx]} в файле ответов проигнорирован.")
        elif not stored_text:
            label = qid_raw or f"запись #{idx + 1}"
            warnings.append(
                f"Ответ для {label} проигнорирован: в файле ответов не указан текст вопроса (поле text), "
                "невозможно проверить соответствие вопросу из предыдущего отчёта."
            )
        elif qid_raw and qid_raw.upper() in by_qid:
            warnings.append(
                f"Ответ для {qid_raw} («{_short_label(stored_text)}») проигнорирован: текст вопроса не совпадает "
                f"с {qid_raw} в предыдущем отчёте и не найден среди остальных вопросов."
            )
        else:
            label = qid_raw or f"«{_short_label(stored_text)}»"
            warnings.append(
                f"Ответ для {label} проигнорирован: подходящий вопрос не найден в предыдущем отчёте."
            )

    return answers, warnings


# ---------------------------------------------------------------- previous <-> current diff


def _finding_blocks(f: BaseFinding) -> set[str]:
    return {b for e in f.evidence for b in e.block_ids}


def _finding_reqs(f: BaseFinding) -> set[str]:
    return set(getattr(f, "requirement_ids", []) or [])


def _match_score(prev: BaseFinding, curr: BaseFinding) -> float:
    """Heuristic similarity score between a previous finding and a newly produced finding."""
    from validator.pipeline.postprocess import stem_set

    prev_blocks = _finding_blocks(prev)
    curr_blocks = _finding_blocks(curr)
    shared_blocks = bool(prev_blocks & curr_blocks)
    quote_match = any(
        quotes_overlap(ea.quote, eb.quote, 0.5)
        for ea in prev.evidence
        for eb in curr.evidence
    )
    prev_reqs = _finding_reqs(prev)
    curr_reqs = _finding_reqs(curr)
    shared_reqs = bool(prev_reqs & curr_reqs)
    same_section = bool(prev.section and curr.section and prev.section == curr.section)

    sa, sb = stem_set(prev.explanation), stem_set(curr.explanation)
    expl_sim = len(sa & sb) / max(1, min(len(sa), len(sb))) if (sa and sb) else 0.0
    qa, qb = stem_set(prev.clarifying_question), stem_set(curr.clarifying_question)
    q_sim = len(qa & qb) / max(1, min(len(qa), len(qb))) if (qa and qb) else 0.0

    same_crit = prev.criterion == curr.criterion
    prev_sc = getattr(prev, "set_check", None)
    curr_sc = getattr(curr, "set_check", None)
    same_sc = prev_sc is not None and prev_sc == curr_sc
    same_class = prev.defect_class == curr.defect_class

    if not (shared_blocks or quote_match or shared_reqs or same_section):
        return 0.0

    if (shared_blocks or quote_match) and (same_crit or same_sc or expl_sim >= 0.30 or q_sim >= 0.40):
        return 2.0 + (0.6 if quote_match else 0.0) + (0.4 if same_crit else 0.0) + (0.4 if same_sc else 0.0) + expl_sim
    if shared_reqs and (same_crit or same_sc or same_class) and (expl_sim >= 0.35 or q_sim >= 0.45):
        return 1.2 + (0.4 if same_crit else 0.0) + expl_sim
    if same_section and (same_crit or same_sc) and expl_sim >= 0.50:
        return 1.0 + expl_sim
    return 0.0


def annotate_candidates_with_previous_ids(
    cands: list[pp.Candidate],
    previous_set_findings: list[SetFinding],
) -> None:
    """Attach matching previous set-finding IDs to candidates before review so the critic can reference them."""
    if not previous_set_findings:
        return
    for c in cands:
        if c.previous_ids or not c.is_set:
            continue
        best_id = ""
        best_score = 0.0
        probe = SetFinding(
            id=c.cid or "S-000",
            criterion=c.criterion,
            severity=c.severity,
            defect_class=defect_class_for(c.criterion, c.set_check),
            evidence=c.evidence,
            explanation=c.explanation,
            recommendation=c.recommendation,
            clarifying_question=c.question,
            answer_options=c.options,
            source=c.source,
            set_check=c.set_check or SetCheck.OTHER,
            requirement_ids=c.requirement_ids,
        )
        for prev in previous_set_findings:
            score = _match_score(prev, probe)
            if score > best_score:
                best_score = score
                best_id = prev.id
        if best_id and best_score >= 2.0:
            c.previous_ids.append(best_id)


def _is_changed(prev: BaseFinding, curr: BaseFinding) -> tuple[bool, str]:
    from validator.pipeline.postprocess import stem_set

    reasons: list[str] = []
    if curr.severity != prev.severity:
        reasons.append(f"критичность: {severity_label(prev.severity)} → {severity_label(curr.severity)}")
    if curr.criterion != prev.criterion:
        reasons.append("изменился критерий")
    prev_sc = getattr(prev, "set_check", None)
    curr_sc = getattr(curr, "set_check", None)
    if prev_sc is not None and curr_sc is not None and prev_sc != curr_sc:
        reasons.append("изменился тип расхождения")
    if _finding_reqs(prev) != _finding_reqs(curr):
        reasons.append("изменился состав затронутых требований")
    if curr.review_note and curr.review_note != prev.review_note:
        reasons.append(curr.review_note)
    sa, sb = stem_set(prev.explanation), stem_set(curr.explanation)
    expl_sim = len(sa & sb) / max(1, min(len(sa), len(sb))) if (sa and sb) else 1.0
    if expl_sim < 0.55 and not quotes_overlap(prev.explanation, curr.explanation, 0.65):
        reasons.append("изменилась формулировка дефекта")
    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def _severity_counts(findings: list[BaseFinding]) -> dict[str, int]:
    return {s.value: sum(1 for f in findings if f.severity == s) for s in Severity}


def build_refinement_summary(
    previous_report: Report,
    new_findings: list[Finding],
    new_set_findings: list[SetFinding],
    new_verdict: Verdict,
    answers: list[AuthorAnswer],
    *,
    curr_lineage: dict[str, list[str]] | None = None,
    closed_by_review: dict[str, list[str]] | None = None,
    previous_report_name: str = "",
) -> RefinementSummary:
    """Compare previous report findings against new run results and summarize the effect of author answers."""
    prev_all: list[BaseFinding] = [*previous_report.set_findings, *previous_report.findings]
    curr_all: list[BaseFinding] = [*new_set_findings, *new_findings]
    lineage = curr_lineage or {}
    closed_map = closed_by_review or {}

    matched_prev_to_curr: dict[str, BaseFinding] = {}
    used_curr: set[str] = set()

    # 1. Exact lineage from reused/annotated candidates.
    curr_by_id = {f.id: f for f in curr_all}
    for curr_id, prev_ids in lineage.items():
        curr_obj = curr_by_id.get(curr_id)
        if curr_obj is None:
            continue
        for pid in prev_ids:
            if pid not in matched_prev_to_curr:
                matched_prev_to_curr[pid] = curr_obj
                used_curr.add(curr_id)

    # 2. Content-based greedy 1-to-1 matching for remaining findings (especially freshly re-run set checks).
    scored_pairs: list[tuple[float, str, str]] = []
    for prev in prev_all:
        if prev.id in matched_prev_to_curr:
            continue
        for curr in curr_all:
            if curr.id in used_curr:
                continue
            score = _match_score(prev, curr)
            if score > 0.0:
                scored_pairs.append((score, prev.id, curr.id))
    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _score, pid, cid in scored_pairs:
        if pid in matched_prev_to_curr or cid in used_curr:
            continue
        matched_prev_to_curr[pid] = curr_by_id[cid]
        used_curr.add(cid)

    # Index which answers target which previous findings or requirements.
    answers_by_fid: dict[str, list[str]] = {}
    answers_by_rid: dict[str, list[str]] = {}
    for ans in answers:
        for fid in ans.finding_ids:
            answers_by_fid.setdefault(fid, [])
            if ans.question_id not in answers_by_fid[fid]:
                answers_by_fid[fid].append(ans.question_id)
        for rid in ans.requirement_ids:
            answers_by_rid.setdefault(rid, [])
            if ans.question_id not in answers_by_rid[rid]:
                answers_by_rid[rid].append(ans.question_id)

    deltas: list[FindingDelta] = []
    delta_by_pid: dict[str, FindingDelta] = {}
    for prev in prev_all:
        curr = matched_prev_to_curr.get(prev.id)
        linked_qids: list[str] = []
        for qid in [*closed_map.get(prev.id, []), *answers_by_fid.get(prev.id, [])]:
            if qid not in linked_qids:
                linked_qids.append(qid)
        if not linked_qids:
            for rid in _finding_reqs(prev):
                for qid in answers_by_rid.get(rid, []):
                    if qid not in linked_qids:
                        linked_qids.append(qid)

        from validator.pipeline.postprocess import first_sentence

        short_summary = first_sentence(prev.explanation, 160)
        if curr is None:
            note = (
                f"Закрыто ответом на {', '.join(linked_qids)}"
                if linked_qids
                else "Снято после повторной проверки с учётом ответов автора"
            )
            delta = FindingDelta(
                previous_id=prev.id,
                status=FindingDeltaStatus.CLOSED,
                previous_severity=prev.severity,
                current_id=None,
                current_severity=None,
                summary=short_summary,
                answered_by=linked_qids,
                note=note,
            )
        else:
            changed, reason = _is_changed(prev, curr)
            if changed:
                delta = FindingDelta(
                    previous_id=prev.id,
                    status=FindingDeltaStatus.CHANGED,
                    previous_severity=prev.severity,
                    current_id=curr.id,
                    current_severity=curr.severity,
                    summary=short_summary,
                    answered_by=linked_qids,
                    note=reason[:1].upper() + reason[1:] if reason else "Изменилось после уточнений",
                )
            else:
                id_note = f" (новый ID: {curr.id})" if curr.id != prev.id else ""
                delta = FindingDelta(
                    previous_id=prev.id,
                    status=FindingDeltaStatus.REMAINED,
                    previous_severity=prev.severity,
                    current_id=curr.id,
                    current_severity=curr.severity,
                    summary=short_summary,
                    answered_by=linked_qids,
                    note=f"Остаётся открытым{id_note}",
                )
        deltas.append(delta)
        delta_by_pid[prev.id] = delta

    # Build per-question effect table.
    question_effects: list[QuestionEffectRow] = []
    for ans in answers:
        related_pids: list[str] = [fid for fid in ans.finding_ids if fid in delta_by_pid]
        for d in deltas:
            if ans.question_id in d.answered_by and d.previous_id not in related_pids:
                related_pids.append(d.previous_id)
        closed_ids = [pid for pid in related_pids if delta_by_pid[pid].status == FindingDeltaStatus.CLOSED]
        changed_ids = [pid for pid in related_pids if delta_by_pid[pid].status == FindingDeltaStatus.CHANGED]
        remained_ids = [pid for pid in related_pids if delta_by_pid[pid].status == FindingDeltaStatus.REMAINED]

        effect_parts: list[str] = []
        if closed_ids:
            effect_parts.append(f"закрыто ответом: {', '.join(closed_ids)}")
        if changed_ids:
            changed_desc = []
            for pid in changed_ids:
                d = delta_by_pid[pid]
                if d.current_severity is not None and d.current_severity != d.previous_severity:
                    changed_desc.append(
                        f"{pid} ({severity_label(d.previous_severity)} → {severity_label(d.current_severity)})"
                    )
                else:
                    changed_desc.append(pid)
            effect_parts.append(f"изменилось: {', '.join(changed_desc)}")
        if remained_ids:
            effect_parts.append(f"осталось: {', '.join(remained_ids)}")
        if not effect_parts:
            effect_text = "Учтено в контексте проверки набора и пересмотра"
        else:
            joined = "; ".join(effect_parts)
            effect_text = joined[0].upper() + joined[1:]

        question_effects.append(
            QuestionEffectRow(
                question_id=ans.question_id,
                question_text=ans.question_text,
                answer=ans.answer,
                finding_ids=related_pids,
                closed_ids=closed_ids,
                changed_ids=changed_ids,
                remained_ids=remained_ids,
                effect=effect_text,
            )
        )

    prev_counts = _severity_counts(prev_all)
    curr_counts = _severity_counts(curr_all)
    return RefinementSummary(
        previous_report_name=previous_report_name,
        answers_total=len(answers),
        previous_findings_by_severity=prev_counts,
        current_findings_by_severity=curr_counts,
        previous_findings_total=len(prev_all),
        current_findings_total=len(curr_all),
        previous_verdict_development=previous_report.verdict.development,
        previous_verdict_testing=previous_report.verdict.testing,
        current_verdict_development=new_verdict.development,
        current_verdict_testing=new_verdict.testing,
        previous_verdict_headline=previous_report.verdict.headline,
        current_verdict_headline=new_verdict.headline,
        finding_deltas=deltas,
        question_effects=question_effects,
    )
