"""Rule boundary_tests_missing: numeric ranges and thresholds without acceptance rows at and just past the edge."""

from __future__ import annotations

import re
from dataclasses import dataclass

from validator.models import Block, BlockKind, Criterion, Document, RuleFinding, SetCheck, Severity
from validator.rules.gaps import _make_finding, _quote_for_span

RULE = "boundary_tests_missing"
MAX_LISTED = 8

_ACCEPTANCE_TITLE = re.compile(r"(критери\w*\s+при[её]мк|acceptance|тест-?кейс|сценари\w*\s+(испытан|тестир))",
                               re.IGNORECASE)
# goals and quality attributes are measured in operation, not by acceptance rows
_NOT_TESTED_HERE = re.compile(r"(\bцел[ьиея]\w*|нефункцион|качеств|производительн|доступност|надёжн|надежн|"
                              r"метрик|kpi|sla)", re.IGNORECASE)
_NUM = r"(\d[\d  ]*(?:[.,]\d+)?)"
_RANGE = re.compile(rf"\bот\s+{_NUM}\s*(?:[^\W\d]+\.?\s+)?до\s+{_NUM}", re.IGNORECASE)
_THRESHOLD = re.compile(rf"(не\s+более|не\s+менее|не\s+больше|не\s+меньше|не\s+выше|не\s+ниже|максимум|минимум|"
                        rf"≥|≤|>=|<=)\s*{_NUM}", re.IGNORECASE)
_INCLUSIVE = re.compile(rf"{_NUM}\s*(?:[^\W\d]+\.?\s+){{0,2}}включительно", re.IGNORECASE)
_ANY_NUM = re.compile(r"(?<![\d.,])(\d[\d  ]*(?:[.,]\d+)?)(?![\d])")
_CLOCK = re.compile(r"^\s*[:.]\d\d")  # 9:00, 31.01: a time or a date, not a quantity


@dataclass
class _Edge:
    block: Block
    start: int
    end: int
    values: list[str]  # boundary values as written

    @property
    def label(self) -> str:
        return re.sub(r"\s+", " ", self.block.text[self.start:self.end]).strip()


def _to_float(raw: str) -> float | None:
    try:
        return float(re.sub(r"[\s ]", "", raw).replace(",", "."))
    except ValueError:
        return None


def _step(raw: str) -> float:
    s = re.sub(r"[\s ]", "", raw).replace(",", ".")
    return 10 ** -len(s.split(".", 1)[1]) if "." in s else 1.0


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _ANY_NUM.finditer(text):
        v = _to_float(m.group(1))
        if v is not None:
            out.add(round(v, 6))
    return out


def _path(doc: Document, block: Block) -> list[str]:
    sec = next((s for s in doc.sections if s.id == block.section_id), None)
    return sec.path if sec else block.section_path


def _is_acceptance(doc: Document, block: Block) -> bool:
    return any(_ACCEPTANCE_TITLE.search(t) for t in _path(doc, block))


def _edges(block: Block) -> list[_Edge]:
    text = block.text
    found: list[_Edge] = []
    taken: list[tuple[int, int]] = []

    def add(m: re.Match, values: list[str]) -> None:
        if any(a < m.end() and m.start() < b for a, b in taken):
            return
        groups = [i for i in range(1, (m.lastindex or 0) + 1) if m.group(i) and _to_float(m.group(i)) is not None]
        if any(_CLOCK.match(text[m.end(i):]) for i in groups):
            return  # a time or a date, not a quantity
        taken.append((m.start(), m.end()))
        found.append(_Edge(block, m.start(), m.end(), values))

    for m in _RANGE.finditer(text):
        add(m, [m.group(1), m.group(2)])
    for m in _THRESHOLD.finditer(text):
        add(m, [m.group(2)])
    for m in _INCLUSIVE.finditer(text):
        add(m, [m.group(1)])
    return found


def _covered(edge: _Edge, nums: set[float]) -> bool:
    for raw in edge.values:
        v = _to_float(raw)
        if v is None:
            continue
        step = _step(raw)
        if round(v, 6) not in nums:
            return False
        if round(v - step, 6) not in nums and round(v + step, 6) not in nums:
            return False
    return True


def check_boundary_tests(doc: Document) -> list[RuleFinding]:
    """One aggregated set finding when the acceptance section lacks rows at and just past numeric edges."""
    acceptance = [b for b in doc.blocks if _is_acceptance(doc, b)]
    if not acceptance:
        return []
    nums = set().union(*(_numbers(b.text) for b in acceptance))
    missing: list[_Edge] = []
    for b in doc.blocks:
        if b.kind in (BlockKind.CODE, BlockKind.HEADING, BlockKind.TABLE_HEADER) or b in acceptance:
            continue
        if any(_NOT_TESTED_HERE.search(t) for t in _path(doc, b)):
            continue
        for e in _edges(b):
            key = tuple(_to_float(v) for v in e.values)
            if not _covered(e, nums) and key not in {tuple(_to_float(v) for v in x.values) for x in missing}:
                missing.append(e)
    if not missing:
        return []
    first = missing[0]
    listed = "; ".join(f"«{e.label}»" for e in missing[:MAX_LISTED])
    more = f" и ещё {len(missing) - MAX_LISTED}" if len(missing) > MAX_LISTED else ""
    return [_make_finding(
        rule=RULE, block_ids=list(dict.fromkeys(e.block.id for e in missing[:MAX_LISTED])),
        criterion=Criterion.SET_COMPLETE, severity=Severity.MAJOR, set_check=SetCheck.ACCEPTANCE_GAP,
        quote=_quote_for_span(first.block.text, first.start, first.end),
        explanation=(f"Для числовых границ {listed}{more} в критериях приёмки нет строк со значением ровно на "
                     "границе и сразу за ней: ошибку включительности нельзя поймать приёмкой."),
        recommendation="Добавить в критерии приёмки строки для каждого значения на границе и на шаг за ней.",
        clarifying_question="Какой результат ожидается ровно на каждой границе и на шаг за ней?",
        answer_options=["Строки на границе и за ней будут добавлены в критерии приёмки",
                        "Граница проверяется другим способом: указать каким"],
    )]
