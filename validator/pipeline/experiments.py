"""Switchable experiments for quality measurements. Every switch is off unless its environment variable is "1".

EXP_REPEAT_INSTRUCTION    the stage's task is repeated in front of the document (task, document, task).
EXP_SCHEMA_REASONING_FIRST a short `reasoning` field comes first in every finding and decision object.
EXP_ENTITY_PASS           one extra call lists entities (quantities, flags, parameters, events, messages) with their
                          places; S2 checks every group with a fixed checklist. One check batch less.
EXP_DIMENSION_PASS        per-requirement check batches are replaced by two passes over the whole document, one
                          per group of completeness dimensions.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Literal

from pydantic import BaseModel


def enabled(name: str) -> bool:
    return os.getenv(name, "").strip() == "1"


def active() -> list[str]:
    names = ["EXP_REPEAT_INSTRUCTION", "EXP_SCHEMA_REASONING_FIRST", "EXP_ENTITY_PASS", "EXP_DIMENSION_PASS",
             "EXP_PAIR_ENUM", "EXP_TWO_PASS"]
    return [n for n in names if enabled(n)]


# ---------------------------------------------------------------- pair enumeration

_IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})`|\b([a-z]+[A-Z][A-Za-z0-9]*|[a-z]+_[a-z0-9_]+)\b")
_STATUS = re.compile(r"(?<![\d.,])([1-5]\d\d)(?![\d.,])")
_NUM_UNIT = re.compile(r"(\d[\d\s]*(?:[.,]\d+)?)\s*(₸|тенге|руб\w*|тиын\w*|коп\w*|%|мс|сек\w*|с\b|мин\w*|час\w*|"
                       r"дн\w*|сут\w*|мес\w*|МБ|КБ|ГБ)", re.IGNORECASE)
_PERIOD = re.compile(r"(в месяц|в день|в сутки|в неделю|в час|за календарный \w+|за период|не чаще|не более|"
                     r"максимум|один раз|лимит|ограничени\w*)", re.IGNORECASE)
_FLAG = re.compile(r"(0 или 1|вкл\w*|выкл\w*|переключател\w*|признак\w*|флаг\w*|enabled|disabled)", re.IGNORECASE)
_RESET = re.compile(r"(независимо от|при изменении|сбрасыва\w*|заново|повторн\w*)", re.IGNORECASE)


def _keys(text: str) -> set[str]:
    keys: set[str] = set()
    for a, b in _IDENT.findall(text):
        keys.add("поле " + (a or b).lower())
    keys |= {"код " + c for c in _STATUS.findall(text)}
    for num, unit in _NUM_UNIT.findall(text):
        keys.add("величина " + re.sub(r"\s", "", num) + " " + unit.lower()[:4])
    keys |= {"период/лимит" for _ in _PERIOD.findall(text)[:1]}
    keys |= {"флаг" for _ in _FLAG.findall(text)[:1]}
    keys |= {"сброс/повтор" for _ in _RESET.findall(text)[:1]}
    return keys


def pair_candidates(reqs, limit: int = 25) -> list[tuple[str, str, list[str]]]:
    """Pairs of requirements that share an entity: a field, a status code, a value with a unit, a period or
    limit, a flag. Code enumerates, the model checks each pair."""
    keyed = [(r, _keys(f"{r.title} {r.quote}")) for r in reqs]
    pairs = []
    for i, (a, ka) in enumerate(keyed):
        for b, kb in keyed[i + 1:]:
            shared = sorted(ka & kb)
            if not shared:
                continue
            # rule-like overlaps (limits, resets, flags) and pairs across sections first
            weight = len(shared) + 2 * len({"период/лимит", "сброс/повтор", "флаг"} & set(shared))
            weight += 1 if a.section != b.section else 0
            pairs.append((weight, a.id, b.id, shared))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return [(a, b, s) for _, a, b, s in pairs[:limit]]


def pairs_text(pairs: list[tuple[str, str, list[str]]], *, reverse: bool = False) -> str:
    items = list(reversed(pairs)) if reverse else pairs
    return "\n".join(f"P{i}: {a} ↔ {b} — общее: {', '.join(s)}" for i, (a, b, s) in enumerate(items, 1))


PAIRS_INSTRUCTION = """Обязательно проверь КАЖДУЮ пару из <pairs>: для пары составь сценарий, в котором оба \
требования применяются к одному и тому же случаю, и сравни, что требует каждое. Разный результат — находка \
contradiction с двумя цитатами и сценарием; согласованные пары находок не дают."""

SECOND_PASS_NOTE = """Это второй, независимый проход: иди по документу от последней части к первой и начинай с \
пар правил, которые ограничивают одно событие (лимиты, сбросы, флаги, коды ответов)."""


# ---------------------------------------------------------------- repeat the instruction


def repeat_instruction(prompt: str) -> str:
    """Task, then the document and the rest, then the task again."""
    idx = prompt.rfind("\nЗадание")
    if idx < 0:
        return prompt
    task = prompt[idx + 1:]
    return f"{task}\n\n{prompt}"


# ---------------------------------------------------------------- reasoning first in schemas

_LABEL_FIELDS = {"severity", "verdict"}


def reasoning_first(schema: dict) -> dict:
    """Objects with a severity or verdict get a short `reasoning` field in first position."""

    def walk(node):
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items()}
            props = out.get("properties")
            if out.get("type") == "object" and isinstance(props, dict) and _LABEL_FIELDS & set(props):
                out["properties"] = {"reasoning": {"type": "string"}, **props}
                out["required"] = ["reasoning", *[r for r in out.get("required", []) if r != "reasoning"]]
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(copy.deepcopy(schema))


REASONING_NOTE = ("Поле reasoning — одна-две фразы рассуждения до выбора критичности или решения; в отчёт оно не "
                  "попадает.")


# ---------------------------------------------------------------- entity pass


class LLMMention(BaseModel):
    block_id: str
    quote: str
    unit: str


class LLMEntity(BaseModel):
    name: str
    kind: Literal["quantity", "flag", "parameter", "event", "message", "other"]
    mentions: list[LLMMention]


class LLMEntities(BaseModel):
    entities: list[LLMEntity]


ENTITY_TASK = """Задание: выпиши сущности документа, о которых говорят несколько мест: величины и суммы, флаги и \
переключатели, параметры интерфейса, события и сообщения пользователю или другим системам. Для каждой сущности \
перечисли ВСЕ места (block_id и дословный короткий фрагмент) и единицу измерения, если она указана (иначе пустая \
строка). Не оценивай качество, только собери места. Не больше 25 сущностей."""

ENTITY_CHECKLIST = """Для каждой группы из <entities> проверь по фиксированному списку:
1) одна величина в разных единицах — есть ли правило пересчёта и округления;
2) флаг или переключатель — обязательны ли остальные параметры, когда он выключен, и учтён ли он везде, где \
действует;
3) сумма или величина во внешнем сообщении — однозначно ли, что именно передаётся и в каком формате;
4) повтор или повторная доставка события — указан ли ключ, по которому распознаётся повтор, и окно хранения;
5) событие или процесс — описаны ли ветки отказа.
Каждое нарушение оформи находкой в findings (с цитатами мест из группы)."""


def entity_groups(entities: LLMEntities, block_text: dict[str, str]) -> str:
    """Entities with verified places, one line each, for the S2 prompt."""
    lines = []
    for i, e in enumerate(entities.entities, 1):
        places = []
        for m in e.mentions:
            if m.block_id not in block_text:
                continue
            unit = f" [{m.unit}]" if m.unit.strip() else ""
            places.append(f"{m.block_id} «{m.quote[:120]}»{unit}")
        if places:
            lines.append(f"E{i} {e.name} ({e.kind}): " + "; ".join(places))
    return "\n".join(lines)


# ---------------------------------------------------------------- dimension passes

DIMENSIONS = [
    ("ошибки и отказы (что происходит при сбое, недоступности, неверном вводе), состояния и флаги (что при каждом "
     "значении), повторы и идемпотентность (ключ, окно, результат повтора)"),
    ("единицы и форматы (одна величина в разных единицах, правило пересчёта), сообщения пользователю и другим "
     "системам (что именно передаётся, шаблон, маскирование), проверка ввода (границы, включительность, значения "
     "по умолчанию)"),
]

DIMENSION_TASK = """Задание: пройди по ВСЕМУ документу и найди пробелы только по измерениям: {dims}.
Для каждой находки укажи requirement_id того требования из <requirements>, к которому она относится, и дословную \
цитату. В results верни записи только по требованиям с находками. Другие измерения не проверяй."""
