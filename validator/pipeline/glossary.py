"""Terms that conditions and calculations depend on, found by code (EXP_GLOSSARY): no model call."""

from __future__ import annotations

import re

from validator.models import ReqKind, Requirement

MAX_TERMS = 25

_QUOTED = re.compile(r"[«\"“]([^«»\"“”\n]{3,60})[»\"”]")
_COMPARE = re.compile(r"((?:[^\W\d_][\w-]*\s+)?[^\W\d_][\w-]*)\s+(?:не\s+)?(?:превыша\w*|превысит\w*|больше|меньше|"
                      r"выше|ниже|достига\w*|достигнет|равн\w*|составля\w*|не\s+менее|не\s+более)\b", re.IGNORECASE)
_BEFORE_NUMBER = re.compile(r"([^\W\d_][\w-]{3,})\s+(?:(?:не\s+)?(?:более|менее|свыше|от|до|превышает)\s+)?\d",
                            re.IGNORECASE)
_ABBREV = re.compile(r"(?<![\w-])([A-ZА-ЯЁ]{2,6})(?![\w-])")
_CAPITALISED = re.compile(r"(?<=[^.!?:;|\s] )([A-ZА-ЯЁ][a-zа-яё]{2,}(?:\s+[A-ZА-ЯЁ][a-zа-яё]{2,})+)")
_LEAD = re.compile(r"^(?:если|когда|при|и|или|а|что|то|в|на|с|по|для|если\s+же|только|все|его|её|их)\s+",
                   re.IGNORECASE)
_VERB_END = re.compile(r"(ют|ят|ет|ит|ть|ться|тся|ется|ал|ил|ло|ли)$")
# rules and data: interface examples, screens and acceptance rows reuse the same terms
_KINDS = {ReqKind.FUNCTIONAL, ReqKind.BUSINESS_GOAL, ReqKind.DATA, ReqKind.NONFUNCTIONAL, ReqKind.CONSTRAINT,
          ReqKind.INTEGRATION, ReqKind.PROCESS}
_STOP = {"она", "он", "оно", "они", "это", "этот", "нет", "при", "для", "или", "который", "которая", "система",
         "системы", "систему", "пользователь", "пользователя", "требование", "требования", "значение", "значения",
         "данные", "данных", "случае", "должна", "должен", "должно", "может", "будет", "является", "менее",
         "более", "равно", "равным", "равной", "ниже", "выше", "течение", "срок", "время", "раз", "число",
         "количество", "диапазоне", "точностью", "длительностью", "уровня", "действия", "ответа", "результат"}


def _clean(term: str) -> str:
    term = re.sub(r"\s+", " ", term).strip(" .,;:()-")
    while True:
        cut = _LEAD.sub("", term)
        if cut == term:
            break
        term = cut
    return term


def _keep(term: str) -> bool:
    low = term.lower()
    words = low.split()
    return (3 <= len(term) <= 60 and low not in _STOP and not low.isdigit() and len(words) <= 4
            and not _VERB_END.search(words[-1]) and words[-1] not in _STOP)


def extract_terms(reqs: list[Requirement]) -> dict[str, list[str]]:
    """Term -> requirement ids where a condition or a calculation depends on it, in document order."""
    out: dict[str, list[str]] = {}
    for r in reqs:
        if r.kind not in _KINDS:
            continue
        text = r.quote
        found = [m.group(1) for m in _QUOTED.finditer(text)]
        found += [m.group(1) for m in _COMPARE.finditer(text)]
        found += [m.group(1) for m in _BEFORE_NUMBER.finditer(text)]
        found += [m.group(1) for m in _ABBREV.finditer(text) if not re.match(r"[A-ZА-ЯЁ]+-\d", text[m.start():])]
        found += [m.group(1) for m in _CAPITALISED.finditer(text)]
        for raw in found:
            term = _clean(raw)
            if not _keep(term):
                continue
            key = next((k for k in out if k.lower() == term.lower()), term)
            ids = out.setdefault(key, [])
            if r.id not in ids:
                ids.append(r.id)
    return out


def terms_block(batch: list[Requirement], all_reqs: list[Requirement]) -> str:
    """The terms used by this batch, each with every requirement that depends on it."""
    terms = extract_terms(all_reqs)
    ids = {r.id for r in batch}
    used = [(t, rids) for t, rids in terms.items() if ids & set(rids)]
    used.sort(key=lambda x: -len(x[1]))
    lines = [f"- {t} ({', '.join(rids[:6])})" for t, rids in used[:MAX_TERMS]]
    return "\n".join(lines)
