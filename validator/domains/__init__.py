"""Pluggable domain knowledge profiles: `validator/domains/<name>.yaml`.

A profile is a checklist of hints for the model (topic + check), never a finding by itself. It is chosen
by name, or automatically by the stems of its `detect_keywords` found in the document.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

DOMAINS_DIR = Path(__file__).resolve().parent
MIN_DISTINCT_STEMS = 3  # auto selection: at least this many different stems must occur in the document
MAX_BLOCK_CHARS = 7000  # about 2.5k tokens of Russian text


class DomainItem(BaseModel):
    id: str
    topic: str
    applies_when: str = ""
    check: str
    typical_gap: str = ""
    criterion: str = ""


class DomainProfile(BaseModel):
    name: str
    domain: str
    detect_keywords: list[str] = Field(default_factory=list)
    items: list[DomainItem] = Field(default_factory=list)


def available() -> list[str]:
    return sorted(p.stem for p in DOMAINS_DIR.glob("*.yaml"))


def load(name: str) -> DomainProfile:
    from validator.answers import parse_yaml_document

    path = DOMAINS_DIR / f"{name}.yaml"
    if not re.fullmatch(r"[a-z0-9_-]+", name) or not path.is_file():
        raise ValueError(f"доменный профиль {name!r} не найден; доступны: {', '.join(available()) or 'нет'}, "
                         "а также auto и none")
    data = parse_yaml_document(path.read_text(encoding="utf-8")) or {}
    try:
        return DomainProfile(name=name, **data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"доменный профиль {path.name} не разобран: {exc}") from exc


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def score(profile: DomainProfile, text: str) -> tuple[int, int]:
    """(distinct stems found, total occurrences)."""
    low = _norm(text)
    counts = [low.count(_norm(s)) for s in profile.detect_keywords if s.strip()]
    return sum(1 for c in counts if c), sum(counts)


def detect(text: str) -> tuple[DomainProfile | None, str]:
    """The profile whose stems occur most, or None below the threshold; with a note for the run passport."""
    best: tuple[tuple[int, int], DomainProfile] | None = None
    for name in available():
        prof = load(name)
        s = score(prof, text)
        if s[0] >= MIN_DISTINCT_STEMS and (best is None or s > best[0]):
            best = (s, prof)
    if best is None:
        return None, f"доменный профиль не выбран: ни один профиль не набрал {MIN_DISTINCT_STEMS} основ в документе"
    (distinct, total), prof = best
    return prof, f"доменный профиль {prof.name} выбран автоматически: основ найдено {distinct}, вхождений {total}"


def select(choice: str, text: str) -> tuple[DomainProfile | None, str]:
    """`choice`: a profile name, "auto" or "none" (or empty)."""
    choice = (choice or "none").strip().lower()
    if choice == "none":
        return None, ""
    if choice == "auto":
        return detect(text)
    prof = load(choice)
    return prof, f"доменный профиль {prof.name} задан явно"
