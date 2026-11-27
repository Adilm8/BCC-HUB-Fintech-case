"""The eval and organiser documents must not leak into the prompts (docs/architecture.md section 9.3)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eval.score import normalize

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "validator" / "pipeline" / "prompts"

# Distinctive terms of the organisers' document and of the synthetic holdout documents.
ORGANISER_TERMS = [
    "автоплат", "alertscontroller", "savelimit", "limit_kzt", "тиын", "алматы", "kzt", "школа тестирования",
    "уведомление о расходах", "месячный порог", "gitlab",
]
AUTOPAY_TERMS = [
    "quartz", "autopay_rules", "autopayrule", "грейс", "кешбэк", "gateway_timeout", "billing.autopay",
    "цифробанк", "srs-pay", "национальный цифровой банк",
]
P2P_TERMS = [
    "phonetransfer", "limits:daily", "idempotency-key", "recipients/lookup", "избранные контакты",
    "профиль риска", "расчётный центр", "srs-trf", "maskedrecipientname", "limit_exceeded", "dailyremaining",
    "сбп",
]
TERMS = ORGANISER_TERMS + AUTOPAY_TERMS + P2P_TERMS

DOCUMENTS = [ROOT / "docs" / "requirements.md", *sorted((ROOT / "eval" / "docs").glob("*.md"))]
GOLD_FILES = [*sorted((ROOT / "eval" / "gold").glob("*.json")),
              *sorted((ROOT / "eval" / "gold_private").glob("*.json"))]


DOMAINS = ROOT / "validator" / "domains"


def prompt_files() -> list[Path]:
    files = [p for p in PROMPTS.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return sorted(files + list(DOMAINS.glob("*.yaml")))


def prompt_text() -> str:
    return normalize("\n".join(p.read_text(encoding="utf-8") for p in prompt_files()))


def test_prompts_exist():
    assert prompt_files(), "validator/pipeline/prompts is empty or missing"


@pytest.mark.parametrize("term", TERMS)
def test_domain_term_is_absent_from_prompts(term):
    assert normalize(term) not in prompt_text(), f"term from an eval or organiser document found in prompts: {term}"


def _gold_fragments() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in GOLD_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        for d in data.get("defects", []):
            out += [(f"{path.name}:{d['id']}", h) for h in d.get("blocks_hint", [])]
        out += [(f"{path.name}:{c['id']}", c["anchor"]) for c in data.get("clean", [])]
    return out


def test_gold_quotes_are_absent_from_prompts():
    text = prompt_text()
    fragments = _gold_fragments()
    assert fragments, "no gold fragments found"
    for where, fragment in fragments:
        norm = normalize(fragment)
        if len(norm) >= 20:
            assert norm not in text, f"gold fragment from {where} is in prompts: {fragment[:60]!r}"


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[\w.:/-]+", normalize(text))
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def test_no_long_phrase_is_copied_from_documents_into_prompts():
    prompt_grams = _word_ngrams(prompt_text(), 7)
    for doc in DOCUMENTS:
        if not doc.is_file():
            continue
        shared = prompt_grams & _word_ngrams(doc.read_text(encoding="utf-8"), 7)
        assert not shared, f"{doc.name}: 7-word phrase copied into prompts: {' '.join(min(shared))!r}"
