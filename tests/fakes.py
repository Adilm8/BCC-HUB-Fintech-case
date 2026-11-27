"""FakeLLM: an offline client with the same complete() interface, scripted per stage."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from typing import Any

from validator.llm import LLMError, LLMResponse

Responder = Callable[[str, dict], Any]  # (prompt, schema) -> dict | str | Exception


def stage_of(schema: dict | None) -> str:
    props = set((schema or {}).get("properties", {}))
    if {"requirements", "skipped"} <= props:
        return "extract"
    if "results" in props:
        return "check"
    if "decisions" in props:
        return "review"
    if "coverage" in props:
        return "set"
    return "other"


_BLOCK = re.compile(r"^\[(B\d+)\] (.*)$", re.MULTILINE)
_REQ = re.compile(r"^(R-\d+) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| блоки ([^|]*) \| (.*)$", re.MULTILINE)


def section(prompt: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL)
    return m.group(1) if m else ""


def default_extract(prompt: str, schema: dict) -> dict:
    reqs = []
    for bid, text in _BLOCK.findall(prompt):
        if text.startswith("#"):
            continue
        kind = "acceptance_criterion" if "→" in text and "|" in text else "functional"
        if text.startswith("| "):
            kind = "interface"
        reqs.append({"block_ids": [bid], "doc_id": "", "kind": kind, "title": text[:60], "quote": ""})
    return {"requirements": reqs, "skipped": []}


def default_check(prompt: str, schema: dict) -> dict:
    results = []
    for m in _REQ.finditer(section(prompt, "requirements")):
        rid, text = m.group(1), m.group(7).strip()
        findings = []
        words = text.split()
        if len(words) >= 6:
            findings.append({
                "quote": " ".join(words[:6]),
                "explanation": f"Во фрагменте «{' '.join(words[:3])}» не задано измеримое значение и условие проверки.",
                "criterion": "verifiable", "severity": "major",
                "recommendation": "Указать число, единицу и условие измерения.",
                "clarifying_question": "Какое значение считается выполнением требования", "addressee": "analyst",
                "answer_options": ["вариант 1", "вариант 2"], "related_requirement_ids": [],
            })
        results.append({"requirement_id": rid, "findings": findings})
    return {"results": results}


def default_set(prompt: str, schema: dict) -> dict:
    rows = list(_REQ.finditer(section(prompt, "requirements")))
    findings = []
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        findings.append({
            "evidence": [{"requirement_id": a.group(1), "quote": a.group(7)[:60]},
                         {"requirement_id": b.group(1), "quote": b.group(7)[:60]}],
            "scenario": "Входные данные X: по первому месту результат A, по второму результат B.",
            "explanation": "Два места документа описывают одно и то же поведение по-разному, реализация неоднозначна.",
            "set_check": "contradiction", "criterion": "set_consistent", "severity": "critical",
            "recommendation": "Согласовать оба места.", "clarifying_question": "Какое из двух описаний верное?",
            "addressee": "business",
            "answer_options": ["первое", "второе"],
        })
    coverage = [{"requirement_id": r.group(1), "acceptance_ids": [], "status": "none", "note": ""}
                for r in rows if r.group(3).strip() == "functional"]
    return {"findings": findings, "coverage": coverage}


def default_review(prompt: str, schema: dict) -> dict:
    ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
    return {
        "decisions": [{"finding_id": i, "verdict": "keep", "merge_into": "", "severity": "major", "drop_reason": "none",
                       "counter_quote": "", "improved_question": ""} for i in ids],
        "missed": [], "fix_first": ids[:3], "verdict_rationale": "Есть противоречия, которые нужно снять до начала.",
    }


DEFAULTS: dict[str, Responder] = {
    "extract": default_extract, "check": default_check, "set": default_set, "review": default_review,
}


class FakeLLM:
    """Responds per stage. `script[stage]` is a responder, a dict/str answer, an exception, or a list of those
    (consumed in order, the last one repeats)."""

    provider = "fake"

    def __init__(self, script: dict[str, Any] | None = None, model: str = "fake-model",
                 truncated: set[str] | None = None):
        self.script = dict(script or {})
        self.model = model
        self.truncated = truncated or set()
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}

    def _next(self, stage: str):
        item = self.script.get(stage, DEFAULTS.get(stage))
        if isinstance(item, list):
            with self._lock:
                n = self._counters.get(stage, 0)
                self._counters[stage] = n + 1
            item = item[min(n, len(item) - 1)]
        return item

    def complete(self, prompt: str, *, system: str | None = None, json_schema: dict | None = None,
                 temperature: float | None = None, effort: str | None = None,
                 max_output_tokens: int | None = None, deadline: float | None = None) -> LLMResponse:
        stage = stage_of(json_schema)
        with self._lock:
            self.calls.append((stage, prompt))
        item = self._next(stage)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(prompt, json_schema or {})
            if isinstance(item, BaseException):
                raise item
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return LLMResponse(text=text, model=self.model, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4,
                           truncated=stage in self.truncated)

    def generate(self, prompt: str, *, system: str | None = None, json_schema: dict | None = None) -> str:
        return self.complete(prompt, system=system, json_schema=json_schema).text


def quota_error() -> LLMError:
    return LLMError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDay", "quota_daily")
