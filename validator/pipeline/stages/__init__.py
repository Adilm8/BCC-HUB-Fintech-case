"""Model stages: each builds its prompt and makes one structured call."""

from __future__ import annotations

from dataclasses import dataclass

from validator.config import Settings
from validator.llm import LLMClient
from validator.pipeline.structured import Budget, StageCache, StructuredResult, call_structured

# per-stage depth for providers that support it
EFFORT = {"extract": "medium", "check": "medium", "set_check": "high", "review": "medium"}
TEMPERATURE = {"extract": 0.0, "check": 0.2, "set_check": 0.2, "review": 0.0}


@dataclass
class StageContext:
    client: LLMClient
    settings: Settings
    budget: Budget
    cache: StageCache
    deadline: float

    def call(self, stage: str, model_cls, prompt: str, system: str) -> StructuredResult:
        return call_structured(
            self.client, model_cls, prompt, system=system, budget=self.budget, cache=self.cache,
            deadline=self.deadline, effort=self.settings.effort or EFFORT.get(stage),
            temperature=TEMPERATURE.get(stage), max_output_tokens=self.settings.max_output_tokens,
        )

    def include_document(self, doc_chars: int) -> bool:
        return doc_chars <= self.settings.max_context_chars
