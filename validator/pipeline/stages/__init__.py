"""Model stages: each builds its prompt and makes one structured call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from validator.config import Settings
from validator.llm import LLMClient
from validator.pipeline.structured import Budget, StageCache, StructuredResult, call_structured

# per-stage depth for providers that support it
EFFORT = {"extract": "medium", "check": "medium", "set_check": "high", "review": "medium"}
# the free provider's model needs deeper reasoning on the per-requirement check to find gaps
EFFORT_BY_PROVIDER: dict[str, dict[str, str]] = {}
TEMPERATURE = {"extract": 0.0, "check": 0.2, "set_check": 0.2, "review": 0.0}


@dataclass
class StageContext:
    client: LLMClient
    settings: Settings
    budget: Budget
    cache: StageCache
    deadline: float
    review_reserve: float = 0.0  # seconds kept for the critic: other stages stop earlier
    note: Callable[[str, str], None] | None = None  # (stage, note) progress sink

    def call(self, stage: str, model_cls, prompt: str, system: str) -> StructuredResult:
        sink = getattr(self.client, "set_note_sink", None)
        if sink is not None:
            sink((lambda text: self.note(stage, text)) if self.note else None)
        deadline = self.deadline if stage == "review" else self.deadline - self.review_reserve
        return call_structured(
            self.client, model_cls, prompt, system=system, budget=self.budget, cache=self.cache,
            deadline=deadline, effort=self.settings.effort or self.effort(stage),
            temperature=self.temperature(stage), max_output_tokens=self.settings.max_output_tokens,
        )

    def effort(self, stage: str) -> str | None:
        provider = getattr(self.client, "provider", "")
        return EFFORT_BY_PROVIDER.get(provider, {}).get(stage) or EFFORT.get(stage)

    def temperature(self, stage: str) -> float | None:
        # the paid provider's model rejects a sampling temperature, and the newest free models degrade at low
        # temperatures: send (and hash) none for them
        provider = getattr(self.client, "provider", "")
        model = getattr(self.client, "model", "") or ""
        if provider == "anthropic" or model.startswith("gemini-3"):
            return None
        return TEMPERATURE.get(stage)

    def include_document(self, doc_chars: int) -> bool:
        return doc_chars <= self.settings.max_context_chars
