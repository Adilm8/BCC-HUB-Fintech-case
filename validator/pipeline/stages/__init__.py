"""Model stages: each builds its prompt and makes one structured call."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from validator.config import Settings, family
from validator.llm import LLMClient
from validator.pipeline.structured import Budget, StageCache, StructuredResult, call_structured

# per-stage depth for providers that support it
EFFORT = {"extract": "medium", "check": "medium", "set_check": "high", "review": "medium"}
# the free provider's model needs deeper reasoning on the per-requirement check to find gaps
EFFORT_BY_PROVIDER: dict[str, dict[str, str]] = {}
# env variables that route one stage to another model of the same provider
STAGE_MODEL_ENV = {"extract": "LLM_MODEL_EXTRACT", "check": "LLM_MODEL_CHECK", "set_check": "LLM_MODEL_SET",
                   "review": "LLM_MODEL_REVIEW"}
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
        from validator.pipeline import experiments as exp

        if exp.enabled("EXP_REPEAT_INSTRUCTION"):
            prompt = exp.repeat_instruction(prompt)
        if exp.enabled("EXP_SCHEMA_REASONING_FIRST"):
            system = f"{system}\n\n{exp.REASONING_NOTE}"
        return call_structured(
            self.client, model_cls, prompt, system=system, budget=self.budget, cache=self.cache,
            deadline=deadline, effort=self.settings.effort or self.effort(stage),
            temperature=self.temperature(stage), max_output_tokens=self.settings.max_output_tokens,
            model=self.model(stage),
        )

    def model(self, stage: str) -> str | None:
        """The model a stage is routed to, or None for the run's model."""
        name = STAGE_MODEL_ENV.get(stage)
        value = os.getenv(name, "").strip() if name else ""
        return value or None

    def effort(self, stage: str) -> str | None:
        provider = getattr(self.client, "provider", "")
        return EFFORT_BY_PROVIDER.get(provider, {}).get(stage) or EFFORT.get(stage)

    def temperature(self, stage: str) -> float | None:
        # the paid provider's model rejects a sampling temperature, and the newest free models degrade at low
        # temperatures: send (and hash) none for them
        provider = family(getattr(self.client, "provider", ""))
        model = self.model(stage) or getattr(self.client, "model", "") or ""
        if provider != "gemini" or model.startswith("gemini-3"):
            return None
        return TEMPERATURE.get(stage)

    def include_document(self, doc_chars: int) -> bool:
        return doc_chars <= self.settings.max_context_chars
