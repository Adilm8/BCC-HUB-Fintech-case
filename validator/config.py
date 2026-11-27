from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

PROVIDERS = {
    # provider: (key variable, model variable, default model, fallback variable)
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.5-flash", "GEMINI_FALLBACK_MODELS"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-5-5", "ANTHROPIC_FALLBACK_MODELS"),
}

DEFAULT_PROFILE = {"gemini": "standard", "anthropic": "standard"}
DEFAULT_TIMEOUT = {"gemini": 240.0, "anthropic": 300.0}  # one attempt, seconds
# the free tier answers slower and runs calls one by one: it needs a longer run
DEFAULT_DEADLINE = {"gemini": 900.0, "anthropic": 480.0}


@dataclass(frozen=True)
class Profile:
    """Numbers that differ between profiles; the pipeline itself is the same."""

    name: str
    batch_size: int  # target requirements per check batch (0 = all in one batch)
    max_batches: int
    merge_set_checks: bool  # S1 and S2 in one call
    review: bool
    max_calls: int  # hard ceiling of model calls per run
    parallel: int


PROFILES: dict[str, Profile] = {
    "economy": Profile("economy", batch_size=0, max_batches=1, merge_set_checks=True, review=True,
                       max_calls=6, parallel=1),
    "standard": Profile("standard", batch_size=20, max_batches=3, merge_set_checks=False, review=True,
                        max_calls=10, parallel=2),
    "deep": Profile("deep", batch_size=12, max_batches=5, merge_set_checks=False, review=True,
                    max_calls=16, parallel=4),
}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str = field(repr=False)
    model: str = "gemini-3.5-flash"
    fallback_models: tuple[str, ...] = ()
    max_retries: int = 4
    retry_base_seconds: float = 8.0
    profile_name: str = "standard"
    timeout_seconds: float = 0.0  # one attempt; 0 means the provider default (DEFAULT_TIMEOUT)
    run_deadline_seconds: float = 0.0  # whole run: no new calls after it; 0 means DEFAULT_DEADLINE
    max_calls: int | None = None  # overrides the profile ceiling
    max_output_tokens: int = 32000
    effort: str | None = None  # overrides per-stage effort
    cache_enabled: bool = True
    cache_dir: str = ".cache/validator"
    max_context_chars: int = 120_000

    @property
    def run_deadline(self) -> float:
        return self.run_deadline_seconds or DEFAULT_DEADLINE.get(self.provider, 480.0)

    @property
    def attempt_timeout(self) -> float:
        return self.timeout_seconds or DEFAULT_TIMEOUT.get(self.provider, 240.0)

    @property
    def profile(self) -> Profile:
        base = PROFILES[self.profile_name]
        if self.max_calls is not None:
            base = replace(base, max_calls=self.max_calls)
        return base

    def with_overrides(self, **changes) -> Settings:
        name = changes.get("profile_name")
        if name is not None and name not in PROFILES:
            raise ValueError(f"профиль {name!r} не поддерживается, доступны: {', '.join(PROFILES)}")
        return replace(self, **changes)

    @classmethod
    def from_env(cls) -> Settings:
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"провайдер {provider!r} не поддерживается, доступны: {', '.join(PROVIDERS)}")
        key_var, model_var, default_model, fallback_var = PROVIDERS[provider]
        key = os.getenv(key_var, "").strip()
        if not key:
            raise ValueError(f"не задан {key_var}: скопируйте .env.example в .env и впишите ключ")
        fallbacks = tuple(m.strip() for m in os.getenv(fallback_var, "").split(",") if m.strip())
        profile = os.getenv("PIPELINE_PROFILE", "").strip().lower() or DEFAULT_PROFILE[provider]
        if profile not in PROFILES:
            raise ValueError(f"профиль {profile!r} не поддерживается, доступны: {', '.join(PROFILES)}")
        max_calls = os.getenv("LLM_MAX_CALLS", "").strip()
        cache = os.getenv("PIPELINE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}
        return cls(
            provider=provider,
            api_key=key,
            model=os.getenv(model_var, "").strip() or default_model,
            fallback_models=fallbacks,
            max_retries=_int("LLM_MAX_RETRIES", 4),
            retry_base_seconds=_float("LLM_RETRY_BASE_SECONDS", 8.0),
            profile_name=profile,
            timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 0.0),
            run_deadline_seconds=_float("RUN_DEADLINE_SECONDS", 0.0),
            max_calls=int(max_calls) if max_calls else None,
            max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 32000),
            effort=os.getenv("LLM_EFFORT", "").strip() or None,
            cache_enabled=cache,
            cache_dir=os.getenv("PIPELINE_CACHE_DIR", "").strip() or ".cache/validator",
            max_context_chars=_int("MAX_CONTEXT_CHARS", 120_000),
        )
