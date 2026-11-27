from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

PROVIDERS = {
    # provider: (key variable, model variable, default model, fallback variable)
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.5-flash", "GEMINI_FALLBACK_MODELS"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-5-5", "ANTHROPIC_FALLBACK_MODELS"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "google/gemini-3.8-flash", "OPENROUTER_FALLBACK_MODELS"),
}
# fallback models used when the environment does not name any
DEFAULT_FALLBACKS: dict[str, tuple[str, ...]] = {"gemini": ("gemini-3.5-flash-lite",)}
# list price in USD per 1M tokens (input, output) for the run passport's cost estimate
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {"claude-sonnet-5-5": (2.0, 10.0),
                                                              "google/gemini-3.8-flash": (0.75, 3.75)}
# providers whose models are used on a free tier by default
FREE_TIER_FAMILIES = {"gemini"}
# providers that answer one call at a time
SEQUENTIAL_PROVIDERS = {"gemini", "openrouter"}
# a provider that behaves like one of the built-in ones (scheduling, profile, sampling); the extension hook adds more
PROVIDER_FAMILY: dict[str, str] = {"openrouter": "gemini"}

DEFAULT_PROFILE = {"gemini": "standard", "anthropic": "standard", "openrouter": "standard"}
DEFAULT_TIMEOUT = {"gemini": 240.0, "anthropic": 300.0, "openrouter": 240.0}  # one attempt, seconds
# the free tier answers slower and runs calls one by one: it needs a longer run
DEFAULT_DEADLINE = {"gemini": 900.0, "anthropic": 480.0, "openrouter": 900.0}


def family(provider: str) -> str:
    """The built-in provider whose behaviour (scheduling, profile choices) this provider follows."""
    return PROVIDER_FAMILY.get(provider, provider)


_EXTRA = None


def extra_provider_module():
    """Optional extra provider for local testing: a Python file named by VALIDATOR_EXTRA_PROVIDER.

    The module defines NAME, CONFIG (key, model, default model, fallback variables), FAMILY, TIMEOUT, DEADLINE and
    make_client(settings). Nothing is loaded when the variable is not set."""
    global _EXTRA
    path = os.getenv("VALIDATOR_EXTRA_PROVIDER", "").strip()
    if not path:
        return None
    if _EXTRA is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("validator_extra_provider", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"не удалось загрузить дополнительный провайдер: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = module.NAME
        PROVIDERS[name] = tuple(module.CONFIG)
        PROVIDER_FAMILY[name] = getattr(module, "FAMILY", name)
        fam = PROVIDER_FAMILY[name]
        DEFAULT_PROFILE[name] = DEFAULT_PROFILE.get(fam, "standard")
        DEFAULT_TIMEOUT[name] = float(getattr(module, "TIMEOUT", DEFAULT_TIMEOUT.get(fam, 240.0)))
        DEFAULT_DEADLINE[name] = float(getattr(module, "DEADLINE", DEFAULT_DEADLINE.get(fam, 480.0)))
        if fam in SEQUENTIAL_PROVIDERS:
            SEQUENTIAL_PROVIDERS.add(name)
        _EXTRA = module
    return _EXTRA


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
    domain: str = ""  # domain profile: a name, "auto" or "none"; empty follows EXP_DOMAIN (on by default: auto)

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
    def from_env(cls, *, require_key: bool = True) -> Settings:
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider not in PROVIDERS:
            extra_provider_module()
        if provider not in PROVIDERS:
            raise ValueError(f"провайдер {provider!r} не поддерживается, доступны: {', '.join(PROVIDERS)}")
        key_var, model_var, default_model, fallback_var = PROVIDERS[provider]
        key = os.getenv(key_var, "").strip() if key_var else ""
        if key_var and not key and require_key:
            raise ValueError(f"не задан {key_var}: скопируйте .env.example в .env и впишите ключ")
        if key and not key.isascii():
            raise ValueError(f"{key_var} содержит недопустимые символы: впишите настоящий ключ вместо примера")
        raw_fallbacks = os.getenv(fallback_var)
        if raw_fallbacks is None:
            fallbacks = DEFAULT_FALLBACKS.get(provider, ())
        else:  # an explicitly empty value means "no fallback model"
            fallbacks = tuple(m.strip() for m in raw_fallbacks.split(",") if m.strip())
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
            domain=os.getenv("DOMAIN_PROFILE", "").strip().lower(),
        )


def estimate_cost(tokens_by_model: dict[str, tuple[int, int]], provider: str) -> tuple[float | None, str]:
    """(cost in USD or None, a Russian note for the run passport)."""
    used = {m: io for m, io in tokens_by_model.items() if io[0] or io[1]}
    if not used:
        return None, "модель не вызывалась"
    if all(m in MODEL_PRICES_USD_PER_MTOK for m in used):
        cost = sum(i * MODEL_PRICES_USD_PER_MTOK[m][0] / 1e6 + o * MODEL_PRICES_USD_PER_MTOK[m][1] / 1e6
                   for m, (i, o) in used.items())
        prices = "; ".join(f"{m}: ${MODEL_PRICES_USD_PER_MTOK[m][0]:g} / ${MODEL_PRICES_USD_PER_MTOK[m][1]:g} "
                           "за 1 млн токенов (вход / выход)" for m in used)
        return round(cost, 4), f"оценка по прайс-листу ({prices})"
    if provider in FREE_TIER_FAMILIES:
        return None, "бесплатный тариф; на платном ключе — по прайс-листу провайдера"
    return None, "цена модели неизвестна"
