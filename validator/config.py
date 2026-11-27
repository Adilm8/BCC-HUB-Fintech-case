from __future__ import annotations

import os
from dataclasses import dataclass, field

PROVIDERS = {
    # provider: (key variable, model variable, default model, fallback variable)
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.5-flash", "GEMINI_FALLBACK_MODELS"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-5-5", "ANTHROPIC_FALLBACK_MODELS"),
}


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str = field(repr=False)
    model: str = "gemini-3.5-flash"
    fallback_models: tuple[str, ...] = ()
    max_retries: int = 4
    retry_base_seconds: float = 8.0

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
        return cls(
            provider=provider,
            api_key=key,
            model=os.getenv(model_var, "").strip() or default_model,
            fallback_models=fallbacks,
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "4")),
            retry_base_seconds=float(os.getenv("LLM_RETRY_BASE_SECONDS", "8")),
        )
