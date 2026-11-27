"""Вызов LLM с повторами и запасными моделями. Провайдер выбирается в .env (LLM_PROVIDER)."""

from __future__ import annotations

import json
import time

from validator.config import Settings

RETRYABLE = (
    "429", "500", "502", "503", "529",
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded", "DEADLINE", "rate_limit", "timeout",
)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Общий цикл: модель, повторы с паузой, затем следующая запасная модель."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self.last_model: str | None = None

    def generate(self, prompt: str, *, system: str | None = None, json_schema: dict | None = None) -> str:
        errors: list[str] = []
        for model in (self._settings.model, *self._settings.fallback_models):
            for attempt in range(self._settings.max_retries + 1):
                try:
                    text = self._call(model, prompt, system, json_schema)
                    self.last_model = model
                    return text
                except Exception as exc:  # noqa: BLE001 - SDKs raise several error types
                    msg = f"{type(exc).__name__}: {exc}"
                    errors.append(f"{model}: {msg[:200]}")
                    if not any(code.lower() in msg.lower() for code in RETRYABLE):
                        break
                    if attempt < self._settings.max_retries:
                        time.sleep(self._settings.retry_base_seconds * (2**attempt))
        raise LLMError("модель недоступна: " + " | ".join(errors[-3:]))

    def _call(self, model: str, prompt: str, system: str | None, json_schema: dict | None) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        from google import genai

        self._client = genai.Client(api_key=settings.api_key)

    def _call(self, model, prompt, system, json_schema):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            response_mime_type="application/json" if json_schema else None,
            response_json_schema=json_schema,
        )
        resp = self._client.models.generate_content(model=model, contents=prompt, config=config)
        return resp.text or ""


class AnthropicClient(LLMClient):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        import anthropic

        # SDK retries are off: the shared loop above handles retries and fallbacks
        self._client = anthropic.Anthropic(api_key=settings.api_key, max_retries=0)

    def _call(self, model, prompt, system, json_schema):
        if json_schema:
            prompt += (
                "\n\nОтветь только JSON-объектом без пояснений и без markdown, по схеме:\n"
                + json.dumps(json_schema, ensure_ascii=False)
            )
        kwargs = {"system": system} if system else {}
        resp = self._client.messages.create(
            model=model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def make_client(settings: Settings) -> LLMClient:
    if settings.provider == "anthropic":
        return AnthropicClient(settings)
    return GeminiClient(settings)
