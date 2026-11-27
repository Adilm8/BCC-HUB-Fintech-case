"""Вызов LLM с повторами и запасными моделями. Провайдер выбирается в .env (LLM_PROVIDER)."""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Literal

from validator.config import Settings

ErrorKind = Literal[
    "quota_daily", "rate_limit", "timeout", "server", "auth", "bad_request", "refusal", "deadline", "other"
]

DAILY_QUOTA = ("PerDay", "per day", "GenerateRequestsPerDay", "daily", "quota exceeded for metric")
RATE_LIMIT = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "rate limit", "RateLimitError")
SERVER = ("500", "502", "503", "504", "529", "UNAVAILABLE", "overloaded", "InternalServerError",
          "ServiceUnavailable", "APIConnectionError", "Connection")
TIMEOUT = ("timeout", "timed out", "DEADLINE_EXCEEDED", "Timeout")
AUTH = ("401", "403", "AuthenticationError", "PermissionDenied", "API key not valid", "invalid x-api-key",
        "API_KEY_INVALID", "UNAUTHENTICATED")
BAD_REQUEST = ("400", "BadRequest", "INVALID_ARGUMENT", "invalid_request")

MIN_ATTEMPT_SECONDS = 15.0  # do not start an attempt when less than this remains before the deadline
MAX_RATE_LIMIT_SLEEP = 60.0


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
    seconds: float = 0.0
    attempts: int = 1


class LLMError(RuntimeError):
    def __init__(self, message: str, kind: ErrorKind = "other"):
        super().__init__(message)
        self.kind: ErrorKind = kind


def classify_error(exc: BaseException) -> ErrorKind:
    """Map an SDK exception to an error kind by type name and message."""
    if isinstance(exc, LLMError):
        return exc.kind
    if isinstance(exc, TimeoutError):
        return "timeout"
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403) or any(m.lower() in low for m in AUTH[2:]):
        return "auth"
    if status == 429 or any(m.lower() in low for m in RATE_LIMIT):
        return "quota_daily" if any(m.lower() in low for m in DAILY_QUOTA) else "rate_limit"
    if any(m.lower() in low for m in TIMEOUT):
        return "timeout"
    if isinstance(status, int) and status >= 500 or any(m.lower() in low for m in SERVER):
        return "server"
    if status == 400 or any(m.lower() in low for m in BAD_REQUEST):
        return "bad_request"
    if status in (401, 403):
        return "auth"
    return "other"


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server hint for a per-minute limit: header or 'retry in Ns' / retryDelay in the message."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
            if value:
                return float(value)
        except (TypeError, ValueError):
            pass
    m = re.search(r"retry(?:Delay)?['\"]?\s*(?:in|:)?\s*['\"]?(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else None


class LLMClient:
    """Shared loop: model, retries with pauses, then the next fallback model. Stateless between calls."""

    provider = "base"

    def __init__(self, settings: Settings):
        self._settings = settings
        self.last_model: str | None = None  # only for generate()
        self._lock = threading.Lock()
        self._exhausted: set[str] = set()  # models that hit the daily quota in this process

    @property
    def model(self) -> str:
        return self._settings.model

    def _models(self) -> list[str]:
        models = [m for m in (self._settings.model, *self._settings.fallback_models)]
        with self._lock:
            alive = [m for m in models if m not in self._exhausted]
        return alive or models[-1:]

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        temperature: float | None = None,
        effort: str | None = None,
        max_output_tokens: int | None = None,
        deadline: float | None = None,
    ) -> LLMResponse:
        """One logical call. `deadline` is a time.monotonic() value after which no attempt starts."""
        s = self._settings
        errors: list[str] = []
        last_kind: ErrorKind = "other"
        attempts = 0
        started = time.monotonic()
        for model in self._models():
            timeouts = 0
            for attempt in range(s.max_retries + 1):
                timeout = s.timeout_seconds
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining < MIN_ATTEMPT_SECONDS:
                        raise LLMError("предел времени прогона исчерпан", "deadline")
                    timeout = min(timeout, remaining)
                attempts += 1
                try:
                    resp = self._call(
                        model, prompt, system=system, json_schema=json_schema, temperature=temperature,
                        effort=effort, max_output_tokens=max_output_tokens or s.max_output_tokens, timeout=timeout,
                    )
                    resp.attempts = attempts
                    resp.seconds = time.monotonic() - started
                    return resp
                except Exception as exc:
                    kind = classify_error(exc)
                    last_kind = kind
                    errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:200]}")
                    if kind == "auth":
                        raise LLMError("ключ отклонён провайдером: проверьте ключ в .env", "auth") from exc
                    if kind == "quota_daily":
                        with self._lock:
                            self._exhausted.add(model)
                        break
                    if kind in ("bad_request", "refusal", "other"):
                        break
                    if kind == "timeout":
                        timeouts += 1
                        if timeouts > 1:
                            break
                        continue
                    if attempt >= s.max_retries:
                        break
                    pause = s.retry_base_seconds * (2**attempt)
                    if kind == "rate_limit":
                        hint = retry_after_seconds(exc)
                        pause = min(hint if hint is not None else pause, MAX_RATE_LIMIT_SLEEP)
                    if deadline is not None and time.monotonic() + pause + MIN_ATTEMPT_SECONDS > deadline:
                        raise LLMError("предел времени прогона исчерпан", "deadline") from exc
                    time.sleep(pause)
        raise LLMError("модель недоступна: " + " | ".join(errors[-3:]), last_kind)

    def generate(self, prompt: str, *, system: str | None = None, json_schema: dict | None = None) -> str:
        resp = self.complete(prompt, system=system, json_schema=json_schema)
        self.last_model = resp.model
        return resp.text

    def _call(
        self, model: str, prompt: str, *, system: str | None, json_schema: dict | None,
        temperature: float | None, effort: str | None, max_output_tokens: int, timeout: float,
    ) -> LLMResponse:
        raise NotImplementedError


def strip_schema_keys(schema: dict, keys: tuple[str, ...]) -> dict:
    """Copy of a JSON schema without the given keywords (some providers reject them)."""
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in keys}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(copy.deepcopy(schema))


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=settings.api_key, http_options=types.HttpOptions(timeout=int(settings.timeout_seconds * 1000))
        )

    def _call(self, model, prompt, *, system, json_schema, temperature, effort, max_output_tokens, timeout):
        from google.genai import types

        schema = strip_schema_keys(json_schema, ("maxItems", "minItems", "$comment")) if json_schema else None
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1 if temperature is None else temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if schema else None,
            response_json_schema=schema,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )
        resp = self._client.models.generate_content(model=model, contents=prompt, config=config)
        truncated = False
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            reason = str(getattr(candidates[0], "finish_reason", "") or "")
            truncated = "MAX_TOKENS" in reason
            if "SAFETY" in reason or "PROHIBITED" in reason:
                raise LLMError(f"ответ заблокирован: {reason}", "refusal")
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=resp.text or "",
            model=model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            truncated=truncated,
        )


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        import anthropic

        # SDK retries are off: the shared loop above handles retries and fallbacks
        self._client = anthropic.Anthropic(api_key=settings.api_key, max_retries=0)
        self._native_schema = True  # switched off once if the API rejects structured output

    def _call(self, model, prompt, *, system, json_schema, temperature, effort, max_output_tokens, timeout):
        # temperature is deliberately not sent: the model rejects non-default sampling parameters
        use_native = bool(json_schema) and self._native_schema
        try:
            return self._stream(model, prompt, system, json_schema if use_native else None,
                                json_schema if not use_native else None, effort, max_output_tokens, timeout)
        except Exception as exc:
            if use_native and classify_error(exc) == "bad_request" and _mentions_schema(exc):
                self._native_schema = False
                return self._stream(model, prompt, system, None, json_schema, effort, max_output_tokens, timeout)
            raise

    def _stream(self, model, prompt, system, native_schema, prompt_schema, effort, max_output_tokens, timeout):
        if prompt_schema:
            prompt += (
                "\n\nОтветь только JSON-объектом без пояснений и без markdown, по схеме:\n"
                + json.dumps(prompt_schema, ensure_ascii=False)
            )
        kwargs: dict = {}
        if system:
            kwargs["system"] = system
        output_config: dict = {}
        if native_schema:
            output_config["format"] = {"type": "json_schema", "schema": native_schema}
        if effort:
            output_config["effort"] = effort
        if output_config:
            kwargs["output_config"] = output_config
        # the document comes first in every prompt: mark it as a cacheable prefix shared by all stages
        cut = prompt.find("</document>")
        if cut > 0:
            cut += len("</document>")
            content: list | str = [
                {"type": "text", "text": prompt[:cut], "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt[cut:] or " "},
            ]
        else:
            content = prompt
        t0 = time.monotonic()
        with self._client.messages.stream(
            model=model,
            max_tokens=max_output_tokens,
            messages=[{"role": "user", "content": content}],
            timeout=timeout,
            **kwargs,
        ) as stream:
            for _event in stream:
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"нет ответа за {timeout:.0f} с")
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise LLMError("модель отказалась отвечать", "refusal")
        text = "".join(block.text for block in msg.content if block.type == "text")
        usage = msg.usage
        input_tokens = (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0) + (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            truncated=msg.stop_reason == "max_tokens",
        )


def _mentions_schema(exc: BaseException) -> bool:
    low = str(exc).lower()
    return any(m in low for m in ("output_config", "json_schema", "schema", "format"))


def make_client(settings: Settings) -> LLMClient:
    if settings.provider == "anthropic":
        return AnthropicClient(settings)
    return GeminiClient(settings)
