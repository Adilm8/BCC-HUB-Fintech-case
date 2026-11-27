"""Вызов LLM с повторами и запасными моделями. Провайдер выбирается в .env (LLM_PROVIDER)."""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from validator.config import Settings, extra_provider_module

ErrorKind = Literal[
    "quota_daily", "rate_limit", "timeout", "server", "auth", "bad_request", "invalid_json", "network", "refusal",
    "deadline", "skipped", "disabled", "replay", "other"
]

# only an explicit per-day quota is daily; "quota exceeded for metric" is also sent for per-minute limits
DAILY_QUOTA = ("perday", "per day", "per_day", "requests per day")
RATE_LIMIT = ("resource_exhausted", "rate_limit", "rate limit", "ratelimiterror", "too many requests")
SERVER = ("unavailable", "overloaded", "high demand", "internalservererror", "serviceunavailable", "bad gateway")
NETWORK = ("connecterror", "connectionerror", "apiconnectionerror", "getaddrinfo", "name or service not known",
           "no connection could be made", "connection refused", "network is unreachable", "proxyerror",
           "nodename nor servname", "temporary failure in name resolution", "connection reset")
TIMEOUT = ("timeout", "timed out", "deadline_exceeded")
AUTH = ("authenticationerror", "permissiondenied", "api key not valid", "invalid x-api-key", "api_key_invalid",
        "unauthenticated")
BAD_REQUEST = ("badrequest", "invalid_argument", "invalid_request", "failed_precondition", "not_found",
               "notfounderror")
NETWORK_RETRIES = 1  # a dead connection is not worth a long series of retries
_SECRET = re.compile(r"(AIza[0-9A-Za-z_\-]{10,}|sk-[A-Za-z0-9_\-]{10,}|key=[^&\s]+|Bearer\s+\S+)")


def _code(low: str, *codes: int) -> bool:
    """Status codes matched as whole numbers, never as parts of other numbers."""
    return any(re.search(rf"(?<![\d.]){c}(?![\d.])", low) for c in codes)


_KNOWN_SECRETS: list[str] = []


def register_secret(value: str) -> None:
    """Exact values (the configured key) that are always masked in messages."""
    if value and len(value) >= 6 and value not in _KNOWN_SECRETS:
        _KNOWN_SECRETS.append(value)


def short_error(exc: BaseException, limit: int = 200) -> str:
    """First line of the error, without keys or tokens: safe to show in a report."""
    first = (str(exc).strip().splitlines() or [""])[0]
    for secret in _KNOWN_SECRETS:
        first = first.replace(secret, "***")
    first = re.sub(r"(x-(?:goog-)?api-key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,]+", r"\1***", first, flags=re.IGNORECASE)
    first = _SECRET.sub("***", first)
    text = f"{type(exc).__name__}: {first}" if not isinstance(exc, LLMError) else first
    return text if len(text) <= limit else text[: limit - 1] + "…"

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
    notes: list[str] = field(default_factory=list)  # failed attempts and model switches, in Russian


ATTEMPT_REASONS = {
    "quota_daily": "дневная квота модели исчерпана",
    "rate_limit": "лимит запросов в минуту",
    "timeout": "нет ответа за отведённое время",
    "server": "сервис модели перегружен или недоступен",
    "bad_request": "API отклонило запрос",
    "network": "нет связи с API",
    "refusal": "модель отказалась отвечать",
    "other": "ошибка вызова",
}


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
    if isinstance(exc, UnicodeEncodeError):
        return "auth"  # a key with non-ASCII characters cannot go into an HTTP header
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(status, int):
        status = None
    if status in (401, 403) or any(m in low for m in AUTH) or (status is None and _code(low, 401) and
                                                              "unauth" in low):
        return "auth"
    if any(m in low for m in NETWORK):
        return "network"
    if status == 429 or _code(low, 429) or any(m in low for m in RATE_LIMIT) or "exceeded your current quota" in low:
        hint = retry_after_seconds(exc)
        # a per-day quota id, or a wait far beyond a minute, will not recover within this run
        if any(m in low for m in DAILY_QUOTA) or (hint is not None and hint > 300):
            return "quota_daily"
        return "rate_limit"
    if any(m in low for m in TIMEOUT):
        return "timeout"
    if (status is not None and status >= 500) or _code(low, 500, 502, 503, 504, 529) or any(m in low for m in SERVER):
        return "server"
    if status in (400, 404, 422) or _code(low, 400, 404) or any(m in low for m in BAD_REQUEST):
        return "bad_request"
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
    # "retry in 41.5s", "retry in 1h10m14.3s", "retryDelay": "41s"
    m = re.search(r"retry(?:Delay)?['\"]?\s*(?:in|:)?\s*['\"]?((?:\d+h)?(?:\d+m(?!s))?(?:\d+(?:\.\d+)?s)?)",
                  str(exc), re.IGNORECASE)
    if not m or not m.group(1):
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1))
    factor = {"h": 3600.0, "m": 60.0, "s": 1.0}
    return sum(float(v) * factor[u] for v, u in parts) if parts else None


class LLMClient:
    """Shared loop: model, retries with pauses, then the next fallback model. Stateless between calls."""

    provider = "base"

    def __init__(self, settings: Settings):
        self._settings = settings
        register_secret(settings.api_key)
        self.last_model: str | None = None  # only for generate()
        self._lock = threading.Lock()
        self._exhausted: set[str] = set()  # models that hit the daily quota in this process
        self._local = threading.local()  # per-thread sink for attempt notes (progress output)

    def set_note_sink(self, fn) -> None:
        """Receive a short Russian note about every failed attempt or model switch in this thread."""
        self._local.fn = fn

    def _note(self, notes: list[str], text: str) -> None:
        notes.append(text)
        fn = getattr(self._local, "fn", None)
        if fn is not None:
            try:
                fn(text)
            except Exception:  # noqa: BLE001 - progress output must never break a call
                notes.append("")

    @property
    def model(self) -> str:
        return self._settings.model

    def cache_models(self, primary: str | None = None) -> list[str]:
        """Models whose cached answers may stand in for `primary`: itself, and the fallbacks once it is out of
        daily quota in this process (a fresh answer of the primary model is preferred while it works)."""
        first = primary or self._settings.model
        with self._lock:
            out_of_quota = first in self._exhausted
        if not out_of_quota:
            return [first]
        return [first, *(m for m in self._settings.fallback_models if m != first)]

    def _models(self, primary: str | None = None) -> list[str]:
        first = primary or self._settings.model
        models = [first, *(m for m in self._settings.fallback_models if m != first)]
        with self._lock:
            alive = [m for m in models if m not in self._exhausted]
        if not alive:
            raise LLMError("исчерпана дневная квота всех моделей", "quota_daily")
        return alive

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
        model: str | None = None,
    ) -> LLMResponse:
        """One logical call. `deadline` is a time.monotonic() value after which no attempt starts; `model`
        replaces the primary model for this call (the fallbacks stay)."""
        s = self._settings
        errors: list[str] = []
        last_kind: ErrorKind = "other"
        attempts = 0
        started = time.monotonic()
        notes: list[str] = []
        chain = self._models(model)
        for index, model in enumerate(chain):  # noqa: PLR1704
            if index > 0:
                self._note(notes, f"переход на запасную модель {model}")
            timeouts = 0
            for attempt in range(s.max_retries + 1):
                timeout = s.attempt_timeout
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
                    resp.notes = [n for n in notes if n]
                    return resp
                except Exception as exc:
                    kind = classify_error(exc)
                    last_kind = kind
                    errors.append(f"{model}: {short_error(exc)}")
                    spent = time.monotonic() - started
                    self._note(notes, f"{model}: {ATTEMPT_REASONS.get(kind, ATTEMPT_REASONS['other'])} "
                                      f"(попытка {attempt + 1}, {spent:.0f} с от начала вызова)")
                    if kind == "auth":
                        raise LLMError("ключ отклонён провайдером: проверьте ключ в .env", "auth") from exc
                    if kind == "network":
                        if attempt < NETWORK_RETRIES:
                            time.sleep(2.0)
                            continue
                        # another model will not fix a dead connection
                        raise LLMError("нет связи с API модели: " + short_error(exc), "network") from exc
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
                        self._note(notes, f"{model}: пауза {pause:.0f} с по требованию сервера")
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
            api_key=settings.api_key, http_options=types.HttpOptions(timeout=int(settings.attempt_timeout * 1000))
        )
        self._no_thinking: set[str] = set()

    _THINKING: ClassVar[dict[str, str]] = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}

    def _call(self, model, prompt, *, system, json_schema, temperature, effort, max_output_tokens, timeout):
        from google.genai import types

        schema = strip_schema_keys(json_schema, ("maxItems", "minItems", "$comment")) if json_schema else None
        level = self._THINKING.get((effort or "").lower()) if model not in self._no_thinking else None

        def config(with_thinking: bool):
            extra = {}
            if with_thinking and level:
                extra["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            if temperature is not None:
                extra["temperature"] = temperature
            return types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json" if schema else None,
                response_json_schema=schema,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
                **extra,
            )

        try:
            resp = self._client.models.generate_content(model=model, contents=prompt, config=config(True))
        except Exception as exc:
            if not level or classify_error(exc) != "bad_request" or "think" not in str(exc).lower():
                raise
            self._no_thinking.add(model)  # this model does not accept a thinking level
            resp = self._client.models.generate_content(model=model, contents=prompt, config=config(False))
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


class DisabledClient(LLMClient):
    """No model at all: every call fails at once, so the run keeps only the rule checks."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.provider = settings.provider

    def complete(self, prompt: str, **kw) -> LLMResponse:
        raise LLMError("запуск в режиме только правил", "disabled")

    def _call(self, model, prompt, **kw) -> LLMResponse:  # pragma: no cover - complete() never gets here
        raise LLMError("запуск в режиме только правил", "disabled")


class OpenRouterError(RuntimeError):
    """An HTTP error answer: carries the status and headers for classify_error and retry_after_seconds."""

    def __init__(self, message: str, status_code: int, headers: dict[str, str]):
        super().__init__(message)
        self.status_code = status_code
        self.response = type("Response", (), {"headers": headers})()


class OpenRouterClient(LLMClient):
    """OpenAI-compatible Chat Completions on openrouter.ai; standard library only."""

    provider = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._native_schema = True  # switched off once if a model rejects response_format

    def _post(self, body: dict, timeout: float) -> tuple[int, dict[str, str], bytes]:
        """One HTTP request; replaced in tests."""
        import urllib.error
        import urllib.request

        req = urllib.request.Request(self.URL, data=json.dumps(body).encode("utf-8"), method="POST", headers={
            "Authorization": f"Bearer {self._settings.api_key}", "Content-Type": "application/json",
            "X-Title": "requirements-validator"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # a fixed https URL
                return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}, exc.read() or b""

    def _call(self, model, prompt, *, system, json_schema, temperature, effort, max_output_tokens, timeout):
        use_native = bool(json_schema) and self._native_schema
        try:
            return self._request(model, prompt, system, json_schema if use_native else None,
                                 json_schema if not use_native else None, temperature, effort, max_output_tokens,
                                 timeout)
        except OpenRouterError as exc:
            if use_native and exc.status_code == 400:
                self._native_schema = False  # the schema goes into the prompt from now on
                return self._request(model, prompt, system, None, json_schema, temperature, effort,
                                     max_output_tokens, timeout)
            raise

    def _request(self, model, prompt, system, native_schema, prompt_schema, temperature, effort, max_output_tokens,
                 timeout) -> LLMResponse:
        if prompt_schema:
            prompt += ("\n\nОтветь только JSON-объектом без пояснений и без markdown, по схеме:\n"
                       + json.dumps(prompt_schema, ensure_ascii=False))
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict = {"model": model, "messages": messages, "max_tokens": max_output_tokens}
        if effort:
            body["reasoning"] = {"effort": {"minimal": "low"}.get(effort, effort)}
        if temperature is not None:
            body["temperature"] = temperature
        if native_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "out", "strict": False, "schema": native_schema}}
        status, headers, raw = self._post(body, timeout)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        err = data.get("error")
        if status >= 400 or err:
            # an upstream failure may also come as status 200 with an error object
            err = err if isinstance(err, dict) else {}
            code = status if status >= 400 else (int(err["code"]) if str(err.get("code", "")).isdigit() else 500)
            text = f"{code} {err.get('message') or raw[:200].decode('utf-8', 'replace')}"
            if code == 402:
                # no credits: another attempt or model will not help within this run
                raise LLMError(f"на счёте OpenRouter недостаточно кредитов: {short_error(RuntimeError(text))}",
                               "quota_daily")
            raise OpenRouterError(text, code, headers)
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if isinstance(content, list):  # some models answer with content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        usage = data.get("usage") or {}
        # the requested name, not the routed variant: cache keys and fallbacks compare by it
        return LLMResponse(text=content, model=model,
                           input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                           truncated=choice.get("finish_reason") == "length")


def make_client(settings: Settings) -> LLMClient:
    if settings.provider == "anthropic":
        return AnthropicClient(settings)
    if settings.provider == "openrouter":
        return OpenRouterClient(settings)
    extra = extra_provider_module()
    if extra is not None and settings.provider == getattr(extra, "NAME", None):
        return extra.make_client(settings)
    return GeminiClient(settings)
