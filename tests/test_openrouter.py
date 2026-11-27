import json

import pytest

from tests.test_pipeline_run import raw_doc
from validator.config import SEQUENTIAL_PROVIDERS, Settings, estimate_cost, family
from validator.llm import LLMError, OpenRouterClient, classify_error, make_client
from validator.pipeline import run_pipeline


def _settings(**kw):
    base = {"provider": "openrouter", "api_key": "sk-or-secret-123456", "model": "google/gemini-3.8-flash",
            "fallback_models": ("vendor/backup",), "max_retries": 1, "retry_base_seconds": 0.0, "cache_enabled": False}
    base.update(kw)
    return Settings(**base)


class Mocked(OpenRouterClient):
    """Answers from a script of (status, headers, body) or a callable(body) -> that tuple."""

    def __init__(self, settings, script):
        super().__init__(settings)
        self.script = list(script)
        self.bodies = []

    def _post(self, body, timeout):
        self.bodies.append(body)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if callable(item):
            item = item(body)
        status, headers, data = item
        return status, headers, json.dumps(data).encode() if isinstance(data, dict) else data


def ok(text, finish="stop"):
    return 200, {}, {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def test_request_body_and_response_parsing():
    c = Mocked(_settings(), [ok('{"a": 1}')])
    resp = c.complete("p", system="s", json_schema={"type": "object"}, effort="high", max_output_tokens=99)
    body = c.bodies[0]
    assert body["model"] == "google/gemini-3.8-flash" and body["max_tokens"] == 99
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]
    assert body["reasoning"] == {"effort": "high"}
    assert body["response_format"]["json_schema"] == {"name": "out", "strict": False, "schema": {"type": "object"}}
    assert resp.text == '{"a": 1}' and resp.input_tokens == 11 and resp.output_tokens == 7 and not resp.truncated
    assert Mocked(_settings(), [ok("x", "length")]).complete("p").truncated


def test_schema_goes_into_the_prompt_after_a_400():
    c = Mocked(_settings(), [(400, {}, {"error": {"code": 400, "message": "response_format not supported"}}),
                             ok("{}")])
    c.complete("p", json_schema={"type": "object"})
    assert "response_format" not in c.bodies[1] and "по схеме" in c.bodies[1]["messages"][-1]["content"]
    c.complete("q", json_schema={"type": "object"})
    assert "response_format" not in c.bodies[2]


def test_error_kinds():
    with pytest.raises(LLMError) as e:
        Mocked(_settings(), [(401, {}, {"error": {"code": 401, "message": "No auth"}})]).complete("p")
    assert e.value.kind == "auth"
    c = Mocked(_settings(), [(402, {}, {"error": {"code": 402, "message": "Insufficient credits"}})])
    with pytest.raises(LLMError) as e:
        c.complete("p")
    assert e.value.kind == "quota_daily" and len(c.bodies) == 2  # each model once, no retries
    c = Mocked(_settings(), [(429, {"retry-after": "0"}, {"error": {"code": 429, "message": "rate"}}), ok("y")])
    assert c.complete("p").text == "y" and len(c.bodies) == 2
    c = Mocked(_settings(), [(503, {}, b"busy"), (503, {}, b"busy"), ok("z")])
    resp = c.complete("p")
    assert resp.text == "z" and resp.model == "vendor/backup"
    c = Mocked(_settings(max_retries=0), [(200, {}, {"error": {"code": 502, "message": "upstream"}}), ok("w")])
    assert c.complete("p").text == "w"


def test_key_never_reaches_messages():
    c = Mocked(_settings(), [(401, {}, {"error": {"message": "bad key sk-or-secret-123456"}})])
    with pytest.raises(LLMError) as e:
        c.complete("p")
    assert "sk-or-secret-123456" not in str(e.value)


def test_provider_family_price_and_pipeline():
    assert family("openrouter") == "gemini" and "openrouter" in SEQUENTIAL_PROVIDERS
    assert isinstance(make_client(_settings()), OpenRouterClient)
    cost, _note = estimate_cost({"google/gemini-3.8-flash": (1_000_000, 1_000_000)}, "openrouter")
    assert cost == 4.5
    assert estimate_cost({"vendor/x": (5, 5)}, "openrouter") == (None, "цена модели неизвестна")
    from tests.fakes import default_check, default_extract, default_review, default_set, stage_of

    def answer(body):
        schema = (body.get("response_format") or {}).get("json_schema", {}).get("schema")
        prompt = body["messages"][-1]["content"]
        fn = {"extract": default_extract, "check": default_check, "set": default_set,
              "review": default_review}[stage_of(schema)]
        return ok(json.dumps(fn(prompt, schema), ensure_ascii=False))

    c = Mocked(_settings(), [answer])
    rep = run_pipeline(raw_doc(), _settings(), client=c)
    assert rep.run.status.value == "complete" and rep.run.provider == "openrouter"
    assert all("temperature" not in b for b in c.bodies)  # a gemini-3 model gets no sampling temperature
    assert rep.run.cost_usd is not None and rep.run.cost_usd > 0


def test_classify_known_statuses():
    from validator.llm import OpenRouterError

    assert classify_error(OpenRouterError("403 forbidden", 403, {})) == "auth"
    assert classify_error(OpenRouterError("429 slow down", 429, {})) == "rate_limit"
    assert classify_error(OpenRouterError("500 oops", 500, {})) == "server"
    assert classify_error(OpenRouterError("400 bad", 400, {})) == "bad_request"
