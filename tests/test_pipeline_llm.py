import time
from types import SimpleNamespace

import pytest

from validator.config import Settings
from validator.llm import (
    LLMClient,
    LLMError,
    LLMResponse,
    classify_error,
    retry_after_seconds,
    strip_schema_keys,
)


class Scripted(LLMClient):
    def __init__(self, settings, outcomes):
        super().__init__(settings)
        self.outcomes = list(outcomes)
        self.models_called = []

    def _call(self, model, prompt, **kw):
        self.models_called.append(model)
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(text=item, model=model)


def _settings(**kw):
    base = {"provider": "gemini", "api_key": "k", "model": "m1", "fallback_models": ("m2",), "max_retries": 2,
                "retry_base_seconds": 0.0}
    base.update(kw)
    return Settings(**base)


def test_classify_errors():
    assert classify_error(RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDay")) == "quota_daily"
    assert classify_error(RuntimeError("429 RESOURCE_EXHAUSTED per minute")) == "rate_limit"
    assert classify_error(RuntimeError("503 UNAVAILABLE")) == "server"
    assert classify_error(TimeoutError("x")) == "timeout"
    assert classify_error(RuntimeError("API key not valid")) == "auth"
    assert classify_error(RuntimeError("400 INVALID_ARGUMENT")) == "bad_request"


def test_retry_after_from_message():
    assert retry_after_seconds(RuntimeError("Please retry in 12.5s.")) == 12.5


def test_daily_quota_switches_model_and_sticks():
    c = Scripted(_settings(), [RuntimeError("429 GenerateRequestsPerDay"), "ok", "ok2"])
    r = c.complete("p")
    assert r.text == "ok" and r.model == "m2" and r.attempts == 2
    r2 = c.complete("p")
    assert r2.model == "m2"  # exhausted model is not retried in this run
    assert c.models_called == ["m1", "m2", "m2"]


def test_server_error_retries_same_model():
    c = Scripted(_settings(), [RuntimeError("503 UNAVAILABLE"), "ok"])
    assert c.complete("p").model == "m1"


def test_auth_error_is_immediate():
    c = Scripted(_settings(), [RuntimeError("401 API key not valid")])
    with pytest.raises(LLMError) as err:
        c.complete("p")
    assert err.value.kind == "auth"


def test_deadline_prevents_new_attempt():
    c = Scripted(_settings(), ["ok"])
    with pytest.raises(LLMError) as err:
        c.complete("p", deadline=time.monotonic() + 1)
    assert err.value.kind == "deadline" and c.models_called == []


def test_all_models_fail_reports_last_kind():
    c = Scripted(_settings(max_retries=0), [RuntimeError("503 UNAVAILABLE"), RuntimeError("503 UNAVAILABLE")])
    with pytest.raises(LLMError) as err:
        c.complete("p")
    assert err.value.kind == "server"


def test_strip_schema_keys():
    s = {"type": "array", "maxItems": 3, "items": {"type": "object", "maxItems": 1}}
    assert strip_schema_keys(s, ("maxItems",)) == {"type": "array", "items": {"type": "object"}}


def test_gemini_thinking_level_falls_back_when_rejected(monkeypatch):
    from validator.llm import GeminiClient

    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(config.thinking_config)
            if config.thinking_config is not None:
                raise RuntimeError("400 INVALID_ARGUMENT thinking_level is not supported")

            return SimpleNamespace(text='{"ok": true}', candidates=[], usage_metadata=None)

    c = GeminiClient.__new__(GeminiClient)
    LLMClient.__init__(c, _settings())
    c._client = type("C", (), {"models": Models()})()
    c._no_thinking = set()
    r = c.complete("p", effort="high")
    assert r.text == '{"ok": true}' and calls[0] is not None and calls[-1] is None
    c.complete("p", effort="high")
    assert calls[-1] is None and len(calls) == 3  # the model is remembered as not accepting it


def test_a_cached_fallback_answer_is_used_once_the_primary_is_out_of_quota(tmp_path):
    from validator.pipeline.prompts.io import LLMCheck
    from validator.pipeline.structured import Budget, StageCache, call_structured

    ok = '{"results": []}'
    quota = RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDay")
    cache = StageCache(tmp_path)
    c = Scripted(_settings(), [quota, ok])
    r1 = call_structured(c, LLMCheck, "p1", system="s", budget=Budget(9), cache=cache)
    assert r1.responses[-1].model == "m2" and c.models_called == ["m1", "m2"]
    # same process, primary exhausted: the fallback's cached answer stands in, no call
    r2 = call_structured(c, LLMCheck, "p1", system="s", budget=Budget(9), cache=cache)
    assert r2.cached and c.models_called == ["m1", "m2"]
    # a fresh client whose primary works asks the primary model
    fresh = Scripted(_settings(), [ok])
    r3 = call_structured(fresh, LLMCheck, "p1", system="s", budget=Budget(9), cache=cache)
    assert not r3.cached and fresh.models_called == ["m1"]


def test_high_demand_503_is_retried_then_falls_back():
    busy = RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand. Please try again later.")
    assert classify_error(busy) == "server"
    c = Scripted(_settings(max_retries=1), [busy, busy, "ok"])
    resp = c.complete("p")
    assert resp.model == "m2" and c.models_called == ["m1", "m1", "m2"]
