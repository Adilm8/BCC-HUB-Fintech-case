"""Reliability on the jury's machine: quota vs rate limit, network outage, schemas, CLI paths."""

import time

import pytest

from tests.fakes import FakeLLM
from tests.test_pipeline_llm import Scripted, _settings
from tests.test_pipeline_run import raw_doc, settings
from validator.cli import main
from validator.llm import LLMError, classify_error, short_error
from validator.models import RunStatus, StageKey, StageState
from validator.pipeline import run_pipeline
from validator.pipeline.prompts.io import LLMExtraction
from validator.pipeline.structured import provider_schema

PER_MINUTE = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota. Quota exceeded for metric: "
              "generativelanguage.googleapis.com/generate_content_free_tier_requests, quotaId: "
              "GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 41.5s.")
PER_DAY = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
           "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")


def test_per_minute_limit_is_not_a_daily_quota():
    assert classify_error(RuntimeError(PER_MINUTE)) == "rate_limit"
    assert classify_error(RuntimeError(PER_DAY)) == "quota_daily"


def test_status_codes_are_whole_numbers():
    assert classify_error(RuntimeError("ReadError: response of 4290 tokens")) == "other"
    assert classify_error(RuntimeError("port 15003 closed")) == "other"
    assert classify_error(RuntimeError("HTTP 503 Service temporarily")) == "server"


def test_network_errors_fail_fast_without_fallback(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    err = RuntimeError("ConnectError: [Errno 11001] getaddrinfo failed")
    c = Scripted(_settings(max_retries=4), [err, err, err, err, err])
    with pytest.raises(LLMError) as e:
        c.complete("p")
    assert e.value.kind == "network" and c.models_called == ["m1", "m1"]


def test_network_outage_stops_remaining_stages():
    err = LLMError("нет связи с API модели: ConnectError: getaddrinfo failed", "network")
    rep = run_pipeline(raw_doc(), settings(), client=FakeLLM({"extract": err, "check": err, "set": err,
                                                              "review": err}))
    st = {s.key: s for s in rep.run.stages}
    assert st[StageKey.EXTRACT].state == StageState.FAILED
    assert "нет связи" in st[StageKey.EXTRACT].detail and "getaddrinfo" in st[StageKey.EXTRACT].detail
    assert st[StageKey.CHECK].state == StageState.SKIPPED
    assert rep.run.status == RunStatus.RULES_ONLY


def test_short_error_hides_keys():
    msg = short_error(RuntimeError("400 bad key=AIzaSyA1234567890abcdef in request\nsecond line"))
    assert "AIza" not in msg and "second line" not in msg


def test_schema_keeps_a_field_named_title():
    schema = provider_schema(LLMExtraction)
    item = schema["properties"]["requirements"]["items"]
    assert "title" in item["properties"] and "title" in item["required"]


def test_long_document_is_cut_by_rules_without_marking_the_run_partial():
    fake = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(max_context_chars=50), client=fake)
    assert all(stage != "extract" for stage, _ in fake.calls)
    assert rep.run.status == RunStatus.COMPLETE
    assert any("длиннее" in w for w in rep.run.warnings)


def test_cli_refuses_to_overwrite_the_input(tmp_path, capsys):
    doc = tmp_path / "spec.md"
    doc.write_text("# Т\n\n- REQ-1. Система делает что-то важное.\n", encoding="utf-8")
    assert main([str(doc), str(doc)]) == 2
    assert "совпадает" in capsys.readouterr().err
    assert doc.read_text(encoding="utf-8").startswith("# Т")


def test_cli_json_never_overwrites_markdown(tmp_path):
    from types import SimpleNamespace

    from validator.cli import _json_path

    out = tmp_path / "report.json"
    path = _json_path(SimpleNamespace(no_json=False, json=None, output=out))
    assert path != out and path.name == "report.report.json"


class SequentialFake(FakeLLM):
    provider = "gemini"


def test_sequential_provider_runs_set_checks_before_check_batches():
    fake = SequentialFake()
    run_pipeline(raw_doc(), settings(), client=fake)
    order = [s for s, _ in fake.calls if s in ("check", "set")]
    assert order[:2] == ["set", "set"] and "check" in order[2:]


def test_check_batches_are_dropped_first_and_review_keeps_its_slot():
    fake = SequentialFake()
    rep = run_pipeline(raw_doc(), settings(run_deadline_seconds=50), client=fake)
    st = {s.key: s for s in rep.run.stages}
    assert st[StageKey.SET_CHECK].state == StageState.DONE
    assert st[StageKey.REVIEW].state == StageState.DONE
    assert all(r.check_status == "not_checked" for r in rep.requirements)
    assert any("пропущено" in w or "Не проверено" in w for w in rep.run.warnings)


def test_attempt_notes_reach_warnings(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    c = Scripted(_settings(), [RuntimeError(PER_MINUTE), "ok"])
    seen = []
    c.set_note_sink(seen.append)
    r = c.complete("p")
    assert r.text == "ok" and any("лимит запросов в минуту" in n for n in r.notes) and seen


def test_provider_deadlines():
    from validator.config import Settings

    assert Settings(provider="gemini", api_key="k").run_deadline == 900
    assert Settings(provider="anthropic", api_key="k").run_deadline == 480


REAL_DAILY = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota. * Quota exceeded for metric: "
              "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: m, "
              "description: Generate content requests per day limit exceeded on Free Tier.\nPlease retry in "
              "1h10m14.349582645s. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")


def test_real_daily_quota_message_with_long_retry_is_daily():
    from validator.llm import retry_after_seconds

    assert abs(retry_after_seconds(RuntimeError(REAL_DAILY)) - 4214.35) < 0.1
    assert retry_after_seconds(RuntimeError(PER_MINUTE)) == 41.5
    assert classify_error(RuntimeError(REAL_DAILY)) == "quota_daily"


def test_daily_quota_switches_to_fallback_without_waiting(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    c = Scripted(_settings(), [RuntimeError(REAL_DAILY), "ok"])
    r = c.complete("p")
    assert r.model == "m2" and not slept
    assert any("дневная квота" in n for n in r.notes) and any("запасную модель" in n for n in r.notes)


def test_gemini_runs_within_seven_calls_without_a_separate_pair_call():
    fake = SequentialFake()
    rep = run_pipeline(raw_doc(), settings(), client=fake)
    prompts = [p for s, p in fake.calls if s == "set"]
    assert not any("S1b" in p for p in prompts)  # the rule-pair lens lives inside S1
    assert rep.run.llm_calls <= 7


def test_other_providers_do_not_get_the_rule_pair_call():
    fake = FakeLLM()
    run_pipeline(raw_doc(), settings(), client=fake)
    assert not any("S1b" in p for s, p in fake.calls if s == "set")


def test_linter_flags_code_declared_but_other_code_prescribed_under_its_heading():
    from pathlib import Path

    from validator.loaders import load_document
    from validator.rules import run_prechecks
    from validator.structure import parse_structure

    root = Path(__file__).resolve().parents[1]
    doc = parse_structure(load_document(root / "docs" / "requirements.md"))
    found = [f for f in run_prechecks(doc).findings if f.rule == "api_status_code_conflict"]
    assert found and found[0].severity == "critical"
    assert "403" in found[0].explanation and "404" in found[0].explanation


def test_default_gemini_fallback(monkeypatch):
    from validator.config import Settings

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    assert Settings.from_env().fallback_models == ("gemini-3.5-flash-lite",)
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    assert Settings.from_env().fallback_models == ()



def test_extra_provider_hook(tmp_path, monkeypatch):
    from validator import config
    from validator.config import Settings
    from validator.llm import make_client

    mod = tmp_path / "extra.py"
    mod.write_text(
        "from tests.fakes import FakeLLM\n"
        "NAME = 'devlocal'\nCONFIG = ('', 'DEVLOCAL_MODEL', 'm-dev', 'DEVLOCAL_FALLBACK')\n"
        "FAMILY = 'gemini'\nTIMEOUT = 11.0\nDEADLINE = 22.0\n"
        "def make_client(settings):\n    return FakeLLM(model=settings.model)\n", encoding="utf-8")
    monkeypatch.setattr(config, "_EXTRA", None)
    monkeypatch.setenv("VALIDATOR_EXTRA_PROVIDER", str(mod))
    monkeypatch.setenv("LLM_PROVIDER", "devlocal")
    s = Settings.from_env()
    assert s.provider == "devlocal" and s.model == "m-dev" and s.attempt_timeout == 11.0
    assert config.family("devlocal") == "gemini" and "devlocal" in config.SEQUENTIAL_PROVIDERS
    assert make_client(s).model == "m-dev"


def test_all_models_out_of_quota_fail_at_once(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    c = Scripted(_settings(), [RuntimeError(PER_DAY), RuntimeError(PER_DAY)])
    with pytest.raises(LLMError):
        c.complete("p")
    with pytest.raises(LLMError) as e:
        c.complete("p")  # no model left: no new request at all
    assert e.value.kind == "quota_daily" and c.models_called == ["m1", "m2"]


def test_fatal_error_skips_waiting_wave_tasks():
    from tests.fakes import quota_error

    fake = SequentialFake({"set": quota_error()})
    rep = run_pipeline(raw_doc(), settings(), client=fake)
    assert not any(s == "check" for s, _ in fake.calls)  # check batches were not started after the fatal error
    assert rep.verdict.development != "yes"


def test_non_ascii_key_is_rejected(monkeypatch):
    from validator.config import Settings

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "ваш_ключ")
    with pytest.raises(ValueError, match="недопустимые"):
        Settings.from_env()


def test_experiments_are_off_by_default_and_each_toggle_works(monkeypatch):
    from validator.pipeline import experiments as exp

    base = FakeLLM()
    run_pipeline(raw_doc(), settings(), client=base)
    base_stages = [s for s, _ in base.calls]
    assert "entities" not in base_stages and exp.active() == []

    monkeypatch.setenv("EXP_REPEAT_INSTRUCTION", "1")
    f = FakeLLM()
    run_pipeline(raw_doc(), settings(), client=f)
    check_prompt = next(p for s, p in f.calls if s == "check")
    assert check_prompt.startswith("Задание") and check_prompt.count("Задание: проверь") == 2
    monkeypatch.delenv("EXP_REPEAT_INSTRUCTION")

    monkeypatch.setenv("EXP_SCHEMA_REASONING_FIRST", "1")
    seen = {}

    class Spy(FakeLLM):
        def complete(self, prompt, **kw):
            if stage_of(kw.get("json_schema")) == "check":
                seen["schema"] = kw["json_schema"]
            return super().complete(prompt, **kw)

    from tests.fakes import stage_of

    rep = run_pipeline(raw_doc(), settings(), client=Spy())
    item = seen["schema"]["properties"]["results"]["items"]["properties"]["findings"]["items"]
    assert next(iter(item["properties"])) == "reasoning" and rep.findings
    monkeypatch.delenv("EXP_SCHEMA_REASONING_FIRST")

    monkeypatch.setenv("EXP_ENTITY_PASS", "1")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    stages = [s for s, _ in f.calls]
    assert "entities" in stages and rep.run.llm_calls <= 7
    s2 = [p for s, p in f.calls if s == "set" and "<entities>" in p]
    assert s2
    monkeypatch.delenv("EXP_ENTITY_PASS")

    monkeypatch.setenv("EXP_DIMENSION_PASS", "1")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    checks = [p for s, p in f.calls if s == "check"]
    assert len(checks) == 2 and all("измерениям" in p for p in checks)
    assert all(r.check_status == "checked" for r in rep.requirements)


def test_two_pass_set_check_adds_one_reordered_s1_call(monkeypatch):
    base = FakeLLM()
    rep0 = run_pipeline(raw_doc(), settings(), client=base)
    monkeypatch.setenv("EXP_TWO_PASS", "1")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    sets = [p for s, p in f.calls if s == "set"]
    assert len(sets) == len([s for s, _ in base.calls if s == "set"]) + 1
    assert any("второй, независимый проход" in p for p in sets)
    assert rep.run.llm_calls == rep0.run.llm_calls + 1 <= 8
    assert rep.run.status == rep0.run.status
