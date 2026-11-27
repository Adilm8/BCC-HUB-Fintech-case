import json

import pytest

from tests.fakes import FakeLLM
from tests.test_pipeline_run import TEXT, raw_doc, settings
from validator.cli import main
from validator.llm import LLMError
from validator.pipeline import run_pipeline
from validator.trace import ReplayClient, TracingClient


def _ids(rep):
    return [(f.id, f.explanation) for f in rep.findings] + [(f.id, f.explanation) for f in rep.set_findings]


def test_a_trace_replays_the_same_report_offline(tmp_path):
    s = settings(api_key="sk-secret-value-123")
    rec = run_pipeline(raw_doc(), s, client=TracingClient(FakeLLM(), tmp_path, s))
    files = sorted(p.name for p in tmp_path.glob("call_*.json"))
    assert files and (tmp_path / "manifest.json").is_file()
    first = json.loads((tmp_path / files[0]).read_text(encoding="utf-8"))
    assert {"stage", "prompt_sha256", "model", "text", "input_tokens", "output_tokens"} <= set(first)
    assert "api_key" not in (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert s.api_key not in "".join(p.read_text(encoding="utf-8") for p in tmp_path.iterdir())

    again = run_pipeline(raw_doc(), s, client=ReplayClient(tmp_path))
    assert _ids(again) == _ids(rec) and again.verdict == rec.verdict
    assert again.run.llm_calls == rec.run.llm_calls


def test_a_changed_prompt_stops_the_replay_with_a_clear_error(tmp_path):
    s = settings()
    run_pipeline(raw_doc(), s, client=TracingClient(FakeLLM(), tmp_path, s))
    changed = raw_doc().model_copy(update={"text": TEXT.replace("240 минут", "300 минут")})
    with pytest.raises(LLMError) as err:
        run_pipeline(changed, s, client=ReplayClient(tmp_path))
    assert err.value.kind == "replay" and "--save-trace" in str(err.value)


def test_cli_replay_needs_no_key(tmp_path, monkeypatch, capsys):
    doc = tmp_path / "rooms.md"
    doc.write_text(TEXT, encoding="utf-8")
    trace = tmp_path / "trace"
    s = settings()
    rec = run_pipeline(doc, s, client=TracingClient(FakeLLM(), trace, s))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    out = tmp_path / "out.md"
    code = main([str(doc), str(out), "--replay", str(trace)])
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert code == (0 if rec.run.status.value == "complete" else 1)
    assert [f["explanation"] for f in data["findings"]] == [f.explanation for f in rec.findings]
    doc.write_text(TEXT + "\n- **BR-4.** Новое правило.\n", encoding="utf-8")
    assert main([str(doc), str(out), "--replay", str(trace)]) == 2
    assert "промпт изменился" in capsys.readouterr().err
    assert main([str(doc), str(out), "--replay", str(tmp_path / "none")]) == 2
