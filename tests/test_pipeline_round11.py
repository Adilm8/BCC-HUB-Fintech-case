from validator.pipeline.prompts.examples import SET_EXAMPLES
from validator.pipeline.prompts.io import LLMSetFinding
from validator.pipeline.prompts.setcheck import S1_TASK


def test_mixed_levels_lens_is_on_and_accepts_appropriate():
    assert "criterion appropriate" in S1_TASK
    assert "смешения уровней" in SET_EXAMPLES
    finding = LLMSetFinding(evidence=[], scenario="", explanation="x", set_check="other",
                            criterion="appropriate", severity="major", recommendation="r",
                            clarifying_question="q", answer_options=[])
    assert finding.criterion == "appropriate"


def _req(rid, section, text):
    from validator.models import Requirement

    return Requirement(id=rid, id_assigned=True, kind="functional", part="p", section=section,
                       block_ids=[], title=rid, quote=text)


def test_pairs_share_an_entity_and_are_capped():
    from validator.pipeline import experiments as exp

    reqs = [_req("R1", "s1", "Продление выдачи не более 3 раз в месяц."),
            _req("R2", "s2", "Счётчик продлений сбрасывается при изменении тарифа, лимит в месяц прежний."),
            _req("R3", "s1", "Поле `reader_id` обязательно, при ошибке код 404."),
            _req("R4", "s3", "Если `reader_id` не найден, вернуть 400."),
            _req("R5", "s3", "Экран показывает логотип.")]
    pairs = exp.pair_candidates(reqs)
    ids = {(a, b) for a, b, _ in pairs}
    assert ("R1", "R2") in ids and ("R3", "R4") in ids
    assert all("R5" not in p for p in ids)
    assert pairs[0][:2] == ("R1", "R2")  # rule-like overlap first
    many = [_req(f"R{i}", f"s{i}", "лимит 5 в месяц") for i in range(12)]
    assert len(exp.pair_candidates(many)) == 25


def test_pair_enum_toggle_adds_the_pair_list_to_s1(monkeypatch):
    from types import SimpleNamespace

    from validator.pipeline.stages import setcheck

    reqs = [_req("R1", "s1", "не более 3 раз в месяц"), _req("R2", "s2", "лимит в месяц сбрасывается")]
    seen = []
    ctx = SimpleNamespace(include_document=lambda n: False, call=lambda *a: seen.append(a[2]))
    doc = SimpleNamespace(char_count=10, parts=[])
    monkeypatch.setattr(setcheck, "build_s1_prompt",
                        lambda d, r, i, **kw: f"{kw['pairs']}|{kw['second_pass']}")
    setcheck.run_s1(ctx, doc, reqs, [])
    monkeypatch.setenv("EXP_PAIR_ENUM", "1")
    setcheck.run_s1(ctx, doc, reqs, [])
    assert seen[0] == "|False"
    assert seen[1].startswith("P1: R1 ↔ R2")


def test_a_routed_call_uses_its_model_first_and_keeps_the_fallbacks():
    from tests.test_pipeline_llm import Scripted, _settings
    from validator.llm import LLMError

    c = Scripted(_settings(), ["ok"])
    assert c.complete("p", model="m9").model == "m9"
    c = Scripted(_settings(), [LLMError("x", "bad_request"), "ok"])
    assert c.complete("p", model="m9").model == "m2"
    assert c.models_called == ["m9", "m2"]
    c = Scripted(_settings(), ["ok"])
    assert c.complete("p").model == "m1"


def test_stage_model_env_routes_only_that_stage(monkeypatch):
    from tests.fakes import FakeLLM, stage_of
    from tests.test_pipeline_run import raw_doc, settings
    from validator.pipeline import run_pipeline

    seen = []

    class Spy(FakeLLM):
        def complete(self, prompt, *, model=None, **kw):
            seen.append((stage_of(kw.get("json_schema")), model))
            return super().complete(prompt, **kw)

    run_pipeline(raw_doc(), settings(), client=Spy())
    assert all(m is None for _, m in seen)  # no routing: nothing extra is sent
    seen.clear()
    monkeypatch.setenv("LLM_MODEL_REVIEW", "gemini-pro-x")
    monkeypatch.setenv("LLM_MODEL_EXTRACT", "gemini-lite-x")
    rep = run_pipeline(raw_doc(), settings(), client=Spy())
    routed = dict(seen)
    assert routed["review"] == "gemini-pro-x" and routed["extract"] == "gemini-lite-x"
    assert routed["check"] is None and routed["set"] is None
    assert any("LLM_MODEL_REVIEW" in w for w in rep.run.warnings)


def test_rules_only_flag_writes_a_report_without_a_key(tmp_path, monkeypatch, capsys):
    import json

    from validator.cli import main

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")  # an empty value is not replaced from .env
    doc = tmp_path / "spec.md"
    doc.write_text("# Т\n\n## Функции\n\n- REQ-1. Система должна быстро обрабатывать заявку и т.д.\n",
                   encoding="utf-8")
    out = tmp_path / "r.md"
    assert main([str(doc), str(out)]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
    assert main([str(doc), str(out), "--rules-only"]) == 1
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["run"]["status"] == "rules_only" and data["run"]["llm_calls"] == 0
    assert data["verdict"]["headline"].startswith("Итог не определён: проверка неполная")
    assert data["verdict"]["development"] == "no"
    assert out.read_text(encoding="utf-8")
