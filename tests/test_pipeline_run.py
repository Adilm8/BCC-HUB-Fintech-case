import json
import re
from pathlib import Path

import pytest

from tests.fakes import FakeLLM, default_check, quota_error, section
from validator.config import Settings
from validator.llm import LLMError
from validator.models import RawDocument, Report, RunStatus, StageKey, StageState
from validator.pipeline import run_pipeline
from validator.report.markdown import render_markdown

ROOT = Path(__file__).resolve().parents[1]

TEXT = """# Бронирование переговорных

## Функции

- **BR-1.** Сотрудник бронирует переговорную на срок от 15 до 240 минут включительно.
- **BR-2.** Бронь отменяется автоматически, если никто не отметился в течение 10 минут.
- **BR-3.** Система быстро подтверждает бронирование.

## Критерии приёмки

| Условие | Действие | Результат |
| :--- | :--- | :--- |
| Бронь на 9:00 | Нет отметки до 9:20 | Бронь активна |
"""


def raw_doc() -> RawDocument:
    return RawDocument(path="rooms.md", name="rooms.md", title="Бронирование переговорных", format="md", text=TEXT)


def settings(**kw) -> Settings:
    base = {"provider": "gemini", "api_key": "k", "profile_name": "standard", "cache_enabled": False}
    base.update(kw)
    return Settings(**base)


def run(fake, **kw) -> Report:
    return run_pipeline(raw_doc(), settings(**kw), client=fake)


def test_happy_path_complete_report():
    fake = FakeLLM()
    events = []
    rep = run_pipeline(raw_doc(), settings(), client=fake, on_event=events.append)
    assert rep.run.status == RunStatus.COMPLETE
    assert rep.run.llm_calls == len(fake.calls) <= settings().profile.max_calls
    assert [s.key for s in rep.run.stages] == list(StageKey)
    assert all(r.check_status == "checked" for r in rep.requirements)
    assert rep.findings and rep.set_findings
    assert [f.id for f in rep.findings] == [f"D-{i:03d}" for i in range(1, len(rep.findings) + 1)]
    assert rep.set_findings[0].id == "S-001"
    assert all(q.text.endswith("?") for q in rep.questions)
    # every quote comes from the document
    for f in [*rep.findings, *rep.set_findings]:
        for e in f.evidence:
            assert e.quote.split(" … ")[0][:10] in TEXT.replace("**", "")
    assert rep.verdict.development == "no"  # fake set check returns a critical contradiction
    assert any(e.stage == StageKey.CHECK for e in events)
    Report.model_validate_json(rep.model_dump_json())
    assert "## Итог" in render_markdown(rep)


def test_requirements_get_doc_ids_and_numbering():
    rep = run(FakeLLM())
    ids = [r.doc_id for r in rep.requirements]
    assert "BR-1" in ids and "BR-3" in ids
    assert [r.id for r in rep.requirements][:2] == ["R-001", "R-002"]


def test_invented_quote_is_dropped_or_anchored():
    def check(prompt, schema):
        out = default_check(prompt, schema)
        for r in out["results"]:
            for f in r["findings"]:
                f["quote"] = "Этого текста в документе нет совсем никогда"
        return out

    rep = run(FakeLLM({"check": check}))
    for f in rep.findings:
        # anchored findings survive only when the critic confirms them, and never as critical
        assert all(e.quote_status == "anchored" for e in f.evidence)
        assert f.severity != "critical"


def test_daily_quota_on_set_check_gives_partial_report():
    rep = run(FakeLLM({"set": quota_error()}))
    assert rep.run.status == RunStatus.PARTIAL
    st = {s.key: s for s in rep.run.stages}
    assert st[StageKey.SET_CHECK].state == StageState.FAILED
    assert st[StageKey.REVIEW].state == StageState.SKIPPED
    assert rep.findings  # check results are kept
    assert "неполный" in rep.run.warnings[0]


def test_model_unavailable_gives_rules_only():
    err = LLMError("503 UNAVAILABLE", "server")
    rep = run(FakeLLM({"extract": err, "check": err, "set": err, "review": err}))
    assert rep.run.status == RunStatus.RULES_ONLY
    assert rep.requirements and all(r.source == "rule" for r in rep.requirements)
    assert all(r.check_status == "not_checked" for r in rep.requirements)


def test_auth_error_propagates():
    with pytest.raises(LLMError) as err:
        run(FakeLLM({"extract": LLMError("bad key", "auth")}))
    assert err.value.kind == "auth"


def test_invalid_json_is_repaired_locally():
    def extract(prompt, schema):
        from tests.fakes import default_extract

        return "```json\n" + json.dumps(default_extract(prompt, schema), ensure_ascii=False)[:-1] + ",}\n```"

    rep = run(FakeLLM({"extract": extract}))
    assert rep.run.stages[3].state == StageState.DONE
    assert any(r.source == "llm" for r in rep.requirements)


def test_missing_requirement_gets_topup_call():
    calls = {"n": 0}

    def check(prompt, schema):
        calls["n"] += 1
        out = default_check(prompt, schema)
        if calls["n"] == 1:
            out["results"] = out["results"][:1]  # the model skipped the rest
        return out

    fake = FakeLLM({"check": check})
    rep = run(fake)
    assert calls["n"] == 2
    assert all(r.check_status == "checked" for r in rep.requirements)


def test_truncated_check_answer_marks_not_checked_without_budget():
    def check(prompt, schema):
        out = default_check(prompt, schema)
        out["results"] = out["results"][:1]
        return out

    rep = run(FakeLLM({"check": check}), max_calls=4)  # extract, check, s1, s2: no room for a top-up
    assert any(r.check_status == "not_checked" for r in rep.requirements)
    assert rep.summary.requirements_not_checked >= 1


def test_critic_drop_without_counter_quote_keeps_finding():
    def review(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": "drop", "merge_into": "", "severity": "minor",
                               "drop_reason": "document_specifies", "counter_quote": "выдуманная цитата из воздуха",
                               "improved_question": ""} for i in ids],
                "missed": [], "fix_first": [], "verdict_rationale": ""}

    rep = run(FakeLLM({"review": review}))
    assert rep.findings and rep.summary.dropped_by_review == 0
    assert all(f.review_note for f in rep.findings)


def test_critic_drop_with_verified_counter_quote_removes_finding():
    def review(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": "drop", "merge_into": "", "severity": "minor",
                               "drop_reason": "document_specifies",
                               "counter_quote": "на срок от 15 до 240 минут включительно",
                               "improved_question": ""} for i in ids[:1]],
                "missed": [], "fix_first": [], "verdict_rationale": ""}

    base = run(FakeLLM())
    rep = run(FakeLLM({"review": review}))
    assert rep.summary.dropped_by_review == 1
    assert rep.summary.findings_total == base.summary.findings_total - 1


def test_critic_cannot_downgrade_two_quote_contradiction():
    def review(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": "keep", "merge_into": "", "severity": "minor",
                               "drop_reason": "none", "counter_quote": "", "improved_question": ""} for i in ids],
                "missed": [], "fix_first": [], "verdict_rationale": ""}

    rep = run(FakeLLM({"review": review}))
    contra = [f for f in rep.set_findings if f.defect_class == "contradictory"]
    assert contra and contra[0].severity == "critical"


def test_economy_profile_merges_set_checks():
    fake = FakeLLM()
    rep = run(fake, profile_name="economy")
    stages = [s for s, _ in fake.calls]
    assert stages.count("set") == 1 and stages.count("check") == 1
    assert rep.run.llm_calls <= 6


def test_real_task_document_end_to_end_offline():
    rep = run_pipeline(ROOT / "docs" / "requirements.md", settings(profile_name="deep"), client=FakeLLM())
    assert rep.run.status == RunStatus.COMPLETE
    assert rep.run.llm_calls <= 16
    assert len(rep.run.document_sha256) == 64
    assert rep.requirements


def test_long_document_mode_sends_only_batch_blocks():
    fake = FakeLLM()
    rep = run(fake, max_context_chars=50)
    assert rep.run.status == RunStatus.COMPLETE
    check_prompts = [p for s, p in fake.calls if s == "check"]
    assert check_prompts and all("<requirements>" in p for p in check_prompts)


def test_image_hint_becomes_run_warning(monkeypatch):
    from validator.models import Hint, PrecheckResult
    from validator.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_prechecks", lambda doc: PrecheckResult(
        hints=[Hint(rule="image_link", block_id="B003", match="![схема](a.png)", message="изображение")]))
    rep = run(FakeLLM())
    assert any("изображения" in w for w in rep.run.warnings)
    assert all("a.png" not in p for _, p in FakeLLM().calls)


def test_rule_finding_without_blocks_is_kept_by_section_title(monkeypatch):
    from validator.models import PrecheckResult, RuleFinding
    from validator.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_prechecks", lambda doc: PrecheckResult(findings=[RuleFinding(
        rule="empty_section", block_ids=[], criterion="set_complete", severity="major", set_check="empty_section",
        quote="Критерии приёмки", explanation="Раздел «Критерии приёмки» не содержит заполненных требований.",
        recommendation="Заполнить раздел.", clarifying_question="Что должно быть в разделе?")]))
    rep = run(FakeLLM())
    assert any(f.set_check == "empty_section" for f in rep.set_findings)


def test_critic_cannot_merge_distinct_contradictions():
    def set_check(prompt, schema):
        rows = re.findall(r"^(R-\d+) \|.*?\| блоки [^|]* \| (.*)$", section(prompt, "requirements"), re.MULTILINE)
        mk = lambda a, b, text: {
            "evidence": [{"requirement_id": a[0], "quote": a[1][:50]}, {"requirement_id": b[0], "quote": b[1][:50]}],
            "scenario": "Входные данные: по первому месту одно, по второму другое.", "explanation": text,
            "set_check": "contradiction", "criterion": "set_consistent", "severity": "critical",
            "recommendation": "Согласовать.", "clarifying_question": "Какое место верное?", "answer_options": []}
        return {"findings": [mk(rows[0], rows[1], "Первое противоречие между сроком брони и отменой без отметки."),
                             mk(rows[2], rows[3], "Второе противоречие между подтверждением и критерием приёмки.")],
                "coverage": []}

    def review(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": "merge", "merge_into": ids[0], "severity": "critical",
                               "drop_reason": "duplicate", "counter_quote": "", "improved_question": ""}
                              for i in ids[1:]],
                "missed": [], "fix_first": [], "verdict_rationale": ""}

    rep = run(FakeLLM({"set": set_check, "review": review}))
    assert sum(1 for f in rep.set_findings if f.defect_class == "contradictory") >= 2


def test_block_ids_are_removed_from_texts():
    def check(prompt, schema):
        out = default_check(prompt, schema)
        for r in out["results"]:
            for f in r["findings"]:
                f["explanation"] += " См. блок (B003) и B999."
        return out

    rep = run(FakeLLM({"check": check}))
    assert all(not re.search(r"\bB\d{3}\b", f.explanation) for f in rep.findings)


def test_critic_generic_drop_of_major_finding_downgrades_it():
    def review(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": "drop", "merge_into": "", "severity": "major",
                               "drop_reason": "generic", "counter_quote": "", "improved_question": ""} for i in ids],
                "missed": [], "fix_first": [], "verdict_rationale": ""}

    base = run(FakeLLM())
    rep = run(FakeLLM({"review": review}))
    # verified model findings stay (as minor); rule findings and minor ones may be removed
    kept = [f for f in [*base.findings, *base.set_findings] if f.source == "llm" and f.severity != "minor"]
    assert rep.summary.findings_total >= len(kept)
    assert all(f.severity == "minor" for f in rep.findings)
