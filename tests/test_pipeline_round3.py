"""Report hygiene: internal ids, duplicates, protected severities, questions, cache and rendering."""

import re

from tests.fakes import FakeLLM, default_check, section
from tests.test_pipeline_run import raw_doc, run, settings
from validator.models import Evidence, QuoteStatus, Severity
from validator.pipeline import postprocess as pp
from validator.pipeline import run_pipeline
from validator.pipeline.prompts.io import LLMCheck
from validator.pipeline.structured import Budget, StageCache, call_structured
from validator.report.markdown import render_markdown


def _set(prompt, findings):
    return {"ledger": [], "findings": findings, "coverage": []}


def _rows(prompt):
    return re.findall(r"^(R-\d+) \| .* \| (.*)$", section(prompt, "requirements"), re.MULTILINE)


def test_internal_candidate_ids_never_reach_the_report():
    def check(prompt, schema):
        out = default_check(prompt, schema)
        for r in out["results"]:
            for f in r["findings"]:
                f["explanation"] += " Это уточняет C10, см. C07."
                f["recommendation"] = "См. C10."
        return out

    def review(prompt, schema):
        rows = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        dup = {"evidence": [{"requirement_id": "", "quote": "Сотрудник бронирует переговорную на срок"}],
               "scenario": "", "explanation": "Дублирует C10, поэтому не добавляется отдельно, суть та же.",
               "set_check": "other", "criterion": "set_complete", "severity": "critical", "recommendation": "См. C10",
               "clarifying_question": "Что?", "addressee": "analyst", "answer_options": []}
        return {"decisions": [], "missed": [dup], "fix_first": rows[:1], "verdict_rationale": ""}

    rep = run(FakeLLM({"check": check, "review": review}))
    md = render_markdown(rep)
    assert not re.search(r"\bC\d{2,3}\b", md)
    assert all("Дублирует" not in f.explanation for f in rep.set_findings)


def test_rule_critical_gap_is_not_lowered_by_critic(monkeypatch):
    from validator.models import PrecheckResult, RuleFinding
    from validator.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_prechecks", lambda doc: PrecheckResult(findings=[RuleFinding(
        rule="placeholder", block_ids=[], criterion="set_complete", severity="critical", set_check="empty_section",
        quote="Критерии приёмки", explanation="Раздел «Критерии приёмки» содержит только заглушку вместо требований.",
        recommendation="Заполнить раздел.", clarifying_question="Кто заполняет раздел?")]))

    def review(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": "drop", "merge_into": "", "severity": "minor",
                               "drop_reason": "generic", "counter_quote": "", "improved_question": ""} for i in ids],
                "missed": [], "fix_first": [], "verdict_rationale": ""}

    rep = run(FakeLLM({"review": review}))
    gap = [f for f in rep.set_findings if f.set_check == "empty_section"]
    assert gap and gap[0].severity == "critical"


def test_set_findings_about_the_same_blocks_are_merged():
    def set_check(prompt, schema):
        rows = _rows(prompt)
        a, b = rows[0], rows[1]
        base = {"scenario": "", "criterion": "set_consistent", "recommendation": "Исправить метод.",
                "clarifying_question": "Какой метод используется для сохранения брони?", "addressee": "analyst",
                "answer_options": []}
        return _set(prompt, [
            {**base, "evidence": [{"requirement_id": a[0], "quote": a[1][:50]}, {"requirement_id": b[0], "quote": b[1][:50]}],
             "explanation": "Метод сохранения брони описан как чтение, что нарушает семантику метода и правила бронирования.",
             "set_check": "other", "severity": "critical"},
            {**base, "evidence": [{"requirement_id": a[0], "quote": a[1][:45]}, {"requirement_id": b[0], "quote": b[1][:45]}],
             "explanation": "Сохранение брони выполняется методом чтения, нарушена семантика метода бронирования.",
             "set_check": "missing_requirement", "criterion": "set_complete", "severity": "major"},
        ])

    rep = run(FakeLLM({"set": set_check}))
    about_method = [f for f in rep.set_findings if "метод" in f.explanation.lower()]
    assert len(about_method) == 1


def test_status_code_contradiction_in_one_part_is_critical():
    doc = raw_doc()
    text = doc.text + "\n## Ответы\n\nКод 403 — ответ, если доступ запрещён.\n\nДля чужой брони возвращается 404.\n"
    doc = doc.model_copy(update={"text": text})

    def set_check(prompt, schema):
        return _set(prompt, [{
            "evidence": [{"requirement_id": "", "quote": "Код 403 — ответ, если доступ запрещён."},
                         {"requirement_id": "", "quote": "Для чужой брони возвращается 404."}],
            "scenario": "Запрос чужой брони: по перечню 403, по тексту 404.",
            "explanation": "Перечень статусов называет 403 для запрета доступа, а текст требует 404 для чужой брони.",
            "set_check": "contradiction", "criterion": "set_consistent", "severity": "major",
            "recommendation": "Выбрать один код.", "clarifying_question": "Какой код возвращается для чужой брони?",
            "addressee": "analyst", "answer_options": ["403", "404"]}])

    rep = run_pipeline(doc, settings(), client=FakeLLM({"set": set_check}))
    codes = [f for f in rep.set_findings if "403" in f.explanation]
    assert codes and codes[0].severity == "critical"


def test_questions_have_addressee_distinct_options_and_no_duplicates():
    def check(prompt, schema):
        out = default_check(prompt, schema)
        for r in out["results"]:
            for f in r["findings"]:
                f["answer_options"] = ["дата по UTC+5", "Дата по UTC+5 "]
        return out

    rep = run(FakeLLM({"check": check}))
    assert rep.questions and all(q.addressee for q in rep.questions)
    assert all(len(q.options) != 1 for q in rep.questions)
    assert all(not f.answer_options for f in rep.findings)  # identical options are no choice
    texts = [q.text for q in rep.questions]
    assert len(texts) == len(set(texts))
    assert rep.questions[0].blocking


def test_distinct_options_helper():
    assert pp.distinct_options(["A вариант", "a вариант", "Б другой"]) == ["A вариант", "Б другой"]
    assert pp.distinct_options(["один"]) == []


def test_short_text_cuts_at_word_boundary():
    t = pp.short_text("Клиент получает уведомление не чаще одного раза за календарный месяц по каждой карте", 50)
    assert t.endswith("…") and not t[:-1].endswith("календа") and len(t) <= 50


def test_no_cache_does_not_write(tmp_path):
    fake = FakeLLM({"check": {"results": []}})
    call_structured(fake, LLMCheck, "p", system="s", budget=Budget(5), cache=StageCache(None))
    assert not any(tmp_path.iterdir())


def test_cache_key_depends_on_effort_and_model():
    k1 = StageCache.key("m1", "s", "p", {}, effort="high")
    assert k1 != StageCache.key("m1", "s", "p", {}, effort="medium")
    assert k1 != StageCache.key("m2", "s", "p", {}, effort="high")


def test_fallback_model_answer_is_not_cached(tmp_path):
    class Fallback(FakeLLM):
        def complete(self, *a, **kw):
            r = super().complete(*a, **kw)
            r.model = "fallback-model"
            return r

    fake = Fallback({"check": {"results": []}})
    call_structured(fake, LLMCheck, "p", system="s", budget=Budget(5), cache=StageCache(tmp_path))
    assert not list(tmp_path.glob("*.json"))


def test_merge_keeps_the_gist_of_a_different_finding():
    ev = Evidence(quote="Блок согласующих", section="s", block_ids=["B1"], quote_status=QuoteStatus.VERIFIED)
    a = pp.Candidate("set_complete", Severity.MAJOR, [ev], "Чек-лист ревью не заполнен: отметки отсутствуют.",
                     "r", "q?", [], [], "rule", set_check="empty_section")
    b = pp.Candidate("set_complete", Severity.MAJOR, [ev], "Раздел согласующих пуст: нет подписей и ответственных.",
                     "r", "q?", [], [], "rule", set_check="empty_section")
    pp.merge_into(a, b)
    assert "согласующих" in a.explanation


def test_report_collapses_clean_requirements_and_shows_set_refs():
    rep = run(FakeLLM())
    md = render_markdown(rep)
    assert "### Требования без собственных нарушений" in md
    assert "Затронуто в наборе" in md or "затронуты находками проверки набора" in md
    assert "выполняются параллельно" in md
    assert "Объединено с другими находками как дубли" in md
    assert "**Адресат:**" in md
    assert rep.summary.requirements_with_findings == len({r for f in rep.findings for r in f.requirement_ids})
