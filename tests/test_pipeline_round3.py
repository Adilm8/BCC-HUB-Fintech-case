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
    return re.findall(r"^(R-\d+) \|.*?\| блоки [^|]* \| (.*)$", section(prompt, "requirements"), re.MULTILINE)


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
    assert any("согласующих" in n for n in a.related)
    assert "Также" not in a.explanation


def test_report_collapses_clean_requirements_and_shows_set_refs():
    rep = run(FakeLLM())
    md = render_markdown(rep)
    assert "### Требования без собственных нарушений" in md
    assert "Затронуто в наборе" in md or "затронуты находками проверки набора" in md
    assert "выполняются параллельно" in md
    assert "Объединено с другими находками как дубли" in md
    assert "**Адресат:**" in md
    assert rep.summary.requirements_with_findings == len({r for f in rep.findings for r in f.requirement_ids})


def test_candidate_balance_adds_up():
    rep = run(FakeLLM())
    s = rep.summary
    assert s.candidates_total == s.findings_total + s.merged_duplicates + s.dropped_by_review + s.dropped_unconfirmed
    assert "Баланс кандидатов" in render_markdown(rep)


def test_first_sentence_does_not_cut_inside_quotes_or_after_letters():
    t = "В разделе «Уведомление › Часть A. Новый функционал» нет сценариев. Второе предложение."
    assert pp.first_sentence(t) == "В разделе «Уведомление › Часть A. Новый функционал» нет сценариев."
    assert pp.first_sentence("По п. 5.2.5 требование неполное. Дальше.") == "По п. 5.2.5 требование неполное."


def test_empty_test_section_is_major_when_acceptance_criteria_exist():
    from validator.models import ReqKind, Requirement
    from validator.pipeline.orchestrator import _soften_test_section_gaps

    ev = Evidence(quote="заполняются тестировщиками", section="Требования к тестированию", block_ids=["B9"],
                  quote_status=QuoteStatus.VERIFIED)
    c = pp.Candidate("set_complete", Severity.CRITICAL, [ev], "Раздел тестирования содержит заглушку.", "r", "q?",
                     [], [], "llm", set_check="empty_section", addressee="analyst")
    acc = Requirement(id="R-001", id_assigned=True, kind=ReqKind.ACCEPTANCE_CRITERION, part="", section="s",
                      block_ids=["B1"], title="t", quote="q")
    _soften_test_section_gaps([c], [acc])
    assert c.severity == Severity.MAJOR and c.addressee == "qa"
    c2 = pp.Candidate("set_complete", Severity.CRITICAL, [ev], "Раздел тестирования содержит заглушку.", "r", "q?",
                      [], [], "llm", set_check="empty_section")
    _soften_test_section_gaps([c2], [])
    assert c2.severity == Severity.CRITICAL


def test_round6_bucket_and_class_rules():
    from validator.models import Criterion, DefectClass, defect_class_for

    assert defect_class_for(Criterion.SET_COMPLETE, "cross_part") == DefectClass.INCOMPLETE


def test_round6_meta_rows_and_preamble_are_skipped():
    from validator.models import RawDocument

    text = ("# Заявки\n\nВводная заметка о комплекте документов.\n\n## Часть 1\n\n### Общие сведения\n\n"
            "| Поле | Значение |\n| --- | --- |\n| Автор | Отдел заявок |\n| Версия | 1.0 |\n\n"
            "### Функции\n\n- **LV-1.** Сотрудник подаёт заявку на отпуск от 1 до 28 дней.\n")
    raw = RawDocument(path="x.md", name="x.md", title="Заявки", format="md", text=text)
    rep = run_pipeline(raw, settings(), client=FakeLLM())
    quotes = " ".join(r.quote for r in rep.requirements)
    assert "Отдел заявок" not in quotes and "Вводная заметка" not in quotes
    assert any(s.reason == "meta" for s in rep.skipped_blocks)


def test_round6_options_reset_when_critic_rewrites_question():
    def check(prompt, schema):
        out = default_check(prompt, schema)
        for r in out["results"]:
            for f in r["findings"]:
                f["clarifying_question"] = "Нужно ли проверять значение?"
                f["answer_options"] = ["Да, проверять", "Нет, не проверять"]
        return out

    def review(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": "keep", "merge_into": "", "severity": "major",
                               "drop_reason": "none", "counter_quote": "",
                               "improved_question": "Какое значение считается допустимым"} for i in ids],
                "missed": [], "fix_first": [], "verdict_rationale": ""}

    rep = run(FakeLLM({"check": check, "review": review}))
    assert rep.findings and all(not f.answer_options for f in rep.findings)


def test_round6_sha256_is_of_the_file_bytes(tmp_path):
    import hashlib

    from validator.loaders import load_document

    f = tmp_path / "a.md"
    f.write_bytes("# Т\r\n\r\nТекст требования.\r\n".encode())
    raw = load_document(f)
    assert raw.sha256 == hashlib.sha256(f.read_bytes()).hexdigest()


def test_critic_does_not_merge_unrelated_findings_of_different_criteria():
    ev = Evidence(quote="Счётчики хранятся в выбранной СУБД", section="s", block_ids=["B1"],
                  quote_status=QuoteStatus.VERIFIED)
    a = pp.Candidate("appropriate", Severity.MINOR, [ev], "В бизнес-требовании зафиксирована конкретная СУБД и "
                     "шаблон ключей, это проектное решение.", "r", "q?", [], ["R-001"], "llm", cid="C01")
    b = pp.Candidate("correct", Severity.MAJOR, [ev], "Срок жизни записи не совпадает с границей календарных "
                     "суток из соседнего правила.", "r", "q?", [], ["R-001"], "llm", cid="C02")
    from validator.pipeline.prompts.io import LLMDecision, LLMReview

    review = LLMReview(decisions=[LLMDecision(finding_id="C01", verdict="merge", merge_into="C02", severity="minor",
                                              drop_reason="duplicate", counter_quote="", improved_question="")],
                       missed=[], fix_first=[], verdict_rationale="")

    class B:
        index = None

        @staticmethod
        def from_set(findings, source):
            return []

    kept, _ = pp.apply_review([a, b], review, B(), pp.Stats())
    assert len(kept) == 2


def _ev(q, b, section="s"):
    return Evidence(quote=q, section=section, block_ids=[b], quote_status=QuoteStatus.VERIFIED)


def test_round7_status_code_from_example_heading_is_critical():
    c = pp.Candidate("set_consistent", Severity.MAJOR,
                     [_ev('{"result":"ERROR","message":"Проверьте данные"}', "B2", "API › 200. Успешный ответ"),
                      _ev("400. Ответ, если параметры заполнены неверно", "B1", "API › Статусы")],
                     "Ошибка приведена под заголовком кода успеха, а перечень требует 400.", "r", "q?", [], [], "llm",
                     set_check="contradiction")
    pp.apply_guards([c], {"B1": "P", "B2": "P"})
    assert c.severity == Severity.CRITICAL


def test_round7_single_place_is_not_a_contradiction():
    c = pp.Candidate("set_consistent", Severity.MAJOR, [_ev("Флаг задаётся в настройках клиента", "B1")],
                     "Флаг не учитывается при отправке, поведение при выключенном флаге не описано.", "r", "q?",
                     [], [], "llm", set_check="contradiction")
    pp.apply_guards([c], {})
    assert c.set_check == "other"


def test_round7_own_gap_is_not_absorbed_and_same_quotes_merge():
    q1, q2 = _ev("Лимит действует один раз в месяц", "B1"), _ev("После смены лимита событие повторяется", "B2")
    contra = pp.Candidate("set_consistent", Severity.CRITICAL, [q1, q2], "Два места требуют разного.", "r", "q?",
                          [], ["R-001", "R-002"], "llm", set_check="contradiction")
    gap = pp.Candidate("complete", Severity.MAJOR, [q2], "Не описано, что происходит, если значение уже выше "
                       "новой границы.", "r", "q?", [], ["R-002"], "llm")
    twin = pp.Candidate("set_consistent", Severity.MAJOR, [q1, q2], "Иначе сформулированное то же самое.", "r",
                        "q?", [], ["R-001"], "llm", set_check="other")
    kept = pp.dedup([contra, gap, twin], pp.Stats())
    assert len(kept) == 2 and gap in kept


def test_round7_rule_questions_get_owner_and_options():
    ev = _ev("Да / Нет", "B1", "Ревью требований")
    c = pp.Candidate("set_complete", Severity.MAJOR, [ev], "Чек-лист ревью не заполнен: стоят шаблонные варианты.",
                     "r", "Кто заполнит чек-лист?", [], [], "rule", set_check="missing_requirement",
                     rule="unfilled_template_table")
    _, sets = pp.finalize([c], [])
    assert sets[0].addressee == "business" and len(sets[0].answer_options) == 2


def test_round7_quote_does_not_cross_a_section_heading():
    from validator.models import RawDocument
    from validator.pipeline._fallback_structure import parse_structure
    from validator.pipeline.quotes import QuoteIndex

    raw = RawDocument(path="x.md", name="x.md", title="T", format="md",
                      text="# T\n\n## A\n\nКонец раздела первого\n\n## B\n\nНачало раздела второго\n")
    idx = QuoteIndex(parse_structure(raw))
    assert idx.find("Конец раздела первого B Начало раздела второго") is None
