from tests.fakes import FakeLLM, default_check
from tests.test_pipeline_run import raw_doc, settings
from validator.pipeline import run_pipeline
from validator.report.markdown import render_markdown


def _with_remarks(prompt, schema):
    out = default_check(prompt, schema)
    for r in out["results"]:
        r["findings"] = []
        r["remarks"] = [{"quote": "Система быстро подтверждает бронирование", "criterion": "verifiable",
                         "text": "Возможно, не задано, что значит «быстро»."},
                        {"quote": "этого текста нет в документе", "criterion": "complete", "text": "x"}]
    return out


def test_remarks_are_off_and_completeness_items_on_by_default(monkeypatch):
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    check = next(p for s, p in f.calls if s == "check")
    assert "remarks" not in check and "Ув —" in check
    assert not any("режим" in w for w in rep.run.warnings)  # defaults are not listed
    monkeypatch.setenv("EXP_COMPLETENESS_ITEMS", "0")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    assert "Ув —" not in next(p for s, p in f.calls if s == "check")
    assert any("Выключен режим EXP_COMPLETENESS_ITEMS" in w for w in rep.run.warnings)


def test_completeness_items_join_the_checklist(monkeypatch):
    monkeypatch.setenv("EXP_COMPLETENESS_ITEMS", "1")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    check = next(p for s, p in f.calls if s == "check")
    assert check.index("Р — расчёты") < check.index("Ув — исходящие") < check.index("Пример: «Г ок")
    assert "Н — нефункциональное требование: точки начала и конца измерения, перцентиль" in check
    assert not any("EXP_COMPLETENESS_ITEMS" in w for w in rep.run.warnings)


def test_remarks_are_verified_minor_flagged_findings_without_questions(monkeypatch):
    monkeypatch.setenv("EXP_REMARKS", "1")
    f = FakeLLM({"check": _with_remarks})
    rep = run_pipeline(raw_doc(), settings(), client=f)
    check = next(p for s, p in f.calls if s == "check")
    assert "remarks — короткие замечания" in check
    remarks = [x for x in rep.findings if x.remark]
    assert len(remarks) == 1  # the invented quote is dropped, duplicates of one place collapse
    r = remarks[0]
    assert r.severity == "minor" and r.quote.startswith("Система быстро")
    assert not any(r.id in q.finding_ids for q in rep.questions)
    assert rep.summary.remarks_total == 1 and r.id not in rep.summary.critical_ids
    md = render_markdown(rep)
    assert "**Замечания (возможные пробелы):**" in md and f"- {r.id}: «Система быстро" in md


def test_a_remark_on_a_place_with_a_finding_is_not_added(monkeypatch):
    monkeypatch.setenv("EXP_REMARKS", "1")

    def both(prompt, schema):
        out = default_check(prompt, schema)
        for r in out["results"]:
            r["remarks"] = [{"quote": f["quote"], "criterion": "complete", "text": "дубль"} for f in r["findings"]]
        return out

    rep = run_pipeline(raw_doc(), settings(), client=FakeLLM({"check": both}))
    assert not [x for x in rep.findings if x.remark]
