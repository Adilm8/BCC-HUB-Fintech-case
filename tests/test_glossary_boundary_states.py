from tests.fakes import FakeLLM
from tests.test_pipeline_run import raw_doc, settings
from validator.models import RawDocument, ReqKind, Requirement
from validator.pipeline import run_pipeline
from validator.pipeline.glossary import extract_terms, terms_block
from validator.rules import run_prechecks
from validator.rules.boundaries import check_boundary_tests
from validator.structure import parse_structure


def _req(rid, text, kind=ReqKind.FUNCTIONAL):
    return Requirement(id=rid, id_assigned=True, kind=kind, part="p", section="s", block_ids=[], title=rid,
                       quote=text)


def test_terms_that_conditions_depend_on():
    reqs = [_req("R-1", "Если «доступный остаток» превышает порог, заявка уходит на проверку."),
            _req("R-2", "Комиссия не более 2 процентов от суммы операции."),
            _req("R-3", "Экран показывает логотип.", ReqKind.UI)]
    terms = extract_terms(reqs)
    assert "доступный остаток" in terms and terms["доступный остаток"] == ["R-1"]
    assert "Комиссия" in terms
    assert not any("R-3" in ids for ids in terms.values())
    block = terms_block(reqs[1:2], reqs)
    assert "Комиссия (R-2)" in block and "доступный остаток" not in block


def test_glossary_toggle_adds_terms_to_the_check_only(monkeypatch):
    raw = RawDocument(path="d.md", name="d.md", title="Сервис", format="md", text=DOC)
    monkeypatch.setenv("EXP_GLOSSARY", "0")
    f = FakeLLM()
    run_pipeline(raw, settings(), client=f)
    assert not any("<terms>" in p for _, p in f.calls)
    monkeypatch.setenv("EXP_GLOSSARY", "1")
    f = FakeLLM()
    run_pipeline(raw, settings(), client=f)
    checks = [p for s, p in f.calls if s == "check"]
    assert checks and all("<terms>" in p and "определён ли он где-то в документе" in p for p in checks)
    assert not any("<terms>" in p for s, p in f.calls if s != "check")


DOC = """# Сервис

## Функции

- **BR-1.** Сумма заявки от 100 до 5 000 включительно.
- **BR-2.** Не более 3 заявок в сутки.

## Цели

- Не менее 10 000 клиентов за год.

## Критерии приёмки

| Условие | Ожидаемый результат |
| :--- | :--- |
| Сумма 100 | Принята |
| Сумма 99 | Отклонена |
| Сумма 5 000 | Принята |
| Сумма 5 001 | Отклонена |
| Третья заявка | Принята |
"""


def _doc(text):
    return parse_structure(RawDocument(path="d.md", name="d.md", title="Сервис", format="md", text=text))


def test_boundary_rule_lists_ranges_without_edge_rows():
    findings = check_boundary_tests(_doc(DOC))
    assert len(findings) == 1
    f = findings[0]
    assert f.rule == "boundary_tests_missing" and f.severity == "major" and f.criterion == "set_complete"
    assert "«Не более 3»" in f.explanation  # neither 3 nor 4 is in the acceptance table
    assert "от 100 до 5 000" not in f.explanation  # covered: 100, 99, 5 000, 5 001
    assert "10 000" not in f.explanation  # goals are not checked by acceptance rows
    assert any(f.quote in b.text for b in _doc(DOC).blocks)


def test_boundary_rule_needs_an_acceptance_section_and_its_toggle(monkeypatch):
    no_acc = DOC.split("## Критерии приёмки")[0]
    assert check_boundary_tests(_doc(no_acc)) == []
    monkeypatch.setenv("EXP_BOUNDARY_RULE", "0")
    assert not [f for f in run_prechecks(_doc(DOC)).findings if f.rule == "boundary_tests_missing"]
    monkeypatch.setenv("EXP_BOUNDARY_RULE", "1")
    assert [f for f in run_prechecks(_doc(DOC)).findings if f.rule == "boundary_tests_missing"]


def test_states_lens_joins_s2_only(monkeypatch):
    monkeypatch.setenv("EXP_STATES", "0")
    f = FakeLLM()
    run_pipeline(raw_doc(), settings(), client=f)
    assert not any("состояние × событие" in p for _, p in f.calls)
    monkeypatch.setenv("EXP_STATES", "1")
    f = FakeLLM()
    run_pipeline(raw_doc(), settings(), client=f)
    with_lens = [p for s, p in f.calls if "состояние × событие" in p]
    assert len(with_lens) == 1 and "Задание S2" in with_lens[0]
    s2 = with_lens[0]
    assert s2.index("6. Состояния") < s2.index("Для находки-пропуска цитата")
