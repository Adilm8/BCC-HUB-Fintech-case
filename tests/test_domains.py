import pytest

from tests.fakes import FakeLLM
from tests.test_pipeline_run import raw_doc, settings
from validator import domains
from validator.models import RawDocument
from validator.pipeline import run_pipeline
from validator.pipeline.prompts.domain import render_domain_block

BANK_TEXT = """# Переводы

## Функции

- **TR-1.** Клиент делает перевод с карты на счёт; комиссия списывается с карты.
- **TR-2.** Лимит переводов в сутки задаёт банк.
"""


def bank_doc() -> RawDocument:
    return RawDocument(path="t.md", name="t.md", title="Переводы", format="md", text=BANK_TEXT)


def test_profiles_load_and_render_topic_and_check_only():
    assert "banking" in domains.available()
    prof = domains.load("banking")
    assert prof.domain == "banking" and len(prof.items) >= 3
    block = render_domain_block(prof)
    assert block.startswith("<domain_checklist>\nДоменный чек-лист (подсказки, не нарушения):")
    assert prof.items[0].topic in block and prof.items[0].check in block
    assert prof.items[0].typical_gap not in block  # topic + check only
    assert "не является" in block
    assert len(render_domain_block(prof, max_chars=60).splitlines()) < len(block.splitlines())


def test_auto_selection_needs_enough_stems():
    prof, note = domains.select("auto", BANK_TEXT)
    assert prof is not None and prof.name == "banking" and "автоматически" in note
    none, note = domains.select("auto", "Сотрудник бронирует переговорную.")
    assert none is None and "не выбран" in note
    assert domains.select("none", BANK_TEXT) == (None, "")
    with pytest.raises(ValueError):
        domains.select("nosuch", BANK_TEXT)
    with pytest.raises(ValueError):
        domains.select("../x", BANK_TEXT)


def _prompts(fake, stage):
    return [p for s, p in fake.calls if s == stage]


def test_domain_block_goes_into_check_and_s2_only(monkeypatch):
    f = FakeLLM()
    run_pipeline(bank_doc(), settings(domain="none"), client=f)
    assert not any("<domain_checklist>" in p for _, p in f.calls)
    monkeypatch.setenv("EXP_DOMAIN", "0")
    f = FakeLLM()
    run_pipeline(bank_doc(), settings(), client=f)
    assert not any("<domain_checklist>" in p for _, p in f.calls)

    monkeypatch.setenv("EXP_DOMAIN", "1")
    f = FakeLLM()
    rep = run_pipeline(bank_doc(), settings(), client=f)
    assert all("<domain_checklist>" in p for p in _prompts(f, "check"))
    sets = _prompts(f, "set")
    assert [("Задание S2" in p) for p in sets if "<domain_checklist>" in p] == [True]
    assert not any("<domain_checklist>" in p for p in _prompts(f, "extract") + _prompts(f, "review"))
    assert any("banking выбран автоматически" in w for w in rep.run.warnings)

    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)  # a document of another domain
    assert not any("<domain_checklist>" in p for _, p in f.calls)
    assert any("не выбран" in w for w in rep.run.warnings)


def test_explicit_domain_overrides_auto():
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(domain="banking"), client=f)
    assert any("<domain_checklist>" in p for p in _prompts(f, "check"))
    assert any("banking задан явно" in w for w in rep.run.warnings)
