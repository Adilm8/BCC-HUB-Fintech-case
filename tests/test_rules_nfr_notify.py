import json
from pathlib import Path

import pytest

from eval.score import normalize
from validator.loaders import load_document
from validator.models import RawDocument
from validator.pipeline import toggles
from validator.rules import run_prechecks
from validator.rules.nfr_notify import check_mixed_levels, check_nfr_unmeasurable, check_notification_content
from validator.structure import parse_structure

ROOT = Path(__file__).resolve().parent.parent


def _doc(text):
    return parse_structure(RawDocument(path="d.md", name="d.md", title="Сервис", format="md", text=text))


NFR = """# Сервис

## Нефункциональные требования

- **NFR-1.** Ответ на запрос выдачи справки — не более 2 секунд.
- **NFR-2.** Ответ на запрос — не более 2 секунд для 95% запросов при нагрузке 50 RPS.
- **NFR-3.** Доступность сервиса — не менее 99,9% в календарный месяц.
- **NFR-4.** Ссылка действует не более 15 минут.

## Функции

- **FR-1.** Заявка обрабатывается в течение 10 минут.
"""


def test_nfr_unmeasurable_flags_only_unmeasured_quality_times():
    f = check_nfr_unmeasurable(_doc(NFR))
    assert len(f) == 1 and f[0].quote.startswith("NFR-1.")  # NFR-2..4 and the functional deadline are fine
    assert f[0].criterion == "verifiable" and f[0].severity == "major" and f[0].block_ids
    assert "перцентиль" in f[0].explanation


def test_a_unit_without_a_number_is_a_time_bound_too():
    doc = _doc("# С\n\n## Нефункциональные требования\n\n- Сервис передаёт push на отправку в течение минуты "
               "после получения события.\n")
    f = check_nfr_unmeasurable(doc)
    assert len(f) == 1
    text = f[0].explanation
    assert "перцентиль" in text and "конец измеряемого интервала" in text and "начало и" not in text
    assert "передача на отправку внешнему сервису или доставка получателю" in text


NOTIFY = """# Сервис

## Функции

- **FR-1.** После одобрения заявки система отправляет клиенту push.
- **FR-2.** При списании система отправляет клиенту SMS-сообщение с суммой и остатком.
- **FR-3.** Система отправляет клиенту SMS-код для входа.
"""


def test_notification_content_without_template():
    f = check_notification_content(_doc(NOTIFY))
    assert len(f) == 1 and f[0].rule == "notification_content" and f[0].quote.startswith("FR-1.")
    pii = _doc(NOTIFY.replace(" с суммой и остатком", ", в нём сумма списания и остаток"))
    texts = [x.explanation for x in check_notification_content(pii)]
    assert any("маскирования" in t for t in texts)
    templated = NOTIFY + "\n## Шаблоны\n\n- Шаблон push: «Заявка одобрена, подробности в приложении»\n"
    assert check_notification_content(_doc(templated)) == []


API = """# Сервис

## Метод POST /limits

- Сохраняет лимит клиента. Поле `limit` обязательно.
- Изменение лимита сохраняет накопленные расходы за месяц.

## Функции

- **FR-1.** Изменение лимита сохраняет накопленные расходы за месяц.
"""


def test_mixed_levels_in_the_api_part_only():
    f = check_mixed_levels(_doc(API))
    assert len(f) == 1 and f[0].rule == "mixed_levels" and f[0].criterion == "appropriate"
    assert f[0].quote == "Изменение лимита сохраняет накопленные расходы за месяц."
    assert len(f[0].block_ids) == 1  # the functional section is not an interface specification
    assert check_mixed_levels(_doc(API.replace("## Метод POST /limits", "## Хранение"))) == []


@pytest.mark.parametrize("flag,rule,text", [("EXP_NFR_RULE", "nfr_unmeasurable", NFR),
                                            ("EXP_NOTIFY_RULE", "notification_content", NOTIFY),
                                            ("EXP_MIXED_RULE", "mixed_levels", API)])
def test_rules_are_on_by_default_and_switch_off(monkeypatch, flag, rule, text):
    assert flag in toggles.DEFAULT_ON
    assert [x for x in run_prechecks(_doc(text)).findings if x.rule == rule]
    monkeypatch.setenv(flag, "0")
    assert not [x for x in run_prechecks(_doc(text)).findings if x.rule == rule]


def _gold_docs():
    for g in sorted((ROOT / "eval" / "gold").glob("*.gold.json")):
        data = json.loads(g.read_text(encoding="utf-8"))
        doc = ROOT / data["document"]
        if not doc.is_file():
            doc = ROOT / "eval" / "docs" / (g.name.split(".")[0] + ".md")
        if doc.is_file() and data.get("clean"):
            yield g.name, doc, data["clean"]


@pytest.mark.parametrize("name,doc,clean", list(_gold_docs()))
def test_default_on_rules_never_quote_clean_requirements(name, doc, clean):
    anchors = [normalize(a) for c in clean for a in (c["anchor"] if isinstance(c["anchor"], list) else [c["anchor"]])
               if a]
    parsed = parse_structure(load_document(doc))
    for check in (check_nfr_unmeasurable, check_notification_content, check_mixed_levels):
        for f in check(parsed):
            q = normalize(f.quote)
            assert not any(a in q or q in a for a in anchors), (name, f.rule, f.quote)
