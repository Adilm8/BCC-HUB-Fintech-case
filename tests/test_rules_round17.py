import json
from pathlib import Path

import pytest

from eval.score import normalize
from validator.loaders import load_document
from validator.models import RawDocument
from validator.rules import run_prechecks
from validator.rules.extra import check_integration_failures, check_non_singular, check_undefined_terms
from validator.structure import parse_structure

ROOT = Path(__file__).resolve().parent.parent


def _doc(text):
    return parse_structure(RawDocument(path="d.md", name="d.md", title="Сервис", format="md", text=text))


BUNDLED = """# Сервис

## Функции

- **FR-1.** Система должна принимать заявку в рабочее время. Если сумма больше лимита, заявка уходит на \
проверку. Менеджер должен ответить в течение дня. Клиент получает уведомление о решении и может отозвать заявку.
- **FR-2.** Система должна хранить историю заявок не менее года.
"""

SINGLE = """# Сервис

## Функции

- **FR-1.** Система должна принимать заявку в рабочее время, указанное в настройках филиала, с учётом \
праздничных дней из производственного календаря.
"""


def test_non_singular_flags_bundled_items_only():
    f = check_non_singular(_doc(BUNDLED))
    assert len(f) == 1 and f[0].rule == "non_singular" and f[0].criterion == "singular" and f[0].severity == "minor"
    assert "FR-1" in f[0].explanation and "FR-2" not in f[0].explanation
    assert check_non_singular(_doc(SINGLE)) == []


TERMS = """# Сервис

## Термины

| Термин | Определение |
| :--- | :--- |
| ЛК | Личный кабинет клиента |

## Функции

- **FR-1.** Клиент с профилем R2 видит ПДн только в ЛК.
- **FR-2.** Ответ возвращается в JSON по HTTPS; КПЭ (ключевой показатель) считается ежедневно.
- **FR-3.** ОПК — это операционный контур; данные ОПК хранятся год.
"""


def test_undefined_terms_skip_known_and_defined_ones():
    f = check_undefined_terms(_doc(TERMS))
    assert len(f) == 1 and f[0].rule == "undefined_term" and f[0].severity == "minor"
    text = f[0].explanation
    assert "R2" in text and "ПДн" in text
    for ok in ("ЛК", "JSON", "HTTPS", "КПЭ", "ОПК"):
        assert f"{ok} («" not in text


CALLS = """# Сервис

## Интеграции

- **INT-1.** Сервис запрашивает статус карты в процессинге перед списанием.
"""


def test_integration_without_failure_and_its_negatives():
    f = check_integration_failures(_doc(CALLS))
    assert len(f) == 1 and f[0].rule == "integration_without_failure" and f[0].severity == "major"
    covered = CALLS + "- **INT-2.** При тайм-ауте процессинга запрос повторяется один раз.\n"
    assert check_integration_failures(_doc(covered)) == []
    error_section = CALLS + "\n## Обработка ошибок\n\n- Если процессинг недоступен, операция отклоняется.\n"
    assert check_integration_failures(_doc(error_section)) == []
    elsewhere = CALLS.replace("## Интеграции", "## Функции")
    assert check_integration_failures(_doc(elsewhere)) == []


@pytest.mark.parametrize("flag,rule", [("EXP_SINGULAR_RULE", "non_singular"), ("EXP_ABBREV_RULE", "undefined_term"),
                                       ("EXP_FAILURE_RULE", "integration_without_failure")])
def test_each_rule_is_behind_its_flag(monkeypatch, flag, rule):
    doc = _doc(BUNDLED + TERMS.split("## Функции", 1)[1] + CALLS.split("# Сервис", 1)[1])
    assert not [x for x in run_prechecks(doc).findings if x.rule == rule]
    monkeypatch.setenv(flag, "1")
    assert [x for x in run_prechecks(doc).findings if x.rule == rule]


def _gold_docs():
    for g in sorted((ROOT / "eval" / "gold").glob("*.gold.json")):
        data = json.loads(g.read_text(encoding="utf-8"))
        doc = ROOT / data["document"]
        if not doc.is_file():
            doc = ROOT / "eval" / "docs" / (g.name.split(".")[0] + ".md")
        if doc.is_file() and data.get("clean"):
            yield g.name, doc, data["clean"]


@pytest.mark.parametrize("name,doc,clean", list(_gold_docs()))
def test_new_rules_do_not_quote_clean_requirements(name, doc, clean):
    anchors = [normalize(a) for c in clean for a in (c["anchor"] if isinstance(c["anchor"], list) else [c["anchor"]])
               if a]
    parsed = parse_structure(load_document(doc))
    for check in (check_non_singular, check_undefined_terms, check_integration_failures):
        for f in check(parsed):
            assert not any(a in normalize(f.quote) for a in anchors), (name, f.rule, f.quote)
