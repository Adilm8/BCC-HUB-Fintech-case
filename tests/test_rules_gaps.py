"""Tests for deterministic gap rules: units mismatch, idempotency, toggle flags, and one-line diagrams."""

from __future__ import annotations

import json
from pathlib import Path

from validator.loaders import load_document
from validator.models import Criterion, Document, RawDocument, Severity
from validator.rules import run_gap_checks, run_prechecks
from validator.structure import parse_structure

ROOT = Path(__file__).resolve().parents[1]


def _parse_inline(md: str, *, title: str = "Спецификация") -> Document:
    raw = RawDocument(path="spec.md", name="spec.md", title=title, format="md", text=md)
    return parse_structure(raw)


def _load_doc(rel_path: str) -> Document:
    path = ROOT / rel_path
    loaded = load_document(path)
    raw = (
        loaded
        if isinstance(loaded, RawDocument)
        else RawDocument(
            path=str(path),
            name=path.name,
            title=loaded.title,
            format="md",
            text=loaded.text,
        )
    )
    return parse_structure(raw)


def _assert_quotes_exact_substrings(doc: Document, findings: list) -> None:
    block_by_id = {b.id: b for b in doc.blocks}
    for finding in findings:
        assert finding.quote, f"Empty quote in {finding.rule}"
        assert finding.block_ids, f"Empty block_ids in {finding.rule}"
        primary = block_by_id[finding.block_ids[0]]
        assert finding.quote in primary.text, (
            f"Quote {finding.quote!r} is not an exact substring of {primary.id}: {primary.text!r}"
        )


def test_clean_synthetic_spec_produces_zero_gap_findings():
    md = (
        "# Заказы на доставку\n\n"
        "## Термины и сокращения\n\n"
        "| Термин | Определение |\n"
        "| --- | --- |\n"
        "| Идемпотентность | Гарантия отсутствия дублей при повторной доставке. |\n"
        "| Платёжный цикл | Первичная и повторные попытки списания. |\n\n"
        "## Функциональные требования\n"
        "- FR-1. Клиент оформляет заказ на сумму от 500 до 50 000 рублей с точностью до 1 копейки.\n"
        "- FR-2. Время отображения каталога не превышает 1,5 секунды.\n"
        "- FR-3. Время ответа шлюза уведомлений не превышает 200 миллисекунд.\n"
        "- FR-4. Архив заказов хранится 90 дней.\n\n"
        "## Интеграция\n"
        "Повторная доставка события с тем же Idempotency-Key в окне 24 часов сохраняет исходный итог.\n\n"
        "## Параметры запроса\n\n"
        "| Parameter | Format | Mandatory | Logic | Description |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `deliveryLimit` | integer | Да | 500–50 000 | Лимит суммы |\n"
        "| `notifyEnabled` | boolean | Да | true / false | Включение уведомлений |\n\n"
        "При отключённом флаге notifyEnabled = false параметр deliveryLimit становится необязательным "
        "и не проверяется по диапазону.\n\n"
        "## Схема процесса\n"
        "Клиент → сервис заказов → проверка лимита; при ошибке валидации возвращается отказ.\n"
    )
    doc = _parse_inline(md)
    res = run_prechecks(doc)
    assert run_gap_checks(doc, res.inventory) == []


def test_units_mismatch_same_quantity_quotes_both_places_and_ignores_unrelated():
    # Positive 1: same field name "amount" in rubles vs kopecks across two sections -> quotes both places.
    md_curr = (
        "# Биллинг API\n\n"
        "## Создание платежа\n"
        "Тело запроса: `amount` (`number`, сумма платежа в рублях, от 10 до 50000).\n\n"
        "## Статус платежа\n"
        "Ответ: `amount` (`integer`, в копейках).\n"
    )
    doc_curr = _parse_inline(md_curr)
    f_curr = [f for f in run_prechecks(doc_curr).findings if f.rule == "units_mismatch"]
    assert len(f_curr) == 1
    assert f_curr[0].criterion == Criterion.SET_CONSISTENT
    assert f_curr[0].severity == Severity.MAJOR
    assert len(f_curr[0].block_ids) == 2
    assert "в рублях" in f_curr[0].explanation
    assert "в копейках" in f_curr[0].explanation
    _assert_quotes_exact_substrings(doc_curr, f_curr)

    # Negative 1: conversion rule stated -> no finding.
    doc_curr_ok = _parse_inline(md_curr + "\nПравило пересчёта: 1 рубль = 100 копеек.\n")
    assert [f for f in run_prechecks(doc_curr_ok).findings if f.rule == "units_mismatch"] == []

    # Negative 2: precision phrasing ("с точностью до 1 копейки") is NOT a units mismatch.
    md_precision = (
        "# Переводы\n\n"
        "## Требования\n"
        "Сумма перевода находится в диапазоне от 10 до 50 000 рублей включительно с точностью до 1 копейки.\n"
    )
    assert [f for f in run_prechecks(_parse_inline(md_precision)).findings if f.rule == "units_mismatch"] == []

    # Negative 3: two DIFFERENT time quantities in the same NFR section -> no finding.
    md_diff_time = (
        "# Нефункциональные требования\n\n"
        "## 4. Нефункциональные требования\n"
        "- NFR-1. Формирование списка правил не должно превышать 1,5 секунды.\n"
        "- NFR-2. Синхронное обращение к внешним шлюзам не должно превышать 10 миллисекунд.\n"
    )
    assert [f for f in run_prechecks(_parse_inline(md_diff_time)).findings if f.rule == "units_mismatch"] == []

    # Positive 2: same time quantity (timeout) in seconds vs milliseconds -> fires.
    md_time = (
        "# Интеграция\n\n"
        "## Сетевые ограничения\n"
        "Таймаут внешнего вызова составляет 30 секунд; параметр `gateway_timeout` задаётся в миллисекундах.\n"
    )
    doc_time = _parse_inline(md_time)
    f_time = [f for f in run_prechecks(doc_time).findings if f.rule == "units_mismatch"]
    assert len(f_time) == 1
    assert "секунд" in f_time[0].explanation
    assert "миллисекундах" in f_time[0].explanation
    _assert_quotes_exact_substrings(doc_time, f_time)

    # Positive 3: same size quantity (file upload) in KB vs MB -> fires.
    md_size = (
        "# Загрузка файлов\n\n"
        "## Ограничения\n"
        "Минимальный размер файла — 10 КБ, максимальный размер файла — 5 МБ.\n"
    )
    doc_size = _parse_inline(md_size)
    f_size = [f for f in run_prechecks(doc_size).findings if f.rule == "units_mismatch"]
    assert len(f_size) == 1
    _assert_quotes_exact_substrings(doc_size, f_size)


def test_idempotency_without_key_skips_glossary_and_retries():
    # Positive: normative claim without key or window.
    md_missing = (
        "# Платёжный шлюз\n\n"
        "## Надёжность\n"
        "Повторная доставка уведомления от шлюза не приводит к повторному списанию.\n"
    )
    doc_missing = _parse_inline(md_missing)
    f_missing = [f for f in run_prechecks(doc_missing).findings if f.rule == "idempotency_without_key"]
    assert len(f_missing) == 1
    assert f_missing[0].criterion == Criterion.COMPLETE
    assert f_missing[0].severity == Severity.MAJOR
    _assert_quotes_exact_substrings(doc_missing, f_missing)

    # Negative 1: glossary section & term-definition table must NOT trigger.
    md_glossary = (
        "# Спецификация\n\n"
        "## Термины и сокращения\n\n"
        "| Термин / сокращение | Определение |\n"
        "| --- | --- |\n"
        "| Платёжный цикл | Первичная и повторные попытки списания. |\n"
        "| Идемпотентность | Гарантия неизменности итога при повторной доставке. |\n"
    )
    assert [f for f in run_prechecks(_parse_inline(md_glossary)).findings if f.rule == "idempotency_without_key"] == []

    # Negative 2: key explicitly named via payment_id / event_id -> no finding.
    md_ok = (
        "# Платёжный шлюз\n\n"
        "## Интеграция\n"
        "Интеграция должна обеспечивать идемпотентную обработку за счёт передачи "
        "уникального идентификатора платежа (`payment_id`).\n"
    )
    assert [f for f in run_prechecks(_parse_inline(md_ok)).findings if f.rule == "idempotency_without_key"] == []


def test_flag_with_mandatory_params_positive_and_negative():
    md_missing = (
        "# Настройки подписки\n\n"
        "## Параметры запроса\n\n"
        "| Parameter | Format | Mandatory | Logic | Description |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `maxCount` | integer | Да | 1–100 | Максимальное число сообщений |\n"
        "| `enabled` | boolean | Да | true / false | Включение рассылки |\n"
    )
    doc_missing = _parse_inline(md_missing)
    f_missing = [f for f in run_prechecks(doc_missing).findings if f.rule == "flag_with_mandatory_params"]
    assert len(f_missing) == 1
    assert f_missing[0].criterion == Criterion.SET_COMPLETE
    assert f_missing[0].severity == Severity.MAJOR
    _assert_quotes_exact_substrings(doc_missing, f_missing)

    md_ok = md_missing + "\nЕсли enabled = false, параметр maxCount не проверяется и может быть опущен.\n"
    assert [f for f in run_prechecks(_parse_inline(md_ok)).findings if f.rule == "flag_with_mandatory_params"] == []


def test_one_line_process_aggregates_multiple_sections_into_single_finding():
    md_bad = (
        "# Обработка заявок\n\n"
        "## BPMN схема процесса\n"
        "Создание заявки → проверка данных → сохранение → завершение.\n\n"
        "## Sequence Diagram\n"
        "Клиент -> шлюз -> сервис -> база.\n"
    )
    doc_bad = _parse_inline(md_bad)
    f_bad = [f for f in run_prechecks(doc_bad).findings if f.rule == "one_line_process"]
    assert len(f_bad) == 1
    assert f_bad[0].criterion == Criterion.COMPLETE
    assert f_bad[0].severity == Severity.MINOR
    assert len(f_bad[0].block_ids) == 2
    assert "BPMN схема процесса" in f_bad[0].explanation
    assert "Sequence Diagram" in f_bad[0].explanation
    _assert_quotes_exact_substrings(doc_bad, f_bad)

    md_ok = (
        "# Обработка заявок\n\n"
        "## Схема процесса\n"
        "Создание заявки → проверка данных → сохранение; при ошибке вернуть отказ.\n\n"
        "## Sequence Diagram\n"
        "```text\n"
        "Клиент -> шлюз: запрос\n"
        "шлюз -> клиент: ответ или ошибка\n"
        "```\n"
    )
    assert [f for f in run_prechecks(_parse_inline(md_ok)).findings if f.rule == "one_line_process"] == []


def test_sample_requirements_doc_fires_all_four_rules_with_aggregated_process():
    doc = _load_doc("docs/requirements.md")
    res = run_prechecks(doc)
    gap_findings = run_gap_checks(doc, res.inventory)
    _assert_quotes_exact_substrings(doc, gap_findings)

    rules = {f.rule for f in gap_findings}
    assert {
        "units_mismatch",
        "idempotency_without_key",
        "flag_with_mandatory_params",
        "one_line_process",
    } <= rules

    proc_findings = [f for f in gap_findings if f.rule == "one_line_process"]
    assert len(proc_findings) == 1
    assert len(proc_findings[0].block_ids) == 3


def test_holdout_docs_have_no_gap_false_positives_on_clean_requirements():
    for doc_rel, gold_rel in (
        ("eval/docs/doc1_autopay.md", "eval/gold/doc1_autopay.gold.json"),
        ("eval/docs/doc2_p2p.md", "eval/gold/doc2_p2p.gold.json"),
    ):
        doc_path = ROOT / doc_rel
        gold_path = ROOT / gold_rel
        if not doc_path.exists() or not gold_path.exists():
            continue
        doc = _load_doc(doc_rel)
        res = run_prechecks(doc)
        gap_findings = run_gap_checks(doc, res.inventory)
        _assert_quotes_exact_substrings(doc, gap_findings)

        clean_anchors = [
            entry["anchor"].lower()
            for entry in json.loads(gold_path.read_text(encoding="utf-8")).get("clean", [])
        ]
        block_by_id = {b.id: b for b in doc.blocks}
        for finding in gap_findings:
            hit_texts = [finding.quote.lower()] + [
                block_by_id[bid].text.lower() for bid in finding.block_ids if bid in block_by_id
            ]
            for anchor in clean_anchors:
                assert not any(anchor in txt for txt in hit_texts), (
                    f"Gap finding {finding.rule} on {doc_rel} hit clean requirement anchor {anchor!r}: "
                    f"blocks={finding.block_ids}, quote={finding.quote!r}"
                )
