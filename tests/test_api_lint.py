"""Tests for the domain-neutral REST/API specification linter."""

from __future__ import annotations

from pathlib import Path

from validator.loaders import load_document
from validator.models import Criterion, Document, PrecheckResult, RawDocument, Severity
from validator.rules import run_api_lint, run_prechecks
from validator.structure import parse_structure

ROOT = Path(__file__).resolve().parents[1]


def _parse_inline(md: str, *, title: str = "API Spec") -> Document:
    raw = RawDocument(path="spec.md", name="spec.md", title=title, format="md", text=md)
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


def test_clean_api_spec_produces_zero_api_lint_findings():
    md = (
        "# Заказы — REST API\n\n"
        "## Общие параметры\n\n"
        "| Поле | Значение |\n"
        "| --- | --- |\n"
        "| **Description** | Получение информации о заказе и его статусе |\n"
        "| **REST type** | GET |\n"
        "| **Request URL** | `/api/v1/orders/{orderId}` |\n\n"
        "## Параметры запроса\n\n"
        "| Parameter | Format | Mandatory | Description |\n"
        "| --- | --- | --- | --- |\n"
        "| `orderId` | string | Да | Идентификатор заказа |\n"
        "| `includeItems` | boolean | Нет | Включать позиции |\n\n"
        "## Request body\n"
        "Тело не используется.\n\n"
        "## Ответ\n"
        "Если результат 200 — успешный. При ошибке возвращается 400 или 404.\n\n"
        "| Parameter | Format | Description |\n"
        "| --- | --- | --- |\n"
        "| `orderId` | string | Идентификатор заказа |\n"
        "| `total` | integer | Итоговая сумма |\n\n"
        "## Статусы\n\n"
        "**200.** Успешный ответ.\n"
        "**400.** Некорректный формат идентификатора.\n"
        "**404.** Заказ не найден.\n\n"
        "### 200. OK\n"
        "```json\n"
        '{"orderId": "ord-1", "total": 1500}\n'
        "```\n\n"
        "### 400. Bad Request\n"
        "```json\n"
        '{"error": "invalid_id"}\n'
        "```\n\n"
        "### 404. Not Found\n"
        "```json\n"
        '{"error": "not_found"}\n'
        "```\n"
    )
    doc = _parse_inline(md)
    res: PrecheckResult = run_prechecks(doc)
    api_findings = run_api_lint(doc, res.inventory)
    assert api_findings == []


def test_ordinary_prose_produces_zero_api_lint_findings():
    md = (
        "# Бронирование переговорных\n\n"
        "## Функциональные требования\n"
        "- FR-1. Сотрудник может сохранить бронь переговорной на 60 минут.\n"
        "- FR-2. При отмене брони система отправляет уведомление участникам.\n"
        "- FR-3. Комиссия за аренду составляет 404 или 500 ₸ в час.\n"
        "- FR-4. Приоритет: высокий. Для получения списка используется GET-запрос к каталогу.\n"
    )
    doc = _parse_inline(md)
    res = run_prechecks(doc)
    assert run_api_lint(doc, res.inventory) == []


def test_get_write_side_effect_major_and_critical_with_query():
    md_major = (
        "# Профиль — REST API\n\n"
        "## Общие параметры\n\n"
        "| Поле | Значение |\n"
        "| --- | --- |\n"
        "| **Description** | Обновление имени пользователя |\n"
        "| **Method** | GET |\n"
        "| **Request URL** | `/api/v1/profile/updateName` |\n"
    )
    doc_major = _parse_inline(md_major)
    res_major = run_prechecks(doc_major)
    f_major = [f for f in res_major.findings if f.rule == "api_get_write_side_effect"]
    assert len(f_major) == 1
    assert f_major[0].criterion == Criterion.CORRECT
    assert f_major[0].severity == Severity.MAJOR
    _assert_quotes_exact_substrings(doc_major, f_major)

    md_critical = (
        md_major
        + "\n## Query-параметры\n\n"
        "| Parameter | Format | Description |\n"
        "| --- | --- | --- |\n"
        "| `fullName` | string | Новое имя |\n"
    )
    doc_crit = _parse_inline(md_critical)
    res_crit = run_prechecks(doc_crit)
    f_crit = [f for f in res_crit.findings if f.rule == "api_get_write_side_effect"]
    assert len(f_crit) == 1
    assert f_crit[0].severity == Severity.CRITICAL
    _assert_quotes_exact_substrings(doc_crit, f_crit)

    # Read-only GET on "/api/v1/assets" or "/api/v1/settings" must NOT trigger false positive on "set".
    md_safe = (
        "# Справочники — REST API\n\n"
        "| Method | Path | Description |\n"
        "| --- | --- | --- |\n"
        "| GET | `/api/v1/assets` | Получение списка активов |\n"
        "| GET | `/api/v1/settings` | Чтение настроек пользователя |\n"
        "| POST | `/api/v1/settings/save` | Сохранение настроек |\n"
    )
    doc_safe = _parse_inline(md_safe)
    assert [f for f in run_api_lint(doc_safe, []) if f.rule == "api_get_write_side_effect"] == []


def test_status_code_consistency_undeclared_error_in_2xx_and_mixed_heading():
    md = (
        "# Билеты — REST API\n\n"
        "## Ответ\n"
        "При успехе возвращается 200. При конфликте броней returns 409.\n\n"
        "## Статусы\n\n"
        "**200.** Успешное выполнение.\n"
        "**400.** Неверные параметры запроса.\n"
        "**500.** Внутренняя ошибка.\n\n"
        "### 200. Successful — Created\n"
        "```json\n"
        '{"status": "OK", "ticketId": "t-1"}\n'
        "```\n"
        "или\n"
        "```json\n"
        '{"status": "ERROR", "message": "Некорректная дата"}\n'
        "```\n\n"
        "### 400. Bad Request\n"
        "```json\n"
        '{"message": "Invalid input"}\n'
        "```\n"
    )
    doc = _parse_inline(md)
    res = run_prechecks(doc)
    api_findings = run_api_lint(doc, res.inventory)
    _assert_quotes_exact_substrings(doc, api_findings)

    undeclared = [f for f in api_findings if f.rule == "api_undeclared_status_code"]
    assert len(undeclared) == 1
    assert undeclared[0].criterion == Criterion.SET_CONSISTENT
    assert undeclared[0].severity == Severity.MAJOR
    assert "409" in undeclared[0].explanation

    err_in_2xx = [f for f in api_findings if f.rule == "api_success_code_error_body"]
    assert len(err_in_2xx) == 1
    assert err_in_2xx[0].criterion == Criterion.SET_CONSISTENT
    assert err_in_2xx[0].severity == Severity.CRITICAL

    heading_mix = [f for f in api_findings if f.rule == "api_status_heading_mismatch"]
    assert len(heading_mix) == 1
    assert heading_mix[0].criterion == Criterion.UNAMBIGUOUS
    assert "201" in heading_mix[0].explanation


def test_request_vs_response_type_mismatch():
    md = (
        "# Платежи — REST API\n\n"
        "## Параметры запроса\n\n"
        "| Parameter | Format | Description |\n"
        "| --- | --- | --- |\n"
        "| `sum` | integer | Сумма операции |\n"
        "| `currency` | string | Код валюты |\n"
        "| `retry` | boolean | Признак повтора |\n\n"
        "## Ответ\n\n"
        "| Parameter | Format | Description |\n"
        "| --- | --- | --- |\n"
        "| `sum` | string | Сумма строкой |\n"
        "| `currency` | строка | Код валюты (синоним типа — не конфликт) |\n"
        "| `retry` | integer | 0 или 1 |\n"
    )
    doc = _parse_inline(md)
    res = run_prechecks(doc)
    mismatches = [f for f in res.findings if f.rule == "api_req_resp_type_mismatch"]
    _assert_quotes_exact_substrings(doc, mismatches)

    fields = {f.explanation.split("«", 1)[1].split("»", 1)[0] for f in mismatches}
    assert fields == {"sum", "retry"}
    assert all(f.criterion == Criterion.SET_CONSISTENT and f.severity == Severity.MAJOR for f in mismatches)


def test_unused_body_with_content_type_and_get_with_json_body():
    md_unused = (
        "# Сервис — REST API\n\n"
        "## Request body\n"
        "- **Content-Type:** `application/json`\n"
        "- Тело не используется.\n"
    )
    doc_unused = _parse_inline(md_unused)
    f_unused = [f for f in run_prechecks(doc_unused).findings if f.rule == "api_body_content_type_unused"]
    assert len(f_unused) == 1
    assert f_unused[0].severity == Severity.MINOR
    _assert_quotes_exact_substrings(doc_unused, f_unused)

    md_get_body = (
        "# Поиск — REST API\n\n"
        "## Общие параметры\n\n"
        "| Поле | Значение |\n"
        "| --- | --- |\n"
        "| **Method** | GET |\n"
        "| **Request URL** | `/api/v1/catalog/search` |\n\n"
        "## Request body\n"
        "```json\n"
        '{"query": "book", "limit": 20}\n'
        "```\n"
    )
    doc_get_body = _parse_inline(md_get_body)
    f_get_body = [f for f in run_prechecks(doc_get_body).findings if f.rule == "api_get_with_body"]
    assert len(f_get_body) == 1
    assert f_get_body[0].severity == Severity.MINOR
    _assert_quotes_exact_substrings(doc_get_body, f_get_body)


def test_sample_requirements_doc_catches_api_items_1_and_2():
    req_path = ROOT / "docs" / "requirements.md"
    loaded = load_document(req_path)
    raw = (
        loaded
        if isinstance(loaded, RawDocument)
        else RawDocument(
            path=str(req_path),
            name=req_path.name,
            title=loaded.title,
            format="md",
            text=loaded.text,
        )
    )
    doc = parse_structure(raw)
    res = run_prechecks(doc)
    api_findings = run_api_lint(doc, res.inventory)
    _assert_quotes_exact_substrings(doc, api_findings)

    rules = {f.rule for f in api_findings}
    # Item 1: GET method with write side effect + query parameters -> critical.
    get_write = [f for f in api_findings if f.rule == "api_get_write_side_effect"]
    assert len(get_write) == 1
    assert get_write[0].severity == Severity.CRITICAL

    # Item 2: undeclared status code (e.g. 404/201), error body under 2xx (critical), and mixed heading.
    assert "api_undeclared_status_code" in rules
    assert any(f.rule == "api_success_code_error_body" and f.severity == Severity.CRITICAL for f in api_findings)
    assert "api_status_heading_mismatch" in rules

    # Bonus verification for items 3 and 4 on the sample document.
    assert "api_req_resp_type_mismatch" in rules
    assert "api_body_content_type_unused" in rules
