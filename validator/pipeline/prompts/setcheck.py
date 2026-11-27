"""Prompts for the set inspector: S1 (consistency, parts) and S2 (acceptance criteria, completeness)."""

from __future__ import annotations

from validator.models import Document, InventoryItem, ReqKind, Requirement, SkippedBlock
from validator.pipeline.prompts.examples import SET_EXAMPLES
from validator.pipeline.prompts.render import inventory_table, render_document, requirement_table

S1_TASK = """Задание S1: найди противоречия внутри набора и между частями документа.
Шаг 1 (про себя, в ответ не выводи): составь реестр общих понятий — для каждого значения, единицы, \
даты и временной базы, лимита на период, метода и кода ответа, обязательного поля и флага, а также \
«кто что читает и записывает» выпиши ВСЕ места документа, где оно упоминается.
Шаг 2: сравни места попарно. Обязательно проверь:
- флаги и переключатели, которые одна часть определяет, а другая игнорирует;
- операции, которых требует одна часть (чтение, сохранение, отображение, выключение), но другая часть \
(спецификация интерфейса, интеграция) не предоставляет;
- проверки, которых требуют бизнес-правила, но нет в логике интерфейса;
- одинаковые величины с разными значениями, единицами, форматами или типами в разных местах;
- коды ответов и результаты, которые по-разному описаны в логике, перечне статусов и примерах;
- назначение, которое описано по-разному в разных полях одной спецификации;
- бизнес-логику другой системы, записанную внутри спецификации интерфейса (смешение уровней).
Правила:
- Каждая находка о противоречии содержит минимум две цитаты из разных мест в evidence и конкретный сценарий \
в scenario: входные данные → что следует из места X → что следует из места Y. Нет сценария — нет находки.
- set_check cross_part — если места в разных частях документа; contradiction — внутри одной части; \
terminology — разные термины для одного понятия.
- Не сверяй требования с критериями приёмки: это делает другой этап.
- coverage верни пустым списком.
- Не больше 12 находок, самые существенные."""

S2_TASK = """Задание S2: сверь требования с критериями приёмки и оцени полноту набора.
1. Для каждой строки критериев приёмки примени текст требований буквально к её исходным условиям и действию \
и сравни полученный результат с ожидаемым результатом строки. Расхождение — находка acceptance_mismatch \
(criterion set_consistent) с двумя цитатами (требование и строка критерия) и сценарием в scenario.
2. coverage: для каждого требования типов functional, nonfunctional, integration, data, interface перечисли \
ID покрывающих критериев приёмки (R-…) и статус covered | partial | none; в note — чего не хватает.
3. Непокрытые сценарии (missing_scenario): границы диапазонов, отказы и ошибки, значения по умолчанию, \
повторы и отмены, которых нет ни в требованиях, ни в критериях. Объединяй однотипные пробелы в одну находку.
4. Отсутствующие классы требований (missing_requirement): по обобщённому чек-листу полноты ниже — \
только то, что действительно нужно этому документу.
5. Пустые разделы и заглушки (empty_section), незаполненные шаблоны.
Для находки-пропуска цитата — место, где пропуск виден (заголовок пустого раздела, соседнее требование, фраза, \
из которой следует потребность). scenario можно оставить пустым, кроме acceptance_mismatch.
Не больше 12 находок.

Обобщённый чек-лист полноты:
- функциональные: предусловия, границы, отказы, значения по умолчанию, повтор и отмена;
- нефункциональные: производительность с метрикой, доступность, безопасность, аудит, хранение и удаление данных;
- интерфейсы: операции под каждую потребность, схемы запроса и ответа, коды ошибок, идемпотентность, права;
- документ: версия, приоритеты требований, согласование, откат."""

MERGED_TASK = """Выполни оба задания в одном ответе: сначала S1, затем S2. Общий лимит — 16 находок."""


def _common(doc: Document, reqs: list[Requirement], include_document: bool) -> list[str]:
    parts = []
    if include_document:
        parts.append(render_document(doc))
    parts.append("<requirements>\n" + requirement_table(reqs, with_text=not include_document) + "\n</requirements>")
    return parts


def build_s1_prompt(doc: Document, reqs: list[Requirement], inventory: list[InventoryItem], *,
                    include_document: bool = True, merged_with_s2: str | None = None) -> str:
    parts = _common(doc, reqs, include_document)
    parts.append("<parts>\n" + "\n".join(f"- {p}" for p in doc.parts) + "\n</parts>" if doc.parts else
                 "<parts>документ из одной части</parts>")
    if inventory:
        parts.append("<inventory>\n" + inventory_table(inventory) + "\n</inventory>")
    parts.append(SET_EXAMPLES)
    parts.append(S1_TASK)
    if merged_with_s2:
        parts.append(merged_with_s2)
    return "\n\n".join(parts)


def build_s2_prompt(doc: Document, reqs: list[Requirement], skipped: list[SkippedBlock], *,
                    include_document: bool = True) -> str:
    parts = _common(doc, reqs, include_document)
    acc = [r for r in reqs if r.kind == ReqKind.ACCEPTANCE_CRITERION]
    parts.append("<acceptance_criteria>\n" + (requirement_table(acc) if acc else "в документе нет критериев приёмки")
                 + "\n</acceptance_criteria>")
    if skipped:
        parts.append("<skipped_blocks>\n" + "\n".join(f"{s.block_id} | {s.reason} | {s.text[:150]}" for s in skipped[:80])
                     + "\n</skipped_blocks>")
    parts.append(SET_EXAMPLES)
    parts.append(S2_TASK)
    return "\n\n".join(parts)


def build_merged_prompt(doc: Document, reqs: list[Requirement], inventory: list[InventoryItem],
                        skipped: list[SkippedBlock], *, include_document: bool = True) -> str:
    s2 = build_s2_prompt(doc, reqs, skipped, include_document=False).split("<acceptance_criteria>", 1)[1]
    s1 = build_s1_prompt(doc, reqs, inventory, include_document=include_document)
    s1 = s1.replace("coverage верни пустым списком.", "")
    return s1 + "\n\n<acceptance_criteria>" + s2 + "\n\n" + MERGED_TASK
