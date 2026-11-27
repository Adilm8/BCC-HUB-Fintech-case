"""Prompt for the critic: keep / drop / merge, severity alignment, missed critical defects."""

from __future__ import annotations

from validator.models import Document
from validator.pipeline.prompts.render import render_document

TASK = """Задание: ты — критик. Перед тобой находки-кандидаты других рецензентов (<candidates>).
1. По каждой находке верни решение в decisions:
   - keep — дефект реальный; severity — выровненная по всему набору критичность;
   - drop — с причиной (для keep и merge drop_reason = none): document_specifies (документ уже это определяет — обязательно приведи дословную \
counter_quote из документа), generic (общая фраза без конкретики), out_of_scope (не о требованиях), \
duplicate (дубль — укажи в merge_into ID оставляемой находки);
   - merge — то же, что duplicate: merge_into обязателен.
   - improved_question — переписанный вопрос автору (конкретный, с вопросительным знаком), если исходный слабый; \
иначе пустая строка.
2. Не удаляй и не понижай противоречие с двумя цитатами без counter_quote, которая его снимает.
   Находку о пробеле (границы, состояния, ошибки, единицы, источник данных) не удаляй как generic, если она \
называет конкретный недостающий факт: если пробел реален, но мелкий, оставь её с severity minor.
   Придирку к корректному требованию (в нём заданы значение, единица, границы и поведение при нарушении, \
а находка требует лишних подробностей реализации) удаляй как generic.
3. Шкала критичности: critical — основной сценарий нельзя однозначно реализовать или проверить, или два места \
требуют разного; major — альтернативные сценарии и границы; minor — оформление. Оформление, единичность, \
ранжирование — не выше minor.
4. missed — до 5 пропущенных КРИТИЧНЫХ дефектов (особенно противоречий между частями и с критериями приёмки), \
в формате находки набора: минимум одна дословная цитата, для противоречия — две и сценарий.
5. fix_first — до 7 ID находок (C…) в порядке, в котором автору стоит их исправлять.
6. verdict_rationale — 2–3 фразы: можно ли начинать разработку и тестирование и почему."""


def build_review_prompt(doc: Document, candidates_text: str, *, include_document: bool = True,
                        task: str | None = None) -> str:
    parts = []
    if include_document:
        parts.append(render_document(doc))
    parts.append("<candidates>\n" + candidates_text + "\n</candidates>")
    parts.append(task or TASK)
    return "\n\n".join(parts)
