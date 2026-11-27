"""Prompt for the per-requirement inspector."""

from __future__ import annotations

from collections import defaultdict

from validator.models import Document, Hint, Requirement
from validator.pipeline.prompts.examples import CHECK_EXAMPLES
from validator.pipeline.prompts.render import render_document, requirement_table

TASK = """Задание: проверь КАЖДОЕ требование из <requirements> по критериям отдельного требования.
- Верни ровно одну запись в results на каждый ID из списка, в том же порядке, даже если находок нет \
(тогда findings пустой).
- Для каждого требования мысленно попробуй написать тест и реализацию: где возникнет развилка или догадка — \
это находка. Проверь границы и включительность, значения по умолчанию, временную базу, единицы, \
ошибочные и повторные сценарии, однозначность того, что именно учитывается.
- Для строк критериев приёмки проверяй только их собственную полноту и проверяемость; сверку с требованиями \
делает другой этап.
- Для требований к интерфейсу применяй чек-лист интерфейсов из системных правил.
- Противоречие с другим требованием отмечай criterion set_consistent и заполняй related_requirement_ids (ID R-…).
- <hints> — кандидаты от автоматических правил: подтверди находкой или проигнорируй, если это не дефект.
- Цитату бери из текста самого требования (или из блока, где виден дефект)."""


def build_check_prompt(doc: Document, batch: list[Requirement], all_reqs: list[Requirement],
                       hints: list[Hint], *, include_document: bool = True) -> str:
    by_block: dict[str, list[Hint]] = defaultdict(list)
    for h in hints:
        by_block[h.block_id].append(h)
    hint_lines = []
    for r in batch:
        for bid in r.block_ids:
            for h in by_block.get(bid, []):
                hint_lines.append(f"{r.id}: «{h.match}» — {h.message}")
    parts = []
    if include_document:
        parts.append(render_document(doc))
    else:
        parts.append(render_document(doc, {b for r in batch for b in r.block_ids}))
    parts.append("<all_requirements>\n" + requirement_table(all_reqs, with_text=False) + "\n</all_requirements>")
    parts.append("<requirements>\n" + requirement_table(batch) + "\n</requirements>")
    if hint_lines:
        parts.append("<hints>\n" + "\n".join(hint_lines[:120]) + "\n</hints>")
    parts.append(CHECK_EXAMPLES)
    parts.append(TASK)
    return "\n\n".join(parts)
