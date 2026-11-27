"""Tests for deterministic lexicon matchers and rule prechecks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from validator.models import (
    Block,
    BlockKind,
    Criterion,
    Document,
    PrecheckResult,
    Section,
    Severity,
)
from validator.rules import extract_image_warnings, run_prechecks
from validator.rules.lexicon import (
    HINT_CATEGORIES,
    PLACEHOLDER_INLINE_RE,
    PLACEHOLDER_LABELED_RE,
    PLACEHOLDER_VALUE_RE,
)

ROOT = Path(__file__).resolve().parents[1]


def _make_doc(
    sections: list[Section],
    blocks: list[Block],
    *,
    title: str = "Тестовое ТЗ",
    text: str = "",
) -> Document:
    full_text = text or "\n".join(b.raw for b in blocks)
    parts = list(dict.fromkeys(s.part for s in sections if s.part))
    return Document(
        path="inline.md",
        name="inline.md",
        title=title,
        format="md",
        sha256=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        char_count=len(full_text),
        text=full_text,
        sections=sections,
        blocks=blocks,
        parts=parts,
    )


def _assert_quotes_are_exact_substrings(doc: Document, result: PrecheckResult) -> None:
    block_by_id = {b.id: b for b in doc.blocks}
    for finding in result.findings:
        assert finding.quote, f"Empty quote in finding {finding.rule}"
        if finding.block_ids:
            primary = block_by_id[finding.block_ids[0]]
            assert finding.quote in primary.text, (
                f"Quote {finding.quote!r} is not an exact substring of block {primary.id}: {primary.text!r}"
            )
        else:
            assert finding.rule == "empty_section"
            assert finding.quote in doc.text


def _parse_markdown_fallback(path: Path) -> Document:
    """Minimal deterministic Markdown-to-Document parser used when validator.structure is absent."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    title = path.stem
    for line in lines:
        if line.strip().startswith("#"):
            title = line.strip().lstrip("#").strip()
            break

    sections: list[Section] = []
    blocks: list[Block] = []
    heading_stack: list[Section] = []
    sec_counter = 0
    blk_counter = 0
    in_code = False
    code_lines: list[str] = []
    code_start = 0

    def current_section() -> Section:
        if heading_stack:
            return heading_stack[-1]
        nonlocal sec_counter
        sec_counter += 1
        root = Section(id=f"S{sec_counter:02d}", title=title, level=1, path=[title], part=title)
        sections.append(root)
        heading_stack.append(root)
        return root

    def add_block(
        kind: BlockKind,
        clean_text: str,
        raw: str,
        line_start: int,
        line_end: int,
        *,
        doc_id: str | None = None,
        cells: list[str] | None = None,
    ) -> Block:
        nonlocal blk_counter
        sec = current_section()
        blk_counter += 1
        block = Block(
            id=f"B{blk_counter:03d}",
            kind=kind,
            text=clean_text,
            raw=raw,
            section_id=sec.id,
            section_path=list(sec.path),
            part=sec.part,
            line_start=line_start,
            line_end=line_end,
            doc_id=doc_id,
            cells=cells,
        )
        blocks.append(block)
        sec.block_ids.append(block.id)
        return block

    idx = 0
    in_table = False
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        line_no = idx + 1

        if stripped.startswith("```"):
            in_table = False
            if not in_code:
                in_code = True
                code_lines = []
                code_start = line_no
            else:
                in_code = False
                joined = "\n".join(code_lines)
                add_block(BlockKind.CODE, joined, f"```\n{joined}\n```", code_start, line_no)
            idx += 1
            continue

        if in_code:
            code_lines.append(line)
            idx += 1
            continue

        if not stripped or re.fullmatch(r"[-_*]{3,}", stripped):
            in_table = False
            idx += 1
            continue

        m_head = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m_head:
            in_table = False
            level = len(m_head.group(1))
            h_title = m_head.group(2).strip()
            while heading_stack and heading_stack[-1].level >= level:
                heading_stack.pop()
            parent = heading_stack[-1] if heading_stack else None
            path_list = [*(parent.path if parent else []), h_title]
            part = path_list[1] if len(path_list) >= 2 else path_list[0]
            sec_counter += 1
            sec = Section(
                id=f"S{sec_counter:02d}",
                title=h_title,
                level=level,
                path=path_list,
                parent_id=parent.id if parent else None,
                part=part,
            )
            sections.append(sec)
            heading_stack.append(sec)
            add_block(BlockKind.HEADING, h_title, line, line_no, line_no)
            idx += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            raw_cells = [c.strip() for c in stripped.strip("|").split("|")]
            if any(raw_cells) and all(re.fullmatch(r":?-+:?", c) for c in raw_cells if c):
                idx += 1
                continue
            clean_cells = [re.sub(r"\*\*|`", "", c).strip() for c in raw_cells]
            kind = BlockKind.TABLE_HEADER if not in_table else BlockKind.TABLE_ROW
            in_table = True
            add_block(kind, " | ".join(clean_cells), line, line_no, line_no, cells=clean_cells)
            idx += 1
            continue

        in_table = False
        m_list = re.match(r"^(?:[\*\-•]|\d+\.)\s+(.+)$", stripped)
        if m_list:
            clean = re.sub(r"\*\*|`", "", m_list.group(1)).strip()
            m_id = re.match(r"^([A-Za-zА-Яа-яЁё]{1,6}[-_]\d{1,5})[\.\:\s]", clean)
            add_block(
                BlockKind.LIST_ITEM,
                clean,
                line,
                line_no,
                line_no,
                doc_id=m_id.group(1) if m_id else None,
            )
            idx += 1
            continue

        clean = re.sub(r"\*\*|`", "", stripped).strip()
        add_block(BlockKind.PARAGRAPH, clean, line, line_no, line_no)
        idx += 1

    parts = list(dict.fromkeys(s.part for s in sections if s.level == 2)) or [title]
    return Document(
        path=str(path),
        name=path.name,
        title=title,
        format="md",
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        char_count=len(text),
        text=text,
        sections=sections,
        blocks=blocks,
        parts=parts,
    )


def _load_requirements_doc() -> Document:
    req_path = ROOT / "docs" / "requirements.md"
    try:
        from validator.loaders import load_document
        from validator.models import RawDocument
        from validator.structure import parse_structure  # type: ignore[import-not-found]
    except ImportError:
        return _parse_markdown_fallback(req_path)

    loaded = load_document(req_path)
    if isinstance(loaded, RawDocument):
        return parse_structure(loaded)
    raw = RawDocument(
        path=str(req_path),
        name=req_path.name,
        title=loaded.title,
        format="md",
        text=loaded.text,
    )
    return parse_structure(raw)


# ---------------------------------------------------------------- lexicon tests


def test_lexicon_word_boundaries_and_russian_morphology():
    by_rule = {cat.rule: cat.pattern for cat in HINT_CATEGORIES}

    assert by_rule["vague_word"].search("Система должна работать быстро и надёжно.")
    assert by_rule["vague_word"].search("Интерфейс должен быть удобным и интуитивным.")
    assert by_rule["vague_word"].search("The search must be fast and user-friendly.")
    # Morphology guards: nouns like быстродействие / надёжность / пространство must not match.
    assert not by_rule["vague_word"].search("Требования к быстродействию и надёжности пространства.")

    assert by_rule["weak_modality"].search("Сервис должен стараться отвечать вовремя и может кэшировать.")
    assert by_rule["weak_modality"].search("Выполнить по возможности в фоновом режиме.")
    assert by_rule["weak_modality"].search("The service may cache results on a best effort basis.")
    assert not by_rule["weak_modality"].search("Это поможет пользователю завершить операцию.")

    assert by_rule["open_ended_list"].search("Поддерживаются форматы PDF, DOCX и т.д.")
    assert by_rule["open_ended_list"].search("Ошибки сети, таймауты и т. п. логируются.")
    assert by_rule["open_ended_list"].search("Export supports CSV, JSON, etc.")
    assert not by_rule["open_ended_list"].search("Нужно fetch данные и другая логика.")

    assert by_rule["loophole"].search("Выполнять аудит при необходимости и если возможно.")
    assert by_rule["loophole"].search("Retry if applicable and as needed.")

    assert by_rule["superlative_comparative"].search("Обеспечить наилучшее качество и быстрее аналогов.")
    assert by_rule["superlative_comparative"].search("Deliver the fastest and better experience.")

    assert by_rule["and_or"].search("Отправить письмо и/или SMS уведомление.")
    assert by_rule["and_or"].search("Notify via email and / or webhook.")

    assert PLACEHOLDER_INLINE_RE.search("Значение уточняется и будет определено позже.")
    assert PLACEHOLDER_INLINE_RE.search("Раздел заполняются аналитиками, статус TBD.")
    assert PLACEHOLDER_VALUE_RE.match("—")
    assert PLACEHOLDER_LABELED_RE.search("Dev: /api/v1/items . Prod: —")
    # Normal Russian em-dash in prose must not be treated as a placeholder.
    assert not PLACEHOLDER_VALUE_RE.match("Модуль B — дочерняя спецификация.")
    assert not PLACEHOLDER_LABELED_RE.search("Модуль B — дочерняя спецификация.")


# ---------------------------------------------------------------- inline precheck rule tests


def test_lexicon_hints_skip_code_and_headers_and_deduplicate_overlaps():
    sec = Section(
        id="S01",
        title="Общие требования",
        level=2,
        path=["Общие требования"],
        part="Общие требования",
        block_ids=["B001", "B002", "B003", "B004", "B005"],
    )
    blocks = [
        Block(
            id="B001",
            kind=BlockKind.HEADING,
            text="Общие требования",
            raw="## Общие требования",
            section_id="S01",
            section_path=["Общие требования"],
            part="Общие требования",
            line_start=1,
            line_end=1,
        ),
        Block(
            id="B002",
            kind=BlockKind.PARAGRAPH,
            text="Сервис должен работать быстро, по возможности отправлять отчёт и/или архив и т.д.",
            raw="Сервис должен работать быстро, по возможности отправлять отчёт и/или архив и т.д.",
            section_id="S01",
            section_path=["Общие требования"],
            part="Общие требования",
            line_start=2,
            line_end=2,
        ),
        Block(
            id="B003",
            kind=BlockKind.PARAGRAPH,
            text="При необходимости вернуть самый быстрый ответ.",
            raw="При необходимости вернуть самый быстрый ответ.",
            section_id="S01",
            section_path=["Общие требования"],
            part="Общие требования",
            line_start=3,
            line_end=3,
        ),
        Block(
            id="B004",
            kind=BlockKind.CODE,
            text='{"status": "fast", "note": "etc."}',
            raw='```json\n{"status": "fast", "note": "etc."}\n```',
            section_id="S01",
            section_path=["Общие требования"],
            part="Общие требования",
            line_start=4,
            line_end=6,
        ),
        Block(
            id="B005",
            kind=BlockKind.TABLE_HEADER,
            text="Fast | Optional | Best",
            raw="| Fast | Optional | Best |",
            section_id="S01",
            section_path=["Общие требования"],
            part="Общие требования",
            line_start=7,
            line_end=7,
            cells=["Fast", "Optional", "Best"],
        ),
    ]
    doc = _make_doc([sec], blocks)
    res = run_prechecks(doc)

    rules_b002 = {h.rule for h in res.hints if h.block_id == "B002"}
    assert rules_b002 == {"vague_word", "weak_modality", "and_or", "open_ended_list"}

    # "самый быстрый" must match superlative_comparative once, not also vague_word on "быстрый".
    b003_hints = [h for h in res.hints if h.block_id == "B003"]
    assert {h.rule for h in b003_hints} == {"loophole", "superlative_comparative"}

    # CODE and TABLE_HEADER blocks must not generate hints.
    assert not any(h.block_id in {"B004", "B005"} for h in res.hints)


def test_unfilled_template_table_emits_single_finding_and_ignores_urls():
    s1 = Section(
        id="S01",
        title="Чек-лист",
        level=2,
        path=["Чек-лист"],
        part="Чек-лист",
        block_ids=["B001", "B002", "B003", "B004", "B005"],
    )
    s2 = Section(
        id="S02",
        title="Эндпоинты",
        level=2,
        path=["Эндпоинты"],
        part="Эндпоинты",
        block_ids=["B006", "B007", "B008", "B009", "B010"],
    )
    blocks = [
        Block(
            id="B001",
            kind=BlockKind.HEADING,
            text="Чек-лист",
            raw="## Чек-лист",
            section_id="S01",
            section_path=["Чек-лист"],
            part="Чек-лист",
            line_start=1,
            line_end=1,
        ),
        Block(
            id="B002",
            kind=BlockKind.TABLE_HEADER,
            text="Пункт | Готово | Примечание",
            raw="| Пункт | Готово | Примечание |",
            section_id="S01",
            section_path=["Чек-лист"],
            part="Чек-лист",
            line_start=2,
            line_end=2,
            cells=["Пункт", "Готово", "Примечание"],
        ),
        Block(
            id="B003",
            kind=BlockKind.TABLE_ROW,
            text="Проверка 1 | Да / Нет | ",
            raw="| Проверка 1 | Да / Нет | |",
            section_id="S01",
            section_path=["Чек-лист"],
            part="Чек-лист",
            line_start=4,
            line_end=4,
            cells=["Проверка 1", "Да / Нет", ""],
        ),
        Block(
            id="B004",
            kind=BlockKind.TABLE_ROW,
            text="Проверка 2 | Да / Нет | ",
            raw="| Проверка 2 | Да / Нет | |",
            section_id="S01",
            section_path=["Чек-лист"],
            part="Чек-лист",
            line_start=5,
            line_end=5,
            cells=["Проверка 2", "Да / Нет", ""],
        ),
        Block(
            id="B005",
            kind=BlockKind.TABLE_ROW,
            text="Проверка 3 | Да / Нет | ",
            raw="| Проверка 3 | Да / Нет | |",
            section_id="S01",
            section_path=["Чек-лист"],
            part="Чек-лист",
            line_start=6,
            line_end=6,
            cells=["Проверка 3", "Да / Нет", ""],
        ),
        Block(
            id="B006",
            kind=BlockKind.HEADING,
            text="Эндпоинты",
            raw="## Эндпоинты",
            section_id="S02",
            section_path=["Эндпоинты"],
            part="Эндпоинты",
            line_start=8,
            line_end=8,
        ),
        Block(
            id="B007",
            kind=BlockKind.TABLE_HEADER,
            text="Метод | Путь",
            raw="| Метод | Путь |",
            section_id="S02",
            section_path=["Эндпоинты"],
            part="Эндпоинты",
            line_start=9,
            line_end=9,
            cells=["Метод", "Путь"],
        ),
        Block(
            id="B008",
            kind=BlockKind.TABLE_ROW,
            text="GET | /api/v1/orders",
            raw="| GET | /api/v1/orders |",
            section_id="S02",
            section_path=["Эндпоинты"],
            part="Эндпоинты",
            line_start=11,
            line_end=11,
            cells=["GET", "/api/v1/orders"],
        ),
        Block(
            id="B009",
            kind=BlockKind.TABLE_ROW,
            text="POST | /api/v1/orders",
            raw="| POST | /api/v1/orders |",
            section_id="S02",
            section_path=["Эндпоинты"],
            part="Эндпоинты",
            line_start=12,
            line_end=12,
            cells=["POST", "/api/v1/orders"],
        ),
        Block(
            id="B010",
            kind=BlockKind.TABLE_ROW,
            text="DELETE | /api/v1/orders",
            raw="| DELETE | /api/v1/orders |",
            section_id="S02",
            section_path=["Эндпоинты"],
            part="Эндпоинты",
            line_start=13,
            line_end=13,
            cells=["DELETE", "/api/v1/orders"],
        ),
    ]
    doc = _make_doc([s1, s2], blocks)
    res = run_prechecks(doc)

    # Exactly one finding for the whole unfilled template table, and none for the URL paths table.
    table_and_cell_findings = [
        f for f in res.findings if f.rule in {"unfilled_template_table", "placeholder", "empty_cell"}
    ]
    assert len(table_and_cell_findings) == 1
    f = table_and_cell_findings[0]
    assert f.rule == "unfilled_template_table"
    assert f.criterion == Criterion.SET_COMPLETE
    assert f.severity == Severity.MAJOR
    assert f.block_ids == ["B003", "B004", "B005"]
    _assert_quotes_are_exact_substrings(doc, res)


def test_empty_sections_placeholders_and_empty_cells():
    parent = Section(
        id="S01",
        title="Спецификация",
        level=2,
        path=["Спецификация"],
        part="Спецификация",
        block_ids=["B001"],
    )
    empty_leaf = Section(
        id="S02",
        title="Безопасность",
        level=3,
        path=["Спецификация", "Безопасность"],
        parent_id="S01",
        part="Спецификация",
        block_ids=["B002"],
    )
    filled_leaf = Section(
        id="S03",
        title="Параметры",
        level=3,
        path=["Спецификация", "Параметры"],
        parent_id="S01",
        part="Спецификация",
        block_ids=["B003", "B004", "B005", "B006", "B007"],
    )
    blocks = [
        Block(
            id="B001",
            kind=BlockKind.HEADING,
            text="Спецификация",
            raw="## Спецификация",
            section_id="S01",
            section_path=["Спецификация"],
            part="Спецификация",
            line_start=1,
            line_end=1,
        ),
        Block(
            id="B002",
            kind=BlockKind.HEADING,
            text="Безопасность",
            raw="### Безопасность",
            section_id="S02",
            section_path=["Спецификация", "Безопасность"],
            part="Спецификация",
            line_start=2,
            line_end=2,
        ),
        Block(
            id="B003",
            kind=BlockKind.HEADING,
            text="Параметры",
            raw="### Параметры",
            section_id="S03",
            section_path=["Спецификация", "Параметры"],
            part="Спецификация",
            line_start=3,
            line_end=3,
        ),
        Block(
            id="B004",
            kind=BlockKind.PARAGRAPH,
            text="Лимиты хранения: TBD.",
            raw="Лимиты хранения: TBD.",
            section_id="S03",
            section_path=["Спецификация", "Параметры"],
            part="Спецификация",
            line_start=4,
            line_end=4,
        ),
        Block(
            id="B005",
            kind=BlockKind.TABLE_HEADER,
            text="Поле | Значение",
            raw="| Поле | Значение |",
            section_id="S03",
            section_path=["Спецификация", "Параметры"],
            part="Спецификация",
            line_start=5,
            line_end=5,
            cells=["Поле", "Значение"],
        ),
        Block(
            id="B006",
            kind=BlockKind.TABLE_ROW,
            text="Документация | —",
            raw="| Документация | — |",
            section_id="S03",
            section_path=["Спецификация", "Параметры"],
            part="Спецификация",
            line_start=7,
            line_end=7,
            cells=["Документация", "—"],
        ),
        Block(
            id="B007",
            kind=BlockKind.TABLE_ROW,
            text="Ответственный | ",
            raw="| Ответственный | |",
            section_id="S03",
            section_path=["Спецификация", "Параметры"],
            part="Спецификация",
            line_start=8,
            line_end=8,
            cells=["Ответственный", ""],
        ),
    ]
    doc = _make_doc([parent, empty_leaf, filled_leaf], blocks)
    res = run_prechecks(doc)

    rules = {(f.rule, tuple(f.block_ids), f.criterion, f.severity) for f in res.findings}
    assert ("empty_section", ("B002",), Criterion.SET_COMPLETE, Severity.MAJOR) in rules
    # Parent section S01 has child sections so it must NOT be reported as empty.
    assert not any(f.rule == "empty_section" and "B001" in f.block_ids for f in res.findings)
    assert ("placeholder", ("B004",), Criterion.SET_COMPLETE, Severity.MAJOR) in rules
    assert ("placeholder", ("B006",), Criterion.SET_COMPLETE, Severity.MAJOR) in rules
    assert ("empty_cell", ("B007",), Criterion.SET_COMPLETE, Severity.MAJOR) in rules
    _assert_quotes_are_exact_substrings(doc, res)


def test_duplicate_ids_numbering_gaps_and_dangling_references():
    s1 = Section(
        id="S01",
        title="1. Функциональные требования",
        level=2,
        path=["1. Функциональные требования"],
        part="1. Функциональные требования",
        block_ids=["B001", "B002", "B003", "B004", "B005", "B006"],
    )
    blocks = [
        Block(
            id="B001",
            kind=BlockKind.HEADING,
            text="1. Функциональные требования",
            raw="## 1. Функциональные требования",
            section_id="S01",
            section_path=["1. Функциональные требования"],
            part="1. Функциональные требования",
            line_start=1,
            line_end=1,
        ),
        Block(
            id="B002",
            kind=BlockKind.LIST_ITEM,
            text="FR-1. Создание заявки.",
            raw="* FR-1. Создание заявки.",
            section_id="S01",
            section_path=["1. Функциональные требования"],
            part="1. Функциональные требования",
            line_start=2,
            line_end=2,
            doc_id="FR-1",
        ),
        Block(
            id="B003",
            kind=BlockKind.LIST_ITEM,
            text="FR-01. Повторный идентификатор заявки.",
            raw="* FR-01. Повторный идентификатор заявки.",
            section_id="S01",
            section_path=["1. Функциональные требования"],
            part="1. Функциональные требования",
            line_start=3,
            line_end=3,
            doc_id="FR-01",
        ),
        Block(
            id="B004",
            kind=BlockKind.LIST_ITEM,
            text="FR-3. Отмена заявки, см. FR-9 и раздел 9.9 и раздел «Архив».",
            raw="* FR-3. Отмена заявки, см. FR-9 и раздел 9.9 и раздел «Архив».",
            section_id="S01",
            section_path=["1. Функциональные требования"],
            part="1. Функциональные требования",
            line_start=4,
            line_end=4,
            doc_id="FR-3",
        ),
        Block(
            id="B005",
            kind=BlockKind.PARAGRAPH,
            text=(
                "Ссылки на существующие FR-1, раздел 1, раздел «Функциональные требования» "
                "и внешний стандарт ISO/IEC/IEEE 29148:2018, п. 5.2.5 корректны."
            ),
            raw="...",
            section_id="S01",
            section_path=["1. Функциональные требования"],
            part="1. Функциональные требования",
            line_start=5,
            line_end=5,
        ),
        Block(
            id="B006",
            kind=BlockKind.PARAGRAPH,
            text="См. требование REQ-77 для деталей.",
            raw="См. требование REQ-77 для деталей.",
            section_id="S01",
            section_path=["1. Функциональные требования"],
            part="1. Функциональные требования",
            line_start=6,
            line_end=6,
        ),
    ]
    doc = _make_doc([s1], blocks)
    res = run_prechecks(doc)

    dup = [f for f in res.findings if f.rule == "duplicate_id"]
    assert len(dup) == 1
    assert dup[0].criterion == Criterion.TRACEABLE
    assert dup[0].severity == Severity.MAJOR
    assert dup[0].block_ids == ["B002", "B003"]

    gaps = [f for f in res.findings if f.rule == "numbering_gap"]
    assert len(gaps) == 1
    assert gaps[0].criterion == Criterion.TRACEABLE
    assert gaps[0].severity == Severity.MINOR
    assert "FR-2" in gaps[0].explanation

    dangling = [f for f in res.findings if f.rule == "dangling_reference"]
    dangling_targets = " ".join(f.explanation for f in dangling)
    assert "FR-9" in dangling_targets
    assert "9.9" in dangling_targets
    assert "Архив" in dangling_targets
    assert "REQ-77" in dangling_targets
    # Valid references and standard clause references must NOT be flagged.
    assert not any("B005" in f.block_ids for f in dangling)
    assert all(f.criterion == Criterion.TRACEABLE and f.severity == Severity.MAJOR for f in dangling)
    _assert_quotes_are_exact_substrings(doc, res)


def test_inventory_quantities_http_codes_identifiers_and_image_warnings():
    s1 = Section(
        id="S01",
        title="API и лимиты",
        level=2,
        path=["API и лимиты"],
        part="API и лимиты",
        block_ids=["B001", "B002", "B003", "B004"],
    )
    blocks = [
        Block(
            id="B001",
            kind=BlockKind.HEADING,
            text="API и лимиты",
            raw="## API и лимиты",
            section_id="S01",
            section_path=["API и лимиты"],
            part="API и лимиты",
            line_start=1,
            line_end=1,
        ),
        Block(
            id="B002",
            kind=BlockKind.PARAGRAPH,
            text=(
                "Комиссия 500 ₸ или 1 500 тенге (100 тиын), доступность 99.9%, "
                "таймаут 30 секунд, обработка за 5 минут, хранение 14 дней."
            ),
            raw="...",
            section_id="S01",
            section_path=["API и лимиты"],
            part="API и лимиты",
            line_start=2,
            line_end=2,
        ),
        Block(
            id="B003",
            kind=BlockKind.PARAGRAPH,
            text=(
                "Метод POST /api/v1/orders передаёт orderId, user_id и заголовок "
                "Content-Type в валюте KZT; коды ответа: 201, 400, 404. "
                "Lowercase verb get should not count as HTTP method."
            ),
            raw=(
                "Метод POST `/api/v1/orders` передаёт `orderId`, `user_id` и заголовок "
                "Content-Type в валюте KZT; коды ответа: 201, 400, 404. "
                "Lowercase verb get should not count as HTTP method."
            ),
            section_id="S01",
            section_path=["API и лимиты"],
            part="API и лимиты",
            line_start=3,
            line_end=3,
        ),
        Block(
            id="B004",
            kind=BlockKind.PARAGRAPH,
            text="Диаграмма потока",
            raw="![Диаграмма потока](flow.png)",
            section_id="S01",
            section_path=["API и лимиты"],
            part="API и лимиты",
            line_start=4,
            line_end=4,
        ),
    ]
    doc = _make_doc([s1], blocks)
    res = run_prechecks(doc)

    quantities = {(i.value, i.unit) for i in res.inventory if i.kind == "quantity"}
    assert ("500", "₸") in quantities
    assert ("1 500", "тенге") in quantities
    assert ("100", "тиын") in quantities
    assert ("99.9", "%") in quantities
    assert ("30", "секунд") in quantities
    assert ("5", "минут") in quantities
    assert ("14", "дней") in quantities

    codes = {(i.value, i.unit) for i in res.inventory if i.kind == "code"}
    assert ("POST", "HTTP method") in codes
    assert ("GET", "HTTP method") not in codes
    assert ("201", "HTTP status") in codes
    assert ("400", "HTTP status") in codes
    assert ("404", "HTTP status") in codes
    # "500 ₸" must NOT be mistaken for HTTP 500.
    assert ("500", "HTTP status") not in codes

    identifiers = {i.value for i in res.inventory if i.kind == "identifier"}
    assert {"/api/v1/orders", "orderId", "user_id", "Content-Type", "KZT"} <= identifiers

    assert any(h.rule == "image_link" and h.block_id == "B004" for h in res.hints)
    warnings = extract_image_warnings(doc)
    assert len(warnings) == 1
    assert "B004" in warnings[0]


# ---------------------------------------------------------------- real sample document test


def test_prechecks_on_sample_requirements_document():
    doc = _load_requirements_doc()
    res = run_prechecks(doc)

    # Every finding must quote an exact substring of its primary block.
    _assert_quotes_are_exact_substrings(doc, res)

    # Unfilled review checklist table must produce one finding for the table.
    template_findings = [f for f in res.findings if f.rule == "unfilled_template_table"]
    assert len(template_findings) == 1
    assert len(template_findings[0].block_ids) >= 3

    # Empty section (approval table with only empty row) and placeholder sections must be caught.
    assert any(f.rule == "empty_section" for f in res.findings)
    placeholder_quotes = " ".join(f.quote for f in res.findings if f.rule == "placeholder")
    assert "заполняются" in placeholder_quotes

    # Numbered step lists and HTTP status headings must NOT trigger false duplicate_id or numbering_gap.
    assert not any(f.rule == "duplicate_id" for f in res.findings)
    assert not any(f.rule == "numbering_gap" for f in res.findings)

    # Inventory must capture currencies/units, HTTP method/codes, and Latin identifiers.
    quantity_units = {i.unit for i in res.inventory if i.kind == "quantity"}
    assert "₸" in quantity_units
    assert any(u.startswith("минут") for u in quantity_units)

    code_values = {(i.value, i.unit) for i in res.inventory if i.kind == "code"}
    assert ("GET", "HTTP method") in code_values
    assert {("200", "HTTP status"), ("400", "HTTP status"), ("404", "HTTP status"), ("500", "HTTP status")} <= code_values

    identifiers = {i.value for i in res.inventory if i.kind == "identifier"}
    assert "KZT" in identifiers
    assert any("_" in ident for ident in identifiers)
    assert any(ident.startswith("/") for ident in identifiers)


def test_parse_structure_blockless_empty_section_and_compound_ids():
    from validator.models import RawDocument
    from validator.structure import parse_structure

    md = (
        "# Заявки на отпуск\n\n"
        "## Функциональные требования\n"
        "- SYS-REQ-1. Сотрудник подаёт заявку.\n"
        "- SYS-REQ-3. Руководитель утверждает заявку.\n\n"
        "## Безопасность\n\n"
        "## Интеграция\n"
        "Отправка уведомлений через сервис рассылки.\n"
    )
    raw = RawDocument(path="leave.md", name="leave.md", title="Заявки на отпуск", format="md", text=md)
    doc = parse_structure(raw)
    res = run_prechecks(doc)

    _assert_quotes_are_exact_substrings(doc, res)
    empty = [f for f in res.findings if f.rule == "empty_section"]
    assert len(empty) == 1
    assert empty[0].quote == "Безопасность"

    gaps = [f for f in res.findings if f.rule == "numbering_gap"]
    assert len(gaps) == 1
    assert "SYS-REQ-2" in gaps[0].explanation


# ---------------------------------------------------------------- follow-up rules tests


def test_missing_req_ids_triggers_only_when_ids_are_partial_and_majority_unnumbered():
    from validator.models import RawDocument
    from validator.structure import parse_structure

    # Partial IDs: 1 numbered vs 3 normative/requirement-section blocks without IDs -> triggers.
    md_partial = (
        "# Бронирование переговорных\n\n"
        "## Функциональные требования\n"
        "- FR-1. Сотрудник создаёт бронь переговорной.\n"
        "- Система должна проверять пересечение интервалов.\n"
        "- Сервис обязан отправлять подтверждение по почте.\n\n"
        "## Нефункциональные требования\n"
        "Отклик каталога не превышает 200 мс. Приоритет: высокий.\n"
    )
    doc_partial = parse_structure(
        RawDocument(path="rooms.md", name="rooms.md", title="Бронирование", format="md", text=md_partial)
    )
    res_partial = run_prechecks(doc_partial)
    _assert_quotes_are_exact_substrings(doc_partial, res_partial)

    missing = [f for f in res_partial.findings if f.rule == "missing_req_ids"]
    assert len(missing) == 1
    assert missing[0].criterion == Criterion.TRACEABLE
    assert missing[0].severity == Severity.MINOR
    assert len(missing[0].block_ids) == 3

    # Majority numbered (2 with ID, 1 without) -> must NOT trigger.
    md_mostly_numbered = (
        "# Библиотека\n\n"
        "## Функциональные требования\n"
        "- FR-1. Читатель ищет книгу по каталогу.\n"
        "- FR-2. Система выдаёт книгу на 14 дней.\n"
        "- Сервис должен логировать возврат.\n"
    )
    doc_mostly = parse_structure(
        RawDocument(path="lib.md", name="lib.md", title="Библиотека", format="md", text=md_mostly_numbered)
    )
    res_mostly = run_prechecks(doc_mostly)
    assert not any(f.rule == "missing_req_ids" for f in res_mostly.findings)

    # Zero IDs anywhere -> must NOT trigger missing_req_ids.
    md_none = (
        "# Заявки\n\n"
        "## Требования\n"
        "Система должна сохранять заявку.\n"
        "Система обязана уведомлять менеджера.\n"
    )
    doc_none = parse_structure(
        RawDocument(path="req.md", name="req.md", title="Заявки", format="md", text=md_none)
    )
    res_none = run_prechecks(doc_none)
    assert not any(f.rule == "missing_req_ids" for f in res_none.findings)


def test_no_ranking_triggers_without_priority_markers_and_clears_with_markers():
    from validator.models import RawDocument
    from validator.structure import parse_structure

    md_unranked = (
        "# Каталог книг\n\n"
        "## Функциональные требования\n"
        "- FR-1. Система отображает список книг.\n"
        "- FR-2. Система ищет книги по названию.\n"
    )
    doc_unranked = parse_structure(
        RawDocument(path="cat.md", name="cat.md", title="Каталог книг", format="md", text=md_unranked)
    )
    res_unranked = run_prechecks(doc_unranked)
    _assert_quotes_are_exact_substrings(doc_unranked, res_unranked)

    ranking = [f for f in res_unranked.findings if f.rule == "no_ranking"]
    assert len(ranking) == 1
    assert ranking[0].criterion == Criterion.RANKED
    assert ranking[0].severity == Severity.MINOR

    for marker_line in (
        "Приоритет: высокий.",
        "Критичность: P1.",
        "The system must authenticate callers.",
        "Уровень: средний.",
    ):
        md_ranked = md_unranked + f"\n{marker_line}\n"
        doc_ranked = parse_structure(
            RawDocument(path="cat.md", name="cat.md", title="Каталог книг", format="md", text=md_ranked)
        )
        res_ranked = run_prechecks(doc_ranked)
        assert not any(f.rule == "no_ranking" for f in res_ranked.findings), (
            f"Unexpected no_ranking when marker present: {marker_line!r}"
        )


def test_critical_severity_for_empty_or_stub_requirement_test_acceptance_sections():
    from validator.models import RawDocument
    from validator.structure import parse_structure

    md = (
        "# Учёт заявок\n\n"
        "## Функциональные требования\n\n"
        "## Критерии приёмки\n"
        "TBD\n\n"
        "## Тестирование требований\n"
        "заполняется позже\n\n"
        "## Acceptance Criteria\n\n"
        "## Примечания\n\n"
        "## История изменений\n"
        "уточняется\n\n"
        "## Обзор\n"
        "Приоритет: P1. Сервис хранит заявки сотрудников.\n"
    )
    doc = parse_structure(
        RawDocument(path="orders.md", name="orders.md", title="Учёт заявок", format="md", text=md)
    )
    res = run_prechecks(doc)
    _assert_quotes_are_exact_substrings(doc, res)

    by_quote = {f.quote: f for f in res.findings if f.rule in {"empty_section", "placeholder"}}
    assert by_quote["Функциональные требования"].severity == Severity.CRITICAL
    assert by_quote["TBD"].severity == Severity.CRITICAL
    assert by_quote["заполняется позже"].severity == Severity.CRITICAL
    assert by_quote["Acceptance Criteria"].severity == Severity.CRITICAL
    # Non-critical empty/stub sections stay major.
    assert by_quote["Примечания"].severity == Severity.MAJOR
    assert by_quote["уточняется"].severity == Severity.MAJOR


def test_inventory_parameter_definitions_capture_field_and_declared_types():
    from validator.models import RawDocument
    from validator.structure import parse_structure

    md = (
        "# API заказов\n\n"
        "## Параметры запроса\n\n"
        "| Parameter | Format | Mandatory | Description |\n"
        "| --- | --- | --- | --- |\n"
        "| `orderId` | string | Да | Идентификатор заказа |\n"
        "| `quantity` | integer | Да | Количество позиций |\n"
        "| `urgent` | boolean | Нет | Флаг срочности |\n\n"
        "## Параметры ответа\n\n"
        "| Поле | Тип | Описание |\n"
        "| --- | --- | --- |\n"
        "| `quantity` | string | Количество строкой |\n"
        "| `total` | число | Итоговая сумма |\n"
        "| `comment` | строка | Комментарий |\n"
    )
    doc = parse_structure(
        RawDocument(path="api.md", name="api.md", title="API заказов", format="md", text=md)
    )
    res = run_prechecks(doc)

    typed_items = {(i.value, i.unit) for i in res.inventory if i.kind == "identifier" and i.unit}
    assert ("orderId", "string") in typed_items
    assert ("quantity", "integer") in typed_items
    assert ("urgent", "boolean") in typed_items
    assert ("quantity", "string") in typed_items
    assert ("total", "число") in typed_items
    assert ("comment", "строка") in typed_items

    # Context text preserves section name so request vs response mismatches can be compared.
    qty_texts = [i.text for i in res.inventory if i.kind == "identifier" and i.value == "quantity"]
    assert any("integer" in t and "Параметры запроса" in t for t in qty_texts)
    assert any("string" in t and "Параметры ответа" in t for t in qty_texts)


def test_sample_requirements_doc_new_rules_integration():
    doc = _load_requirements_doc()
    res = run_prechecks(doc)
    _assert_quotes_are_exact_substrings(doc, res)

    # Stub section under a testing/requirements heading must be critical.
    stub_findings = [f for f in res.findings if f.rule == "placeholder" and "заполняются" in f.quote]
    assert len(stub_findings) == 1
    assert stub_findings[0].severity == Severity.CRITICAL

    # Sample document has no priority/importance ranking markers -> no_ranking minor finding.
    ranking_findings = [f for f in res.findings if f.rule == "no_ranking"]
    assert len(ranking_findings) == 1
    assert ranking_findings[0].criterion == Criterion.RANKED
    assert ranking_findings[0].severity == Severity.MINOR

    # Parameter definition inventory captures differing declared types across request and response tables.
    typed_by_name: dict[str, set[str]] = {}
    for item in res.inventory:
        if item.kind == "identifier" and item.unit:
            typed_by_name.setdefault(item.value, set()).add(item.unit)
    assert any({"integer", "string"} <= types for types in typed_by_name.values())
