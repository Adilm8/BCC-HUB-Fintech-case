import hashlib
from pathlib import Path

import pytest

from validator.loaders import load_document
from validator.models import BlockKind, Document, RawDocument
from validator.structure import parse_structure

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DOCS = Path(__file__).resolve().parent / "fixtures" / "docs"


def _raw(text: str, *, title: str = "Тестовый документ", fmt: str = "md") -> RawDocument:
    return RawDocument(
        path=f"mem/{title}.{fmt}",
        name=f"{title}.{fmt}",
        title=title,
        format=fmt,  # type: ignore[arg-type]
        text=text,
    )


def test_parse_requirements_md_structure():
    raw = load_document(ROOT / "docs" / "requirements.md")
    doc = parse_structure(raw)

    assert isinstance(doc, Document)
    assert doc.sha256 == raw.sha256 == hashlib.sha256(Path(raw.path).read_bytes()).hexdigest()
    assert doc.char_count == len(raw.text)
    assert doc.parts == ["Часть A. Новый функционал", "Часть B. REST API"]
    assert len(doc.sections) == 34

    # Block IDs are sequential B001.. and bijectively referenced by sections.
    assert [b.id for b in doc.blocks] == [f"B{i:03d}" for i in range(1, len(doc.blocks) + 1)]
    sec_block_ids = [bid for sec in doc.sections for bid in sec.block_ids]
    assert sec_block_ids == [b.id for b in doc.blocks]

    # Line numbers match raw slices exactly.
    lines = doc.text.splitlines()
    for blk in doc.blocks:
        assert 1 <= blk.line_start <= blk.line_end <= len(lines)
        assert "\n".join(lines[blk.line_start - 1 : blk.line_end]) == blk.raw

    # Section hierarchy and parts.
    by_title = {s.title: s for s in doc.sections}
    assert by_title["Уведомление о расходах по карте"].parent_id is None
    assert by_title["Часть A. Новый функционал"].parent_id == by_title["Уведомление о расходах по карте"].id
    assert by_title["Функциональные требования"].path == [
        "Уведомление о расходах по карте",
        "Часть A. Новый функционал",
        "Требования к изменениям",
        "Функциональные требования",
    ]
    assert by_title["Функциональные требования"].part == "Часть A. Новый функционал"
    assert by_title["Query-параметры"].part == "Часть B. REST API"

    # Expected block kinds are present.
    kinds = {b.kind for b in doc.blocks}
    assert {
        BlockKind.PARAGRAPH,
        BlockKind.LIST_ITEM,
        BlockKind.TABLE_HEADER,
        BlockKind.TABLE_ROW,
        BlockKind.CODE,
    }.issubset(kinds)

    # Document IDs extracted cleanly without false positives.
    found_ids = [b.doc_id for b in doc.blocks if b.doc_id]
    assert found_ids == ["Ф-1", "Ф-2", "Ф-3", "Ф-4"]

    # Empty data row in sign-off table is preserved for downstream rules.
    signoff_blocks = [
        b for b in doc.blocks if b.section_id == by_title["Блок согласующих"].id
    ]
    assert any(b.kind == BlockKind.TABLE_HEADER and b.cells == ["Подразделение", "ФИО ответственного", "Подпись"] for b in signoff_blocks)
    assert any(b.kind == BlockKind.TABLE_ROW and b.cells == ["", "", ""] for b in signoff_blocks)

    # Checklist table preserves trailing empty notes cell.
    review_rows = [
        b
        for b in doc.blocks
        if b.section_id == by_title["Ревью требований"].id and b.kind == BlockKind.TABLE_ROW
    ]
    assert len(review_rows) == 14
    assert all(b.cells is not None and len(b.cells) == 4 for b in review_rows)
    assert review_rows[0].cells[3] == ""


@pytest.mark.parametrize("fixture_name", ["requirements.docx", "requirements.pdf"])
def test_parse_docx_and_pdf_fixtures(fixture_name):
    raw = load_document(FIXTURES_DOCS / fixture_name)
    doc = parse_structure(raw)

    assert doc.parts == ["Часть A. Новый функционал", "Часть B. REST API"]
    found_ids = [b.doc_id for b in doc.blocks if b.doc_id]
    assert found_ids == ["Ф-1", "Ф-2", "Ф-3", "Ф-4"]
    assert len(doc.sections) >= 30
    assert any(b.kind == BlockKind.TABLE_ROW for b in doc.blocks)


def test_parts_rule_variations():
    # Single H1 + multiple H2 -> H2 headings are parts.
    doc1 = parse_structure(
        _raw(
            "# Документ\n\nВводная.\n\n## Раздел A\n\nТекст A.\n\n## Раздел B\n\n### Подпункт\n\nТекст B."
        )
    )
    assert doc1.parts == ["Раздел A", "Раздел B"]
    assert doc1.blocks[0].part == "Документ"
    assert doc1.blocks[1].part == "Раздел A"
    assert doc1.blocks[2].part == "Раздел B"

    # Multiple H1 -> H1 headings themselves are parts.
    doc2 = parse_structure(
        _raw("# Часть 1\n\n## Детали 1\n\nТекст 1.\n\n# Часть 2\n\n## Детали 2\n\nТекст 2.")
    )
    assert doc2.parts == ["Часть 1", "Часть 2"]
    assert [b.part for b in doc2.blocks] == ["Часть 1", "Часть 2"]

    # Single H1 without subheadings -> single part.
    doc3 = parse_structure(_raw("# Единственный раздел\n\nТекст."))
    assert doc3.parts == ["Единственный раздел"]
    assert doc3.blocks[0].part == "Единственный раздел"

    # Document without headings -> single synthetic section and part.
    doc4 = parse_structure(_raw("Первый абзац.\n\nВторой абзац.", title="Без заголовков"))
    assert len(doc4.sections) == 1
    assert doc4.sections[0].title == "Без заголовков"
    assert doc4.parts == ["Без заголовков"]
    assert len(doc4.blocks) == 2
    assert all(b.section_id == "S01" and b.part == "Без заголовков" for b in doc4.blocks)


def test_doc_id_patterns_and_false_positives():
    text = """# Требования

- **Ф-1.** Функциональное требование.
- FR-12: English functional requirement.
- [REQ_001] Bracketed requirement.
- (НФТ-2) Нефункциональное требование.
- SYS-REQ-1.2 Multi-prefix requirement.

3.2.1 Нумерованное требование в абзаце.
3.2.2 Второе нумерованное без пустой строки.
1.2. Двухуровневый номер с точкой.
1.3 Требование с заглавной буквы.

| ID | Описание |
| --- | --- |
| BR-04 | Бизнес-правило в таблице |
| 4.1.2 | Табличный цифровой код |

27.11.2026 дата выпуска документа.

99.9% доступность сервиса.

1.5 секунды максимальное время ожидания.

UTF-8 кодировка используется для всех файлов.

HTTP-404 ошибка возвращается при отсутствии ресурса.

TLS-1.3 обязателен для всех соединений.

200. Успешный ответ сервера.

1. Первый шаг алгоритма.
2. Второй шаг алгоритма.

```text
FR-99 внутри блока кода не считается doc_id
```"""
    doc = parse_structure(_raw(text))
    found = [(b.kind, b.doc_id, b.text) for b in doc.blocks if b.doc_id]
    assert [(k, d) for k, d, _ in found] == [
        (BlockKind.LIST_ITEM, "Ф-1"),
        (BlockKind.LIST_ITEM, "FR-12"),
        (BlockKind.LIST_ITEM, "REQ_001"),
        (BlockKind.LIST_ITEM, "НФТ-2"),
        (BlockKind.LIST_ITEM, "SYS-REQ-1.2"),
        (BlockKind.PARAGRAPH, "3.2.1"),
        (BlockKind.PARAGRAPH, "3.2.2"),
        (BlockKind.PARAGRAPH, "1.2"),
        (BlockKind.PARAGRAPH, "1.3"),
        (BlockKind.TABLE_ROW, "BR-04"),
        (BlockKind.TABLE_ROW, "4.1.2"),
    ]


def test_tables_escaped_pipes_and_headerless():
    text = r"""# Таблицы

| **Поле** | `Описание` | Примечание |
| :--- | :---: | ---: |
| `a_b` | Значение 1 \| Значение 2 | |

| Без | заголовка |
| Строка | вторая |"""
    doc = parse_structure(_raw(text))
    assert doc.blocks[0].kind == BlockKind.TABLE_HEADER
    assert doc.blocks[0].cells == ["Поле", "Описание", "Примечание"]
    assert doc.blocks[0].text == "Поле | Описание | Примечание"

    assert doc.blocks[1].kind == BlockKind.TABLE_ROW
    assert doc.blocks[1].cells == ["a_b", "Значение 1 | Значение 2", ""]

    assert doc.blocks[2].kind == BlockKind.TABLE_ROW
    assert doc.blocks[2].cells == ["Без", "заголовка"]
    assert doc.blocks[3].kind == BlockKind.TABLE_ROW
    assert doc.blocks[3].cells == ["Строка", "вторая"]


def test_code_blocks_ignore_markdown_and_handle_unclosed():
    text = """# Раздел

```python
# Не заголовок
- Не список
| Не | таблица |
```

~~~json
{"unclosed": true}"""
    doc = parse_structure(_raw(text))
    assert len(doc.sections) == 1
    assert len(doc.blocks) == 2
    assert doc.blocks[0].kind == BlockKind.CODE
    assert "# Не заголовок" in doc.blocks[0].text
    assert doc.blocks[0].line_start == 3
    assert doc.blocks[0].line_end == 7
    assert doc.blocks[1].kind == BlockKind.CODE
    assert doc.blocks[1].text == '{"unclosed": true}'
    assert doc.blocks[1].line_start == 9
    assert doc.blocks[1].line_end == 10


def test_lists_continuations_and_inline_stripping():
    text = """# Списки

- Первый пункт с **жирным**, _курсивом_, `card_id` и [ссылкой](https://example.com)
  продолжение первого пункта с limit_kzt и 2 * 3 * 4.
  - Вложенный подпункт.
1. Нумерованный шаг."""
    doc = parse_structure(_raw(text))
    assert len(doc.blocks) == 3
    assert all(b.kind == BlockKind.LIST_ITEM for b in doc.blocks)
    assert (
        doc.blocks[0].text
        == "Первый пункт с жирным, курсивом, card_id и ссылкой продолжение первого пункта с limit_kzt и 2 * 3 * 4."
    )
    assert doc.blocks[0].line_start == 3
    assert doc.blocks[0].line_end == 4
    assert doc.blocks[1].text == "Вложенный подпункт."
    assert doc.blocks[2].text == "Нумерованный шаг."


@pytest.mark.parametrize(
    "weird_text",
    [
        "",
        "   \n\n\t  ",
        "---\n***\n___",
        "#### Глубокий\n\nТекст\n\n# Корень\n\n###### Очень глубокий\n\n## Уровень 2",
        "###    \n\nТекст после пустого заголовка",
        "Преамбула до заголовка\n\n# Первый заголовок\n\nСодержимое",
        "Заголовок Setext\n================\n\nПодраздел\n---------\n\nТекст",
        "| --- | --- |\n| одинокая строка",
        "Незакрытый **жирный и `код и [ссылка(без скобки",
        "> Цитата строка 1\n> Цитата строка 2\n\nОбычный текст\r\nвторая строка\r\n",
    ],
)
def test_odd_markup_never_raises(weird_text):
    doc = parse_structure(_raw(weird_text))
    assert isinstance(doc, Document)
    assert len(doc.sections) >= 1
    assert len(doc.parts) >= 1
    total_lines = len(weird_text.splitlines())
    for blk in doc.blocks:
        assert 1 <= blk.line_start <= blk.line_end <= total_lines
