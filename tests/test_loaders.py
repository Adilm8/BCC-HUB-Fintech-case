import codecs
from pathlib import Path

import docx
import pytest

from tests.fixtures.make_fixtures import build_minimal_pdf_bytes
from validator.loaders import load_document
from validator.models import RawDocument

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DOCS = Path(__file__).resolve().parent / "fixtures" / "docs"


def test_loads_task_markdown():
    doc = load_document(ROOT / "docs" / "requirements.md")
    assert "Ф-2" in doc.text
    assert doc.title


def test_rejects_unknown_format(tmp_path):
    f = tmp_path / "a.xyz"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="не поддерживается"):
        load_document(f)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_document(tmp_path / "nope.md")


def test_missing_key_message(monkeypatch):
    from validator.config import Settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Settings.from_env()


def test_loads_markdown_returns_raw_document():
    raw = load_document(ROOT / "docs" / "requirements.md")
    assert isinstance(raw, RawDocument)
    assert raw.format == "md"
    assert raw.name == "requirements.md"
    assert raw.title == "Уведомление о расходах по карте"
    assert "\r" not in raw.text
    assert raw.warnings == []


def test_loads_markdown_and_txt_with_bom(tmp_path):
    md = tmp_path / "spec.markdown"
    md.write_bytes(b"\xef\xbb\xbf# Spec Title\r\n\r\nBody line.\r\n")
    raw_md = load_document(md)
    assert raw_md.format == "md"
    assert raw_md.title == "Spec Title"
    assert not raw_md.text.startswith("\ufeff")
    assert "\r" not in raw_md.text

    txt = tmp_path / "plain.txt"
    txt.write_bytes(b"\xef\xbb\xbf\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n")
    raw_txt = load_document(txt)
    assert raw_txt.format == "txt"
    assert raw_txt.text.strip() == "Привет"
    assert raw_txt.title == "Привет"


def test_rejects_empty_bad_encoding_no_ext_and_directory(tmp_path):
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("   \n\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="нет текста"):
        load_document(empty_md)

    bad_encoding = tmp_path / "broken.txt"
    bad_encoding.write_bytes(b"\x98\x98\xff\x00")
    with pytest.raises(ValueError, match="UTF-8"):
        load_document(bad_encoding)

    broken_utf16 = tmp_path / "broken_utf16.txt"
    broken_utf16.write_bytes(b"\xff\xfe\x41")
    with pytest.raises(ValueError, match="UTF-16"):
        load_document(broken_utf16)

    no_ext = tmp_path / "README"
    no_ext.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="без расширения"):
        load_document(no_ext)

    with pytest.raises(ValueError, match="каталог"):
        load_document(tmp_path)


def test_loads_txt_and_md_in_utf16_and_cp1251_with_warnings(tmp_path):
    utf16_le = tmp_path / "spec_le.txt"
    utf16_le.write_bytes("# Требования\r\nСистема хранит заявки.".encode("utf-16"))
    raw_le = load_document(utf16_le)
    assert raw_le.format == "txt"
    assert raw_le.title == "Требования"
    assert "Система хранит заявки." in raw_le.text
    assert any("UTF-16" in w for w in raw_le.warnings)

    utf16_be = tmp_path / "spec_be.md"
    utf16_be.write_bytes(codecs.BOM_UTF16_BE + "# Заголовок\nТекст в UTF-16 BE.".encode("utf-16-be"))
    raw_be = load_document(utf16_be)
    assert raw_be.format == "md"
    assert "Текст в UTF-16 BE." in raw_be.text
    assert any("UTF-16" in w for w in raw_be.warnings)

    cp1251_txt = tmp_path / "ansi.txt"
    cp1251_txt.write_bytes("# Общие требования\r\nПорог задаётся в «рублях» — целое число.".encode("cp1251"))
    raw_cp = load_document(cp1251_txt)
    assert raw_cp.format == "txt"
    assert raw_cp.title == "Общие требования"
    assert "«рублях»" in raw_cp.text
    assert any("cp1251" in w for w in raw_cp.warnings)


def test_docx_headings_lists_and_tables_order(tmp_path):
    d = docx.Document()
    d.add_paragraph("Спецификация сервиса", style="Title")
    d.add_heading("Раздел 1", level=2)
    d.add_paragraph("Вводный абзац.")
    d.add_paragraph("Пункт списка один", style="List Bullet")
    d.add_paragraph("Пункт списка два", style="List Bullet")
    d.add_paragraph("Шаг первый", style="List Number")
    d.add_paragraph("Шаг второй", style="List Number")
    tbl = d.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "Поле"
    tbl.cell(0, 1).text = "Значение"
    tbl.cell(1, 0).text = "A | B"
    tbl.cell(1, 1).text = "Строка 1\nСтрока 2"

    path = tmp_path / "rich.docx"
    d.save(path)

    raw = load_document(path)
    assert raw.format == "docx"
    assert raw.title == "Спецификация сервиса"
    assert "# Спецификация сервиса" in raw.text
    assert "## Раздел 1" in raw.text
    assert "- Пункт списка один" in raw.text
    assert "1. Шаг первый" in raw.text
    assert "2. Шаг второй" in raw.text
    assert "| A / B | Строка 1 Строка 2 |" in raw.text
    assert raw.text.index("Вводный абзац.") < raw.text.index("- Пункт списка один")
    assert raw.text.index("2. Шаг второй") < raw.text.index("| A / B |")


def test_docx_nested_tables_and_merged_cells(tmp_path):
    d = docx.Document()
    d.add_heading("Требования", level=1)
    tbl = d.add_table(rows=4, cols=3)
    tbl.cell(0, 0).text = "Раздел"
    tbl.cell(0, 1).text = "Описание"
    tbl.cell(0, 2).text = "Примечание"

    # Horizontally merged row spanning all 3 columns.
    merged = tbl.cell(1, 0).merge(tbl.cell(1, 2))
    merged.text = "Общая строка"

    # Row with a nested table inside one cell.
    tbl.cell(2, 0).text = "Безопасность"
    parent_cell = tbl.cell(2, 1)
    parent_cell.text = "Основные меры"
    nested = parent_cell.add_table(rows=2, cols=2)
    nested.cell(0, 0).text = "REQ-1.1"
    nested.cell(0, 1).text = "Шифрование данных"
    nested.cell(1, 0).text = "REQ-1.2"
    nested.cell(1, 1).text = "Срок хранения 5 лет"
    tbl.cell(2, 2).text = "Обязательно"

    # Separate identical cells (NOT merged) must stay distinct.
    tbl.cell(3, 0).text = "Проверка"
    tbl.cell(3, 1).text = "Да / Нет"
    tbl.cell(3, 2).text = "Да / Нет"

    path = tmp_path / "nested_merged.docx"
    d.save(path)

    raw = load_document(path)
    assert "| Общая строка |" in raw.text
    assert "| Общая строка | Общая строка |" not in raw.text
    assert "| REQ-1.1 | Шифрование данных |" in raw.text
    assert "| REQ-1.2 | Срок хранения 5 лет |" in raw.text
    assert "| Проверка | Да / Нет | Да / Нет |" in raw.text
    assert raw.text.index("Основные меры") < raw.text.index("REQ-1.1") < raw.text.index("Проверка")
    assert any("вложенные таблицы" in w for w in raw.warnings)


def test_rejects_corrupted_and_empty_docx(tmp_path):
    bad = tmp_path / "broken.docx"
    bad.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="DOCX"):
        load_document(bad)

    empty = tmp_path / "empty.docx"
    docx.Document().save(empty)
    with pytest.raises(ValueError, match="нет текста"):
        load_document(empty)


def test_loads_docx_fixture():
    raw = load_document(FIXTURES_DOCS / "requirements.docx")
    assert raw.format == "docx"
    assert raw.title == "Уведомление о расходах по карте"
    assert "## Часть A. Новый функционал" in raw.text
    assert "## Часть B. REST API" in raw.text
    assert "Ф-1." in raw.text
    assert "Ф-4." in raw.text
    assert "| Расходы 48 500 ₸, порог 50 000 ₸ |" in raw.text
    assert "| cardId | string |" in raw.text


def test_pdf_dehyphenation(tmp_path):
    pages = [
        [
            (16.0, "Техническое задание"),
            (10.0, "Система обеспечивает функцио-"),
            (10.0, "нальные требо-"),
            (10.0, "вания и проверяет поро\u00ad"),
            (10.0, "ги через HTTP-"),
            (10.0, "запросы и require-"),
            (10.0, "ments check."),
            (10.0, "Ф-"),
            (10.0, "1. Сумма не меньше нуля."),
        ]
    ]
    pdf_path = tmp_path / "hyphen.pdf"
    pdf_path.write_bytes(build_minimal_pdf_bytes(pages))

    raw = load_document(pdf_path)
    assert raw.format == "pdf"
    assert "функциональные требования" in raw.text
    assert "пороги" in raw.text
    assert "HTTP-запросы" in raw.text
    assert "requirements check." in raw.text
    assert "Ф-" in raw.text


def test_pdf_strips_repeated_headers_and_footers(tmp_path):
    pages = [
        [
            (9.0, "Служебный колонтитул — версия 1.0"),
            (16.0, "Описание сервиса"),
            (10.0, "Уникальный текст первой страницы."),
            (9.0, "Страница 1 из 3"),
        ],
        [
            (9.0, "Служебный колонтитул — версия 1.0"),
            (10.0, "Уникальный текст второй страницы."),
            (9.0, "Страница 2 из 3"),
        ],
        [
            (9.0, "Служебный колонтитул — версия 1.0"),
            (10.0, "Уникальный текст третьей страницы."),
            (9.0, "Страница 3 из 3"),
        ],
    ]
    pdf_path = tmp_path / "headers.pdf"
    pdf_path.write_bytes(build_minimal_pdf_bytes(pages))

    raw = load_document(pdf_path)
    assert "Служебный колонтитул" not in raw.text
    assert "Страница 1 из 3" not in raw.text
    assert "Страница 2 из 3" not in raw.text
    assert "Страница 3 из 3" not in raw.text
    assert "Уникальный текст первой страницы." in raw.text
    assert "Уникальный текст второй страницы." in raw.text
    assert "Уникальный текст третьей страницы." in raw.text


def test_pdf_heading_heuristics_text_patterns(tmp_path):
    pages = [
        [
            (10.0, "Раздел 1. Общие положения"),
            (10.0, "1.1 Требования к системе"),
            (10.0, "Система обрабатывает заявки пользователей."),
            (10.0, "1. Проверить входные данные."),
            (10.0, "2. Сохранить запись в базе."),
            (10.0, "НЕФУНКЦИОНАЛЬНЫЕ ТРЕБОВАНИЯ"),
            (10.0, "Время отклика не превышает 200 мс."),
        ]
    ]
    pdf_path = tmp_path / "heuristics.pdf"
    pdf_path.write_bytes(build_minimal_pdf_bytes(pages))

    raw = load_document(pdf_path)
    assert "# Раздел 1. Общие положения" in raw.text
    assert "## 1.1 Требования к системе" in raw.text
    assert "## НЕФУНКЦИОНАЛЬНЫЕ ТРЕБОВАНИЯ" in raw.text
    assert "# 1. Проверить входные данные." not in raw.text
    assert "# 2. Сохранить запись в базе." not in raw.text


def test_pdf_without_headings_and_with_images_warnings(tmp_path):
    pages = [
        [
            (10.0, "Обычный абзац без заголовков и номеров разделов."),
            (10.0, "Второй обычный абзац с описанием бизнес-правила."),
        ]
    ]
    plain_pdf = tmp_path / "plain.pdf"
    plain_pdf.write_bytes(build_minimal_pdf_bytes(pages))
    raw_plain = load_document(plain_pdf)
    assert any("без заголовков" in w for w in raw_plain.warnings)

    img_pdf = tmp_path / "with_image.pdf"
    img_pdf.write_bytes(build_minimal_pdf_bytes(pages, include_image_xobject=True))
    raw_img = load_document(img_pdf)
    assert any("изображения" in w for w in raw_img.warnings)


def test_pdf_scanned_or_corrupted_raises_russian_value_error(tmp_path, capsys):
    scanned = tmp_path / "scan.pdf"
    scanned.write_bytes(build_minimal_pdf_bytes([[]], include_image_xobject=True))
    with pytest.raises(ValueError, match="скан"):
        load_document(scanned)

    corrupted = tmp_path / "broken.pdf"
    corrupted.write_bytes(b"%PDF-broken-garbage")
    with pytest.raises(ValueError, match="PDF"):
        load_document(corrupted)
    captured = capsys.readouterr()
    assert "invalid pdf header" not in captured.err.lower()
    assert "eof marker" not in captured.err.lower()


def test_loads_pdf_fixture():
    raw = load_document(FIXTURES_DOCS / "requirements.pdf")
    assert raw.format == "pdf"
    assert raw.title == "Уведомление о расходах по карте"
    assert "Спецификация требований — копия для проверки" not in raw.text
    assert "Страница 1 из" not in raw.text
    assert "# Уведомление о расходах по карте" in raw.text
    assert "## Часть A. Новый функционал" in raw.text
    assert "## Часть B. REST API" in raw.text
    assert "Ф-1." in raw.text
    assert "Ф-4." in raw.text
    assert "cardId" in raw.text


def test_loads_real_tables_pdf_without_literal_pipes():
    from pypdf import PdfReader

    from validator.models import BlockKind
    from validator.structure import parse_structure

    pdf_path = FIXTURES_DOCS / "requirements_tables.pdf"
    # Verify the fixture itself has zero literal pipe characters when read as raw text.
    raw_pypdf_text = "\n".join(pg.extract_text() or "" for pg in PdfReader(str(pdf_path)).pages)
    assert "|" not in raw_pypdf_text

    raw = load_document(pdf_path)
    assert raw.format == "pdf"
    assert raw.title == "Уведомление о расходах по карте"
    assert "Спецификация требований — копия для проверки" not in raw.text
    assert "Страница 1 из" not in raw.text
    assert raw.text.count("Провести покупку на 1 500 ₸") == 1

    doc = parse_structure(raw)
    assert doc.parts == ["Часть A. Новый функционал", "Часть B. REST API"]
    assert [b.doc_id for b in doc.blocks if b.doc_id] == ["Ф-1", "Ф-2", "Ф-3", "Ф-4"]

    # Acceptance-criteria table extracted as pipe table with exact header and cells.
    ac_blocks = [b for b in doc.blocks if b.section_path[-1] == "Критерии приемки"]
    ac_headers = [b for b in ac_blocks if b.kind == BlockKind.TABLE_HEADER]
    ac_rows = [b for b in ac_blocks if b.kind == BlockKind.TABLE_ROW]
    assert len(ac_headers) == 1
    assert ac_headers[0].cells == ["Исходные условия", "Действие", "Ожидаемый результат"]
    assert [r.cells for r in ac_rows] == [
        ["Расходы 48 500 ₸, порог 50 000 ₸", "Провести покупку на 1 500 ₸", "Итог 50 000 ₸ и один push"],
        [
            "Покупка 31 января в 23:58 на 4 000 ₸",
            "Провести 1 февраля в 00:03 по Алматы",
            "Февральский счётчик вырос на 4 000 ₸",
        ],
        ["Расходы 19 900 ₸, порог 20 000 ₸", "Перевести 500 ₸", "Счётчик показывает 19 900 ₸"],
    ]

    # Query-parameters table extracted as 7-column pipe table with exact header and cells.
    qp_blocks = [b for b in doc.blocks if b.section_path[-1] == "Query-параметры"]
    qp_headers = [b for b in qp_blocks if b.kind == BlockKind.TABLE_HEADER]
    qp_rows = [b for b in qp_blocks if b.kind == BlockKind.TABLE_ROW]
    assert len(qp_headers) == 1
    assert qp_headers[0].cells == [
        "Parameter",
        "Format",
        "Size",
        "Mandatory",
        "Logic",
        "Description",
        "DB field",
    ]
    assert [r.cells for r in qp_rows] == [
        ["cardId", "string", "1–64", "Да", "Карта клиента", "Идентификатор карты", "card_id"],
        ["amount", "integer", "32 bit", "Да", "1 000–1 000 000", "Порог в тенге", "limit_kzt"],
        ["enabled", "integer", "1 знак", "Да", "0 или 1", "Признак включения", "enabled"],
    ]

    # Multi-page review checklist table merged intact across page break, and empty signoff row kept.
    review_rows = [
        b
        for b in doc.blocks
        if b.section_path[-1] == "Ревью требований" and b.kind == BlockKind.TABLE_ROW
    ]
    assert len(review_rows) == 14
    signoff_rows = [
        b
        for b in doc.blocks
        if b.section_path[-1] == "Блок согласующих" and b.kind == BlockKind.TABLE_ROW
    ]
    assert any(r.cells == ["", "", ""] for r in signoff_rows)


def test_pdf_falls_back_to_pypdf_when_pdfplumber_unavailable(monkeypatch):
    import validator.loaders as loaders_mod

    def _boom(*_args, **_kwargs):
        raise ImportError("simulated missing pdfplumber")

    monkeypatch.setattr(loaders_mod, "_extract_pdf_pages_pdfplumber", _boom)
    raw = load_document(FIXTURES_DOCS / "requirements.pdf")
    assert raw.format == "pdf"
    assert raw.title == "Уведомление о расходах по карте"
    assert "## Часть A. Новый функционал" in raw.text
    assert "## Часть B. REST API" in raw.text
    assert "Ф-1." in raw.text
