import docx

from validator.loaders import load_document


def test_docx_keeps_headings_and_tables(tmp_path):
    d = docx.Document()
    d.add_heading("Функциональные требования", level=2)
    d.add_paragraph("Ф-1. Клиент задаёт порог.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Условие", "Результат"
    t.cell(1, 0).text, t.cell(1, 1).text = "Порог 50 000", "Один push"
    path = tmp_path / "spec.docx"
    d.save(path)

    text = load_document(path).text
    assert "## Функциональные требования" in text
    assert "| Порог 50 000 | Один push |" in text
    assert text.index("Ф-1") < text.index("Порог 50 000")
