from validator.models import Document, QuoteStatus, RawDocument
from validator.pipeline._fallback_structure import parse_structure
from validator.pipeline.quotes import QuoteIndex, normalize, quotes_overlap

TEXT = """# Бронирование переговорных

## Функции

- **BR-1.** Сотрудник бронирует переговорную на срок от 15 до 240 минут включительно.
- **BR-2.** Бронь отменяется, если никто не отметился в течение 10 минут.

| Условие | Действие | Результат |
| :--- | :--- | :--- |
| Бронь на 9:00 | Нет отметки до 9:20 | Бронь активна |
"""


def _doc() -> Document:
    raw = RawDocument(path="x.md", name="x.md", title="Бронирование переговорных", format="md", text=TEXT)
    return parse_structure(raw)


def test_normalize_unifies_quotes_dashes_yo_and_markup():
    assert normalize("«Ёлка» — **жирный** `код`") == normalize('"елка" - жирный код')


def test_exact_and_normalized_match_returns_document_text():
    idx = QuoteIndex(_doc())
    m = idx.find("бронирует переговорную на срок от 15 до 240 минут")
    assert m is not None and m.status == QuoteStatus.VERIFIED
    assert m.text == "бронирует переговорную на срок от 15 до 240 минут"
    m2 = idx.find("СОТРУДНИК  БРОНИРУЕТ переговорную")
    assert m2 is not None and m2.text.startswith("Сотрудник бронирует")


def test_fuzzy_match_replaces_quote_with_real_fragment():
    idx = QuoteIndex(_doc())
    m = idx.find("Бронь отменяется, если никто не отметился в течении 10 минут")
    assert m is not None and m.status == QuoteStatus.REPAIRED
    assert "в течение 10 минут" in m.text


def test_ellipsis_fragments_in_order():
    idx = QuoteIndex(_doc())
    m = idx.find("Бронь отменяется … в течение 10 минут")
    assert m is not None and m.status == QuoteStatus.VERIFIED


def test_invented_quote_is_rejected():
    idx = QuoteIndex(_doc())
    assert idx.find("Система отправляет письмо руководителю каждые 5 минут") is None


def test_short_quote_needs_unique_match_in_referenced_blocks():
    doc = _doc()
    idx = QuoteIndex(doc)
    assert idx.find("Бронь") is None  # ambiguous across the document
    br2 = next(b for b in doc.blocks if b.doc_id == "BR-2")
    assert idx.find("10 минут", [br2.id]) is not None


def test_quote_spanning_table_cells():
    idx = QuoteIndex(_doc())
    m = idx.find("Нет отметки до 9:20 | Бронь активна")
    assert m is not None


def test_overlap():
    assert quotes_overlap("от 15 до 240 минут", "на срок от 15 до 240 минут включительно")
    assert not quotes_overlap("от 15 до 240 минут", "никто не отметился")
