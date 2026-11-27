# Агент проверки требований

Агент статической проверки документа требований по ISO/IEC/IEEE 29148:2018 (п. 5.2.5, 5.2.6) и IEEE 830-1998 (п. 4.3). На входе файл требований (Markdown, PDF, DOCX, TXT), на выходе отчёт о качестве требований на русском языке.

> Статус: каркас. Конвейер проверки в работе.

## Модель

По умолчанию Google Gemini API, модель `gemini-3.5-flash` (бесплатный тариф Google AI Studio), запасная `gemini-3.5-flash-lite`. Также поддерживается Anthropic API (`LLM_PROVIDER=anthropic`, модель по умолчанию `claude-sonnet-5-5`, ключ в `ANTHROPIC_API_KEY`). Провайдер и модели задаются в `.env`.

## Как получить ключ

1. Открыть https://aistudio.google.com/apikey и создать ключ.
2. Скопировать `.env.example` в `.env` и вписать ключ в `GEMINI_API_KEY`.

```bash
cp .env.example .env
```

Файл `.env` в git не попадает.

## Установка и запуск

Нужен Python 3.11+.

```bash
pip install -r requirements.txt
python -m validator docs/requirements.md reports/report.md
```

Или через [uv](https://docs.astral.sh/uv/): `uv run python -m validator docs/requirements.md reports/report.md`.

Первый аргумент: путь к документу требований. Второй: куда записать отчёт (папка создаётся сама).

## Тесты

```bash
uv run python -m pytest -q
```

Без uv: `pip install -r requirements-dev.txt`, затем `python -m pytest -q` и `ruff check .`.
