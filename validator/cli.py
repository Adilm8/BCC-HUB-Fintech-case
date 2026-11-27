"""Точка входа: python -m validator <файл требований> <файл отчёта>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from validator.config import Settings
from validator.llm import LLMError
from validator.loaders import load_document
from validator.pipeline import run


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m validator", description="Проверка качества требований")
    parser.add_argument("input", type=Path, help="путь к документу требований (.md, .txt, .pdf, .docx)")
    parser.add_argument("output", type=Path, help="путь к файлу отчёта (.md)")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        settings = Settings.from_env()
        doc = load_document(args.input)
        report = run(doc, settings)
    except (ValueError, FileNotFoundError, LLMError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Отчёт записан: {args.output}")
    return 0
