"""Конвейер проверки. Пока каркас: паспорт прогона и пробный вызов модели."""

from __future__ import annotations

import subprocess
from datetime import datetime

from validator import __version__
from validator.config import Settings
from validator.llm import make_client
from validator.loaders import Document


def agent_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{__version__} ({sha})"
    except (OSError, subprocess.CalledProcessError):
        return __version__


def run(doc: Document, settings: Settings) -> str:
    client = make_client(settings)
    probe = client.generate(
        "Ответь одной строкой по-русски: сколько примерно требований в документе ниже?\n\n" + doc.text[:20000]
    )
    return "\n".join(
        [
            "# Отчёт о качестве требований",
            "",
            "## Паспорт прогона",
            "",
            "| Поле | Значение |",
            "|---|---|",
            f"| Документ | {doc.title} (`{doc.path.name}`) |",
            f"| Дата | {datetime.now():%d.%m.%Y %H:%M} |",
            f"| Модель | {client.last_model} |",
            f"| Версия агента | {agent_version()} |",
            "",
            "## Пробный ответ модели",
            "",
            probe.strip(),
            "",
        ]
    )
