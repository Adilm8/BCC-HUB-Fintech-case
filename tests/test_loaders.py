from pathlib import Path

import pytest

from validator.loaders import load_document

ROOT = Path(__file__).resolve().parents[1]


def test_loads_task_markdown():
    doc = load_document(ROOT / "docs" / "requirements.md")
    assert "Ф-2" in doc.text
    assert doc.title


def test_rejects_unknown_format(tmp_path):
    f = tmp_path / "a.xyz"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(f)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "nope.md")


def test_missing_key_message(monkeypatch):
    from validator.config import Settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Settings.from_env()
