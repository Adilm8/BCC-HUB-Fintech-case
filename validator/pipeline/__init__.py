"""Конвейер проверки требований. Публичный вход ядра: run_pipeline."""

from __future__ import annotations

from validator.pipeline.orchestrator import agent_version, run_pipeline

__all__ = ["agent_version", "run_pipeline"]
