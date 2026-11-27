"""Deterministic precheck rules package."""

from __future__ import annotations

from validator.models import Document, PrecheckResult
from validator.rules.api_lint import run_api_lint
from validator.rules.checks import extract_image_warnings
from validator.rules.checks import run_prechecks as _run_base_prechecks


def run_prechecks(doc: Document) -> PrecheckResult:
    """Run all deterministic prechecks, including domain-neutral REST/API lint rules."""
    base = _run_base_prechecks(doc)
    api_findings = run_api_lint(doc, base.inventory)
    if not api_findings:
        return base
    return PrecheckResult(
        hints=base.hints,
        findings=[*base.findings, *api_findings],
        inventory=base.inventory,
    )


__all__ = ["extract_image_warnings", "run_api_lint", "run_prechecks"]
