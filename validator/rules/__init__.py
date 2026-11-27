"""Deterministic precheck rules package."""

from __future__ import annotations

from validator.models import Document, PrecheckResult
from validator.rules.api_lint import run_api_lint
from validator.rules.checks import extract_image_warnings
from validator.rules.checks import run_prechecks as _run_base_prechecks
from validator.rules.gaps import run_gap_checks


def run_prechecks(doc: Document) -> PrecheckResult:
    """Run all deterministic prechecks, including REST/API lint and completeness gap rules."""
    base = _run_base_prechecks(doc)
    extra_findings = [
        *run_api_lint(doc, base.inventory),
        *run_gap_checks(doc, base.inventory),
    ]
    if not extra_findings:
        return base
    return PrecheckResult(
        hints=base.hints,
        findings=[*base.findings, *extra_findings],
        inventory=base.inventory,
    )


__all__ = ["extract_image_warnings", "run_api_lint", "run_gap_checks", "run_prechecks"]
