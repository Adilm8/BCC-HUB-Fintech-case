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
    from validator.pipeline import toggles

    if toggles.enabled("EXP_BOUNDARY_RULE"):
        from validator.rules.boundaries import check_boundary_tests

        extra_findings.extend(check_boundary_tests(doc))
    from validator.rules import extra

    for flag, check in (("EXP_SINGULAR_RULE", extra.check_non_singular), ("EXP_ABBREV_RULE", extra.check_undefined_terms),
                        ("EXP_FAILURE_RULE", extra.check_integration_failures)):
        if toggles.enabled(flag):
            extra_findings.extend(check(doc))
    from validator.rules import nfr_notify

    for flag, check in (("EXP_NFR_RULE", nfr_notify.check_nfr_unmeasurable),
                        ("EXP_NOTIFY_RULE", nfr_notify.check_notification_content),
                        ("EXP_MIXED_RULE", nfr_notify.check_mixed_levels)):
        if toggles.enabled(flag):
            extra_findings.extend(check(doc))
    if not extra_findings:
        return base
    return PrecheckResult(
        hints=base.hints,
        findings=[*base.findings, *extra_findings],
        inventory=base.inventory,
    )


__all__ = ["extract_image_warnings", "run_api_lint", "run_gap_checks", "run_prechecks"]
