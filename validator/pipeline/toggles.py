"""Quality switches. A switch in DEFAULT_ON is on unless its environment variable is "0"; any other switch is
off unless its variable is "1"."""

from __future__ import annotations

import os

# EXP_REMARKS             the per-requirement check may add short remarks: possible gaps noted in doubt,
#                         kept as minor findings flagged `remark`
# EXP_COMPLETENESS_ITEMS  three more items in the per-requirement checklist: terms of calculations, NFR
#                         measurement points, content of outgoing messages
# EXP_DOMAIN              the domain checklist is chosen automatically (validator/domains) unless --domain or
#                         DOMAIN_PROFILE names one
# EXP_NO_REVIEW           the report without the critic (eval/ablate.py)
# EXP_BOUNDARY_RULE       rule boundary_tests_missing (validator/rules/boundaries.py)
# EXP_SINGULAR_RULE, EXP_ABBREV_RULE, EXP_FAILURE_RULE  rules non_singular, undefined_term and
#                         integration_without_failure (validator/rules/extra.py)
# EXP_NFR_RULE, EXP_NOTIFY_RULE, EXP_MIXED_RULE  rules nfr_unmeasurable, notification_content and mixed_levels
#                         (validator/rules/nfr_notify.py): no hits on clean requirements of the evaluation set
# EXP_GLOSSARY            terms that conditions depend on, found by code, go to the check with "is it defined?"
# EXP_STATES              S2 also builds a state x event table for entities with statuses or flags
TOGGLES = ("EXP_REMARKS", "EXP_COMPLETENESS_ITEMS", "EXP_DOMAIN", "EXP_BOUNDARY_RULE", "EXP_GLOSSARY",
           "EXP_STATES", "EXP_SINGULAR_RULE", "EXP_ABBREV_RULE", "EXP_FAILURE_RULE", "EXP_NFR_RULE",
           "EXP_NOTIFY_RULE", "EXP_MIXED_RULE", "EXP_NO_REVIEW")


# measured gain on the evaluation set: on by default
DEFAULT_ON = frozenset({"EXP_DOMAIN", "EXP_GLOSSARY", "EXP_BOUNDARY_RULE", "EXP_STATES", "EXP_COMPLETENESS_ITEMS",
                        "EXP_NFR_RULE", "EXP_NOTIFY_RULE", "EXP_MIXED_RULE"})


def enabled(name: str) -> bool:
    value = os.getenv(name, "").strip()
    if not value:
        return name in DEFAULT_ON
    return value == "1"


def effective() -> dict[str, str]:
    """Every switch with its effective value ("1" or "0"): recorded with a trace."""
    return {n: "1" if enabled(n) else "0" for n in TOGGLES}


def non_default() -> list[tuple[str, bool]]:
    """Switches set away from their default, with their value."""
    return [(n, enabled(n)) for n in TOGGLES if enabled(n) != (n in DEFAULT_ON)]
