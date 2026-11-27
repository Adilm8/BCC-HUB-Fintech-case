"""Summary, questions to the author and the readiness verdict, computed without the model."""

from __future__ import annotations

from collections import Counter

from validator.models import (
    SEVERITY_RANK,
    CoverageRow,
    DefectClass,
    Finding,
    FindingSource,
    FixItem,
    Question,
    Readiness,
    ReqKind,
    Requirement,
    SetFinding,
    Severity,
    Summary,
    Verdict,
)
from validator.pipeline.quotes import normalize

MAX_QUESTIONS = 20
CLASS_ORDER = [DefectClass.CONTRADICTORY, DefectClass.INCOMPLETE, DefectClass.UNVERIFIABLE,
               DefectClass.AMBIGUOUS, DefectClass.OTHER]
CLASS_GENITIVE = {
    DefectClass.CONTRADICTORY: "противоречивых",
    DefectClass.INCOMPLETE: "неполных",
    DefectClass.UNVERIFIABLE: "непроверяемых",
    DefectClass.AMBIGUOUS: "неоднозначных",
    DefectClass.OTHER: "прочих",
}

AnyFinding = Finding | SetFinding


def _all(findings: list[Finding], set_findings: list[SetFinding]) -> list[AnyFinding]:
    return [*set_findings, *findings]


def _question_priority(f: AnyFinding) -> tuple:
    """Blocking contradictions first, then other critical, then major; template questions last."""
    if f.severity == Severity.CRITICAL:
        tier = 0 if f.defect_class == DefectClass.CONTRADICTORY else 1
    else:
        tier = 2 + SEVERITY_RANK[f.severity]
    return (tier, 0 if f.answer_options else 1, 1 if f.source == FindingSource.RULE else 0,
            0 if isinstance(f, SetFinding) else 1, f.id)


def _same_topic(q: Question, f: AnyFinding, q_class: DefectClass) -> bool:
    """A question covers a finding when the texts match, or they share requirements and the topic."""
    text = f.clarifying_question.strip()
    if normalize(text) == normalize(q.text):
        return True
    from validator.pipeline.postprocess import stem_set

    wa, wb = stem_set(q.text), stem_set(text)
    if not (wa and wb):
        return False
    jaccard = len(wa & wb) / len(wa | wb)
    if set(q.requirement_ids) & set(f.requirement_ids):
        return jaccard >= 0.35
    # the same person asked nearly the same thing about different places
    return q.addressee is not None and q.addressee == f.addressee and jaccard >= 0.5


def build_questions(findings: list[Finding], set_findings: list[SetFinding]) -> list[Question]:
    items = sorted(_all(findings, set_findings), key=_question_priority)
    out: list[Question] = []
    classes: dict[str, DefectClass] = {}
    places: dict[str, set[str]] = {}  # question id -> blocks quoted by its findings
    for f in items:
        text = f.clarifying_question.strip()
        if not text:
            continue
        blocks = {b for e in f.evidence for b in e.block_ids}
        q = next((x for x in out if _same_topic(x, f, classes[x.id]) or _same_place_question(x, f, places[x.id],
                                                                                            blocks)), None)
        if q is not None:
            q.finding_ids.append(f.id)
            q.requirement_ids.extend(r for r in f.requirement_ids if r not in q.requirement_ids)
            if not q.options and f.answer_options:
                q.options = list(f.answer_options)
                q.option_consequences = list(f.option_consequences[: len(q.options)])
            elif q.options and not q.option_consequences and f.option_consequences:
                q.option_consequences = list(f.option_consequences[: len(q.options)])
            continue
        if len(out) >= MAX_QUESTIONS:
            continue
        opts = list(f.answer_options)
        cons = list(f.option_consequences[: len(opts)]) if opts else []
        q = Question(id=f"Q-{len(out) + 1:03d}", text=text, options=opts, option_consequences=cons,
                     finding_ids=[f.id], requirement_ids=list(f.requirement_ids), section=f.section,
                     blocking=f.severity == Severity.CRITICAL, addressee=f.addressee)
        classes[q.id] = f.defect_class
        places[q.id] = set(blocks)
        out.append(q)
    for q in out:
        for f in items:
            if f.id in q.finding_ids:
                places[q.id] |= {b for e in f.evidence for b in e.block_ids}
    return out


def _same_place_question(q: Question, f: AnyFinding, q_blocks: set[str], f_blocks: set[str]) -> bool:
    """Questions about findings that quote the same block and share a topic are one question."""
    if not (q_blocks & f_blocks):
        return False
    from validator.pipeline.postprocess import stem_set

    wa, wb = stem_set(q.text), stem_set(f.clarifying_question)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.4


def build_summary(reqs: list[Requirement], findings: list[Finding], set_findings: list[SetFinding],
                  questions: list[Question], *, dropped_unverified: int, dropped_by_review: int,
                  merged: int = 0, candidates: dict[str, int] | None = None, dropped_unconfirmed: int = 0) -> Summary:
    every = _all(findings, set_findings)
    known = {r.id for r in reqs}
    # counted the same way the report shows them: own findings vs set-level findings only
    checked = {r.id for r in reqs if r.check_status == "checked"}
    with_findings = {r for f in findings for r in f.requirement_ids} & known & checked
    in_sets = {r for f in set_findings for r in f.requirement_ids} & known & checked
    critical = [f for f in every if f.severity == Severity.CRITICAL]
    by_class: dict[str, list[str]] = {}
    for cls in CLASS_ORDER:
        ids = [f.id for f in critical if f.defect_class == cls]
        if ids:
            by_class[cls.value] = ids
    return Summary(
        remarks_total=sum(1 for f in findings if getattr(f, "remark", False)),
        requirements_total=len(reqs),
        requirements_by_kind=dict(Counter(r.kind.value for r in reqs)),
        requirements_with_findings=len(with_findings),
        requirements_in_set_findings_only=len(in_sets - with_findings),
        merged_duplicates=merged,
        requirements_clean=sum(1 for r in reqs if r.id not in with_findings and r.id not in in_sets
                               and r.check_status == "checked"),
        candidates_total=sum((candidates or {}).values()),
        questions_in_findings=sum(1 for f in every if f.clarifying_question.strip()),
        candidates_by_source=dict(candidates or {}),
        dropped_unconfirmed=dropped_unconfirmed,
        requirements_not_checked=sum(1 for r in reqs if r.check_status == "not_checked"),
        findings_total=len(every),
        findings_by_severity={s.value: sum(1 for f in every if f.severity == s) for s in Severity},
        findings_by_criterion=dict(Counter(f.criterion.value for f in every)),
        critical_ids=[f.id for f in critical],
        critical_by_class=by_class,
        questions_total=len(questions),
        dropped_unverified_quotes=dropped_unverified,
        dropped_by_review=dropped_by_review,
    )


def _fix_rank(f: AnyFinding) -> int:
    from validator.models import Criterion, FindingSource, SetCheck

    if f.defect_class == DefectClass.CONTRADICTORY:
        return 0
    template = f.source == FindingSource.RULE and (
        getattr(f, "set_check", None) in (SetCheck.EMPTY_SECTION, SetCheck.MISSING_REQUIREMENT, SetCheck.TRACEABILITY)
        or f.criterion in (Criterion.CONFORMING, Criterion.TRACEABLE, Criterion.RANKED))
    return 2 if template else 1


def _short_title(f: AnyFinding, limit: int = 400) -> str:
    from validator.pipeline.postprocess import first_sentence

    return first_sentence(f.explanation, limit).rstrip(".")


def build_verdict(reqs: list[Requirement], findings: list[Finding], set_findings: list[SetFinding],
                  coverage: list[CoverageRow], *, critic_order: list[str] | None = None,
                  rationale: str = "", run_status: str = "complete") -> Verdict:
    verdict = _build_verdict(reqs, findings, set_findings, coverage, critic_order=critic_order, rationale=rationale)
    unchecked = sum(1 for r in reqs if r.check_status == "not_checked")
    if run_status == "rules_only":
        # nothing was checked by the model: no permission to start, whatever the rules found
        return verdict.model_copy(update={
            "development": Readiness.NO, "testing": Readiness.NO,
            "headline": "Итог не определён: проверка неполная — выполнены только проверки по правилам, "
                        "модель не отработала. До полной проверки начинать разработку нельзя."})
    if run_status == "partial" or unchecked:
        cap = {"development": verdict.development, "testing": verdict.testing}
        for k, v in cap.items():
            if v == Readiness.YES:
                cap[k] = Readiness.CONDITIONAL
        if cap != {"development": verdict.development, "testing": verdict.testing}:
            reason = (f"не проверено требований: {unchecked}" if unchecked else "часть этапов модели не выполнена")
            return verdict.model_copy(update={
                **cap, "headline": f"Начинать можно только с условием: отчёт неполный ({reason})."})
    return verdict


def _build_verdict(reqs: list[Requirement], findings: list[Finding], set_findings: list[SetFinding],
                   coverage: list[CoverageRow], *, critic_order: list[str] | None = None,
                   rationale: str = "") -> Verdict:
    every = _all(findings, set_findings)
    critical = [f for f in every if f.severity == Severity.CRITICAL]
    major = [f for f in every if f.severity == Severity.MAJOR]
    has_acceptance = any(r.kind == ReqKind.ACCEPTANCE_CRITERION for r in reqs)
    no_cov = sum(1 for c in coverage if c.status == "none")

    if critical:
        development = Readiness.NO
    elif major:
        development = Readiness.CONDITIONAL
    else:
        development = Readiness.YES
    if critical or not has_acceptance or (coverage and no_cov * 2 > len(coverage)):
        testing = Readiness.NO
    elif major:
        testing = Readiness.CONDITIONAL
    else:
        testing = Readiness.YES

    if not reqs:
        headline = "Начинать разработку нельзя: в документе не найдено ни одного требования."
        development = testing = Readiness.NO
    elif critical:
        classes = Counter(f.defect_class for f in critical)
        detail = ", ".join(f"{CLASS_GENITIVE[c]} — {classes[c]}" for c in CLASS_ORDER if classes.get(c))
        headline = f"Начинать разработку нельзя: критичных дефектов — {len(critical)} ({detail})."
    elif major:
        headline = (f"Разработку можно начинать с условием: критичных дефектов нет, существенных — {len(major)}; "
                    "их нужно снять ответами автора по ходу работы.")
    else:
        headline = "Разработку можно начинать: критичных и существенных дефектов не выявлено."

    by_id = {f.id: f for f in every}
    order: list[AnyFinding] = []
    for fid in critic_order or []:
        f = by_id.get(fid)
        if f is not None and f not in order and f.severity != Severity.MINOR:
            order.append(f)
    # every critical defect comes before any major one; the critic's order is kept inside each level
    default = sorted(critical, key=lambda f: (CLASS_ORDER.index(f.defect_class), -len(f.requirement_ids), f.id))
    order += [f for f in default if f not in order]
    if len(order) < 7:
        order += [f for f in sorted(major, key=lambda f: (CLASS_ORDER.index(f.defect_class), f.id))
                  if f not in order]
    # stable: keeps the critic's order within a level; behaviour conflicts before formatting and templates
    order.sort(key=lambda f: (SEVERITY_RANK[f.severity], _fix_rank(f)))
    limit = max(7, len(critical))
    fix_first = [FixItem(finding_id=f.id, title=_short_title(f)) for f in order[:limit]]
    return Verdict(development=development, testing=testing, headline=headline, rationale=rationale,
                   fix_first=fix_first)
