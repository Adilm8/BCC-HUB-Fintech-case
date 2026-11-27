from pathlib import Path

from validator.models import CoverageRow, Report
from validator.pipeline.verdict import build_questions, build_summary, build_verdict

FIXTURE = Path(__file__).parent / "fixtures" / "sample_report.json"


def load() -> Report:
    return Report.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_valid_report():
    rep = load()
    assert rep.schema_version == "1.0"
    assert rep.set_findings[0].quote == rep.set_findings[0].evidence[0].quote


def test_verdict_no_when_critical():
    rep = load()
    v = build_verdict(rep.requirements, rep.findings, rep.set_findings, rep.coverage)
    assert v.development == "no" and v.testing == "no"
    assert v.fix_first[0].finding_id == "S-001"


def test_verdict_conditional_without_critical():
    rep = load()
    sets = [f for f in rep.set_findings if f.severity != "critical"]
    cov = [CoverageRow(requirement_id="R-001", acceptance_ids=["R-004"], status="covered")]
    v = build_verdict(rep.requirements, rep.findings, sets, cov)
    assert v.development == "conditional" and v.testing == "conditional"


def test_verdict_testing_no_without_acceptance_criteria():
    rep = load()
    reqs = [r for r in rep.requirements if r.kind != "acceptance_criterion"]
    v = build_verdict(reqs, [], [], [])
    assert v.development == "yes" and v.testing == "no"


def test_critic_order_is_used_when_valid():
    rep = load()
    v = build_verdict(rep.requirements, rep.findings, rep.set_findings, rep.coverage, critic_order=["D-001", "S-001"])
    assert [x.finding_id for x in v.fix_first][:2] == ["S-001", "D-001"]  # critical first


def test_questions_blocking_first_and_summary_counts():
    rep = load()
    qs = build_questions(rep.findings, rep.set_findings)
    assert qs[0].blocking and qs[0].finding_ids == ["S-001"]
    assert all(q.id == f"Q-{i:03d}" for i, q in enumerate(qs, 1))
    s = build_summary(rep.requirements, rep.findings, rep.set_findings, qs, dropped_unverified=2, dropped_by_review=0)
    assert s.findings_total == 4 and s.critical_ids == ["S-001"]
    assert s.critical_by_class == {"contradictory": ["S-001"]}
    assert s.requirements_not_checked == 0  # the unchecked one is touched by a set finding
    total = s.requirements_with_findings + s.requirements_in_set_findings_only + s.requirements_clean
    assert total + s.requirements_not_checked == s.requirements_total


def test_batches_cut_on_section_boundaries():
    from validator.config import PROFILES
    from validator.pipeline.stages.check import make_batches

    rep = load()
    base = rep.requirements[0]
    reqs = [base.model_copy(update={"id": f"R-{i:03d}", "section": f"S{i // 10}"}) for i in range(1, 50)]
    batches = make_batches(reqs, PROFILES["deep"])
    assert sum(len(b) for b in batches) == 49 and len(batches) <= 5
    for b in batches[:-1]:
        assert b[-1].section != reqs[reqs.index(b[-1]) + 1].section
    assert make_batches(reqs, PROFILES["economy"]) == [reqs]


def test_compact_evidence_drops_contained_quotes_and_caps():
    from validator.models import Evidence
    from validator.pipeline.postprocess import compact_evidence

    def ev(q, b):
        return Evidence(quote=q, section="s", block_ids=[b], quote_status="verified")

    out = compact_evidence([ev("Тело не используется.", "B1"), ev("Тело не используется", "B1"),
                            ev("a b c d e", "B2"), ev("f g h i", "B3"), ev("j k l m", "B4"), ev("n o p", "B5")])
    assert len(out) == 4
    assert sum("Тело" in e.quote for e in out) == 1
