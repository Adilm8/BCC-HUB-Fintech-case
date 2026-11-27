"""Snapshot-style and edge-case tests for Markdown report rendering and enum labels."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from validator.models import (
    BlockKind,
    Criterion,
    DefectClass,
    FindingSource,
    QuoteStatus,
    Readiness,
    Report,
    ReqKind,
    RunInfo,
    RunStatus,
    SetCheck,
    Severity,
    StageKey,
    StageState,
    StageStatus,
    Summary,
    Verdict,
)
from validator.report.labels import (
    BLOCK_KIND_LABELS,
    CRITERION_LABELS,
    CRITERION_META,
    CRITERION_REFS,
    DEFECT_CLASS_LABELS,
    FINDING_SOURCE_LABELS,
    QUOTE_STATUS_LABELS,
    READINESS_LABELS,
    REQ_KIND_LABELS,
    RUN_STATUS_LABELS,
    SET_CHECK_LABELS,
    SEVERITY_LABELS,
    STAGE_STATE_LABELS,
    criterion_ref,
)
from validator.report.markdown import render_markdown

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_report.json"

EXPECTED_SECTION_HEADINGS = [
    "## Паспорт прогона",
    "## Коротко",
    "## Нарезка требований",
    "## Проверка каждого требования",
    "## Проверка набора требований",
    "## Сводка",
    "## Вопросы к автору ТЗ",
    "## Итог",
    "## Приложение",
]


def _load_sample_report() -> Report:
    return Report.model_validate_json(FIXTURE_PATH.read_text(encoding="utf-8"))


def _extract_h2_headings(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("## ")]


def _assert_all_markdown_tables_balanced(markdown: str) -> None:
    """Verify every Markdown table has the same unescaped pipe count across all its rows."""
    lines = markdown.splitlines()
    table_pipe_count: int | None = None
    for idx, line in enumerate(lines, start=1):
        if line.startswith("|") and line.endswith("|"):
            unescaped_pipes = len(re.findall(r"(?<!\\)\|", line))
            if table_pipe_count is None:
                table_pipe_count = unescaped_pipes
            else:
                assert unescaped_pipes == table_pipe_count, (
                    f"Unbalanced table row at line {idx}: expected {table_pipe_count} pipes, "
                    f"got {unescaped_pipes} in {line!r}"
                )
        else:
            table_pipe_count = None


def test_labels_cover_every_enum_member():
    assert set(CRITERION_META) == set(Criterion)
    assert set(CRITERION_LABELS) == set(Criterion)
    assert set(CRITERION_REFS) == set(Criterion)
    assert set(SEVERITY_LABELS) == set(Severity)
    assert set(DEFECT_CLASS_LABELS) == set(DefectClass)
    assert set(REQ_KIND_LABELS) == set(ReqKind)
    assert set(BLOCK_KIND_LABELS) == set(BlockKind)
    assert set(SET_CHECK_LABELS) == set(SetCheck)
    assert set(QUOTE_STATUS_LABELS) == set(QuoteStatus)
    assert set(FINDING_SOURCE_LABELS) == set(FindingSource)
    assert set(STAGE_STATE_LABELS) == set(StageState)
    assert set(RUN_STATUS_LABELS) == set(RunStatus)
    assert set(READINESS_LABELS) == set(Readiness)
    assert criterion_ref(Criterion.UNAMBIGUOUS) == "ISO/IEC/IEEE 29148:2018, п. 5.2.5 — однозначность"


def test_render_sample_report_snapshot_and_invariants():
    report = _load_sample_report()
    md = render_markdown(report)

    # Pure deterministic rendering: repeated invocation produces identical text.
    assert md == render_markdown(report)
    assert md.endswith("\n")

    # All 9 required sections present in exact fixed order.
    assert _extract_h2_headings(md) == EXPECTED_SECTION_HEADINGS

    # Complete run must not show partial-run warning banner.
    assert "Внимание: отчёт неполный" not in md

    # Every Markdown table has balanced columns even when quotes contain pipes.
    _assert_all_markdown_tables_balanced(md)
    assert r"Расходы 48 500 ₸, порог 50 000 ₸ \| Провести покупку на 1 500 ₸ \| Итог 50 000 ₸ и один push" in md

    # Passport metadata and stage statuses.
    assert report.run.document_title in md
    assert report.run.document_name in md
    assert report.run.document_sha256 in md
    assert report.run.agent_version in md
    assert "212.4 с" in md

    # Slicing table: every requirement and "присвоен агентом" for unnumbered items.
    for req in report.requirements:
        assert req.id in md
    assert "присвоен агентом" in md
    assert "Служебных и пропущенных блоков: 1 (служебная информация — 1)." in md

    # Per-requirement check: full major, one-line minor, clean requirement, unchecked requirement.
    assert "### Требования без собственных нарушений" in md
    assert "собственных нарушений по п. 5.2.5 нет" in md or "Без нарушений (ни собственных" in md
    assert "не проверено" in md
    assert "ISO/IEC/IEEE 29148:2018, п. 5.2.5 — проверяемость" in md
    assert "#### [Существенный] D-001" in md
    minor_lines = [line for line in md.splitlines() if line.startswith("- **[Незначительный] D-002**")]
    assert len(minor_lines) == 1

    # Set-level subsections: S-001 under cross-part with both quotes, S-002 under completeness, coverage matrix.
    set_section = md.split("## Проверка набора требований", 1)[1].split("## Сводка", 1)[0]
    cross_idx = set_section.index("### Согласованность между частями документа")
    cov_idx = set_section.index("### Требования и критерии приёмки")
    comp_idx = set_section.index("### Полнота")
    assert cross_idx < set_section.index("S-001") < cov_idx
    assert "уточнена по тексту документа" in set_section
    assert comp_idx < set_section.index("S-002")
    assert "Нет критерия на границы порога" in set_section
    assert "Не проверено ограничение одного push в месяц" in set_section

    # Summary numbers consistent with report.summary.
    summary_section = md.split("## Сводка", 1)[1].split("## Вопросы к автору ТЗ", 1)[0]
    assert f"| Всего требований | {report.summary.requirements_total} |" in summary_section
    assert f"| Требований с собственными нарушениями | {report.summary.requirements_with_findings} |" in summary_section
    assert f"| Непроверенных требований | {report.summary.requirements_not_checked} |" in summary_section
    assert f"| Всего нарушений | {report.summary.findings_total} |" in summary_section
    assert f"| Вопросов к автору ТЗ | {report.summary.questions_total} |" in summary_section
    assert "| Критичные | 1 |" in summary_section
    assert "| Существенные | 2 |" in summary_section
    assert "| Незначительные | 1 |" in summary_section
    assert "**Противоречивые** (1):" in summary_section
    assert "**S-001**" in summary_section

    # Questions: blocking Q-001 appears before non-blocking Q-002..Q-004.
    questions_section = md.split("## Вопросы к автору ТЗ", 1)[1].split("## Итог", 1)[0]
    assert "### Блокирующие вопросы" in questions_section
    assert "### Прочие вопросы" in questions_section
    assert questions_section.index("Q-001") < questions_section.index("Q-002")
    assert questions_section.index("Q-002") < questions_section.index("Q-003")
    assert questions_section.index("Q-003") < questions_section.index("Q-004")

    # Verdict and appendix counts.
    verdict_section = md.split("## Итог", 1)[1].split("## Приложение", 1)[0]
    assert report.verdict.headline in verdict_section
    assert report.verdict.rationale in verdict_section
    assert "| Разработка | Нет |" in verdict_section
    assert "| Тестирование | Нет |" in verdict_section
    assert "1. **S-001**" in verdict_section
    assert "2. **D-001**" in verdict_section

    appendix_section = md.split("## Приложение", 1)[1]
    assert f"Отброшено находок без подтверждённой цитаты: {report.summary.dropped_unverified_quotes}" in appendix_section
    assert f"Отброшено на этапе критического пересмотра: {report.summary.dropped_by_review}" in appendix_section


def test_partial_report_banner_and_pipe_escaping():
    report = _load_sample_report()
    partial = report.model_copy(
        deep=True,
        update={
            "run": report.run.model_copy(
                deep=True,
                update={
                    "status": RunStatus.PARTIAL,
                    "warnings": ["Достигнут дневной лимит запросов"],
                    "stages": [
                        *report.run.stages[:-2],
                        StageStatus(
                            key=StageKey.REVIEW,
                            title="Критический пересмотр",
                            state=StageState.SKIPPED,
                            detail="исчерпана квота",
                        ),
                        report.run.stages[-1],
                    ],
                },
            ),
        },
    )
    md = render_markdown(partial)
    assert "Внимание: отчёт неполный (частичный)." in md
    assert "Критический пересмотр: пропущен — исчерпана квота" in md
    assert "Достигнут дневной лимит запросов" in md
    # Banner must appear before Section 1.
    assert md.index("Внимание: отчёт неполный") < md.index("## Паспорт прогона")
    _assert_all_markdown_tables_balanced(md)


def test_empty_report_renders_cleanly():
    now = datetime(2026, 11, 27, 10, 0, tzinfo=UTC)
    empty = Report(
        run=RunInfo(
            document_title="Пустой документ",
            document_name="empty.md",
            document_format="md",
            document_sha256="0" * 64,
            started_at=now,
            finished_at=now,
            duration_seconds=0.1,
            provider="none",
            models_used=[],
            agent_version="0.1.0",
            profile="standard",
            llm_calls=0,
            status=RunStatus.RULES_ONLY,
            stages=[
                StageStatus(
                    key=StageKey.LOAD,
                    title="Чтение документа",
                    state=StageState.DONE,
                ),
                StageStatus(
                    key=StageKey.EXTRACT,
                    title="Нарезка требований",
                    state=StageState.FAILED,
                    detail="провайдер недоступен",
                ),
            ],
            standards=["ISO/IEC/IEEE 29148:2018", "IEEE 830-1998"],
        ),
        requirements=[],
        skipped_blocks=[],
        findings=[],
        set_findings=[],
        coverage=[],
        questions=[],
        summary=Summary(
            requirements_total=0,
            requirements_by_kind={},
            requirements_with_findings=0,
            requirements_not_checked=0,
            findings_total=0,
            findings_by_severity={},
            findings_by_criterion={},
            critical_ids=[],
            critical_by_class={},
            questions_total=0,
            dropped_unverified_quotes=0,
            dropped_by_review=0,
        ),
        verdict=Verdict(
            development=Readiness.NO,
            testing=Readiness.NO,
            headline="Требования в документе не обнаружены; начинать работу нельзя.",
        ),
    )

    md = render_markdown(empty)
    assert _extract_h2_headings(md) == EXPECTED_SECTION_HEADINGS
    assert "Внимание: отчёт неполный (только проверки по правилам)." in md
    assert "Требования в документе не выделены." in md
    assert "Уточняющих вопросов к автору ТЗ нет." in md
    assert "Критичных дефектов не выявлено." in md
    _assert_all_markdown_tables_balanced(md)
