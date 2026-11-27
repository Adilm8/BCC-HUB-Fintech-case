"""Pure deterministic Markdown report renderer implementing the fixed 9-section template."""

from __future__ import annotations

from collections import Counter, defaultdict

from validator.config import SEQUENTIAL_PROVIDERS
from validator.models import (
    SEVERITY_RANK,
    BaseFinding,
    Criterion,
    Evidence,
    Finding,
    FindingDeltaStatus,
    Question,
    QuoteStatus,
    Report,
    Requirement,
    RunStatus,
    SetCheck,
    SetFinding,
    Severity,
    StageState,
)
from validator.report.labels import (
    COVERAGE_STATUS_LABELS,
    CRITERION_META,
    DEFECT_CLASS_ORDER,
    FINDING_SOURCE_LABELS,
    QUOTE_STATUS_LABELS,
    READINESS_LABELS,
    REQ_KIND_ORDER,
    RUN_STATUS_LABELS,
    SEVERITY_LABELS,
    SEVERITY_PLURAL_LABELS,
    SKIPPED_REASON_LABELS,
    STAGE_STATE_LABELS,
    addressee_label,
    criterion_label,
    criterion_ref,
    defect_class_label,
    finding_delta_status_label,
    req_kind_label,
    set_check_label,
    severity_label,
)


def _first_sentence(text: str) -> str:
    from validator.pipeline.postprocess import first_sentence

    return first_sentence(text, 600)


def _short(text: str, limit: int = 120) -> str:
    """Cut at a word boundary and mark the cut with an ellipsis."""
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:—-") + "…"


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split()).strip(" .…")


def _set_refs(report: Report) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = defaultdict(list)
    for sf in report.set_findings:
        for rid in sf.requirement_ids:
            if sf.id not in refs[rid]:
                refs[rid].append(sf.id)
    return refs


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Russian numeral agreement: 1 вопрос, 3 вопроса, 5 вопросов, 21 вопрос."""
    tail = n % 100
    if 11 <= tail <= 14:
        word = many
    elif n % 10 == 1:
        word = one
    elif 2 <= n % 10 <= 4:
        word = few
    else:
        word = many
    return f"{n} {word}"


def _count(n: int) -> str:
    """'1 требования', '3 требований' after the preposition «у»; plain genitive forms."""
    word = "требования" if n % 10 == 1 and n % 100 != 11 else "требований"
    return f"{n} {word}"


def _escape_cell(value: str) -> str:
    """Escape pipes and collapse line breaks so text stays inside a single Markdown table cell."""
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    cleaned = cleaned.replace("|", r"\|")
    return cleaned.strip()


def _blockquote(text: str, *, indent: str = "") -> list[str]:
    """Format possibly multi-line text as a Markdown blockquote."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return [f"{indent}> {line}" if line else f"{indent}>" for line in lines]


def _inline_text(text: str) -> str:
    """Collapse embedded newlines into spaces for single-line list items."""
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def _format_doc_id(req: Requirement) -> str:
    if req.id_assigned or not req.doc_id:
        return "присвоен агентом"
    return req.doc_id


def _render_banner(report: Report) -> list[str]:
    if report.run.status == RunStatus.COMPLETE:
        return []

    status_ru = RUN_STATUS_LABELS.get(report.run.status, str(report.run.status))
    lines = [f"> **Внимание: отчёт неполный ({status_ru.lower()}).**"]

    problem_stages = [
        stage
        for stage in report.run.stages
        if stage.state in (StageState.FAILED, StageState.SKIPPED)
    ]
    if problem_stages:
        lines.append(">")
        lines.append("> Проблемные или пропущенные этапы:")
        for stage in problem_stages:
            state_ru = STAGE_STATE_LABELS.get(stage.state, str(stage.state))
            detail = f" — {stage.detail}" if stage.detail else ""
            lines.append(f"> - {stage.title}: {state_ru}{detail}")

    if report.run.warnings:
        lines.append(">")
        lines.append("> Причины и предупреждения:")
        for warning in report.run.warnings:
            lines.append(f"> - {warning}")

    if report.run.status == RunStatus.RULES_ONLY:
        lines.append(">")
        lines.append(
            "> Выполнены только детерминированные проверки по правилам; рекомендуется "
            "повторить полный запуск после восстановления доступа к провайдеру."
        )

    return lines


def _render_passport(report: Report) -> list[str]:
    run = report.run
    started = run.started_at.isoformat(sep=" ", timespec="seconds")
    finished = run.finished_at.isoformat(sep=" ", timespec="seconds")
    models_str = ", ".join(run.models_used) if run.models_used else "—"
    standards_str = ", ".join(run.standards) if run.standards else "—"
    status_str = RUN_STATUS_LABELS.get(run.status, str(run.status))

    rows = [
        ("Документ", _escape_cell(run.document_title)),
        ("Файл", f"`{_escape_cell(run.document_name)}`"),
        ("Формат", _escape_cell(run.document_format)),
        ("SHA-256", f"`{_escape_cell(run.document_sha256)}`"),
        ("Дата прогона", f"{started} — {finished}"),
        ("Длительность", f"{run.duration_seconds:.1f} с"),
        ("Провайдер", _escape_cell(run.provider)),
        ("Модели", _escape_cell(models_str)),
        ("Версия агента", _escape_cell(run.agent_version)),
        ("Профиль", _escape_cell(run.profile)),
        ("Число вызовов", str(run.llm_calls)),
        ("Статус прогона", _escape_cell(status_str)),
        ("Стандарты", _escape_cell(standards_str)),
    ]

    lines = [
        "## Паспорт прогона",
        "",
        "| Поле | Значение |",
        "| --- | --- |",
    ]
    for field, value in rows:
        lines.append(f"| {field} | {value} |")

    if run.warnings:
        lines.append("")
        lines.append("**Предупреждения:**")
        for warning in run.warnings:
            lines.append(f"- {warning}")

    if run.stages:
        lines.extend(
            [
                "",
                "### Статус этапов",
                "",
                "| Этап | Статус | Модель | Вызовы | Токены (вход / выход) | Время, с | Детали |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for stage in run.stages:
            state_ru = STAGE_STATE_LABELS.get(stage.state, str(stage.state))
            model_cell = _escape_cell(stage.model) if stage.model else "—"
            tokens_cell = (
                f"{stage.input_tokens} / {stage.output_tokens}"
                if (stage.input_tokens or stage.output_tokens)
                else "—"
            )
            detail_cell = _escape_cell(stage.detail) if stage.detail else "—"
            lines.append(
                f"| {_escape_cell(stage.title)} | {_escape_cell(state_ru)} | {model_cell} | "
                f"{stage.llm_calls} | {tokens_cell} | {stage.seconds:.2f} | {detail_cell} |"
            )
        by_key = {s.key: s for s in run.stages}
        check, set_check = by_key.get("check"), by_key.get("set_check")
        sequential = run.provider in SEQUENTIAL_PROVIDERS or run.profile == "economy"  # one call at a time
        if check and set_check and check.llm_calls and set_check.llm_calls and not sequential:
            lines.extend([
                "",
                (
                    "Этапы «Проверка каждого требования» и «Проверка набора требований» выполняются параллельно: "
                    "их время перекрывается, поэтому сумма времени этапов больше общей длительности прогона."
                ),
            ])

    return lines


def _render_short_summary(report: Report) -> list[str]:
    summary = report.summary
    verdict = report.verdict
    crit = summary.findings_by_severity.get(Severity.CRITICAL.value, 0)
    major = summary.findings_by_severity.get(Severity.MAJOR.value, 0)
    minor = summary.findings_by_severity.get(Severity.MINOR.value, 0)
    blocking_q = sum(1 for q in report.questions if q.blocking)

    dev_ready = READINESS_LABELS.get(verdict.development, str(verdict.development))
    test_ready = READINESS_LABELS.get(verdict.testing, str(verdict.testing))

    refinement_link = "[Уточнение: было → стало](#уточнение-было--стало) · " if report.refinement is not None else ""
    return [
        "## Коротко",
        "",
        f"**{verdict.headline}**",
        "",
        (
            f"- **Требований выделено:** {summary.requirements_total} "
            f"(с собственными нарушениями — {summary.requirements_with_findings}, "
            f"затронуты находками проверки набора — {summary.requirements_in_set_findings_only}, "
            f"без нарушений — {summary.requirements_clean}, "
            f"не проверено — {summary.requirements_not_checked})."
        ),
        (
            f"- **Нарушений всего:** {summary.findings_total} "
            f"(критичных — {crit}, существенных — {major}, незначительных — {minor})."
        ),
        f"- **Вопросов к автору ТЗ:** {summary.questions_total} (блокирующих — {blocking_q}).",
        f"- **Готовность:** разработка — {dev_ready}; тестирование — {test_ready}.",
        "",
        (
            "**Переход к разделам:** "
            f"{refinement_link}"
            "[Нарезка требований](#нарезка-требований) · "
            "[Проверка каждого требования](#проверка-каждого-требования) · "
            "[Проверка набора требований](#проверка-набора-требований) · "
            "[Сводка](#сводка) · "
            "[Вопросы к автору ТЗ](#вопросы-к-автору-тз) · "
            "[Итог](#итог) · "
            "[Приложение](#приложение)"
        ),
    ]


def _delta_str(before: int, after: int) -> str:
    diff = after - before
    if diff > 0:
        return f"+{diff}"
    return str(diff)


def _render_refinement(report: Report) -> list[str]:
    ref = report.refinement
    if ref is None:
        return []

    closed_n = sum(1 for d in ref.finding_deltas if d.status == FindingDeltaStatus.CLOSED)
    changed_n = sum(1 for d in ref.finding_deltas if d.status == FindingDeltaStatus.CHANGED)
    remained_n = sum(1 for d in ref.finding_deltas if d.status == FindingDeltaStatus.REMAINED)
    prev_dev = READINESS_LABELS.get(ref.previous_verdict_development, str(ref.previous_verdict_development))
    curr_dev = READINESS_LABELS.get(ref.current_verdict_development, str(ref.current_verdict_development))
    prev_test = READINESS_LABELS.get(ref.previous_verdict_testing, str(ref.previous_verdict_testing))
    curr_test = READINESS_LABELS.get(ref.current_verdict_testing, str(ref.current_verdict_testing))

    source_note = f" (предыдущий отчёт: `{_escape_cell(ref.previous_report_name)}`)" if ref.previous_report_name else ""
    lines = [
        "## Уточнение: было → стало",
        "",
        (
            f"Учтено ответов автора: **{ref.answers_total}**{source_note}. "
            f"По предыдущим находкам: **закрыто ответом** — {closed_n}, "
            f"**изменилось** — {changed_n}, **осталось** — {remained_n}."
        ),
        "",
        "| Показатель | Было | Стало | Изменение |",
        "| --- | ---: | ---: | --- |",
        (
            f"| Открытых нарушений всего | {ref.previous_findings_total} | {ref.current_findings_total} | "
            f"{_delta_str(ref.previous_findings_total, ref.current_findings_total)} |"
        ),
    ]
    for sev in (Severity.CRITICAL, Severity.MAJOR, Severity.MINOR):
        before_val = ref.previous_findings_by_severity.get(sev.value, 0)
        after_val = ref.current_findings_by_severity.get(sev.value, 0)
        lines.append(
            f"| {SEVERITY_PLURAL_LABELS[sev]} | {before_val} | {after_val} | {_delta_str(before_val, after_val)} |"
        )
    dev_change = "без изменений" if prev_dev == curr_dev else f"{prev_dev} → {curr_dev}"
    test_change = "без изменений" if prev_test == curr_test else f"{prev_test} → {curr_test}"
    lines.extend(
        [
            f"| Готовность разработки | {_escape_cell(prev_dev)} | {_escape_cell(curr_dev)} | {_escape_cell(dev_change)} |",
            f"| Готовность тестирования | {_escape_cell(prev_test)} | {_escape_cell(curr_test)} | {_escape_cell(test_change)} |",
        ]
    )

    if ref.previous_verdict_headline or ref.current_verdict_headline:
        lines.append("")
        if ref.previous_verdict_headline:
            lines.append(f"- **Вердикт было:** {ref.previous_verdict_headline}")
        if ref.current_verdict_headline:
            lines.append(f"- **Вердикт стало:** {ref.current_verdict_headline}")

    lines.extend(["", "### Ответы автора и эффект", ""])
    if not ref.question_effects:
        lines.append("Заполненных ответов автора не передано.")
    else:
        lines.extend(
            [
                "| Вопрос | Формулировка вопроса | Ответ автора | Эффект |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in ref.question_effects:
            q_text = row.question_text or "—"
            lines.append(
                f"| {_escape_cell(row.question_id)} | {_escape_cell(q_text)} | "
                f"{_escape_cell(row.answer)} | {_escape_cell(row.effect)} |"
            )

    lines.extend(["", "### Статус предыдущих находок", ""])
    if not ref.finding_deltas:
        lines.append("В предыдущем отчёте открытых находок не было.")
    else:
        lines.extend(
            [
                "| Предыдущий ID | Было (критичность) | Статус | Стало (ID и критичность) | Комментарий |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for d in ref.finding_deltas:
            status_ru = finding_delta_status_label(d.status)
            prev_sev = severity_label(d.previous_severity)
            if d.current_id and d.current_severity is not None:
                curr_cell = f"{d.current_id} ({severity_label(d.current_severity)})"
            elif d.current_id:
                curr_cell = d.current_id
            else:
                curr_cell = "—"
            note_cell = d.note or "—"
            lines.append(
                f"| {_escape_cell(d.previous_id)} | {_escape_cell(prev_sev)} | "
                f"{_escape_cell(status_ru)} | {_escape_cell(curr_cell)} | {_escape_cell(note_cell)} |"
            )

    return lines


def _render_slicing(report: Report) -> list[str]:
    lines = ["## Нарезка требований", ""]

    if not report.requirements:
        lines.append("Требования в документе не выделены.")
    else:
        lines.extend(
            [
                "| ID в отчёте | ID в документе | Раздел | Тип | Краткая формулировка | Цитата |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for req in report.requirements:
            title = _short(req.title)
            nt, nq = _norm(req.title), _norm(req.quote)
            same = nt == nq
            quote_cell = "совпадает с формулировкой" if same and len(req.quote) <= 120 else req.quote
            lines.append(
                f"| {_escape_cell(req.id)} "
                f"| {_escape_cell(_format_doc_id(req))} "
                f"| {_escape_cell(req.section)} "
                f"| {_escape_cell(req_kind_label(req.kind))} "
                f"| {_escape_cell(title)} "
                f"| {_escape_cell(quote_cell)} |"
            )

    lines.append("")
    s = report.summary
    if s.blocks_total:
        lines.append(
            f"Блоки документа: всего {s.blocks_total} = в требованиях {s.blocks_in_requirements} "
            f"+ служебные {s.blocks_skipped} + заголовки разделов и таблиц {s.blocks_structural} "
            f"+ прочие {s.blocks_other} (короткие фрагменты и примеры, не образующие отдельных требований). "
            "Одно требование может состоять из нескольких блоков."
        )
        lines.append("")
    skipped_total = len(report.skipped_blocks)
    if skipped_total == 0:
        lines.append("Служебных и пропущенных блоков: 0.")
    else:
        reason_counts = Counter(sb.reason for sb in report.skipped_blocks)
        breakdown = ", ".join(
            f"{SKIPPED_REASON_LABELS.get(reason, reason)} — {reason_counts[reason]}"
            for reason in ("meta", "template", "empty", "other")
            if reason_counts.get(reason)
        )
        # Preserve any unexpected reason keys deterministically.
        extra_reasons = sorted(r for r in reason_counts if r not in ("meta", "template", "empty", "other"))
        if extra_reasons:
            extra_str = ", ".join(f"{r} — {reason_counts[r]}" for r in extra_reasons)
            breakdown = f"{breakdown}, {extra_str}" if breakdown else extra_str
        lines.append(f"Служебных и пропущенных блоков: {skipped_total} ({breakdown}).")

    return lines


def _render_evidence_lines(evidence_list: list[Evidence]) -> list[str]:
    lines: list[str] = []
    if len(evidence_list) == 1:
        ev = evidence_list[0]
        q_status = QUOTE_STATUS_LABELS.get(ev.quote_status, str(ev.quote_status))
        lines.append(f"- **Раздел:** {ev.section}")
        lines.append(f"- **Цитата** (*{q_status}*):")
        lines.extend(_blockquote(ev.quote, indent="  "))
        return lines

    lines.append("- **Подтверждающие цитаты:**")
    for idx, ev in enumerate(evidence_list, start=1):
        q_status = QUOTE_STATUS_LABELS.get(ev.quote_status, str(ev.quote_status))
        req_part = f", требование `{ev.requirement_id}`" if ev.requirement_id else ""
        lines.append(f"  {idx}. Раздел: {ev.section}{req_part} (*{q_status}*)")
        lines.extend(_blockquote(ev.quote, indent="     "))
    return lines


def _render_full_finding(finding: BaseFinding, heading_title: str) -> list[str]:
    sev_ru = SEVERITY_LABELS.get(finding.severity, str(finding.severity))
    src_ru = FINDING_SOURCE_LABELS.get(finding.source, str(finding.source))
    lines = [
        f"#### [{sev_ru}] {finding.id} — {heading_title}",
        "",
        f"- **Критерий:** {criterion_ref(finding.criterion)}",
        f"- **Источник:** {src_ru}",
    ]
    if isinstance(finding, SetFinding) and finding.requirement_ids:
        lines.append(f"- **Затронутые требования:** {', '.join(finding.requirement_ids)}")
    lines.extend(_render_evidence_lines(finding.evidence))
    lines.append(f"- **Объяснение:** {finding.explanation}")
    lines.append(f"- **Рекомендация:** {finding.recommendation}")
    if finding.related_notes:
        lines.append("- **Связанные замечания** (объединены с этой находкой):")
        for note in finding.related_notes:
            lines.append(f"  - {note}")
    lines.append(f"- **Уточняющий вопрос** ({addressee_label(finding.addressee)}): {finding.clarifying_question}")
    if finding.answer_options:
        lines.append("- **Варианты ответа:**")
        for idx, opt in enumerate(finding.answer_options):
            lines.append(f"  - {opt}")
            cons = (
                finding.option_consequences[idx].strip()
                if idx < len(finding.option_consequences)
                else ""
            )
            if cons:
                lines.append(f"    - *Что изменится:* {cons}")
    if finding.review_note:
        lines.append(f"- **Примечание пересмотра:** {finding.review_note}")
    return lines


def _render_minor_finding_line(finding: Finding) -> str:
    sev_ru = SEVERITY_LABELS.get(finding.severity, str(finding.severity))
    quote = _inline_text(finding.quote)
    explanation = _inline_text(finding.explanation)
    recommendation = _inline_text(finding.recommendation)
    question = _inline_text(finding.clarifying_question)
    return (
        f"- **[{sev_ru}] {finding.id}** ({criterion_ref(finding.criterion)}): "
        f"«{quote}» — {explanation} "
        f"Рекомендация: {recommendation} "
        f"Вопрос: {question}"
    )


def _render_per_requirement(report: Report) -> list[str]:
    lines = ["## Проверка каждого требования", ""]

    if not report.requirements:
        lines.append("Список требований пуст.")
        return lines

    req_ids_known = {req.id for req in report.requirements}
    findings_by_req: dict[str, list[Finding]] = defaultdict(list)
    orphan_findings: list[Finding] = []

    also_in: dict[str, list[str]] = defaultdict(list)  # requirement -> findings printed under another one
    for finding in report.findings:
        known = [r for r in finding.requirement_ids if r in req_ids_known]
        if not known:
            orphan_findings.append(finding)
            continue
        findings_by_req[known[0]].append(finding)  # the card is printed once, under its first requirement
        for other in known[1:]:
            also_in[other].append(finding.id)

    set_refs = _set_refs(report)
    clean: list[Requirement] = []
    for req in report.requirements:
        req_findings = sorted(
            findings_by_req.get(req.id, []),
            key=lambda f: (SEVERITY_RANK[f.severity], f.id),
        )
        if not req_findings and req.check_status != "not_checked" and not also_in.get(req.id):
            clean.append(req)
            continue

        doc_id_str = _format_doc_id(req)
        lines.append(f"### {req.id} ({doc_id_str}) — {_short(req.title)}")
        lines.append("")
        status_badge = " · **Статус проверки: не проверено**" if req.check_status == "not_checked" else ""
        lines.append(
            f"- **Раздел:** {req.section} · **Тип:** {req_kind_label(req.kind)}{status_badge}"
        )
        if set_refs.get(req.id):
            lines.append(f"- **Затронуто в наборе:** {', '.join(set_refs[req.id])}")
        if also_in.get(req.id):
            lines.append(f"- **Касается и этого требования:** {', '.join(also_in[req.id])} (карточка выше)")
        if not req_findings and req.check_status != "not_checked":
            lines.append("")
            continue

        if req.check_status == "not_checked" and not req_findings:
            lines.append("")
            lines.append("Требование не проверялось (этап проверки не завершился или запись пропущена).")
            lines.append("")
            continue

        if req.check_status == "not_checked" and req_findings:
            lines.append("")
            lines.append(
                "Внимание: полная проверка требования не выполнялась; ниже приведены доступные замечания."
            )

        major_or_crit = [f for f in req_findings if f.severity in (Severity.CRITICAL, Severity.MAJOR)]
        minors = [f for f in req_findings if f.severity == Severity.MINOR]

        for f in major_or_crit:
            lines.append("")
            lines.extend(_render_full_finding(f, criterion_label(f.criterion)))

        if minors:
            lines.append("")
            for f in minors:
                lines.append(_render_minor_finding_line(f))

        lines.append("")

    if clean:
        touched = [r for r in clean if set_refs.get(r.id)]
        untouched = [r for r in clean if not set_refs.get(r.id)]
        lines.append("### Требования без собственных нарушений по п. 5.2.5")
        lines.append("")
        if touched:
            lines.append(
                f"У {_count(len(touched))} собственных нарушений по п. 5.2.5 нет, но они затронуты находками "
                "проверки набора (см. раздел «Проверка набора требований»):"
            )
            lines.append("")
            for r in touched:
                lines.append(f"- {r.id} ({_escape_cell(_short(r.title, 80))}) — см. {', '.join(set_refs[r.id])}")
            lines.append("")
        if untouched:
            lines.append(
                f"Без нарушений (ни собственных, ни в проверке набора): {_count(len(untouched))} — "
                + ", ".join(r.id for r in untouched) + "."
            )
        lines.append("")

    if orphan_findings:
        lines.append("### Прочие замечания по требованиям")
        lines.append("")
        for f in sorted(orphan_findings, key=lambda item: (SEVERITY_RANK[item.severity], item.id)):
            if f.severity == Severity.MINOR:
                lines.append(_render_minor_finding_line(f))
            else:
                lines.extend(_render_full_finding(f, criterion_label(f.criterion)))
                lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _bucket_set_findings(
    set_findings: list[SetFinding],
) -> tuple[list[SetFinding], list[SetFinding], list[SetFinding], list[SetFinding]]:
    """Partition set findings into 4 mutually exclusive subsections in §8.1 order."""
    contradictions: list[SetFinding] = []
    cross_part: list[SetFinding] = []
    acceptance: list[SetFinding] = []
    completeness: list[SetFinding] = []

    gaps = (SetCheck.EMPTY_SECTION, SetCheck.MISSING_REQUIREMENT, SetCheck.MISSING_SCENARIO)
    for sf in set_findings:
        # the kind of defect decides the chapter; "several parts are quoted" only matters for consistency issues
        if sf.set_check in (SetCheck.ACCEPTANCE_MISMATCH, SetCheck.ACCEPTANCE_GAP):
            acceptance.append(sf)
        elif sf.set_check in gaps or sf.criterion in (Criterion.SET_COMPLETE, Criterion.RANKED,
                                                       Criterion.MODIFIABLE, Criterion.TRACEABLE):
            completeness.append(sf)
        elif sf.set_check == SetCheck.CROSS_PART or (sf.cross_part and sf.criterion == Criterion.SET_CONSISTENT):
            cross_part.append(sf)
        elif (
            sf.set_check in (SetCheck.CONTRADICTION, SetCheck.TERMINOLOGY, SetCheck.DUPLICATION)
            or sf.criterion == Criterion.SET_CONSISTENT
        ):
            contradictions.append(sf)
        else:
            completeness.append(sf)

    def sort_key(item: SetFinding) -> tuple[int, str]:
        return (SEVERITY_RANK[item.severity], item.id)

    return (
        sorted(contradictions, key=sort_key),
        sorted(cross_part, key=sort_key),
        sorted(acceptance, key=sort_key),
        sorted(completeness, key=sort_key),
    )


def _render_set_check(report: Report) -> list[str]:
    req_by_id = {req.id: req for req in report.requirements}
    contradictions, cross_part, acceptance, completeness = _bucket_set_findings(report.set_findings)

    lines = ["## Проверка набора требований", ""]

    # 1. Contradictions.
    lines.append("### Противоречия")
    lines.append("")
    if not contradictions:
        lines.append("Противоречий внутри набора не выявлено.")
    else:
        for idx, sf in enumerate(contradictions):
            if idx > 0:
                lines.append("")
            lines.extend(_render_full_finding(sf, set_check_label(sf.set_check)))

    # 2. Cross-part consistency.
    lines.extend(["", "### Согласованность между частями документа", ""])
    if not cross_part:
        lines.append("Расхождений между частями документа не выявлено.")
    else:
        for idx, sf in enumerate(cross_part):
            if idx > 0:
                lines.append("")
            lines.extend(_render_full_finding(sf, set_check_label(sf.set_check)))

    # 3. Requirements <-> Acceptance criteria (matrix + findings).
    lines.extend(["", "### Требования и критерии приёмки", ""])
    if report.coverage:
        lines.extend(
            [
                "| Требование | Критерии приёмки | Статус покрытия | Примечание |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in report.coverage:
            req = req_by_id.get(row.requirement_id)
            if req is not None and req.doc_id and not req.id_assigned:
                req_cell = f"{req.id} ({req.doc_id})"
            else:
                req_cell = row.requirement_id
            acc_cell = ", ".join(row.acceptance_ids) if row.acceptance_ids else "—"
            status_cell = COVERAGE_STATUS_LABELS.get(row.status, row.status)
            note_cell = row.note if row.note else "—"
            lines.append(
                f"| {_escape_cell(req_cell)} | {_escape_cell(acc_cell)} | "
                f"{_escape_cell(status_cell)} | {_escape_cell(note_cell)} |"
            )
    else:
        lines.append("Матрица покрытия критериями приёмки не сформирована.")

    if acceptance:
        for sf in acceptance:
            lines.append("")
            lines.extend(_render_full_finding(sf, set_check_label(sf.set_check)))
    elif report.coverage:
        lines.append("")
        lines.append("Отдельных расхождений с критериями приёмки не выявлено.")

    # 4. Completeness.
    lines.extend(["", "### Полнота", ""])
    if not completeness:
        lines.append("Пробелов полноты на уровне набора не выявлено.")
    else:
        for idx, sf in enumerate(completeness):
            if idx > 0:
                lines.append("")
            lines.extend(_render_full_finding(sf, set_check_label(sf.set_check)))

    return lines


def _render_summary(report: Report) -> list[str]:
    summary = report.summary
    finding_by_id: dict[str, BaseFinding] = {
        **{f.id: f for f in report.findings},
        **{sf.id: sf for sf in report.set_findings},
    }

    lines = [
        "## Сводка",
        "",
        "| Показатель | Значение |",
        "| --- | ---: |",
        f"| Всего требований | {summary.requirements_total} |",
        f"| Требований с собственными нарушениями | {summary.requirements_with_findings} |",
        f"| Требований, затронутых только находками набора | {summary.requirements_in_set_findings_only} |",
        f"| Требований без нарушений | {summary.requirements_clean} |",
        f"| Непроверенных требований | {summary.requirements_not_checked} |",
        f"| Всего нарушений | {summary.findings_total} |",
        f"| Вопросов к автору ТЗ | {summary.questions_total} |",
    ]

    if summary.requirements_by_kind:
        lines.extend(
            [
                "",
                "### Требования по типам",
                "",
                "| Тип требования | Количество |",
                "| --- | ---: |",
            ]
        )
        rendered_kinds: set[str] = set()
        for kind in REQ_KIND_ORDER:
            if kind.value in summary.requirements_by_kind:
                lines.append(
                    f"| {_escape_cell(req_kind_label(kind))} | {summary.requirements_by_kind[kind.value]} |"
                )
                rendered_kinds.add(kind.value)
        for key in sorted(summary.requirements_by_kind):
            if key not in rendered_kinds:
                lines.append(f"| {_escape_cell(req_kind_label(key))} | {summary.requirements_by_kind[key]} |")

    lines.extend(
        [
            "",
            "### Нарушения по критичности",
            "",
            "| Критичность | Количество |",
            "| --- | ---: |",
        ]
    )
    for sev in (Severity.CRITICAL, Severity.MAJOR, Severity.MINOR):
        count = summary.findings_by_severity.get(sev.value, 0)
        lines.append(f"| {SEVERITY_PLURAL_LABELS[sev]} | {count} |")

    lines.extend(["", "### Нарушения по критериям", ""])
    if not summary.findings_by_criterion:
        lines.append("Нарушений по критериям не зафиксировано.")
    else:
        lines.extend(
            [
                "| Критерий | Пункт стандарта | Количество |",
                "| --- | --- | ---: |",
            ]
        )
        sorted_criteria = sorted(
            summary.findings_by_criterion.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for crit_key, count in sorted_criteria:
            lines.append(
                f"| {_escape_cell(criterion_label(crit_key))} | "
                f"{_escape_cell(criterion_ref(crit_key))} | {count} |"
            )

    lines.extend(["", "### Критичные дефекты по группам", ""])
    if not summary.critical_ids:
        lines.append("Критичных дефектов не выявлено.")
    else:
        listed_any = False
        rendered_classes: set[str] = set()
        ordered_keys = [dc.value for dc in DEFECT_CLASS_ORDER]
        extra_keys = sorted(k for k in summary.critical_by_class if k not in ordered_keys)

        for cls_key in [*ordered_keys, *extra_keys]:
            ids = summary.critical_by_class.get(cls_key)
            if not ids:
                continue
            listed_any = True
            rendered_classes.add(cls_key)
            label = defect_class_label(cls_key)
            lines.append(f"- **{label}** ({len(ids)}):")
            for fid in ids:
                finding = finding_by_id.get(fid)
                if finding is not None:
                    lines.append(
                        f"  - **{fid}** ({finding.section}) — {_first_sentence(_inline_text(finding.explanation))}"
                    )
                else:
                    lines.append(f"  - **{fid}**")

        if not listed_any:
            for fid in summary.critical_ids:
                finding = finding_by_id.get(fid)
                if finding is not None:
                    lines.append(f"- **{fid}** ({finding.section}) — {_inline_text(finding.explanation)}")
                else:
                    lines.append(f"- **{fid}**")

    return lines


def _render_question_block(question: Question) -> list[str]:
    place_parts: list[str] = []
    if question.requirement_ids:
        word = "требование" if len(question.requirement_ids) == 1 else "требования"
        place_parts.append(f"{word} {', '.join(question.requirement_ids)}")
    if question.section:
        place_parts.append(f"раздел «{question.section}»")
    place = "; ".join(place_parts) if place_parts else "—"
    defects = ", ".join(question.finding_ids) if question.finding_ids else "—"

    lines = [
        f"#### {question.id}. {question.text}",
        "",
        f"- **Адресат:** {addressee_label(question.addressee)}",
        f"- **Место:** {place}",
        f"- **Связанные дефекты:** {defects}",
    ]
    if question.options:
        lines.append("- **Варианты ответа:**")
        for idx, opt in enumerate(question.options):
            lines.append(f"  - {opt}")
            cons = (
                question.option_consequences[idx].strip()
                if idx < len(question.option_consequences)
                else ""
            )
            if cons:
                lines.append(f"    - *Что изменится:* {cons}")
    return lines


def _render_questions(report: Report) -> list[str]:
    lines = ["## Вопросы к автору ТЗ", ""]

    if not report.questions:
        lines.append("Уточняющих вопросов к автору ТЗ нет.")
        return lines

    blocking = [q for q in report.questions if q.blocking]
    regular = [q for q in report.questions if not q.blocking]
    in_cards = report.summary.questions_in_findings
    if in_cards > len(report.questions):
        lines.append(
            f"Здесь собраны {len(report.questions)} главных вопросов: блокирующие первыми, близкие по теме "
            f"объединены. Всего в карточках нарушений {_plural(in_cards, 'уточняющий вопрос', 'уточняющих вопроса', 'уточняющих вопросов')}."
        )
        lines.append("")

    if blocking:
        lines.append("### Блокирующие вопросы")
        for q in blocking:
            lines.append("")
            lines.extend(_render_question_block(q))

    if regular:
        if blocking:
            lines.append("")
        lines.append("### Прочие вопросы")
        for q in regular:
            lines.append("")
            lines.extend(_render_question_block(q))

    return lines


def _render_verdict(report: Report) -> list[str]:
    verdict = report.verdict
    dev_ready = READINESS_LABELS.get(verdict.development, str(verdict.development))
    test_ready = READINESS_LABELS.get(verdict.testing, str(verdict.testing))

    lines = [
        "## Итог",
        "",
        f"**{verdict.headline}**",
        "",
        "| Этап | Можно ли начинать |",
        "| --- | --- |",
        f"| Разработка | {_escape_cell(dev_ready)} |",
        f"| Тестирование | {_escape_cell(test_ready)} |",
    ]

    if verdict.rationale:
        lines.extend(["", verdict.rationale])

    if verdict.fix_first:
        lines.extend(["", "### Что исправить первым", ""])
        for idx, item in enumerate(verdict.fix_first, start=1):
            lines.append(f"{idx}. **{item.finding_id}** — {item.title}")

    return lines


_SOURCE_LABELS = {
    "rules": "правила",
    "check": "проверка каждого требования",
    "set": "проверка набора",
    "derived": "сводные находки по матрице покрытия",
    "review": "добавлено критиком",
}


def _candidate_balance(report: Report) -> list[str]:
    """Candidates in = findings out + merged + removed: the numbers in the report add up."""
    s = report.summary
    if not s.candidates_total:
        return []
    parts = ", ".join(f"{_SOURCE_LABELS.get(k, k)} — {v}" for k, v in s.candidates_by_source.items() if v)
    return [
        f"- Не подтверждено критиком (цитата только по тексту требования): {s.dropped_unconfirmed}",
        "",
        (
            f"Баланс кандидатов: всего {s.candidates_total} ({parts}) = итоговые нарушения {s.findings_total} "
            f"+ объединённые дубли {s.merged_duplicates} + удалённые критиком {s.dropped_by_review} "
            f"+ неподтверждённые {s.dropped_unconfirmed}. Находки без подтверждённой цитаты "
            f"({s.dropped_unverified_quotes}) отброшены до попадания в кандидаты."
        ),
    ]


def _render_appendix(report: Report) -> list[str]:
    standards = report.run.standards
    standards_list = ", ".join(standards) if standards else "ISO/IEC/IEEE 29148:2018, IEEE 830-1998"

    quote_counts: Counter[QuoteStatus] = Counter()
    for finding in [*report.findings, *report.set_findings]:
        for ev in finding.evidence:
            quote_counts[ev.quote_status] += 1

    lines = [
        "## Приложение",
        "",
        "### Критерии и стандарты",
        "",
        (
            f"Используемые стандарты: {standards_list} "
            "(IEEE 830-1998 официально заменён ISO/IEC/IEEE 29148 и используется как дополнительная рамка)."
        ),
        "",
        "| Код | Критерий | Пункт стандарта |",
        "| --- | --- | --- |",
    ]
    for crit, meta in CRITERION_META.items():
        ref_str = meta.standard_ref
        if meta.ieee830_ref and meta.ieee830_ref not in ref_str:
            ref_str = f"{ref_str} ({meta.ieee830_ref})"
        lines.append(
            f"| `{crit.value}` | {_escape_cell(meta.label)} | {_escape_cell(ref_str)} |"
        )

    lines.extend(
        [
            "",
            "### Проверка цитат и отсев находок",
            "",
            (
                "Каждая находка проходит детерминированную сверку цитат по тексту документа "
                "(нормализация, точное совпадение, затем нечёткое сопоставление с заменой на "
                "дословный фрагмент документа). Находки без подтверждённой цитаты отбрасываются."
            ),
            "",
            "Цитаты в итоговых нарушениях (у одного нарушения может быть несколько цитат):",
            "",
            f"- Цитаты, совпавшие с текстом дословно (`verified`): {quote_counts.get(QuoteStatus.VERIFIED, 0)}",
            f"- Цитаты, исправленные по тексту документа (`repaired`): {quote_counts.get(QuoteStatus.REPAIRED, 0)}",
            f"- Цитаты, заменённые текстом требования (`anchored`): {quote_counts.get(QuoteStatus.ANCHORED, 0)}",
            "",
            "Находки (кандидаты в нарушения):",
            "",
            f"- Отброшено находок без подтверждённой цитаты: {report.summary.dropped_unverified_quotes}",
            f"- Отброшено на этапе критического пересмотра: {report.summary.dropped_by_review}",
            f"- Объединено с другими находками как дубли: {report.summary.merged_duplicates}",
            *_candidate_balance(report),
            "",
            "### Ограничения",
            "",
            (
                "- Выполняется только статический анализ текста спецификации без доступа к исходному коду, "
                "базам данных и внешним системам."
            ),
            "- Встроенные изображения, схемы и диаграммы в графическом формате не анализируются.",
            (
                "- Бизнес-корректность требований, приоритеты и выбор между противоречащими формулировками "
                "требуют подтверждения со стороны автора ТЗ."
            ),
        ]
    )

    if report.run.warnings:
        for warning in report.run.warnings:
            lines.append(f"- Предупреждение прогона: {warning}")

    return lines


def render_markdown(report: Report) -> str:
    """Render a structured ``Report`` into a deterministic Russian Markdown document."""
    title = f"# Отчёт о качестве требований: {report.run.document_title}" if report.run.document_title else "# Отчёт о качестве требований"
    blocks: list[list[str]] = [[title]]

    banner = _render_banner(report)
    if banner:
        blocks.append(banner)

    blocks.extend(
        [
            _render_passport(report),
            _render_short_summary(report),
        ]
    )
    if report.refinement is not None:
        blocks.append(_render_refinement(report))
    blocks.extend(
        [
            _render_slicing(report),
            _render_per_requirement(report),
            _render_set_check(report),
            _render_summary(report),
            _render_questions(report),
            _render_verdict(report),
            _render_appendix(report),
        ]
    )

    return "\n\n".join("\n".join(section) for section in blocks) + "\n"
