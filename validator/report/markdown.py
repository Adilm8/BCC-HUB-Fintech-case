"""Pure deterministic Markdown report renderer implementing the fixed 9-section template."""

from __future__ import annotations

from collections import Counter, defaultdict

from validator.models import (
    SEVERITY_RANK,
    BaseFinding,
    Criterion,
    Evidence,
    Finding,
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
    criterion_label,
    criterion_ref,
    defect_class_label,
    req_kind_label,
    set_check_label,
)


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

    return [
        "## Коротко",
        "",
        f"**{verdict.headline}**",
        "",
        (
            f"- **Требований выделено:** {summary.requirements_total} "
            f"(с нарушениями — {summary.requirements_with_findings}, "
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
            "[Нарезка требований](#нарезка-требований) · "
            "[Проверка каждого требования](#проверка-каждого-требования) · "
            "[Проверка набора требований](#проверка-набора-требований) · "
            "[Сводка](#сводка) · "
            "[Вопросы к автору ТЗ](#вопросы-к-автору-тз) · "
            "[Итог](#итог) · "
            "[Приложение](#приложение)"
        ),
    ]


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
            lines.append(
                f"| {_escape_cell(req.id)} "
                f"| {_escape_cell(_format_doc_id(req))} "
                f"| {_escape_cell(req.section)} "
                f"| {_escape_cell(req_kind_label(req.kind))} "
                f"| {_escape_cell(req.title)} "
                f"| {_escape_cell(req.quote)} |"
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
    lines.append(f"- **Уточняющий вопрос:** {finding.clarifying_question}")
    if finding.answer_options:
        lines.append("- **Варианты ответа:**")
        for opt in finding.answer_options:
            lines.append(f"  - {opt}")
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

    for finding in report.findings:
        matched = False
        for req_id in finding.requirement_ids:
            if req_id in req_ids_known:
                findings_by_req[req_id].append(finding)
                matched = True
        if not matched:
            orphan_findings.append(finding)

    for req in report.requirements:
        doc_id_str = _format_doc_id(req)
        lines.append(f"### {req.id} ({doc_id_str}) — {req.title}")
        lines.append("")
        status_badge = " · **Статус проверки: не проверено**" if req.check_status == "not_checked" else ""
        lines.append(
            f"- **Раздел:** {req.section} · **Тип:** {req_kind_label(req.kind)}{status_badge}"
        )

        req_findings = sorted(
            findings_by_req.get(req.id, []),
            key=lambda f: (SEVERITY_RANK[f.severity], f.id),
        )

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

        if not req_findings:
            lines.append("")
            lines.append("Нарушений по п. 5.2.5 не выявлено.")
            lines.append("")
            continue

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

    for sf in set_findings:
        if sf.cross_part or sf.set_check == SetCheck.CROSS_PART:
            cross_part.append(sf)
        elif sf.set_check in (SetCheck.ACCEPTANCE_MISMATCH, SetCheck.ACCEPTANCE_GAP):
            acceptance.append(sf)
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
        f"| Требований с нарушениями | {summary.requirements_with_findings} |",
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
                        f"  - **{fid}** ({finding.section}) — {_inline_text(finding.explanation)}"
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
        place_parts.append(f"требования {', '.join(question.requirement_ids)}")
    if question.section:
        place_parts.append(f"раздел «{question.section}»")
    place = "; ".join(place_parts) if place_parts else "—"
    defects = ", ".join(question.finding_ids) if question.finding_ids else "—"

    lines = [
        f"#### {question.id}. {question.text}",
        "",
        f"- **Место:** {place}",
        f"- **Связанные дефекты:** {defects}",
    ]
    if question.options:
        lines.append("- **Варианты ответа:**")
        for opt in question.options:
            lines.append(f"  - {opt}")
    return lines


def _render_questions(report: Report) -> list[str]:
    lines = ["## Вопросы к автору ТЗ", ""]

    if not report.questions:
        lines.append("Уточняющих вопросов к автору ТЗ нет.")
        return lines

    blocking = [q for q in report.questions if q.blocking]
    regular = [q for q in report.questions if not q.blocking]

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
            f"- Подтверждено дословно (`verified`): {quote_counts.get(QuoteStatus.VERIFIED, 0)}",
            f"- Уточнено по тексту документа (`repaired`): {quote_counts.get(QuoteStatus.REPAIRED, 0)}",
            f"- Привязано к тексту требования (`anchored`): {quote_counts.get(QuoteStatus.ANCHORED, 0)}",
            f"- Отброшено находок без подтверждённой цитаты: {report.summary.dropped_unverified_quotes}",
            f"- Отброшено на этапе критического пересмотра: {report.summary.dropped_by_review}",
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
