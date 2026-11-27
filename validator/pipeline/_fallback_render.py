"""Minimal private Markdown renderer, used only while validator.report.markdown is not available."""

from __future__ import annotations

from validator.models import Report, Severity

_SEV = {"critical": "критичный", "major": "существенный", "minor": "незначительный"}
_READY = {"yes": "да", "conditional": "с условием", "no": "нет"}
_STATE = {"pending": "ожидает", "running": "идёт", "done": "выполнен", "failed": "сбой", "skipped": "пропущен",
          "cached": "из кэша"}


def _cell(text: str) -> str:
    return (text or "").replace("|", "/").replace("\n", " ")


def render_markdown(report: Report) -> str:
    r = report.run
    out = ["# Отчёт о качестве требований", ""]
    if r.status != "complete":
        out += [f"> **Отчёт неполный ({r.status}).** " + " ".join(r.warnings[:3]), ""]
    out += ["## 1. Паспорт прогона", "", "| Поле | Значение |", "|---|---|",
            f"| Документ | {_cell(r.document_title)} (`{r.document_name}`, {r.document_format}) |",
            f"| SHA-256 | `{r.document_sha256}` |",
            f"| Дата | {r.started_at:%d.%m.%Y %H:%M} |",
            f"| Провайдер и модели | {r.provider}: {', '.join(r.models_used) or '—'} |",
            f"| Версия агента | {r.agent_version} |",
            f"| Профиль, вызовов модели | {r.profile}, {r.llm_calls} |",
            f"| Длительность | {r.duration_seconds:.0f} с |",
            f"| Стандарты | {', '.join(r.standards)} |", "",
            "| Этап | Состояние | Подробности |", "|---|---|---|"]
    out += [f"| {s.title} | {_STATE.get(s.state, s.state)} | {_cell(s.detail)} |" for s in r.stages]
    s = report.summary
    v = report.verdict
    out += ["", "## 2. Коротко", "", v.headline, "",
            (f"Требований: {s.requirements_total}; нарушений: {s.findings_total} "
             f"(критичных {s.findings_by_severity.get('critical', 0)}, существенных "
             f"{s.findings_by_severity.get('major', 0)}, незначительных {s.findings_by_severity.get('minor', 0)}); "
             f"вопросов: {s.questions_total}."), "",
            "## 3. Как агент нарезал требования", "",
            "| ID | ID в документе | Раздел | Тип | Формулировка |", "|---|---|---|---|---|"]
    for q in report.requirements:
        out.append(f"| {q.id} | {q.doc_id or 'присвоен агентом'} | {_cell(q.section)} | {q.kind} | {_cell(q.title)} |")
    out += ["", f"Служебных блоков: {len(report.skipped_blocks)}.", "", "## 4. Проверка каждого требования", ""]
    by_req: dict[str, list] = {}
    for f in report.findings:
        by_req.setdefault(f.requirement_ids[0], []).append(f)
    for q in report.requirements:
        out.append(f"### {q.id} {q.doc_id or ''} — {_cell(q.title)}")
        items = by_req.get(q.id, [])
        if q.check_status == "not_checked":
            out.append("_Требование не проверено моделью._")
        elif not items:
            out.append("Нарушений по п. 5.2.5 не выявлено.")
        for f in items:
            out += _finding(f)
        out.append("")
    out += ["## 5. Проверка всего набора требований", ""]
    for f in report.set_findings:
        out.append(f"### {f.id} ({f.set_check})")
        out += _finding(f)
        out.append("")
    if report.coverage:
        out += ["Требования и критерии приёмки:", "", "| Требование | Критерии | Статус | Примечание |",
                "|---|---|---|---|"]
        out += [f"| {c.requirement_id} | {', '.join(c.acceptance_ids) or '—'} | {c.status} | {_cell(c.note)} |"
                for c in report.coverage]
        out.append("")
    out += ["## 6. Сводка", "", f"- Требований: {s.requirements_total}",
            f"- Нарушений: {s.findings_total}", f"- Критичные: {', '.join(s.critical_ids) or 'нет'}"]
    for cls, ids in s.critical_by_class.items():
        out.append(f"  - {cls}: {', '.join(ids)}")
    out += ["", "## 7. Вопросы к автору ТЗ", ""]
    for qn in report.questions:
        opts = f" Варианты: {'; '.join(qn.options)}." if qn.options else ""
        mark = " **(блокирующий)**" if qn.blocking else ""
        out.append(f"- **{qn.id}**{mark} [{', '.join(qn.finding_ids)}; {qn.section}] {qn.text}{opts}")
    out += ["", "## 8. Итог", "", f"- Разработка: {_READY[v.development]}", f"- Тестирование: {_READY[v.testing]}"]
    if v.rationale:
        out += ["", v.rationale]
    if v.fix_first:
        out += ["", "Что исправить первым:"]
        out += [f"{i}. {x.finding_id}: {x.title}" for i, x in enumerate(v.fix_first, 1)]
    out += ["", "## 9. Приложение", "",
            (f"Отброшено находок без подтверждённой цитаты: {s.dropped_unverified_quotes}; "
             f"удалено критиком: {s.dropped_by_review}."), ""]
    return "\n".join(out)


def _finding(f) -> list[str]:
    head = f"- **{f.id}** · {f.criterion} · {_SEV[f.severity]}"
    if f.severity == Severity.MINOR:
        return [f"{head}: «{f.quote}» — {f.explanation}"]
    lines = [head]
    for e in f.evidence:
        lines.append(f"  - Цитата ({e.section}): «{e.quote}»")
    lines += [f"  - Объяснение: {f.explanation}", f"  - Рекомендация: {f.recommendation}",
              f"  - Вопрос: {f.clarifying_question}"]
    return lines
