"""Детерминированная оценка отчёта агента по эталонному списку дефектов.

    python -m eval.score <report.json> <gold.json> [--out PATH] [--no-write]

Без сети и без модели: читает сохранённый JSON отчёта (`validator.models.Report`) и эталон
(`eval/gold/*.json`, формат описан ниже).

Формат эталона:

    {"document": "...", "defects": [
        {"id": "G01", "severity": "critical", "criterion": "set_consistent", "description": "...",
         "match": {"where": [["фрагмент 1", "фрагмент 2"]],
                   "what": [["противореч", "несоответств"], ["альтернатива"]]},
         "blocks_hint": ["фрагмент документа, где лежит дефект"]}],
     "clean": [{"id": "C01", "anchor": "фрагмент корректного требования"}]}

Сопоставление. Группа это список альтернатив (должна встретиться хотя бы одна), условие в целом это
«все группы». Условие дефекта состоит из двух частей:

* `where` (где): слова места, фрагменты цитат, номера требований. Ищутся в полном «стоге» находки:
  цитаты всех её доказательств, заголовки требований, на которые она ссылается, объяснение и вопрос.
* `what` (что): слова самого дефекта (противоречие, метода нет, не определено...). Ищутся только в
  собственном тексте находки: заголовок (если есть), объяснение и уточняющий вопрос. Цитаты и заголовки
  требований сюда не входят, поэтому находка на другую тему, которая просто цитирует тот же абзац,
  дефект не засчитывает.

Для обратной совместимости `match.all_of` читается как `where` (поиск в полном «стоге»).

Всё нормализуется (регистр, «ё», тире, пробелы, обратные кавычки). Альтернатива вида `re:<регулярное
выражение>` ищется регулярным выражением (нужно для якорей вроде `\\bget\\b` или `\\b200\\b`, которые как
подстрока сработали бы на чём угодно).

Кредит. Для каждой пары (находка, дефект) считается сила совпадения: число найденных альтернатив
(в `what` с весом 2) плюс бонус, если в цитате находки есть `blocks_hint`. Одна находка засчитывается не
более чем двум дефектам (лучшим по силе), а сводная находка, ссылающаяся больше чем на 8 требований,
только одному. Прочие совпавшие дефекты от этой находки кредита не получают (`uncredited_finding_ids`).
Precision (нижняя граница) считает находку совпавшей, если она подходит хотя бы одному дефекту.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEVERITIES = ("critical", "major", "minor")
_RANK = {"critical": 0, "major": 1, "minor": 2}
_DASHES = re.compile("[‐‑‒–—―−]")
_STRIP = re.compile(r"[`*]")
_SPACES = re.compile(r"\s+")

MAX_PER_FINDING = 2  # столько дефектов эталона может засчитать одна находка
MAX_PER_AGGREGATE = 1  # столько, если находка ссылается на слишком много требований
AGGREGATE_REQUIREMENTS = 8  # «слишком много» значит больше этого числа
WHAT_WEIGHT = 2
HINT_BONUS = 3


def normalize(text: str) -> str:
    """Регистр, «ё», все виды тире и пробелов, разметка Markdown: к единому виду."""
    t = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    t = _DASHES.sub("-", t)
    t = _STRIP.sub("", t)
    return _SPACES.sub(" ", t).strip()


# ---------------------------------------------------------------- gold


@dataclass(frozen=True)
class Alt:
    raw: str
    regex: re.Pattern[str] | None
    needle: str

    def found_in(self, haystack: str) -> bool:
        if self.regex is not None:
            return self.regex.search(haystack) is not None
        return self.needle in haystack


def _compile_alt(raw: str) -> Alt:
    if raw.startswith("re:"):
        return Alt(raw, re.compile(raw[3:], re.IGNORECASE), "")
    return Alt(raw, None, normalize(raw))


Groups = tuple[tuple[Alt, ...], ...]


def _holds(groups: Groups, haystack: str) -> bool:
    return all(any(a.found_in(haystack) for a in group) for group in groups)


def _hits(groups: Groups, haystack: str) -> int:
    return sum(1 for group in groups for a in group if a.found_in(haystack))


@dataclass(frozen=True)
class GoldItem:
    id: str
    severity: str
    criterion: str
    description: str
    where: Groups
    what: Groups
    hints: tuple[str, ...]

    def matches(self, view: FindingView) -> bool:
        return _holds(self.where, view.where_hay) and _holds(self.what, view.what_hay)

    def strength(self, view: FindingView) -> int:
        bonus = HINT_BONUS if any(normalize(h) in q for h in self.hints for q in view.quotes) else 0
        return _hits(self.where, view.where_hay) + WHAT_WEIGHT * _hits(self.what, view.what_hay) + bonus


@dataclass(frozen=True)
class Gold:
    document: str
    items: tuple[GoldItem, ...]
    clean: tuple[tuple[str, str], ...]  # (id, нормализованный фрагмент)


def _parse_groups(gid: str, name: str, raw: Any) -> Groups:
    if raw is None:
        return ()
    ok = isinstance(raw, list) and all(
        isinstance(g, list) and g and all(isinstance(a, str) and a for a in g) for g in raw)
    if not ok:
        raise ValueError(f"эталон {gid}: match.{name} должен быть списком непустых списков строк")
    return tuple(tuple(_compile_alt(a) for a in g) for g in raw)


def parse_gold(data: Any) -> Gold:
    if isinstance(data, list):
        data = {"defects": data}
    if not isinstance(data, dict) or not isinstance(data.get("defects"), list):
        raise ValueError("эталон: ожидается объект с ключом defects (список)")  # noqa: TRY004
    items: list[GoldItem] = []
    seen: set[str] = set()
    for raw in data["defects"]:
        gid = raw.get("id")
        if not gid or gid in seen:
            raise ValueError(f"эталон: пустой или повторный id {gid!r}")
        seen.add(gid)
        severity = raw.get("severity")
        if severity not in SEVERITIES:
            raise ValueError(f"эталон {gid}: severity должна быть одной из {SEVERITIES}, а не {severity!r}")
        match = raw.get("match") or {}
        where = _parse_groups(gid, "where", match.get("where")) + _parse_groups(gid, "all_of", match.get("all_of"))
        what = _parse_groups(gid, "what", match.get("what"))
        if not where and not what:
            raise ValueError(f"эталон {gid}: нужен match.where и/или match.what (или устаревший match.all_of)")
        items.append(GoldItem(gid, severity, raw.get("criterion", ""), raw.get("description", ""), where, what,
                              tuple(raw.get("blocks_hint") or ())))
    clean: list[tuple[str, str]] = []
    for i, c in enumerate(data.get("clean") or [], 1):
        anchors = c["anchor"] if isinstance(c.get("anchor"), list) else [c.get("anchor", "")]
        for a in anchors:
            if a:
                clean.append((c.get("id", f"C{i:02d}"), normalize(a)))
    return Gold(str(data.get("document", "")), tuple(items), tuple(clean))


def load_gold(path: Path) -> Gold:
    return parse_gold(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- report


@dataclass(frozen=True)
class FindingView:
    id: str
    severity: str
    criterion: str
    quotes: tuple[str, ...]  # нормализованные
    quote_preview: str
    where_hay: str  # цитаты, заголовки требований, объяснение, вопрос
    what_hay: str  # собственный текст находки: заголовок, объяснение, вопрос (без цитат)
    n_requirements: int

    @property
    def capacity(self) -> int:
        return MAX_PER_AGGREGATE if self.n_requirements > AGGREGATE_REQUIREMENTS else MAX_PER_FINDING


def finding_views(report: dict) -> list[FindingView]:
    titles = {r["id"]: r.get("title", "") for r in report.get("requirements", [])}
    views: list[FindingView] = []
    for f in list(report.get("findings", [])) + list(report.get("set_findings", [])):
        quotes = [e.get("quote", "") for e in f.get("evidence", [])]
        own = [f.get("title") or "", f.get("explanation") or "", f.get("clarifying_question") or ""]
        req_ids = f.get("requirement_ids", [])
        parts = quotes + [titles.get(r, "") for r in req_ids] + own
        views.append(FindingView(
            id=f["id"], severity=f.get("severity", "minor"), criterion=f.get("criterion", ""),
            quotes=tuple(normalize(q) for q in quotes), quote_preview=(quotes[0] if quotes else "")[:100],
            where_hay=normalize("\n".join(parts)), what_hay=normalize("\n".join(own)),
            n_requirements=len(req_ids),
        ))
    return views


def assign_credit(views: list[FindingView], items: tuple[GoldItem, ...]
                  ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Кому из дефектов какие находки засчитать: (credited, candidates), оба по id дефекта.

    Кандидат: находка, удовлетворяющая условию дефекта. Кредит получают самые сильные пары, но не
    больше `capacity` дефектов на находку."""
    pairs: list[tuple[int, str, int, int]] = []  # (-сила, id находки, номер дефекта, номер находки)
    candidates: dict[str, list[str]] = {i.id: [] for i in items}
    for ii, item in enumerate(items):
        for vi, v in enumerate(views):
            if item.matches(v):
                pairs.append((-item.strength(v), v.id, ii, vi))
                candidates[item.id].append(v.id)
    pairs.sort()
    used = [0] * len(views)
    credited: dict[str, list[str]] = {i.id: [] for i in items}
    for _, _, ii, vi in pairs:
        if used[vi] < views[vi].capacity:
            used[vi] += 1
            credited[items[ii].id].append(views[vi].id)
    return credited, candidates


# ---------------------------------------------------------------- scoring


def score(report: dict, gold: Gold) -> dict[str, Any]:
    views = finding_views(report)
    run = report.get("run", {})
    warnings: list[str] = []
    if gold.document and run.get("document_name") and Path(gold.document).name != run["document_name"]:
        warnings.append(f"эталон относится к {Path(gold.document).name}, а отчёт сделан по {run['document_name']}")
    if not gold.items:
        warnings.append("в эталоне нет дефектов")
    if not views:
        warnings.append("в отчёте нет находок")

    credited, candidates = assign_credit(views, gold.items)
    by_id = {v.id: v for v in views}
    matched_findings = {fid for ids in candidates.values() for fid in ids}
    rows: list[dict[str, Any]] = []
    for item in gold.items:
        hits = [by_id[fid] for fid in credited[item.id]]
        hints = [normalize(h) for h in item.hints]
        located = None
        if hints and hits:
            located = any(h in q for v in hits for q in v.quotes for h in hints)
        severity_ok = None
        if hits:
            severity_ok = min(_RANK.get(v.severity, 2) for v in hits) <= _RANK[item.severity]
        rows.append({"id": item.id, "severity": item.severity, "criterion": item.criterion,
                     "description": item.description, "matched": bool(hits),
                     "finding_ids": [v.id for v in hits], "located": located, "severity_ok": severity_ok,
                     "uncredited_finding_ids": [f for f in candidates[item.id] if f not in credited[item.id]]})

    def recall(sev: str | None) -> dict[str, Any]:
        sel = [r for r in rows if sev is None or r["severity"] == sev]
        found = sum(1 for r in sel if r["matched"])
        return {"found": found, "total": len(sel), "value": (found / len(sel)) if sel else None}

    matched_rows = [r for r in rows if r["matched"]]
    located_rows = [r for r in matched_rows if r["located"] is not None]
    noise_ids: list[str] = []
    noise_major: list[str] = []
    if gold.clean:
        for v in views:
            if v.quotes and all(any(a in q for _, a in gold.clean) for q in v.quotes):
                noise_ids.append(v.id)
                if v.severity in ("critical", "major"):
                    noise_major.append(v.id)
    unmatched = [{"id": v.id, "severity": v.severity, "criterion": v.criterion, "quote": v.quote_preview}
                 for v in views if v.id not in matched_findings]
    return {
        "document": run.get("document_name", ""),
        "profile": run.get("profile", ""),
        "run_status": run.get("status", ""),
        "llm_calls": run.get("llm_calls"),
        "findings_total": len(views),
        "recall": recall(None),
        "recall_by_severity": {s: recall(s) for s in SEVERITIES},
        "severity_agreement": {
            "ok": sum(1 for r in matched_rows if r["severity_ok"]), "total": len(matched_rows),
            "value": (sum(1 for r in matched_rows if r["severity_ok"]) / len(matched_rows)) if matched_rows else None,
        },
        "located": {"ok": sum(1 for r in located_rows if r["located"]), "total": len(located_rows)},
        "precision_lower_bound": {
            "matched": len(matched_findings), "total": len(views),
            "value": (len(matched_findings) / len(views)) if views else None,
        },
        "noise": {"available": bool(gold.clean), "all": noise_ids, "major": noise_major},
        "items": rows,
        "unmatched_findings": unmatched,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- output


def _pct(value: float | None) -> str:
    return "н/д" if value is None else f"{value:.0%}"


def _frac(d: dict[str, Any]) -> str:
    return f"{d['found']} из {d['total']} ({_pct(d['value'])})"


def render_markdown(result: dict[str, Any], report_path: str = "", gold_path: str = "") -> str:
    r = result
    lines = [f"# Оценка отчёта: {r['document'] or report_path}", ""]
    meta = [f"профиль: {r['profile'] or '?'}", f"статус прогона: {r['run_status'] or '?'}",
            f"вызовов модели: {r['llm_calls']}", f"находок: {r['findings_total']}"]
    lines += [", ".join(meta)]
    if report_path or gold_path:
        lines += [f"отчёт: `{report_path}`, эталон: `{gold_path}`"]
    for w in r["warnings"]:
        lines += ["", f"Внимание: {w}"]
    sev = r["recall_by_severity"]
    pl, sa, noise = r["precision_lower_bound"], r["severity_agreement"], r["noise"]
    lines += ["", "## Метрики", "", "| Метрика | Значение |", "| --- | --- |",
              f"| recall общий | {_frac(r['recall'])} |",
              f"| recall критичных | {_frac(sev['critical'])} |",
              f"| recall существенных | {_frac(sev['major'])} |",
              f"| recall незначительных | {_frac(sev['minor'])} |",
              f"| согласие по критичности | {sa['ok']} из {sa['total']} ({_pct(sa['value'])}) |",
              f"| precision, нижняя граница | {pl['matched']} из {pl['total']} ({_pct(pl['value'])}) |"]
    if r["located"]["total"]:
        lines += [f"| найдено в нужном месте | {r['located']['ok']} из {r['located']['total']} |"]
    if noise["available"]:
        lines += [f"| шум на корректных (все находки) | {len(noise['all'])} |",
                  f"| шум на корректных (major и critical) | {len(noise['major'])} |"]
    else:
        lines += ["| шум на корректных | н/д (в эталоне нет списка корректных требований) |"]

    def table(rows: list[dict[str, Any]], status: str) -> list[str]:
        out = ["| ID | Критичность | Описание | Находки |", "| --- | --- | --- | --- |"]
        for x in rows:
            desc = x["description"].replace("|", "\\|").replace("\n", " ")
            out.append(f"| {x['id']} | {x['severity']} | {desc} | {', '.join(x['finding_ids']) or status} |")
        return out

    found = [x for x in r["items"] if x["matched"]]
    missed = [x for x in r["items"] if not x["matched"]]
    lines += ["", f"## Не найдено ({len(missed)})", ""]
    lines += table(missed, "-") if missed else ["Все дефекты эталона найдены."]
    lines += ["", f"## Найдено ({len(found)})", ""]
    lines += table(found, "-") if found else ["Ничего."]
    if noise["all"]:
        lines += ["", "## Находки на корректных требованиях", "", ", ".join(noise["all"])]
    um = r["unmatched_findings"]
    lines += ["", f"## Находки без совпадения с эталоном ({len(um)})", "",
              "Для ручной выборки precision: часть из них может быть верной, но не из эталона.", ""]
    if um:
        lines += ["| Находка | Критичность | Критерий | Цитата |", "| --- | --- | --- | --- |"]
        for u in um:
            q = u["quote"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {u['id']} | {u['severity']} | {u['criterion']} | {q} |")
    else:
        lines.append("Нет.")
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], report_path: Path, gold_path: Path, out: Path | None = None) -> Path:
    md_path = out or report_path.with_suffix(".score.md")
    md_path.write_text(render_markdown(result, str(report_path), str(gold_path)), encoding="utf-8")
    md_path.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path


def score_files(report_path: Path, gold_path: Path, out: Path | None = None, write: bool = True) -> tuple[dict, Path | None]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = score(report, load_gold(gold_path))
    md_path = write_outputs(result, report_path, gold_path, out) if write else None
    return result, md_path


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m eval.score", description="Оценка отчёта по эталону дефектов")
    p.add_argument("report", type=Path, help="отчёт JSON (рядом с .md, который пишет python -m validator)")
    p.add_argument("gold", type=Path, help="эталон: eval/gold/<документ>.gold.json")
    p.add_argument("--out", type=Path, default=None, help="куда записать сводку (по умолчанию <отчёт>.score.md)")
    p.add_argument("--no-write", action="store_true", help="только напечатать, файлов не писать")
    args = p.parse_args(argv)
    try:
        result, md_path = score_files(args.report, args.gold, args.out, write=not args.no_write)
    except (OSError, ValueError, KeyError) as exc:  # json.JSONDecodeError is a ValueError
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(result, str(args.report), str(args.gold)))
    if md_path:
        print(f"Сводка записана: {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
