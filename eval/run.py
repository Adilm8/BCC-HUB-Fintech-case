"""Прогон агента по документу и оценка результата.

    python -m eval.run <документ> [--profile P] [--gold PATH] [--no-cache]
    python -m eval.run <документ> --report-json <отчёт.json>      # без прогона, только оценка

Прогон идёт через CLI агента (`python -m validator`) и пишет в `eval/runs/<время>-<документ>/`:
report.md, report.json, report.score.md, report.score.json, meta.json (команда, профиль, хэш промптов).
Эталон ищется как `eval/gold/<документ>.gold.json`, затем `eval/gold_private/<документ>.gold.json`.
Этот модуль сам не обращается к модели и не читает `.env`: ключи нужны только дочернему процессу агента.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from eval.score import render_markdown, score_files

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"
GOLD_DIRS = (ROOT / "eval" / "gold", ROOT / "eval" / "gold_private")
PROMPTS = ROOT / "validator" / "pipeline" / "prompts"

EXIT_NO_GOLD = 3


def find_gold(doc: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for d in GOLD_DIRS:
        cand = d / f"{doc.stem}.gold.json"
        if cand.is_file():
            return cand
    return None


def prompts_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(PROMPTS.rglob("*.py")):
        h.update(p.relative_to(PROMPTS).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_validator(doc: Path, run_dir: Path, profile: str | None, no_cache: bool) -> tuple[int, list[str]]:
    cmd = [sys.executable, "-m", "validator", str(doc), str(run_dir / "report.md")]
    if profile:
        cmd += ["--profile", profile]
    if no_cache:
        cmd.append("--no-cache")
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode, cmd


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m eval.run", description="Прогон агента по документу и оценка")
    p.add_argument("doc", type=Path, help="документ требований (по нему выбирается эталон)")
    p.add_argument("--profile", default=None, help="профиль вызовов модели (передаётся агенту)")
    p.add_argument("--gold", type=Path, default=None, help="эталон (по умолчанию ищется по имени документа)")
    p.add_argument("--report-json", type=Path, default=None, help="готовый отчёт JSON: агента не запускать")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш этапов агента")
    args = p.parse_args(argv)

    gold = find_gold(args.doc, args.gold)
    if gold is None:
        print(f"Эталон для {args.doc.name} не найден (искали eval/gold, eval/gold_private).", file=sys.stderr)
        return EXIT_NO_GOLD

    rc = 0
    if args.report_json is not None:
        report_json = args.report_json
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS / f"{stamp}-{args.doc.stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rc, cmd = run_validator(args.doc, run_dir, args.profile, args.no_cache)
        report_json = run_dir / "report.json"
        (run_dir / "meta.json").write_text(json.dumps({
            "document": str(args.doc), "profile": args.profile, "command": cmd, "validator_exit_code": rc,
            "prompts_sha256": prompts_hash(), "gold": str(gold), "started": stamp,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if not report_json.is_file():
            print(f"Агент не записал отчёт (код {rc}): {report_json}", file=sys.stderr)
            return rc or 1
        if rc != 0:
            print(f"Агент завершился с кодом {rc} (отчёт неполный), оценка всё равно считается.", file=sys.stderr)

    try:
        result, md_path = score_files(report_json, gold)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Ошибка оценки: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(result, str(report_json), str(gold)))
    print(f"Сводка записана: {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
