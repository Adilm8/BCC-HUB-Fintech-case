"""Абляция: один документ, несколько вариантов запуска агента, таблица полноты и шума.

    python -m eval.ablate <документ> --variants rules_only,no_critic,full [--runs 3] [--profile P]

Вариант — имя из VARIANTS или переключатель EXP_… (включается значением 1); части соединяются «+»:
`full+EXP_SCENARIOS`, `no_critic+EXP_RULE_HINTS`. Каждый прогон идёт через CLI агента без кэша и пишет
отчёт в `eval/runs/ablate-<время>-<документ>/<вариант>-<n>/`; итог — `ablate.md` и `ablate.json` там же.
Модуль сам не обращается к модели и не читает `.env`: ключи нужны только дочернему процессу агента.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from eval.run import RUNS, find_gold, prompts_hash
from eval.score import score_files

ROOT = Path(__file__).resolve().parent.parent

# name -> (CLI flags, environment)
VARIANTS: dict[str, tuple[list[str], dict[str, str]]] = {
    "full": ([], {}),
    "rules_only": (["--rules-only"], {}),
    "no_critic": ([], {"EXP_NO_REVIEW": "1"}),
}
_TOGGLE = re.compile(r"^EXP_[A-Z0-9_]+$")


def resolve(variant: str) -> tuple[list[str], dict[str, str]]:
    flags: list[str] = []
    env: dict[str, str] = {}
    for part in variant.split("+"):
        part = part.strip()
        if part in VARIANTS:
            f, e = VARIANTS[part]
            flags += f
            env.update(e)
        elif _TOGGLE.match(part.upper()):
            env[part.upper()] = "1"
        else:
            raise ValueError(f"неизвестный вариант {part!r}: доступны {', '.join(VARIANTS)} или EXP_…")
    return flags, env


def run_variant(doc: Path, out_dir: Path, flags: list[str], env: dict[str, str], profile: str | None) -> int:
    cmd = [sys.executable, "-m", "validator", str(doc), str(out_dir / "report.md"), "--no-cache", *flags]
    if profile and "--rules-only" not in flags:
        cmd += ["--profile", profile]
    child = {k: v for k, v in os.environ.items() if not k.startswith("EXP_")}  # only the variant's toggles
    child.update(env)
    return subprocess.run(cmd, cwd=ROOT, env=child, check=False).returncode


def _num(d: dict | None, key: str = "value") -> float | None:
    return None if not d else d.get(key)


def summarize(result: dict) -> dict:
    noise = result.get("noise", {})
    return {
        "status": result.get("run_status", ""), "calls": result.get("llm_calls"),
        "findings": result.get("findings_total"),
        "recall": _num(result.get("recall")), "found": result["recall"]["found"], "total": result["recall"]["total"],
        "critical": _num(result.get("recall_by_severity", {}).get("critical")),
        "precision": _num(result.get("precision_lower_bound")),
        "noise": len(noise.get("all", [])) if noise.get("available") else None,
    }


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "н/д"
    return f"{v:.0%}" if pct else (f"{v:.1f}" if isinstance(v, float) else str(v))


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def render(doc: Path, table: dict[str, list[dict]]) -> str:
    lines = [f"# Абляция: {doc.name}", "",
             "| Вариант | Прогонов | Полнота | Найдено | Критичные | Точность (нижняя) | Шум | Находок | Вызовов |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, rows in table.items():
        ok = [r for r in rows if r.get("recall") is not None]
        if not ok:
            lines.append(f"| {name} | {len(rows)} | отчёт не записан | | | | | | |")
            continue
        found = "/".join(str(r["found"]) for r in ok) + f" из {ok[0]['total']}"
        lines.append(f"| {name} | {len(ok)} | {_fmt(_mean(ok, 'recall'), True)} | {found} | "
                     f"{_fmt(_mean(ok, 'critical'), True)} | {_fmt(_mean(ok, 'precision'), True)} | "
                     f"{_fmt(_mean(ok, 'noise'))} | {_fmt(_mean(ok, 'findings'))} | {_fmt(_mean(ok, 'calls'))} |")
    lines += ["", "Значения — среднее по прогонам; «Найдено» — по каждому прогону."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m eval.ablate", description="Абляция вариантов агента на документе")
    p.add_argument("doc", type=Path, help="документ требований (по нему выбирается эталон)")
    p.add_argument("--variants", default="rules_only,no_critic,full", help="варианты через запятую")
    p.add_argument("--runs", type=int, default=1, help="прогонов на вариант")
    p.add_argument("--profile", default=None, help="профиль вызовов модели (передаётся агенту)")
    p.add_argument("--gold", type=Path, default=None, help="эталон (по умолчанию ищется по имени документа)")
    p.add_argument("--out", type=Path, default=None, help="папка результатов")
    args = p.parse_args(argv)

    gold = find_gold(args.doc, args.gold)
    if gold is None:
        print(f"Эталон для {args.doc.name} не найден (искали eval/gold, eval/gold_private).", file=sys.stderr)
        return 3
    try:
        plan = {v.strip(): resolve(v) for v in args.variants.split(",") if v.strip()}
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or RUNS / f"ablate-{stamp}-{args.doc.stem}"
    out.mkdir(parents=True, exist_ok=True)

    table: dict[str, list[dict]] = {}
    for name, (flags, env) in plan.items():
        for n in range(1, max(1, args.runs) + 1):
            run_dir = out / f"{re.sub(r'[^A-Za-z0-9_+-]', '_', name)}-{n}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{name} {n}/{args.runs}]", file=sys.stderr, flush=True)
            rc = run_variant(args.doc, run_dir, flags, env, args.profile)
            report = run_dir / "report.json"
            row: dict = {"exit_code": rc, "dir": str(run_dir)}
            if report.is_file():
                result, _ = score_files(report, gold, write=False)
                row.update(summarize(result))
            table.setdefault(name, []).append(row)

    md = render(args.doc, table)
    (out / "ablate.md").write_text(md, encoding="utf-8")
    (out / "ablate.json").write_text(json.dumps({
        "document": str(args.doc), "gold": str(gold), "profile": args.profile, "prompts_sha256": prompts_hash(),
        "variants": {k: {"flags": v[0], "env": v[1]} for k, v in plan.items()}, "runs": table,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"Итог записан: {out / 'ablate.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
