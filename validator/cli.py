"""Точка входа: python -m validator <файл требований> <файл отчёта>."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from validator.config import PROFILES, Settings
from validator.llm import LLMError
from validator.models import Report, RunStatus, StageEvent, StageState

_STATE = {
    StageState.RUNNING: "…", StageState.DONE: "готово", StageState.FAILED: "сбой", StageState.SKIPPED: "пропущен",
    StageState.CACHED: "из кэша", StageState.PENDING: "",
}


def _render(report: Report) -> str:
    try:
        from validator.report.markdown import render_markdown  # type: ignore[import-not-found]
    except ImportError:
        from validator.pipeline._fallback_render import render_markdown
    return render_markdown(report)


def _progress(event: StageEvent) -> None:
    if event.state == StageState.RUNNING and event.total and event.done is not None:
        line = f"  {event.stage.value}: {event.done}/{event.total}"
    elif event.state == StageState.RUNNING and event.message:
        line = f"  {event.stage.value}: {event.message}"  # a retry, a pause or a switch to a fallback model
    else:
        line = f"  {event.stage.value}: {_STATE.get(event.state, event.state.value)}"
        if event.message and event.state != StageState.RUNNING:
            line += f" ({event.message})"
    print(line, file=sys.stderr, flush=True)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _json_path(args) -> Path | None:
    if args.no_json:
        return None
    path = args.json or args.output.with_suffix(".json")
    if _same_file(path, args.output):
        # never overwrite the Markdown report with JSON
        path = args.output.with_name(args.output.stem + ".report.json")
    return path


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # library warnings in English are noise for the user
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr, flush=True)
        # worker threads may sit in a network call; do not wait for them
        os._exit(130)
    except Exception as exc:  # noqa: BLE001 - last line of defence: a message instead of a traceback
        from validator.llm import short_error

        print(f"Непредвиденная ошибка: {short_error(exc, 300)}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog="python -m validator", description="Проверка качества требований")
    parser.add_argument("input", type=Path, help="путь к документу требований (.md, .txt, .pdf, .docx)")
    parser.add_argument("output", type=Path, help="путь к файлу отчёта (.md)")
    parser.add_argument("--json", type=Path, default=None, help="путь к отчёту JSON (по умолчанию рядом с .md)")
    parser.add_argument("--no-json", action="store_true", help="не записывать отчёт JSON")
    parser.add_argument("--no-cache", action="store_true", help="не использовать кэш этапов")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="профиль вызовов модели")
    parser.add_argument("--rules-only", action="store_true",
                        help="проверить только правилами, без модели (ключ не нужен; итог не определён)")
    args = parser.parse_args(argv)
    if _same_file(args.input, args.output) or (args.json is not None and _same_file(args.input, args.json)):
        print("Ошибка: путь отчёта совпадает с входным документом; укажите другой файл отчёта.", file=sys.stderr)
        return 2
    if args.output.is_dir():
        print(f"Ошибка: {args.output} — это папка; укажите путь к файлу отчёта .md.", file=sys.stderr)
        return 2

    load_dotenv()
    try:
        settings = Settings.from_env(require_key=not args.rules_only)
        overrides = {}
        if args.rules_only:
            overrides["cache_enabled"] = False  # a cached model answer would turn this into a model run
        if args.profile:
            overrides["profile_name"] = args.profile
        if args.no_cache:
            overrides["cache_enabled"] = False
        if overrides:
            settings = settings.with_overrides(**overrides)
        from validator.pipeline import run_pipeline

        client = None
        if args.rules_only:
            from validator.llm import DisabledClient

            client = DisabledClient(settings)
            print(f"Проверка {args.input} только правилами, без модели", file=sys.stderr, flush=True)
        else:
            print(f"Проверка {args.input} (профиль {settings.profile.name})", file=sys.stderr, flush=True)
        report = run_pipeline(args.input, settings, on_event=_progress, client=client)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"Ошибка модели: {exc}", file=sys.stderr)
        return 2

    json_path = _json_path(args)
    try:
        # JSON first: if rendering fails, the paid run is still saved
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_render(report), encoding="utf-8")
    except OSError as exc:
        saved = f" Отчёт JSON сохранён: {json_path}." if json_path is not None and json_path.exists() else ""
        print(f"Ошибка записи отчёта: {exc}.{saved}", file=sys.stderr)
        return 2

    run = report.run
    tokens_in = sum(s.input_tokens for s in run.stages)
    tokens_out = sum(s.output_tokens for s in run.stages)
    print(f"Вызовов модели: {run.llm_calls}; токены: вход {tokens_in}, выход {tokens_out}; "
          f"длительность {run.duration_seconds:.0f} с", file=sys.stderr)
    print(f"Отчёт записан: {args.output}" + (f" и {json_path}" if json_path else ""))
    if run.status != RunStatus.COMPLETE:
        print(f"Отчёт неполный ({run.status.value}): {'; '.join(run.warnings[:3])}", file=sys.stderr)
        return 1
    return 0
