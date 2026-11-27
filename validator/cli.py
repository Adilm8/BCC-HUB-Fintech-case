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
    from validator.report.markdown import render_markdown

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
    parser.add_argument("input", nargs="?", type=Path,
                        help="путь к документу требований (.md, .txt, .pdf, .docx)")
    parser.add_argument("output", nargs="?", type=Path, help="путь к файлу отчёта (.md)")
    parser.add_argument("--json", type=Path, default=None, help="путь к отчёту JSON (по умолчанию рядом с .md)")
    parser.add_argument("--no-json", action="store_true", help="не записывать отчёт JSON")
    parser.add_argument("--no-cache", action="store_true", help="не использовать кэш этапов")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="профиль вызовов модели")
    parser.add_argument("--save-trace", type=Path, default=None, metavar="DIR",
                        help="записать все запросы к модели и ответы прогона в папку (без ключей)")
    parser.add_argument("--replay", type=Path, default=None, metavar="DIR",
                        help="повторить прогон по записанной трассе: без ключа и без сети")
    parser.add_argument("--domain", default=None, metavar="NAME|auto|none",
                        help="доменный чек-лист из validator/domains (по умолчанию DOMAIN_PROFILE, иначе auto: "
                             "выбор по ключевым словам документа)")
    parser.add_argument("--rules-only", action="store_true",
                        help="проверить только правилами, без модели (ключ не нужен; итог не определён)")
    parser.add_argument("--questions-template", nargs=2, type=Path, metavar=("REPORT_JSON", "ANSWERS_YAML"),
                        default=None,
                        help="сформировать YAML-шаблон ответов из <report.json> в <answers.yaml>")
    parser.add_argument("--answers", type=Path, default=None,
                        help="путь к YAML-файлу с ответами автора ТЗ (требует --previous)")
    parser.add_argument("--previous", type=Path, default=None,
                        help="путь к предыдущему JSON-отчёту для повторной проверки с --answers")
    args = parser.parse_args(argv)

    if args.questions_template is not None:
        report_json, answers_yaml = args.questions_template
        if _same_file(report_json, answers_yaml):
            print("Ошибка: путь шаблона ответов совпадает с JSON-отчётом; укажите другой файл.", file=sys.stderr)
            return 2
        from validator.answers import write_questions_template

        try:
            count = write_questions_template(report_json, answers_yaml)
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            return 2
        print(f"Шаблон ответов записан: {answers_yaml} (вопросов: {count})")
        return 0

    if args.input is None or args.output is None:
        print("Ошибка: укажите путь к документу требований и путь к файлу отчёта (.md), "
              "либо используйте --questions-template <report.json> <answers.yaml>.", file=sys.stderr)
        return 2

    if args.answers is not None and args.previous is None:
        print("Ошибка: вместе с --answers необходимо указать --previous <report.json>.", file=sys.stderr)
        return 2
    if args.previous is not None and args.answers is None:
        print("Ошибка: флаг --previous используется только вместе с --answers <answers.yaml>.", file=sys.stderr)
        return 2
    if args.answers is not None and args.rules_only:
        print("Ошибка: --answers нельзя совмещать с --rules-only.", file=sys.stderr)
        return 2

    if _same_file(args.input, args.output) or (args.json is not None and _same_file(args.input, args.json)):
        print("Ошибка: путь отчёта совпадает с входным документом; укажите другой файл отчёта.", file=sys.stderr)
        return 2
    if args.output.is_dir():
        print(f"Ошибка: {args.output} — это папка; укажите путь к файлу отчёта .md.", file=sys.stderr)
        return 2

    if sum(bool(x) for x in (args.save_trace, args.replay, args.rules_only)) > 1:
        print("Ошибка: --save-trace, --replay и --rules-only нельзя сочетать.", file=sys.stderr)
        return 2

    json_path = _json_path(args)
    if args.previous is not None and (
        _same_file(args.previous, args.output)
        or (json_path is not None and _same_file(args.previous, json_path))
    ):
        print("Ошибка: путь нового отчёта совпадает с --previous; укажите другой выходной файл.", file=sys.stderr)
        return 2

    load_dotenv()
    try:
        offline = args.rules_only or args.replay is not None
        settings = Settings.from_env(require_key=not offline)
        overrides = {}
        if args.rules_only or args.replay is not None or args.save_trace is not None:
            # a cached model answer would bypass the trace or turn a rules-only run into a model run
            overrides["cache_enabled"] = False
        replay = None
        if args.replay is not None:
            from validator.config import PROVIDERS
            from validator.trace import ReplayClient, read_manifest, restore_prompt_env

            replay = ReplayClient(args.replay)
            m = read_manifest(args.replay)
            restore_prompt_env(m)
            overrides.update(model=m.get("model", settings.model), fallback_models=tuple(m.get("fallback_models", ())),
                             profile_name=m.get("profile", settings.profile_name),
                             max_context_chars=int(m.get("max_context_chars", settings.max_context_chars)),
                             domain=m.get("domain", ""))
            if replay.provider in PROVIDERS:
                overrides["provider"] = replay.provider
        if args.profile:
            overrides["profile_name"] = args.profile
        if args.domain is not None:
            overrides["domain"] = args.domain.strip().lower()
        if args.no_cache:
            overrides["cache_enabled"] = False
        if overrides:
            settings = settings.with_overrides(**overrides)
        from validator.pipeline import run_pipeline

        previous_report = None
        author_answers = None
        answer_warnings: list[str] = []
        if args.answers is not None and args.previous is not None:
            from validator.answers import load_answers, load_previous_report

            previous_report = load_previous_report(args.previous)
            author_answers, answer_warnings = load_answers(args.answers, previous_report)
            for warning in answer_warnings:
                print(f"Предупреждение: {warning}", file=sys.stderr, flush=True)

        client = None
        if replay is not None:
            client = replay
            print(f"Повтор прогона {args.input} по трассе {args.replay}", file=sys.stderr, flush=True)
        elif args.save_trace is not None:
            from validator.llm import make_client
            from validator.trace import TracingClient

            client = TracingClient(make_client(settings), args.save_trace, settings)
            print(f"Проверка {args.input} (профиль {settings.profile.name}), трасса: {args.save_trace}",
                  file=sys.stderr, flush=True)
        elif args.rules_only:
            from validator.llm import DisabledClient

            client = DisabledClient(settings)
            print(f"Проверка {args.input} только правилами, без модели", file=sys.stderr, flush=True)
        elif author_answers is not None:
            print(
                f"Уточняющая проверка {args.input} с ответами {args.answers} (профиль {settings.profile.name})",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"Проверка {args.input} (профиль {settings.profile.name})", file=sys.stderr, flush=True)
        report = run_pipeline(
            args.input,
            settings,
            on_event=_progress,
            client=client,
            previous_report=previous_report,
            answers=author_answers,
            previous_report_name=args.previous.name if args.previous is not None else "",
        )
        for warning in answer_warnings:
            if warning not in report.run.warnings:
                report.run.warnings.append(warning)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"Ошибка модели: {exc}", file=sys.stderr)
        return 2

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
