"""Stage order, worker pool, call budget, run deadline, stage cache and partial reports."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from validator import __version__
from validator.config import Settings
from validator.llm import LLMClient, LLMError, make_client
from validator.models import (
    STAGE_TITLES,
    STANDARDS,
    BlockKind,
    CoverageRow,
    Criterion,
    Document,
    FindingSource,
    PrecheckResult,
    RawDocument,
    Report,
    ReqKind,
    Requirement,
    RunInfo,
    RunStatus,
    SetCheck,
    Severity,
    StageEvent,
    StageKey,
    StageState,
    StageStatus,
)
from validator.pipeline import postprocess as pp
from validator.pipeline.prompts.io import LLMCheck, LLMExtraction, LLMReview, LLMSetCheck
from validator.pipeline.quotes import QuoteIndex
from validator.pipeline.stages import StageContext
from validator.pipeline.stages.check import make_batches, run_check_batch
from validator.pipeline.stages.extract import run_extract
from validator.pipeline.stages.review import run_review
from validator.pipeline.stages.setcheck import run_merged, run_s1, run_s2
from validator.pipeline.structured import Budget, BudgetExceeded, StageCache, StructuredResult
from validator.pipeline.verdict import build_questions, build_summary, build_verdict

LLM_STAGES = (StageKey.EXTRACT, StageKey.CHECK, StageKey.SET_CHECK, StageKey.REVIEW)
FATAL_KINDS = {"quota_daily", "deadline", "auth", "network"}
TOPUP_LIMIT = 40
LONG_DOC_DETAIL = "документ слишком длинный для одного вызова: нарезка правилами"

REASONS = {
    "quota_daily": "исчерпана дневная квота модели",
    "rate_limit": "превышен лимит запросов в минуту",
    "timeout": "модель не ответила вовремя",
    "server": "сервис модели недоступен",
    "bad_request": "API модели отклонило запрос",
    "invalid_json": "модель вернула ответ, который не удалось разобрать",
    "network": "нет связи с API модели (проверьте интернет или прокси)",
    "refusal": "модель отказалась отвечать",
    "deadline": "исчерпан предел времени прогона",
    "skipped": "пропущено из-за нехватки времени",
    "other": "ошибка вызова модели",
}


def describe_error(exc: BaseException) -> str:
    """Generic reason plus the provider's own short message (no keys), and a hint where one helps."""
    from validator.llm import short_error

    kind = getattr(exc, "kind", "other")
    reason = REASONS.get(kind, REASONS["other"])
    if isinstance(exc, BudgetExceeded):
        return "исчерпан бюджет вызовов модели"
    detail = short_error(exc, 160)
    low = detail.lower()
    if kind == "bad_request" and ("not_found" in low or "not found" in low or re.search(r"(?<!\d)404(?!\d)", low)):
        reason += " (проверьте имя модели: GEMINI_MODEL или ANTHROPIC_MODEL в .env)"
    if "location is not supported" in low:
        reason += " (провайдер недоступен из этого региона)"
    return f"{reason}: {detail}" if detail and detail not in reason else reason


def agent_version() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], capture_output=True, text=True,
                             check=True, cwd=root, timeout=5).stdout.strip()
        return f"{__version__} ({out})" if out else __version__
    except (OSError, subprocess.SubprocessError):
        return __version__


def _load(source: Path | RawDocument) -> RawDocument:
    from validator.pipeline._fallback_structure import to_raw

    if isinstance(source, RawDocument):
        return source
    from validator.loaders import load_document

    return to_raw(load_document(Path(source)), Path(source))


def _structure(raw: RawDocument) -> Document:
    try:
        from validator.structure import parse_structure  # type: ignore[import-not-found]
    except ImportError:
        from validator.pipeline._fallback_structure import parse_structure
    return parse_structure(raw)


def _prechecks(doc: Document) -> PrecheckResult:
    try:
        from validator.rules import run_prechecks  # type: ignore[import-not-found]
    except ImportError:
        return PrecheckResult()
    return run_prechecks(doc)


class _Run:
    def __init__(self, settings: Settings, on_event: Callable[[StageEvent], None] | None):
        self.settings = settings
        self.on_event = on_event
        self.lock = threading.Lock()
        self.stages = {k: StageStatus(key=k, title=STAGE_TITLES[k], state=StageState.PENDING) for k in StageKey}
        self.models: list[str] = []
        self.warnings: list[str] = []
        self._t0: dict[StageKey, float] = {}
        self._hits: dict[StageKey, list[bool]] = {}
        self.notes: dict[StageKey, list[str]] = {}

    def note(self, stage: str, text: str) -> None:
        """A failed attempt or a model switch: shown as progress and kept for the run warnings."""
        try:
            key = StageKey(stage)
        except ValueError:
            return
        with self.lock:
            bucket = self.notes.setdefault(key, [])
            if text not in bucket:
                bucket.append(text)
        if self.on_event is not None:
            try:
                self.on_event(StageEvent(stage=key, state=StageState.RUNNING, message=text))
            except Exception:  # noqa: BLE001 - a broken listener must not break the run
                return

    def note_warnings(self) -> None:
        for key, notes in self.notes.items():
            if notes:
                shown = "; ".join(notes[:4]) + (f" и ещё {len(notes) - 4}" if len(notes) > 4 else "")
                self.warnings.append(f"{STAGE_TITLES[key]}: {shown}.")

    def emit(self, key: StageKey, state: StageState, message: str = "", done: int | None = None,
             total: int | None = None) -> None:
        with self.lock:
            st = self.stages[key]
            if state == StageState.RUNNING and key not in self._t0:
                self._t0[key] = time.monotonic()
            if state in (StageState.DONE, StageState.FAILED, StageState.CACHED, StageState.SKIPPED):
                if key in self._t0:
                    st.seconds = round(time.monotonic() - self._t0[key], 2)
                if message:
                    st.detail = message
            st.state = state
        if self.on_event is not None:
            try:
                self.on_event(StageEvent(stage=key, state=state, message=message, done=done, total=total))
            except Exception as exc:  # noqa: BLE001 - a broken listener must not break the run
                self.warnings.append(f"Ошибка обработчика событий: {type(exc).__name__}")

    def account(self, key: StageKey, res: StructuredResult | None, responses=None) -> None:
        with self.lock:
            st = self.stages[key]
            if res is not None:
                st.llm_calls += res.calls
                st.input_tokens += res.input_tokens
                st.output_tokens += res.output_tokens
                self._hits.setdefault(key, []).append(res.cached)
                if res.model:
                    st.model = res.model
                    if res.model not in self.models:
                        self.models.append(res.model)
            for r in responses or []:
                st.llm_calls += 1
                st.input_tokens += r.input_tokens or 0
                st.output_tokens += r.output_tokens or 0

    def finished_state(self, key: StageKey) -> StageState:
        hits = self._hits.get(key, [])
        return StageState.CACHED if hits and all(hits) else StageState.DONE

    def fail(self, key: StageKey, exc: BaseException) -> str:
        reason = describe_error(exc)
        self.account(key, None, getattr(exc, "responses", None))
        self.emit(key, StageState.FAILED, reason)
        return reason


def run_pipeline(source: Path | RawDocument, settings: Settings, *,
                 on_event: Callable[[StageEvent], None] | None = None,
                 client: LLMClient | None = None) -> Report:
    started_at = datetime.now().astimezone()
    t_start = time.monotonic()
    run = _Run(settings, on_event)
    profile = settings.profile

    run.emit(StageKey.LOAD, StageState.RUNNING)
    raw = _load(source)
    run.emit(StageKey.LOAD, StageState.DONE, f"формат {raw.format}, {len(raw.text)} символов")
    run.warnings.extend(raw.warnings)

    run.emit(StageKey.STRUCTURE, StageState.RUNNING)
    doc = _structure(raw)
    run.emit(StageKey.STRUCTURE, StageState.DONE, f"разделов: {len(doc.sections)}, блоков: {len(doc.blocks)}")
    for w in doc.warnings:
        if w not in run.warnings:
            run.warnings.append(w)

    run.emit(StageKey.PRECHECKS, StageState.RUNNING)
    try:
        pre = _prechecks(doc)
        run.emit(StageKey.PRECHECKS, StageState.DONE,
                 f"находок: {len(pre.findings)}, подсказок: {len(pre.hints)}, инвентарь: {len(pre.inventory)}")
    except Exception as exc:  # noqa: BLE001 - rules must never stop the run
        pre = PrecheckResult()
        run.emit(StageKey.PRECHECKS, StageState.FAILED, f"правила не выполнены: {type(exc).__name__}")

    images = sorted({h.block_id for h in pre.hints if h.rule == "image_link"}, key=lambda b: b)
    if images and not any("изображ" in w.lower() for w in run.warnings):
        shown = ", ".join(images[:5]) + (f" и ещё {len(images) - 5}" if len(images) > 5 else "")
        run.warnings.append(f"В документе есть изображения или диаграммы (блоки: {shown}): их содержимое "
                            "не анализировалось.")
    pre = pre.model_copy(update={"hints": [h for h in pre.hints if h.rule != "image_link"]})

    client = client or make_client(settings)
    budget = Budget(profile.max_calls)
    cache = StageCache(Path(settings.cache_dir) if settings.cache_enabled else None)
    run_started = time.monotonic()
    deadline = run_started + settings.run_deadline
    review_reserve = 0.15 * settings.run_deadline if profile.review else 0.0
    ctx = StageContext(client=client, settings=settings, budget=budget, cache=cache, deadline=deadline,
                       review_reserve=review_reserve, note=run.note)
    durations: list[float] = []  # seconds of finished model calls in this run
    index = QuoteIndex(doc)
    stats = pp.Stats()
    fatal: str | None = None

    def fatal_from(exc: BaseException) -> None:
        nonlocal fatal
        kind = getattr(exc, "kind", "other")
        if kind == "auth":
            raise exc
        if kind == "deadline" and time.monotonic() < deadline - 15:
            return  # only the time before the critic's reserve ran out: the critic still gets its slot
        if kind in FATAL_KINDS or isinstance(exc, BudgetExceeded):
            fatal = fatal or REASONS.get(kind, REASONS["other"])

    # ---------------------------------------------------------- extract
    extraction: LLMExtraction | None = None
    run.emit(StageKey.EXTRACT, StageState.RUNNING)
    if doc.char_count > settings.max_context_chars:
        # one call cannot hold the whole document: cut by rules, check batch by batch with their own blocks
        run.emit(StageKey.EXTRACT, StageState.SKIPPED, LONG_DOC_DETAIL)
        run.warnings.append(
            f"Документ длиннее {settings.max_context_chars} символов: нарезка выполнена правилами по блокам, "
            "проверка идёт батчами без полного текста документа.")
    else:
        try:
            t_extract = time.monotonic()
            res = run_extract(ctx, doc)
            durations.append(time.monotonic() - t_extract)
            run.account(StageKey.EXTRACT, res)
            extraction = res.data
            stats.dropped_schema_items += res.dropped_items
            if res.truncated:
                run.warnings.append("Ответ нарезчика обрезан: часть блоков отнесена к требованиям правилами.")
        except LLMError as exc:
            fatal_from(exc)
            reason = run.fail(StageKey.EXTRACT, exc)
            run.warnings.append(f"Нарезка выполнена правилами: {reason}.")
    ext = pp.build_requirements(doc, extraction, index)
    reqs = ext.requirements
    if extraction is not None:
        detail = f"требований: {len(reqs)}, служебных блоков: {len(ext.skipped)}"
        if ext.added_by_code:
            detail += f", добавлено кодом: {ext.added_by_code}"
        run.emit(StageKey.EXTRACT, run.finished_state(StageKey.EXTRACT), detail)

    builder = pp.CandidateBuilder(doc, reqs, index, stats)
    candidates: list[pp.Candidate] = builder.from_rules(pre.findings)
    counts: dict[str, int] = {"rules": len(candidates)}
    coverage: list[CoverageRow] = []
    checked: set[str] = set()
    check_ok = False
    set_ok = False

    # ---------------------------------------------------------- wave: check batches + set checks
    batches = make_batches(reqs, profile)
    if fatal or not reqs:
        for key in (StageKey.CHECK, StageKey.SET_CHECK):
            run.emit(key, StageState.SKIPPED, fatal or "нет требований для проверки")
    else:
        run.emit(StageKey.CHECK, StageState.RUNNING, total=len(batches), done=0)
        run.emit(StageKey.SET_CHECK, StageState.RUNNING)
        def typical_call() -> float:
            done = sorted(durations)
            return max(45.0, 1.2 * done[len(done) // 2]) if done else 90.0

        def timed(fn: Callable[[], StructuredResult], *, optional: bool) -> Callable[[], StructuredResult]:
            def task() -> StructuredResult:
                left = deadline - review_reserve - time.monotonic()
                need = typical_call()
                if optional and left < need:
                    raise LLMError(f"пропущено: осталось {max(left, 0):.0f} с, обычный вызов занимает около "
                                   f"{need:.0f} с", "skipped")
                t0 = time.monotonic()
                res = fn()
                durations.append(time.monotonic() - t0)
                return res
            return task

        check_tasks = [(f"check:{i}", timed(lambda b=b: run_check_batch(ctx, doc, b, reqs, pre.hints), optional=True))
                       for i, b in enumerate(batches)]
        if profile.merge_set_checks:
            set_tasks = [("set:merged", timed(lambda: run_merged(ctx, doc, reqs, pre.inventory, ext.skipped),
                                              optional=False))]
        else:
            set_tasks = [("set:s1", timed(lambda: run_s1(ctx, doc, reqs, pre.inventory), optional=False)),
                         ("set:s2", timed(lambda: run_s2(ctx, doc, reqs, ext.skipped), optional=False))]
        workers = 1 if getattr(client, "provider", "") == "gemini" else max(1, profile.parallel)
        # one call at a time: the set checks find the critical contradictions, so they go first and the
        # per-requirement batches are the ones dropped when time runs out
        tasks = set_tasks + check_tasks if workers == 1 else check_tasks + set_tasks
        results: dict[str, StructuredResult | BaseException] = {}
        done_batches = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[str, Future] = {name: pool.submit(fn) for name, fn in tasks}
            for name, fut in futures.items():
                try:
                    results[name] = fut.result()
                except LLMError as exc:
                    results[name] = exc
                except Exception as exc:  # noqa: BLE001 - one broken stage must not lose the others
                    results[name] = LLMError(f"{type(exc).__name__}: {exc}", "other")
                if name.startswith("check:"):
                    done_batches += 1
                    run.emit(StageKey.CHECK, StageState.RUNNING, done=done_batches, total=len(batches))

        # check results
        failed_batches: list[list[Requirement]] = []
        failures: list[BaseException] = []
        missing: list[Requirement] = []
        for i, b in enumerate(batches):
            r = results[f"check:{i}"]
            if isinstance(r, BaseException):
                failed_batches.append(b)
                failures.append(r)
                run.account(StageKey.CHECK, None, getattr(r, "responses", None))
                continue
            run.account(StageKey.CHECK, r)
            stats.dropped_schema_items += r.dropped_items
            data: LLMCheck = r.data
            cands, seen = builder.from_check(data.results)
            candidates.extend(cands)
            checked |= seen
            missing.extend(x for x in b if x.id not in seen)
        for exc in failures:
            if getattr(exc, "kind", "") == "auth":
                raise exc
            fatal_from(exc)
        # one top-up call for requirements the model skipped or could not reach
        to_retry = missing + [x for b in failed_batches for x in b]
        reserve = 1 if profile.review else 0
        enough_time = deadline - review_reserve - time.monotonic() > typical_call()
        if to_retry and not fatal and budget.remaining > reserve and enough_time:
            topup = to_retry[:TOPUP_LIMIT]
            try:
                r = run_check_batch(ctx, doc, topup, reqs, pre.hints)
                run.account(StageKey.CHECK, r)
                cands, seen = builder.from_check(r.data.results)
                candidates.extend(cands)
                checked |= seen
            except LLMError as exc:
                run.account(StageKey.CHECK, None, getattr(exc, "responses", None))
                fatal_from(exc)
        check_ok = bool(checked)
        not_checked = [x for x in reqs if x.id not in checked]
        if check_ok:
            n_check = sum(1 for c in candidates if c.source == FindingSource.LLM)
            detail = (f"батчей: {len(batches)}, проверено требований: {len(checked)} из {len(reqs)}, "
                      f"кандидатов: {n_check}")
            counts["check"] = n_check
            run.emit(StageKey.CHECK, run.finished_state(StageKey.CHECK), detail)
            if not_checked:
                run.warnings.append(f"Не проверено требований: {len(not_checked)} "
                                    f"({', '.join(x.id for x in not_checked[:15])}).")
        else:
            reason = describe_error(failures[0]) if failures else \
                "модель не вернула результатов"
            run.emit(StageKey.CHECK, StageState.FAILED, reason)
            run.warnings.append(f"Проверка требований не выполнена: {reason}.")

        # set results
        set_failures = []
        set_names = [n for n, _ in tasks if n.startswith("set:")]
        for name in set_names:
            r = results[name]
            if isinstance(r, BaseException):
                set_failures.append((name, r))
                run.account(StageKey.SET_CHECK, None, getattr(r, "responses", None))
                if getattr(r, "kind", "") == "auth":
                    raise r
                fatal_from(r)
                continue
            run.account(StageKey.SET_CHECK, r)
            stats.dropped_schema_items += r.dropped_items
            sdata: LLMSetCheck = r.data
            set_cands = builder.from_set(sdata.findings)
            counts["set"] = counts.get("set", 0) + len(set_cands)
            candidates.extend(set_cands)
            if name != "set:s1":
                coverage.extend(_coverage(builder, sdata, reqs))
        set_ok = len(set_failures) < len(set_names)
        if not set_failures:
            run.emit(StageKey.SET_CHECK, run.finished_state(StageKey.SET_CHECK),
                     f"вызовов: {len(set_names)}, кандидатов: {counts.get('set', 0)}")
        else:
            reason = describe_error(set_failures[0][1])
            lens = ", ".join("S1" if n == "set:s1" else "S2" if n == "set:s2" else "S1+S2" for n, _ in set_failures)
            run.emit(StageKey.SET_CHECK, StageState.FAILED if not set_ok else StageState.DONE,
                     f"не выполнено ({lens}): {reason}")
            run.warnings.append(f"Проверка набора выполнена не полностью ({lens}): {reason}.")

    derived: list[pp.Candidate] = []
    if not check_ok and pre.hints:
        derived.extend(_hints_as_findings(builder, pre))
    derived.extend(_aggregate_gaps(builder, coverage, candidates))
    counts["derived"] = len(derived)
    candidates.extend(derived)
    _soften_test_section_gaps(candidates, reqs)
    pp.clean_texts(candidates, builder)
    parts = {b.id: b.part for b in doc.blocks}
    pp.apply_guards(candidates, parts)
    candidates = pp.dedup(candidates, stats)
    _soften_test_section_gaps(candidates, reqs)  # a merge may have brought back a higher severity
    stats.merged_before_review = stats.merged
    pp.assign_candidate_ids(candidates)

    # ---------------------------------------------------------- review
    fix_order: list[str] = []
    rationale = ""
    reviewed = False
    if not profile.review:
        run.emit(StageKey.REVIEW, StageState.SKIPPED, "не входит в профиль")
    elif fatal:
        run.emit(StageKey.REVIEW, StageState.SKIPPED, fatal)
    elif not candidates:
        run.emit(StageKey.REVIEW, StageState.SKIPPED, "нет находок для пересмотра")
    else:
        run.emit(StageKey.REVIEW, StageState.RUNNING)
        try:
            r = run_review(ctx, doc, pp.candidates_for_review(candidates))
            run.account(StageKey.REVIEW, r)
            review: LLMReview = r.data
            before = len(candidates)
            candidates, outcome = pp.apply_review(candidates, review, builder, stats)
            counts["review"] = len(outcome.missed)
            reviewed = True
            pp.clean_texts(candidates, builder)
            pp.apply_guards(candidates, parts)  # the critic may not lift capped criteria
            _soften_test_section_gaps(candidates, reqs)
            if outcome.missed:
                pp.clean_texts(outcome.missed, builder)
                pp.apply_guards(outcome.missed, parts)
                candidates = pp.dedup(candidates + outcome.missed, stats)
            fix_order = outcome.fix_first
            rationale = outcome.rationale
            entered = sum(counts.get(k, 0) for k in ("rules", "check", "set", "derived"))
            run.emit(StageKey.REVIEW, run.finished_state(StageKey.REVIEW),
                     f"на пересмотр: {before} из {entered} кандидатов, осталось: {len(candidates)} "
                     "(подробный баланс — в приложении)")
        except LLMError as exc:
            if exc.kind == "auth":
                raise
            reason = run.fail(StageKey.REVIEW, exc)
            run.warnings.append(f"Критический пересмотр не выполнен: {reason}.")

    # ---------------------------------------------------------- assemble
    run.emit(StageKey.ASSEMBLE, StageState.RUNNING)
    candidates = pp.drop_unconfirmed_anchored(candidates, reviewed, stats)
    findings, set_findings = pp.finalize(candidates, reqs)
    by_cid = {c.cid: c.final_id for c in candidates if c.cid}
    critic_order = [by_cid[x] for x in fix_order if x in by_cid]
    reqs = [r.model_copy(update={"check_status": "checked" if r.id in checked else "not_checked"}) for r in reqs]
    questions = build_questions(findings, set_findings)
    summary = build_summary(reqs, findings, set_findings, questions, merged=stats.merged,
                            dropped_unverified=stats.dropped_unverified_quotes,
                            dropped_by_review=stats.dropped_by_review, candidates=counts,
                            dropped_unconfirmed=stats.dropped_unconfirmed)
    verdict = build_verdict(reqs, findings, set_findings, coverage, critic_order=critic_order, rationale=rationale)
    in_reqs = {b for r in reqs for b in r.block_ids}
    skipped_ids = {s.block_id for s in ext.skipped} - in_reqs
    structural = {b.id for b in doc.blocks if b.kind in (BlockKind.HEADING, BlockKind.TABLE_HEADER)}
    structural -= in_reqs | skipped_ids
    summary = summary.model_copy(update={
        "blocks_total": len(doc.blocks), "blocks_in_requirements": len(in_reqs), "blocks_skipped": len(skipped_ids),
        "blocks_structural": len(structural),
        "blocks_other": len(doc.blocks) - len(in_reqs) - len(skipped_ids) - len(structural),
    })
    run.emit(StageKey.ASSEMBLE, StageState.DONE,
             f"нарушений: {len(findings) + len(set_findings)}, вопросов: {len(questions)}")

    run.note_warnings()
    status = _status(run, profile.review)
    if status != RunStatus.COMPLETE:
        run.warnings.insert(0, "Отчёт неполный: часть этапов модели не выполнена. Повторите запуск позже, "
                               "с другой моделью или другим ключом; готовые этапы возьмутся из кэша.")
    finished = datetime.now().astimezone()
    info = RunInfo(
        document_title=doc.title, document_name=doc.name, document_format=doc.format, document_sha256=doc.sha256,
        started_at=started_at, finished_at=finished, duration_seconds=round(time.monotonic() - t_start, 2),
        provider=getattr(client, "provider", settings.provider), models_used=run.models,
        agent_version=agent_version(), profile=profile.name,
        llm_calls=sum(s.llm_calls for s in run.stages.values()), status=status,
        stages=[run.stages[k] for k in StageKey], standards=list(STANDARDS), warnings=run.warnings,
    )
    return Report(run=info, requirements=reqs, skipped_blocks=ext.skipped, findings=findings,
                  set_findings=set_findings, coverage=coverage, questions=questions, summary=summary,
                  verdict=verdict)


def _status(run: _Run, review_in_profile: bool) -> RunStatus:
    ok = {StageState.DONE, StageState.CACHED}
    st = {k: run.stages[k].state for k in LLM_STAGES}
    if st[StageKey.EXTRACT] == StageState.SKIPPED and run.stages[StageKey.EXTRACT].detail == LONG_DOC_DETAIL:
        st[StageKey.EXTRACT] = StageState.DONE  # skipped by design for long documents
    if st[StageKey.EXTRACT] not in ok and st[StageKey.CHECK] not in ok:
        return RunStatus.RULES_ONLY
    needed = [StageKey.EXTRACT, StageKey.CHECK, StageKey.SET_CHECK]
    if review_in_profile and not (st[StageKey.REVIEW] == StageState.SKIPPED
                                  and run.stages[StageKey.REVIEW].detail == "нет находок для пересмотра"):
        needed.append(StageKey.REVIEW)
    if any(st[k] not in ok for k in needed):
        return RunStatus.PARTIAL
    if any("выполнена не полностью" in w or "Не проверено" in w for w in run.warnings):
        return RunStatus.PARTIAL
    return RunStatus.COMPLETE


def _coverage(builder: pp.CandidateBuilder, data: LLMSetCheck, reqs: list[Requirement]) -> list[CoverageRow]:
    kinds = {r.id: r.kind for r in reqs}
    rows: dict[str, CoverageRow] = {}
    for c in data.coverage:
        rid = builder.resolve_req(c.requirement_id)
        # interface details (parameters, examples) are covered through the operation, not row by row
        if rid is None or rid in rows or kinds.get(rid) in (ReqKind.ACCEPTANCE_CRITERION, ReqKind.INTERFACE):
            continue
        acc = [a for a in (builder.resolve_req(x) for x in c.acceptance_ids)
               if a and kinds.get(a) == ReqKind.ACCEPTANCE_CRITERION]
        status = c.status
        if status != "none" and not acc:
            status = "none" if status == "covered" else status
        rows[rid] = CoverageRow(requirement_id=rid, acceptance_ids=acc, status=status, note=c.note.strip())
    order = {r.id: i for i, r in enumerate(reqs)}
    return sorted(rows.values(), key=lambda r: order.get(r.requirement_id, 10**6))


def _aggregate_gaps(builder: pp.CandidateBuilder, coverage: list[CoverageRow],
                    candidates: list[pp.Candidate]) -> list[pp.Candidate]:
    """One aggregated finding for requirements without acceptance criteria, when no finding mentions them."""
    kinds = {r.id: r.kind for r in builder.reqs.values()}
    gap_kinds = {ReqKind.FUNCTIONAL, ReqKind.NONFUNCTIONAL, ReqKind.INTEGRATION, ReqKind.DATA}
    uncovered = [c.requirement_id for c in coverage if c.status == "none" and kinds.get(c.requirement_id) in gap_kinds]
    if not uncovered:
        return []
    mentioned = {r for c in candidates if c.set_check in (SetCheck.ACCEPTANCE_GAP, SetCheck.MISSING_SCENARIO)
                 for r in c.requirement_ids}
    rest = [r for r in uncovered if r not in mentioned]
    if not rest:
        return []
    evidence = [builder.requirement_evidence(r) for r in rest[:3]]
    rest = rest[:10]
    ids = ", ".join(rest)
    return [pp.Candidate(
        criterion=Criterion.SET_VALIDATABLE, severity=Severity.MAJOR, evidence=evidence,
        explanation=f"Для требований {ids} в документе нет ни одного критерия приёмки: результат их реализации "
                    "нельзя принять по согласованным сценариям.",
        recommendation="Добавить критерии приёмки с исходными условиями, действием и ожидаемым результатом "
                       "для каждого из перечисленных требований.",
        question="Какие сценарии приёмки подтверждают выполнение перечисленных требований?",
        options=["Критерии приёмки будут добавлены в документ", "Сценарии будут описаны в тест-кейсах QA"],
        requirement_ids=rest, source=FindingSource.RULE, set_check=SetCheck.ACCEPTANCE_GAP, confirmed=True,
        no_merge=True, addressee="qa",
    )]


def _soften_test_section_gaps(candidates: list[pp.Candidate], reqs: list[Requirement]) -> None:
    """An empty testing section is not blocking when the document has filled acceptance criteria."""
    has_acceptance = any(r.kind == ReqKind.ACCEPTANCE_CRITERION for r in reqs)
    for c in candidates:
        if c.set_check not in (SetCheck.EMPTY_SECTION, None) and c.source != FindingSource.RULE:
            continue
        where = " ".join([*(e.section for e in c.evidence), *(e.quote for e in c.evidence), c.explanation])
        if not re.search(r"тестир", where, re.IGNORECASE):
            continue
        if c.set_check == SetCheck.EMPTY_SECTION or c.source == FindingSource.RULE:
            c.addressee = "qa"
        if has_acceptance and c.severity == Severity.CRITICAL:
            c.severity = Severity.MAJOR  # still protected: the critic cannot lower it further


def _hints_as_findings(builder: pp.CandidateBuilder, pre: PrecheckResult) -> list[pp.Candidate]:
    out = []
    for h in pre.hints[:30]:
        ev = builder.evidence(h.match, None) if len(h.match) >= 12 else None
        if ev is None and h.block_id in builder.blocks:
            b = builder.blocks[h.block_id]
            ev = pp.Evidence(quote=b.text[:300], section=builder.section_of([h.block_id]), block_ids=[h.block_id],
                             requirement_id=builder._block_req.get(h.block_id), quote_status=pp.QuoteStatus.VERIFIED)
        if ev is None or ev.requirement_id is None:
            continue
        out.append(pp.Candidate(
            criterion=Criterion.UNAMBIGUOUS, severity=Severity.MINOR, evidence=[ev],
            explanation=f"{h.message} (фрагмент «{h.match}»).", recommendation="Заменить формулировку измеримой.",
            question=f"Что именно означает «{h.match}» в этом требовании?", options=[],
            requirement_ids=[ev.requirement_id], source=FindingSource.RULE, confirmed=True,
        ))
    return out
