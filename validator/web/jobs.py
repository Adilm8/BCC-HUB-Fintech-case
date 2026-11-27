"""In-memory single-worker job manager wrapping the core validation pipeline."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from validator.config import PROFILES, Settings
from validator.llm import LLMClient, LLMError
from validator.models import STAGE_TITLES, Report, StageEvent, StageKey, StageState, StageStatus
from validator.pipeline import run_pipeline
from validator.report.markdown import render_markdown
from validator.web.schemas import RunCreatedDTO, RunStatusDTO

_TERMINAL_STAGE_STATES = frozenset({
    StageState.DONE,
    StageState.FAILED,
    StageState.SKIPPED,
    StageState.CACHED,
})
_MAX_STORED_JOBS = 20


class JobBusyError(RuntimeError):
    """Raised when a new run is requested while another run is active."""


class JobNotFoundError(KeyError):
    """Raised when a requested run identifier does not exist."""


class JobNotReadyError(RuntimeError):
    """Raised when a report is requested before a run finishes successfully."""


class ConfigurationError(RuntimeError):
    """Raised when server settings cannot be resolved for a run."""


@dataclass
class _Job:
    run_id: str
    document_name: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    stages: dict[StageKey, StageStatus] = field(default_factory=dict)
    stage_t0: dict[StageKey, float] = field(default_factory=dict)
    stage_progress: dict[StageKey, tuple[int, int]] = field(default_factory=dict)
    report: Report | None = None
    markdown: str | None = None
    error: str | None = None
    cleanup_dir: Path | None = None


def _initial_stages() -> dict[StageKey, StageStatus]:
    return {
        key: StageStatus(key=key, title=STAGE_TITLES[key], state=StageState.PENDING)
        for key in StageKey
    }


def _sanitize_error(message: str, source_path: Path, document_name: str) -> str:
    """Replace internal temporary filesystem paths with the user-visible document name."""
    cleaned = message
    for candidate in (str(source_path), str(source_path.parent), source_path.name):
        if candidate and candidate in cleaned:
            cleaned = cleaned.replace(candidate, document_name)
    return cleaned


class JobManager:
    """Runs one validation job at a time on a background worker thread."""

    def __init__(
        self,
        settings_resolver: Callable[[], Settings],
        client: LLMClient | None = None,
    ) -> None:
        self._settings_resolver = settings_resolver
        self._client = client
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="validator-web")
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._job_order: list[str] = []
        self._active_run_id: str | None = None

    def submit(
        self,
        source_path: Path,
        document_name: str,
        *,
        profile: str | None = None,
        cleanup_dir: Path | None = None,
    ) -> RunCreatedDTO:
        """Validate configuration, enforce single-run concurrency, and queue the job."""
        effective_settings = self._resolve_settings(profile)

        with self._lock:
            if self._active_run_id is not None:
                self._cleanup(cleanup_dir)
                raise JobBusyError("Уже выполняется другая проверка. Дождитесь её завершения.")

            run_id = uuid.uuid4().hex[:16]
            job = _Job(
                run_id=run_id,
                document_name=document_name,
                status="queued",
                stages=_initial_stages(),
                cleanup_dir=cleanup_dir,
            )
            self._jobs[run_id] = job
            self._job_order.append(run_id)
            self._active_run_id = run_id
            self._evict_old_jobs_locked()

            try:
                self._executor.submit(self._execute, job, source_path, effective_settings)
            except Exception:
                self._active_run_id = None
                job.status = "failed"
                job.error = "Не удалось поставить задачу в очередь."
                self._cleanup(cleanup_dir)
                job.cleanup_dir = None
                raise

        return RunCreatedDTO(run_id=run_id, status="queued")

    def get_status(self, run_id: str) -> RunStatusDTO:
        """Return a thread-safe progress snapshot for the specified run."""
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                raise JobNotFoundError(run_id)
            stages = [job.stages[key].model_copy() for key in StageKey]
            progress = self._compute_progress_locked(job, stages)
            return RunStatusDTO(
                run_id=job.run_id,
                status=job.status,
                document_name=job.document_name,
                stages=stages,
                progress=progress,
                report_status=job.report.run.status if job.report is not None else None,
                summary=job.report.summary if job.report is not None else None,
                verdict=job.report.verdict if job.report is not None else None,
                error=job.error,
            )

    def get_report(self, run_id: str) -> tuple[Report, str]:
        """Return the finished Report and its rendered Markdown, or raise if unavailable."""
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                raise JobNotFoundError(run_id)
            if job.status in ("queued", "running"):
                raise JobNotReadyError("Отчёт ещё формируется.")
            if job.status == "failed" or job.report is None or job.markdown is None:
                detail = job.error or "Проверка завершилась с ошибкой."
                raise JobNotReadyError(detail)
            return job.report, job.markdown

    def shutdown(self, wait: bool = False) -> None:
        """Stop the worker executor and clean up any leftover temporary upload directories."""
        self._executor.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            dirs = [job.cleanup_dir for job in self._jobs.values() if job.cleanup_dir is not None]
            for job in self._jobs.values():
                job.cleanup_dir = None
        for directory in dirs:
            self._cleanup(directory)

    def _resolve_settings(self, profile: str | None) -> Settings:
        try:
            base = self._settings_resolver()
        except ValueError as exc:
            raise ConfigurationError(
                "Ключ доступа не настроен на сервере. Проверьте настройки окружения."
            ) from exc

        if profile:
            normalized = profile.strip().lower()
            if normalized not in PROFILES:
                raise ValueError(
                    f"Неизвестный профиль проверки. Допустимы: {', '.join(sorted(PROFILES))}."
                )
            return base.with_overrides(profile_name=normalized)
        return base

    def _execute(self, job: _Job, source_path: Path, settings: Settings) -> None:
        with self._lock:
            job.status = "running"

        def on_event(event: StageEvent) -> None:
            now = time.monotonic()
            with self._lock:
                current = job.stages.get(event.stage)
                if current is None:
                    return
                if event.state == StageState.RUNNING:
                    if event.stage not in job.stage_t0:
                        job.stage_t0[event.stage] = now
                    if event.total and event.done is not None:
                        job.stage_progress[event.stage] = (event.done, event.total)
                        detail = f"батчей: {event.done} из {event.total}"
                    else:
                        detail = event.message or current.detail
                    job.stages[event.stage] = current.model_copy(
                        update={"state": event.state, "detail": detail}
                    )
                else:
                    seconds = current.seconds
                    if event.stage in job.stage_t0:
                        seconds = round(now - job.stage_t0[event.stage], 2)
                    job.stage_progress.pop(event.stage, None)
                    job.stages[event.stage] = current.model_copy(
                        update={
                            "state": event.state,
                            "detail": event.message if event.message else current.detail,
                            "seconds": seconds,
                        }
                    )

        try:
            report = run_pipeline(source_path, settings, on_event=on_event, client=self._client)
            report = self._normalize_report_document_name(report, source_path, job.document_name)
            markdown = render_markdown(report)
            with self._lock:
                for st in report.run.stages:
                    job.stages[st.key] = st
                job.stage_progress.clear()
                job.report = report
                job.markdown = markdown
                job.status = "done"
        except (ValueError, FileNotFoundError) as exc:
            self._fail_job(job, _sanitize_error(str(exc), source_path, job.document_name))
        except LLMError as exc:
            if exc.kind == "auth":
                msg = "Ключ доступа отклонён провайдером. Проверьте настройки сервера."
            else:
                msg = _sanitize_error(str(exc), source_path, job.document_name)
            self._fail_job(job, msg)
        except Exception as exc:  # noqa: BLE001 - isolate background failure into job state
            self._fail_job(job, f"Внутренняя ошибка при проверке документа ({type(exc).__name__}).")
        finally:
            cleanup_dir: Path | None = None
            with self._lock:
                if self._active_run_id == job.run_id:
                    self._active_run_id = None
                cleanup_dir = job.cleanup_dir
                job.cleanup_dir = None
            self._cleanup(cleanup_dir)

    @staticmethod
    def _normalize_report_document_name(
        report: Report,
        source_path: Path,
        document_name: str,
    ) -> Report:
        """Restore the original uploaded filename inside RunInfo when run on a temp file."""
        updates: dict[str, str] = {}
        if report.run.document_name == source_path.name and source_path.name != document_name:
            updates["document_name"] = document_name
        if report.run.document_title == source_path.stem and source_path.stem != Path(document_name).stem:
            updates["document_title"] = Path(document_name).stem
        if not updates:
            return report
        return report.model_copy(update={"run": report.run.model_copy(update=updates)})

    def _fail_job(self, job: _Job, message: str) -> None:
        now = time.monotonic()
        with self._lock:
            for key, st in list(job.stages.items()):
                if st.state == StageState.RUNNING:
                    seconds = round(now - job.stage_t0[key], 2) if key in job.stage_t0 else st.seconds
                    job.stages[key] = st.model_copy(
                        update={"state": StageState.FAILED, "detail": message, "seconds": seconds}
                    )
            job.stage_progress.clear()
            job.error = message
            job.status = "failed"

    @staticmethod
    def _compute_progress_locked(job: _Job, stages: list[StageStatus]) -> float:
        if job.status == "done":
            return 1.0
        total_stages = len(StageKey)
        if total_stages == 0:
            return 0.0
        completed = sum(1 for st in stages if st.state in _TERMINAL_STAGE_STATES)
        partial = 0.0
        for key, (done_count, total_count) in job.stage_progress.items():
            if total_count > 0 and job.stages[key].state == StageState.RUNNING:
                partial += min(max(done_count, 0) / total_count, 0.95)
        return round(min((completed + partial) / total_stages, 1.0), 3)

    def _evict_old_jobs_locked(self) -> None:
        while len(self._jobs) > _MAX_STORED_JOBS and self._job_order:
            oldest = self._job_order[0]
            if oldest == self._active_run_id:
                break
            self._job_order.pop(0)
            removed = self._jobs.pop(oldest, None)
            if removed and removed.cleanup_dir is not None:
                self._cleanup(removed.cleanup_dir)

    @staticmethod
    def _cleanup(directory: Path | None) -> None:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
