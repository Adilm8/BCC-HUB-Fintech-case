"""FastAPI application factory and HTTP routes for the local web demo."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from validator.config import PROFILES, PROVIDERS, Settings
from validator.llm import LLMClient
from validator.pipeline import agent_version
from validator.web.jobs import (
    ConfigurationError,
    JobBusyError,
    JobManager,
    JobNotFoundError,
    JobNotReadyError,
)
from validator.web.schemas import HealthDTO, RunCreatedDTO, RunStatusDTO

ALLOWED_EXTENSIONS = frozenset({".md", ".txt", ".pdf", ".docx"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_READ_CHUNK_BYTES = 64 * 1024

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCUMENT_PATH = REPO_ROOT / "docs" / "requirements.md"

_UNSAFE_FILENAME_RE = re.compile(r"[^\w.-]+", re.UNICODE)


def _default_provider_name() -> str:
    return next(iter(PROVIDERS))


def _health_from_env() -> tuple[str, str, bool]:
    """Inspect server environment to determine configured provider, model, and key presence."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower() or _default_provider_name()
    spec = PROVIDERS.get(provider)
    if spec is None:
        return provider, "", False
    key_var, model_var, default_model, _fallback_var = spec
    model = os.getenv(model_var, "").strip() or default_model
    key_configured = bool(os.getenv(key_var, "").strip())
    return provider, model, key_configured


def _safe_display_filename(raw_name: str | None) -> str:
    if not raw_name:
        return ""
    # Handle both POSIX and Windows separators sent by browsers.
    normalized = raw_name.replace("\\", "/")
    return Path(normalized).name.strip()


def _download_filename(document_name: str, suffix: str) -> tuple[str, str]:
    """Return (ascii_fallback, utf8_filename) for RFC 5987 Content-Disposition headers."""
    stem = Path(document_name).stem or "report"
    cleaned = _UNSAFE_FILENAME_RE.sub("_", stem).strip("._") or "report"
    utf8_name = f"report_{cleaned}{suffix}"
    ascii_stem = "".join(ch for ch in cleaned if ch.isascii() and (ch.isalnum() or ch in "._-"))
    ascii_name = f"report_{ascii_stem}{suffix}" if ascii_stem else f"report{suffix}"
    return ascii_name, utf8_name


def _content_disposition(document_name: str, suffix: str) -> str:
    ascii_name, utf8_name = _download_filename(document_name, suffix)
    encoded = quote(utf8_name, safe="")
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'


def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an uploaded file in bounded chunks, aborting once size exceeds max_bytes."""
    chunks: list[bytes] = []
    total = 0
    stream = upload.file
    while True:
        chunk = stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="Размер файла превышает ограничение 10 МБ.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_profile(profile: str | None) -> str | None:
    if profile is None or not profile.strip():
        return None
    normalized = profile.strip().lower()
    if normalized not in PROFILES:
        allowed = ", ".join(sorted(PROFILES))
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный профиль проверки «{profile}». Допустимы: {allowed}.",
        )
    return normalized


def create_app(
    settings: Settings | None = None,
    client: LLMClient | None = None,
) -> FastAPI:
    """Create a configured FastAPI application over the core validation pipeline."""
    version_str = agent_version()

    def resolve_settings() -> Settings:
        if settings is not None:
            return settings
        try:
            return Settings.from_env()
        except ValueError:
            if client is not None:
                return Settings(
                    provider=getattr(client, "provider", "test"),
                    api_key="injected-client",
                    model=getattr(client, "model", "test-model"),
                    cache_enabled=False,
                )
            raise

    manager = JobManager(settings_resolver=resolve_settings, client=client)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            manager.shutdown(wait=False)

    app = FastAPI(
        title="Проверка требований",
        version=version_str,
        lifespan=lifespan,
    )
    app.state.job_manager = manager

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            media_type="text/html; charset=utf-8",
        )

    @app.get("/api/health", response_model=HealthDTO)
    def health() -> HealthDTO:
        if settings is not None:
            return HealthDTO(
                status="ok",
                version=version_str,
                provider=settings.provider,
                model=settings.model,
                key_configured=bool(settings.api_key and settings.api_key.strip()),
            )
        provider, model, key_configured = _health_from_env()
        if not key_configured and client is not None:
            provider = getattr(client, "provider", provider)
            model = getattr(client, "model", model)
            key_configured = True
        return HealthDTO(
            status="ok",
            version=version_str,
            provider=provider,
            model=model,
            key_configured=key_configured,
        )

    @app.post("/api/runs", response_model=RunCreatedDTO, status_code=202)
    def create_run(
        file: Annotated[UploadFile, File()],
        profile: Annotated[str | None, Form()] = None,
    ) -> RunCreatedDTO:
        try:
            validated_profile = _validate_profile(profile)
            display_name = _safe_display_filename(file.filename)
            suffix = Path(display_name).suffix.lower()
            if not display_name or suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="Неподдерживаемый формат файла. Допустимы: .md, .txt, .pdf, .docx.",
                )

            payload = _read_upload_limited(file, MAX_UPLOAD_BYTES)
            if not payload or (suffix in {".md", ".txt"} and not payload.strip()):
                raise HTTPException(
                    status_code=400,
                    detail="Загруженный файл пуст.",
                )
        finally:
            file.file.close()

        temp_dir = Path(tempfile.mkdtemp(prefix="reqval_upload_"))
        temp_file = temp_dir / f"{uuid.uuid4().hex}{suffix}"
        try:
            temp_file.write_bytes(payload)
            return manager.submit(
                temp_file,
                display_name,
                profile=validated_profile,
                cleanup_dir=temp_dir,
            )
        except JobBusyError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ConfigurationError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    @app.post("/api/runs/sample", response_model=RunCreatedDTO, status_code=202)
    def create_sample_run(
        profile: str | None = Query(default=None),
    ) -> RunCreatedDTO:
        validated_profile = _validate_profile(profile)
        if not SAMPLE_DOCUMENT_PATH.is_file():
            raise HTTPException(
                status_code=404,
                detail="Встроенный пример документа не найден.",
            )
        try:
            return manager.submit(
                SAMPLE_DOCUMENT_PATH,
                SAMPLE_DOCUMENT_PATH.name,
                profile=validated_profile,
                cleanup_dir=None,
            )
        except JobBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}", response_model=RunStatusDTO)
    def get_run_status(run_id: str) -> RunStatusDTO:
        try:
            return manager.get_status(run_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Проверка не найдена.") from exc

    @app.get("/api/runs/{run_id}/report.json")
    def get_report_json(
        run_id: str,
        download: bool = Query(default=False),
    ) -> Response:
        try:
            report, _markdown = manager.get_report(run_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Проверка не найдена.") from exc
        except JobNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        headers: dict[str, str] = {}
        if download:
            headers["Content-Disposition"] = _content_disposition(
                report.run.document_name,
                ".json",
            )
        return Response(
            content=report.model_dump_json(indent=2),
            media_type="application/json; charset=utf-8",
            headers=headers,
        )

    @app.get("/api/runs/{run_id}/report.md")
    def get_report_markdown(
        run_id: str,
        download: bool = Query(default=False),
    ) -> Response:
        try:
            report, markdown = manager.get_report(run_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Проверка не найдена.") from exc
        except JobNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        headers: dict[str, str] = {}
        if download:
            headers["Content-Disposition"] = _content_disposition(
                report.run.document_name,
                ".md",
            )
        return Response(
            content=markdown,
            media_type="text/markdown; charset=utf-8",
            headers=headers,
        )

    return app
