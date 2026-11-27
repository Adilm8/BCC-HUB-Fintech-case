"""Offline integration tests for the local FastAPI web demo using TestClient and FakeLLM."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from tests.fakes import FakeLLM, default_extract
from validator.config import PROVIDERS, Settings
from validator.llm import LLMError
from validator.models import Report, StageKey
from validator.web.app import MAX_UPLOAD_BYTES, create_app
from validator.web.schemas import RunStatusDTO

SAMPLE_MD = """# Бронирование переговорных

## Функциональные требования

- **BR-1.** Сотрудник бронирует переговорную на срок от 15 до 240 минут включительно.
- **BR-2.** Бронь отменяется автоматически, если никто не отметился в течение 10 минут.
- **BR-3.** Система быстро подтверждает бронирование.

## Критерии приёмки

| Условие | Действие | Результат |
| :--- | :--- | :--- |
| Бронь на 9:00 | Нет отметки до 9:20 | Бронь активна |
"""


def _test_settings(**overrides) -> Settings:
    params = {
        "provider": "fake",
        "api_key": "super-secret-test-key",
        "model": "fake-model",
        "profile_name": "economy",
        "cache_enabled": False,
    }
    params.update(overrides)
    return Settings(**params)


def _poll_until_terminal(client: TestClient, run_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    last_data: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        last_data = resp.json()
        if last_data["status"] in ("done", "failed"):
            return last_data
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not finish within {timeout}s; last state: {last_data}")


def test_index_and_static_assets():
    app = create_app(settings=_test_settings(), client=FakeLLM())
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "text/html" in index.headers["content-type"]
        assert "Проверка требований" in index.text
        assert "ISO/IEC/IEEE 29148" in index.text
        assert "IEEE 830" in index.text
        assert 'type="password"' not in index.text

        js = client.get("/static/app.js")
        assert js.status_code == 200
        assert "innerHTML" not in js.text
        assert "outerHTML" not in js.text
        assert "insertAdjacentHTML" not in js.text

        css = client.get("/static/style.css")
        assert css.status_code == 200


def test_health_reports_key_configured_only_without_leaking_secret(monkeypatch):
    app = create_app(settings=_test_settings(api_key="my-private-token-999"), client=FakeLLM())
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["key_configured"] is True
        assert data["provider"] == "fake"
        assert data["model"] == "fake-model"
        assert data["version"]
        assert "my-private-token-999" not in resp.text
        assert "api_key" not in data

    # Unconfigured environment: key_configured must be False and starting a run must return 503.
    for key_var, _model_var, _default_model, _fallback_var in PROVIDERS.values():
        monkeypatch.delenv(key_var, raising=False)
    unconfigured_app = create_app(settings=None, client=None)
    with TestClient(unconfigured_app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["key_configured"] is False

        run_resp = client.post("/api/runs/sample")
        assert run_resp.status_code == 503


def test_sample_run_poll_and_fetch_reports():
    app = create_app(settings=_test_settings(), client=FakeLLM())
    with TestClient(app) as client:
        start = client.post("/api/runs/sample")
        assert start.status_code == 202
        created = start.json()
        run_id = created["run_id"]
        assert created["status"] in ("queued", "running")

        final_data = _poll_until_terminal(client, run_id)
        dto = RunStatusDTO.model_validate(final_data)
        assert dto.status == "done"
        assert dto.document_name == "requirements.md"
        assert dto.progress == 1.0
        assert dto.error is None
        assert dto.report_status == "complete"
        assert dto.summary is not None
        assert dto.verdict is not None
        assert [stage.key for stage in dto.stages] == list(StageKey)

        json_resp = client.get(f"/api/runs/{run_id}/report.json")
        assert json_resp.status_code == 200
        report = Report.model_validate_json(json_resp.text)
        assert report.run.document_name == "requirements.md"
        assert report.requirements

        json_dl = client.get(f"/api/runs/{run_id}/report.json?download=1")
        assert json_dl.status_code == 200
        assert "attachment;" in json_dl.headers.get("content-disposition", "")
        assert ".json" in json_dl.headers["content-disposition"]

        md_resp = client.get(f"/api/runs/{run_id}/report.md")
        assert md_resp.status_code == 200
        assert "text/markdown" in md_resp.headers["content-type"]
        assert "# Отчёт о качестве требований" in md_resp.text
        assert "## Итог" in md_resp.text

        md_dl = client.get(f"/api/runs/{run_id}/report.md?download=1")
        assert md_dl.status_code == 200
        assert "attachment;" in md_dl.headers.get("content-disposition", "")
        assert ".md" in md_dl.headers["content-disposition"]


def test_upload_preserves_cyrillic_filename_and_deletes_temp_dir(monkeypatch):
    created_dirs: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created_dirs.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)

    app = create_app(settings=_test_settings(), client=FakeLLM())
    with TestClient(app) as client:
        resp = client.post(
            "/api/runs",
            files={"file": ("ТЗ_переговорные.md", SAMPLE_MD.encode("utf-8"), "text/markdown")},
            data={"profile": "economy"},
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        final_data = _poll_until_terminal(client, run_id)
        assert final_data["status"] == "done"
        assert final_data["document_name"] == "ТЗ_переговорные.md"

        report = Report.model_validate_json(client.get(f"/api/runs/{run_id}/report.json").text)
        assert report.run.document_name == "ТЗ_переговорные.md"

        md_dl = client.get(f"/api/runs/{run_id}/report.md?download=1")
        assert md_dl.status_code == 200
        assert "attachment;" in md_dl.headers["content-disposition"]
        assert "UTF-8''" in md_dl.headers["content-disposition"]

    assert created_dirs
    assert all(not directory.exists() for directory in created_dirs)


def test_validation_errors_400_and_413():
    app = create_app(settings=_test_settings(), client=FakeLLM())
    with TestClient(app) as client:
        # Unsupported extensions.
        for bad_name in ("script.exe", "page.html", "no_extension"):
            resp = client.post("/api/runs", files={"file": (bad_name, b"hello", "application/octet-stream")})
            assert resp.status_code == 400

        # Empty and whitespace-only text files.
        empty_resp = client.post("/api/runs", files={"file": ("empty.md", b"", "text/markdown")})
        assert empty_resp.status_code == 400

        blank_resp = client.post("/api/runs", files={"file": ("blank.txt", b"   \n\t  ", "text/plain")})
        assert blank_resp.status_code == 400

        # Unknown profile.
        profile_resp = client.post(
            "/api/runs",
            files={"file": ("spec.md", SAMPLE_MD.encode("utf-8"), "text/markdown")},
            data={"profile": "unknown_profile"},
        )
        assert profile_resp.status_code == 400

        # File larger than 10 MB -> 413.
        oversized = b"a" * (MAX_UPLOAD_BYTES + 1)
        big_resp = client.post("/api/runs", files={"file": ("huge.txt", oversized, "text/plain")})
        assert big_resp.status_code == 413


def test_busy_returns_409_and_report_unavailable_until_done():
    started = threading.Event()
    release = threading.Event()

    def slow_extract(prompt: str, schema: dict) -> dict:
        started.set()
        release.wait(timeout=10.0)
        return default_extract(prompt, schema)

    app = create_app(settings=_test_settings(), client=FakeLLM({"extract": slow_extract}))
    with TestClient(app) as client:
        first = client.post("/api/runs/sample")
        assert first.status_code == 202
        run_id = first.json()["run_id"]

        assert started.wait(timeout=5.0), "Background worker did not enter extract stage"

        # Another run cannot start while the first one is active.
        second_sample = client.post("/api/runs/sample")
        assert second_sample.status_code == 409

        second_upload = client.post(
            "/api/runs",
            files={"file": ("spec.md", SAMPLE_MD.encode("utf-8"), "text/markdown")},
        )
        assert second_upload.status_code == 409

        # Reports are not ready yet.
        assert client.get(f"/api/runs/{run_id}/report.json").status_code == 409
        assert client.get(f"/api/runs/{run_id}/report.md").status_code == 409

        # Unblock worker and verify completion + lock release.
        release.set()
        final_data = _poll_until_terminal(client, run_id)
        assert final_data["status"] == "done"

        # Subsequent run succeeds now that worker is idle.
        third = client.post(
            "/api/runs",
            files={"file": ("spec.md", SAMPLE_MD.encode("utf-8"), "text/markdown")},
        )
        assert third.status_code == 202
        third_final = _poll_until_terminal(client, third.json()["run_id"])
        assert third_final["status"] == "done"


def test_unknown_run_id_returns_404():
    app = create_app(settings=_test_settings(), client=FakeLLM())
    with TestClient(app) as client:
        assert client.get("/api/runs/nonexistent").status_code == 404
        assert client.get("/api/runs/nonexistent/report.json").status_code == 404
        assert client.get("/api/runs/nonexistent/report.md").status_code == 404


def test_failed_run_exposes_sanitized_error_and_releases_lock():
    app = create_app(settings=_test_settings(), client=FakeLLM())
    with TestClient(app) as client:
        # Invalid DOCX triggers a loader ValueError inside the worker thread.
        bad = client.post(
            "/api/runs",
            files={"file": ("broken.docx", b"not-a-valid-zip-archive", "application/octet-stream")},
        )
        assert bad.status_code == 202
        run_id = bad.json()["run_id"]

        final_data = _poll_until_terminal(client, run_id)
        assert final_data["status"] == "failed"
        assert final_data["error"]
        assert "reqval_upload_" not in final_data["error"]
        assert client.get(f"/api/runs/{run_id}/report.json").status_code == 409
        assert client.get(f"/api/runs/{run_id}/report.md").status_code == 409

        # Worker lock must be released after failure.
        follow_up = client.post(
            "/api/runs",
            files={"file": ("ok.md", SAMPLE_MD.encode("utf-8"), "text/markdown")},
        )
        assert follow_up.status_code == 202
        assert _poll_until_terminal(client, follow_up.json()["run_id"])["status"] == "done"


def test_auth_error_fails_job_cleanly():
    fake = FakeLLM({"extract": LLMError("401 unauthorized secret", "auth")})
    app = create_app(settings=_test_settings(), client=fake)
    with TestClient(app) as client:
        resp = client.post("/api/runs/sample")
        assert resp.status_code == 202
        final_data = _poll_until_terminal(client, resp.json()["run_id"])
        assert final_data["status"] == "failed"
        assert final_data["error"]
        assert "secret" not in final_data["error"]
