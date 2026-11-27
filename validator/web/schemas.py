"""API response DTOs for the local web interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from validator.models import RunStatus, StageStatus, Summary, Verdict


class HealthDTO(BaseModel):
    """Service status without exposing credentials."""

    status: Literal["ok"] = "ok"
    version: str
    provider: str
    model: str
    key_configured: bool


class RunCreatedDTO(BaseModel):
    """Accepted run identifier returned immediately on job submission."""

    run_id: str
    status: Literal["queued", "running"]


class RunStatusDTO(BaseModel):
    """Live progress and final summary snapshot for a pipeline run."""

    run_id: str
    status: Literal["queued", "running", "done", "failed"]
    document_name: str
    stages: list[StageStatus]
    progress: float
    report_status: RunStatus | None = None
    summary: Summary | None = None
    verdict: Verdict | None = None
    error: str | None = None
