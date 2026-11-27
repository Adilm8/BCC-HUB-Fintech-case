"""Run trace: every model request and answer of a run as JSON files, and a client that replays them offline.

A trace holds prompts and answers only: no keys, no headers, no provider settings beyond the model names.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from validator.config import Settings, family
from validator.llm import LLMError, LLMResponse, short_error

MANIFEST = "manifest.json"
# switches that change the prompts: recorded with the trace and restored on replay
PROMPT_ENV_PREFIXES = ("EXP_", "LLM_MODEL_")
TRACE_VERSION = 1


def prompt_hash(prompt: str, system: str | None, json_schema: dict | None) -> str:
    payload = json.dumps([system or "", prompt, json_schema or {}], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _StageMixin:
    """The stage name of the call in progress, per thread (set by StageContext.call)."""

    def _init_stage(self) -> None:
        self._stage_local = threading.local()

    def set_stage(self, stage: str) -> None:
        self._stage_local.stage = stage

    @property
    def stage(self) -> str:
        return getattr(self._stage_local, "stage", "other")


class TracingClient(_StageMixin):
    """Wraps a real client and writes one JSON file per logical call into `directory`."""

    def __init__(self, inner, directory: Path, settings: Settings):
        self._inner = inner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        for old in self.directory.glob("call_*.json"):
            old.unlink()  # a trace describes exactly one run
        self._lock = threading.Lock()
        self._seq = 0
        self._init_stage()
        manifest = {"version": TRACE_VERSION, "provider": getattr(inner, "provider", settings.provider),
                    "family": family(getattr(inner, "provider", settings.provider)), "model": settings.model,
                    "fallback_models": list(settings.fallback_models), "profile": settings.profile_name,
                    "max_context_chars": settings.max_context_chars, "domain": settings.domain, "env": prompt_env()}
        (self.directory / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def provider(self) -> str:
        return getattr(self._inner, "provider", "")

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "")

    def set_note_sink(self, fn) -> None:
        sink = getattr(self._inner, "set_note_sink", None)
        if sink is not None:
            sink(fn)

    def complete(self, prompt: str, *, system: str | None = None, json_schema: dict | None = None, **kw) -> LLMResponse:
        key = prompt_hash(prompt, system, json_schema)
        record = {"stage": self.stage, "prompt_sha256": key, "prompt_chars": len(prompt)}
        try:
            resp = self._inner.complete(prompt, system=system, json_schema=json_schema, **kw)
        except LLMError as exc:
            record["error"] = {"kind": exc.kind, "message": short_error(exc, 300)}
            self._write(record)
            raise
        record.update(model=resp.model, text=resp.text, input_tokens=resp.input_tokens,
                      output_tokens=resp.output_tokens, truncated=resp.truncated)
        self._write(record)
        return resp

    def _write(self, record: dict) -> None:
        with self._lock:
            self._seq += 1
            record = {"seq": self._seq, **record}
            name = f"call_{self._seq:03d}_{record['stage']}_{record['prompt_sha256'][:12]}.json"
            (self.directory / name).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def prompt_env() -> dict[str, str]:
    from validator.pipeline import toggles

    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(PROMPT_ENV_PREFIXES) and v.strip()}
    env.update(toggles.effective())  # explicit values: a later change of defaults does not change the replay
    return env


def restore_prompt_env(manifest: dict) -> None:
    """The replay runs with exactly the switches of the recorded run. A switch the manifest does not name was
    off when it was recorded (older traces list only the variables that were set)."""
    from validator.pipeline import toggles

    for k in [k for k in os.environ if k.startswith(PROMPT_ENV_PREFIXES)]:
        del os.environ[k]
    os.environ.update({n: "0" for n in toggles.TOGGLES})
    os.environ.update({k: str(v) for k, v in (manifest.get("env") or {}).items()})


def read_manifest(directory: Path) -> dict:
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise ValueError(f"в {directory} нет трассы прогона ({MANIFEST}); запишите её с --save-trace")
    return json.loads(path.read_text(encoding="utf-8"))


class ReplayClient(_StageMixin):
    """Answers every call from a saved trace by stage and prompt hash; no network, no key."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        manifest = read_manifest(self.directory)
        self.provider = manifest.get("family") or manifest.get("provider", "gemini")
        self.model = manifest.get("model", "")
        self._records: dict[tuple[str, str], list[dict]] = {}
        for path in sorted(self.directory.glob("call_*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            self._records.setdefault((rec["stage"], rec["prompt_sha256"]), []).append(rec)
        self._used: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self._init_stage()

    def set_note_sink(self, fn) -> None:
        return None

    def complete(self, prompt: str, *, system: str | None = None, json_schema: dict | None = None, **kw) -> LLMResponse:
        key = (self.stage, prompt_hash(prompt, system, json_schema))
        with self._lock:
            recs = self._records.get(key)
            if not recs:
                raise LLMError(
                    f"трасса {self.directory} не содержит запроса этапа {self.stage} с таким текстом: промпт "
                    "изменился (другая версия кода, документа или настроек). Запишите трассу заново с --save-trace",
                    "replay")
            n = self._used.get(key, 0)
            self._used[key] = n + 1
            rec = recs[min(n, len(recs) - 1)]
        if "error" in rec:
            raise LLMError(rec["error"].get("message", "ошибка из трассы"), rec["error"].get("kind", "other"))
        return LLMResponse(text=rec["text"], model=rec.get("model") or self.model,
                           input_tokens=rec.get("input_tokens"), output_tokens=rec.get("output_tokens"),
                           truncated=bool(rec.get("truncated")))
