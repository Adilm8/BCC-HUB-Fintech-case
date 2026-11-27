"""JSON answers from the model: provider schema, parsing, local repair, item-wise validation, one retry."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from validator.llm import LLMClient, LLMError, LLMResponse

M = TypeVar("M", bound=BaseModel)


# ---------------------------------------------------------------- schema


def provider_schema(model: type[BaseModel]) -> dict:
    """Flat JSON schema: $ref inlined, no titles/defaults, all fields required, no extra properties."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any, properties: bool = False) -> Any:
        if isinstance(node, dict):
            if properties:
                # keys here are field names (a field may well be called "title"), not schema keywords
                return {k: walk(v) for k, v in node.items()}
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(copy.deepcopy(defs[name]))
            out = {k: walk(v, properties=k == "properties") for k, v in node.items()
                   if k not in ("title", "default", "description")}
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"].keys())
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(raw)


# ---------------------------------------------------------------- parsing and repair

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _close_truncated(text: str) -> list[str]:
    """Candidate texts cut after the last complete element, with open brackets closed."""
    stack: list[str] = []
    in_str = esc = False
    cuts: list[tuple[int, list[str]]] = []
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i, list(stack)))
    out = []
    for i, st in reversed(cuts[-200:]):
        closers = "".join("}" if c == "{" else "]" for c in reversed(st))
        out.append(text[: i + 1] + closers)
    return out


def parse_json_lenient(text: str) -> tuple[Any, bool]:
    """Parse model output. Returns (data, repaired). Raises ValueError when nothing can be recovered."""
    s = _FENCE.sub("", text.strip())
    try:
        return json.loads(s), False
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON-объекта")
    end = s.rfind("}")
    body = s[start: end + 1] if end > start else s[start:]
    for candidate in (body, _TRAILING_COMMA.sub(r"\1", body)):
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass
    tail = _TRAILING_COMMA.sub(r"\1", s[start:])
    for candidate in _close_truncated(tail):
        try:
            return json.loads(_TRAILING_COMMA.sub(r"\1", candidate)), True
        except json.JSONDecodeError:
            continue
    raise ValueError("JSON не удалось восстановить")


def _list_item_model(annotation: Any) -> type[BaseModel] | None:
    origin = typing.get_origin(annotation)
    if origin in (list, typing.List):  # noqa: UP006
        args = typing.get_args(annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    return None


def _empty_value(annotation: Any) -> Any:
    if annotation is str:
        return ""
    origin = typing.get_origin(annotation)
    if origin in (list, typing.List):  # noqa: UP006
        return []
    if origin in (typing.Union, types.UnionType) and type(None) in typing.get_args(annotation):
        return None
    return None


def validate_lenient(model: type[M], data: Any) -> tuple[M | None, int]:
    """Validate; on failure keep valid list items and drop bad ones. Returns (instance, dropped items)."""
    if not isinstance(data, dict):
        return None, 1
    try:
        return model.model_validate(data), 0
    except ValidationError:
        pass
    fixed: dict[str, Any] = {}
    dropped = 0
    for name, info in model.model_fields.items():
        value = data.get(name)
        sub = _list_item_model(info.annotation)
        if sub is not None:
            items = value if isinstance(value, list) else []
            good = []
            for item in items:
                obj, d = validate_lenient(sub, item)
                dropped += d
                if obj is not None:
                    good.append(obj)
            fixed[name] = good
        elif value is None:
            fixed[name] = _empty_value(info.annotation)
        elif isinstance(value, str) and typing.get_origin(info.annotation) is typing.Literal:
            fixed[name] = value.strip().lower()
        elif typing.get_origin(info.annotation) in (list, typing.List) and not isinstance(value, list):  # noqa: UP006
            fixed[name] = [value] if isinstance(value, str) and value else []
        else:
            fixed[name] = value
    try:
        return model.model_validate(fixed), dropped
    except ValidationError:
        return None, dropped + 1


# ---------------------------------------------------------------- budget and cache


class Budget:
    """Thread-safe ceiling of model calls per run."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.used


class StageCache:
    """On-disk cache of raw model answers keyed by model, prompt and schema."""

    def __init__(self, root: Path | None):
        self.root = root  # None: no cache at all, neither read nor written

    @staticmethod
    def key(model: str, system: str | None, prompt: str, schema: dict | None, *, effort: str | None = None,
            temperature: float | None = None) -> str:
        h = hashlib.sha256()
        for part in (model, effort or "", "" if temperature is None else f"{temperature:.2f}", system or "", prompt,
                     json.dumps(schema or {}, sort_keys=True, ensure_ascii=False)):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> dict | None:
        if self.root is None:
            return None
        path = self.root / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: dict) -> None:
        if self.root is None:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f"{key}.tmp"
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.root / f"{key}.json")
        except OSError:
            pass


class BudgetExceeded(LLMError):
    def __init__(self):
        super().__init__("исчерпан бюджет вызовов модели на прогон", "other")


# ---------------------------------------------------------------- structured call


@dataclass
class StructuredResult:
    data: Any
    responses: list[LLMResponse] = field(default_factory=list)
    dropped_items: int = 0
    truncated: bool = False
    repaired: bool = False
    cached: bool = False

    @property
    def calls(self) -> int:
        return 0 if self.cached else len(self.responses)

    @property
    def input_tokens(self) -> int:
        return 0 if self.cached else sum(r.input_tokens or 0 for r in self.responses)

    @property
    def output_tokens(self) -> int:
        return 0 if self.cached else sum(r.output_tokens or 0 for r in self.responses)

    @property
    def model(self) -> str | None:
        return self.responses[-1].model if self.responses else None

    @property
    def notes(self) -> list[str]:
        return [] if self.cached else [n for r in self.responses for n in getattr(r, "notes", [])]


def call_structured(
    client: LLMClient,
    model_cls: type[M],
    prompt: str,
    *,
    system: str | None,
    budget: Budget,
    cache: StageCache,
    deadline: float | None = None,
    effort: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    allow_retry: bool = True,
) -> StructuredResult:
    """One structured call with local repair and at most one corrective retry."""
    schema = provider_schema(model_cls)
    model_name = getattr(client, "model", "") or ""
    key = StageCache.key(model_name, system, prompt, schema, effort=effort, temperature=temperature)
    hit = cache.get(key)
    if hit is not None:
        obj, dropped = validate_lenient(model_cls, hit.get("data"))
        if obj is not None:
            resp = LLMResponse(text="", model=hit.get("model", model_name), truncated=hit.get("truncated", False))
            return StructuredResult(obj, [resp], dropped, resp.truncated, False, cached=True)

    responses: list[LLMResponse] = []
    current = prompt
    last_error = ""
    for attempt in range(2):
        if not budget.try_acquire():
            raise BudgetExceeded()
        resp = client.complete(
            current, system=system, json_schema=schema, temperature=temperature, effort=effort,
            max_output_tokens=max_output_tokens, deadline=deadline,
        )
        responses.append(resp)
        try:
            data, repaired = parse_json_lenient(resp.text)
            obj, dropped = validate_lenient(model_cls, data)
            if obj is None:
                raise ValueError("ответ не соответствует схеме")
            result = StructuredResult(obj, responses, dropped, resp.truncated, repaired or resp.truncated)
            # an answer from a fallback model is not stored under the primary model's key
            if not resp.truncated and (not model_name or resp.model == model_name):
                cache.put(key, {"data": obj.model_dump(mode="json"), "model": resp.model, "truncated": False})
            return result
        except ValueError as exc:
            last_error = str(exc)
            if attempt == 0 and allow_retry:
                current = (
                    prompt
                    + "\n\nПредыдущий ответ не удалось разобрать как JSON по схеме ("
                    + last_error
                    + "). Верни только корректный JSON-объект по схеме, без пояснений."
                )
                continue
            break
    err = LLMError(f"невалидный JSON от модели: {last_error}", "invalid_json")
    err.responses = responses  # type: ignore[attr-defined]
    raise err
