import json

import pytest

from tests.fakes import FakeLLM
from validator.llm import LLMError
from validator.pipeline.prompts.io import LLMCheck, LLMExtraction, LLMReview, LLMSetCheck
from validator.pipeline.structured import (
    Budget,
    BudgetExceeded,
    StageCache,
    call_structured,
    parse_json_lenient,
    provider_schema,
    validate_lenient,
)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("model", [LLMExtraction, LLMCheck, LLMSetCheck, LLMReview])
def test_provider_schema_is_flat_and_strict(model):
    schema = provider_schema(model)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and "maxItems" not in text
    for node in _walk(schema):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


def test_repair_code_fence_and_trailing_commas():
    data, repaired = parse_json_lenient('```json\n{"results": [{"requirement_id": "R-001", "findings": [],},]}\n```')
    assert repaired is False or repaired is True
    assert data["results"][0]["requirement_id"] == "R-001"


def test_repair_truncated_array_keeps_whole_items():
    text = '{"results": [{"requirement_id": "R-001", "findings": []}, {"requirement_id": "R-002", "find'
    data, repaired = parse_json_lenient(text)
    assert repaired
    assert [r["requirement_id"] for r in data["results"]] == ["R-001"]


def test_itemwise_validation_drops_bad_items():
    data = {"results": [
        {"requirement_id": "R-001", "findings": [{"quote": "q", "explanation": "e", "criterion": "bogus",
                                                  "severity": "major", "recommendation": "", "clarifying_question": "",
                                                  "answer_options": [], "related_requirement_ids": []}]},
        {"requirement_id": "R-002", "findings": []},
    ]}
    obj, dropped = validate_lenient(LLMCheck, data)
    assert obj is not None and dropped == 1
    assert [r.requirement_id for r in obj.results] == ["R-001", "R-002"]
    assert obj.results[0].findings == []


def test_retry_once_on_invalid_json_then_succeed():
    fake = FakeLLM({"check": ["это не JSON", {"results": []}]})
    res = call_structured(fake, LLMCheck, "prompt", system=None, budget=Budget(5), cache=StageCache(None))
    assert res.calls == 2 and res.data.results == []
    assert "не удалось разобрать" in fake.calls[1][1]


def test_invalid_twice_raises_bad_request():
    fake = FakeLLM({"check": "мусор"})
    with pytest.raises(LLMError) as err:
        call_structured(fake, LLMCheck, "p", system=None, budget=Budget(5), cache=StageCache(None))
    assert err.value.kind == "bad_request"


def test_budget_is_enforced():
    fake = FakeLLM({"check": {"results": []}})
    budget = Budget(1)
    call_structured(fake, LLMCheck, "p1", system=None, budget=budget, cache=StageCache(None))
    with pytest.raises(BudgetExceeded):
        call_structured(fake, LLMCheck, "p2", system=None, budget=budget, cache=StageCache(None))


def test_cache_hit_spends_no_call(tmp_path):
    fake = FakeLLM({"check": {"results": [{"requirement_id": "R-001", "findings": []}]}})
    cache = StageCache(tmp_path)
    call_structured(fake, LLMCheck, "p", system="s", budget=Budget(5), cache=cache)
    res = call_structured(fake, LLMCheck, "p", system="s", budget=Budget(0), cache=cache)
    assert res.cached and res.calls == 0 and len(fake.calls) == 1
