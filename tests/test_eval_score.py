"""Tests for the offline evaluation harness: eval/score.py, eval/run.py and the committed gold files."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval import run as eval_run
from eval.score import load_gold, main, normalize, parse_gold, render_markdown, score

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "sample_report.json"
GOLD_DIR = ROOT / "eval" / "gold"

TINY_GOLD = {
    "document": "requirements.md",
    "defects": [
        {"id": "T1", "severity": "critical", "criterion": "set_consistent", "description": "чтение порога",
         "match": {"all_of": [["чтение порога"], ["сохранени"]]}, "blocks_hint": ["Сохранение порога"]},
        {"id": "T2", "severity": "major", "criterion": "complete", "description": "порог по умолчанию",
         "match": {"all_of": [["по умолчанию"], ["порог"]]}},
        {"id": "T3", "severity": "major", "criterion": "set_complete", "description": "пустой раздел тестов",
         "match": {"all_of": [["заполняются тестировщиками"], ["заглушк"]]}},
        {"id": "T4", "severity": "critical", "criterion": "set_consistent", "description": "200 против 400",
         "match": {"all_of": [["re:\\b200\\b"], ["re:\\b400\\b"]]}},
        {"id": "T5", "severity": "minor", "criterion": "ranked", "description": "нет приоритетов",
         "match": {"all_of": [["приоритет"], ["требовани"]]}},
    ],
    "clean": [{"id": "C1", "anchor": "в течение минуты после получения события"}],
}


def finding(text: str, explanation: str = "", severity: str = "major") -> dict:
    return {"id": "D1", "severity": severity, "criterion": "complete", "requirement_ids": [],
            "evidence": [{"quote": text}], "explanation": explanation}


def report_of(*findings: dict) -> dict:
    return {"run": {}, "requirements": [], "findings": list(findings), "set_findings": []}


def one_group_gold(*groups: list[str]) -> object:
    return parse_gold({"defects": [{"id": "X", "severity": "major", "match": {"all_of": list(groups)}}]})


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def by_id(result: dict) -> dict:
    return {r["id"]: r for r in result["items"]}


# ---------------------------------------------------------------- normalisation and matching


def test_normalize_unifies_case_yo_dashes_and_spaces():
    assert normalize("Проведённые  `покупки` — 1 000 ₸") == "проведенные покупки - 1 000 ₸"
    assert normalize("Ф‑2") == normalize("ф-2")


def test_thousand_separator_variants_match():
    gold = one_group_gold(["10 000"])
    assert score(report_of(finding("лимит 10 000 рублей")), gold)["recall"]["found"] == 1


def test_regex_alternative_respects_word_boundaries():
    gold = one_group_gold(["re:\\bget\\b"])
    assert score(report_of(finding("REST type | GET")), gold)["recall"]["found"] == 1
    assert score(report_of(finding("widget and target")), gold)["recall"]["found"] == 0


def test_every_group_must_match():
    gold = one_group_gold(["alpha"], ["beta"])
    assert score(report_of(finding("Alpha only", "nothing")), gold)["recall"]["found"] == 0
    assert score(report_of(finding("Alpha only", "also BETA")), gold)["recall"]["found"] == 1


def test_requirement_title_is_part_of_haystack(report):
    # D-001 quotes only the time phrase; the "Передача push..." title comes from requirement R-003
    gold = one_group_gold(["передача push"], ["не указано"])
    assert score(report, gold)["recall"]["found"] == 1


# ---------------------------------------------------------------- metrics on the sample report


def test_metrics_on_sample_report(report):
    result = score(report, parse_gold(TINY_GOLD))
    items = by_id(result)
    assert [i for i, r in items.items() if r["matched"]] == ["T1", "T2", "T3"]
    assert items["T1"]["finding_ids"] == ["S-001"]
    assert result["recall"] == {"found": 3, "total": 5, "value": 0.6}
    assert result["recall_by_severity"]["critical"]["found"] == 1
    assert result["recall_by_severity"]["critical"]["total"] == 2
    assert result["recall_by_severity"]["major"]["found"] == 2
    assert result["recall_by_severity"]["minor"]["value"] == 0.0
    # S-001, D-002, S-002 matched; D-001 not
    assert result["precision_lower_bound"] == {"matched": 3, "total": 4, "value": 0.75}
    assert [u["id"] for u in result["unmatched_findings"]] == ["D-001"]
    assert result["warnings"] == []


def test_location_and_severity_agreement(report):
    result = score(report, parse_gold(TINY_GOLD))
    items = by_id(result)
    assert items["T1"]["located"] is True
    assert items["T2"]["located"] is None  # no blocks_hint
    assert items["T1"]["severity_ok"] is True
    assert items["T2"]["severity_ok"] is False  # D-002 is minor, gold says major
    assert result["severity_agreement"]["total"] == 3
    assert result["severity_agreement"]["ok"] == 2


def test_noise_counts_findings_quoting_only_clean_requirements(report):
    noise = score(report, parse_gold(TINY_GOLD))["noise"]
    assert noise == {"available": True, "all": ["D-001"], "major": ["D-001"]}
    no_clean = copy.deepcopy(TINY_GOLD)
    no_clean["clean"] = []
    assert score(report, parse_gold(no_clean))["noise"] == {"available": False, "all": [], "major": []}


def test_noise_ignores_findings_with_a_non_clean_quote(report):
    rep = copy.deepcopy(report)
    rep["findings"][0]["evidence"].append({"quote": "совсем другая цитата"})
    assert score(rep, parse_gold(TINY_GOLD))["noise"]["all"] == []


def test_document_mismatch_warns(report):
    gold = copy.deepcopy(TINY_GOLD)
    gold["document"] = "eval/docs/other.md"
    assert "other.md" in score(report, parse_gold(gold))["warnings"][0]


def test_scoring_is_deterministic(report):
    gold = parse_gold(TINY_GOLD)
    assert score(report, gold) == score(copy.deepcopy(report), gold)


def test_markdown_summary_lists_missed_and_found(report):
    md = render_markdown(score(report, parse_gold(TINY_GOLD)), "r.json", "g.json")
    assert "recall критичных | 1 из 2 (50%)" in md
    assert "precision, нижняя граница | 3 из 4 (75%)" in md
    assert "## Не найдено (2)" in md
    assert "T4" in md and "T5" in md
    assert "D-001" in md  # unmatched finding and noise


# ---------------------------------------------------------------- gold validation


@pytest.mark.parametrize("bad", [
    {"defects": [{"id": "A", "severity": "huge", "match": {"all_of": [["x"]]}}]},
    {"defects": [{"id": "A", "severity": "major", "match": {"all_of": []}}]},
    {"defects": [{"id": "A", "severity": "major", "match": {"all_of": [[]]}}]},
    {"defects": [{"id": "A", "severity": "major", "match": {"all_of": [["x"]]}},
                 {"id": "A", "severity": "major", "match": {"all_of": [["y"]]}}]},
    {"nothing": []},
])
def test_invalid_gold_is_rejected(bad):
    with pytest.raises(ValueError):
        parse_gold(bad)


# ---------------------------------------------------------------- command line


def test_cli_writes_markdown_and_json_next_to_report(tmp_path, capsys, report):
    rep_path = tmp_path / "out.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    gold_path = tmp_path / "tiny.gold.json"
    gold_path.write_text(json.dumps(TINY_GOLD, ensure_ascii=False), encoding="utf-8")
    assert main([str(rep_path), str(gold_path)]) == 0
    assert "recall общий" in capsys.readouterr().out
    md = (tmp_path / "out.score.md").read_text(encoding="utf-8")
    assert "recall критичных" in md
    data = json.loads((tmp_path / "out.score.json").read_text(encoding="utf-8"))
    assert data["recall"]["found"] == 3


def test_cli_no_write_and_errors(tmp_path, report):
    rep_path = tmp_path / "out.json"
    rep_path.write_text(json.dumps(report), encoding="utf-8")
    gold_path = tmp_path / "g.json"
    gold_path.write_text(json.dumps(TINY_GOLD), encoding="utf-8")
    assert main([str(rep_path), str(gold_path), "--no-write"]) == 0
    assert not (tmp_path / "out.score.md").exists()
    assert main([str(tmp_path / "missing.json"), str(gold_path)]) == 2
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main([str(rep_path), str(broken)]) == 2


def test_run_helper_scores_an_existing_report_without_running(tmp_path, report, monkeypatch):
    rep_path = tmp_path / "report.json"
    rep_path.write_text(json.dumps(report), encoding="utf-8")
    gold_path = tmp_path / "requirements.gold.json"
    gold_path.write_text(json.dumps(TINY_GOLD), encoding="utf-8")

    def boom(*args, **kwargs):  # the validator must not be started
        raise AssertionError("validator was started")

    monkeypatch.setattr(eval_run, "run_validator", boom)
    code = eval_run.main(["docs/requirements.md", "--report-json", str(rep_path), "--gold", str(gold_path)])
    assert code == 0
    assert (tmp_path / "report.score.md").is_file()


def test_run_helper_reports_missing_gold(tmp_path):
    code = eval_run.main([str(tmp_path / "no_such_doc.md"), "--report-json", str(SAMPLE)])
    assert code == eval_run.EXIT_NO_GOLD


# ---------------------------------------------------------------- committed gold files


GOLD_FILES = sorted(GOLD_DIR.glob("*.gold.json"))


def test_synthetic_gold_files_exist():
    assert {p.name for p in GOLD_FILES} >= {"doc1_autopay.gold.json", "doc2_p2p.gold.json"}


def _document_text(path: Path) -> str:
    rel = json.loads(path.read_text(encoding="utf-8"))["document"]
    return (ROOT / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.name)
def test_gold_file_is_consistent_with_its_document(path):
    gold = load_gold(path)
    doc_text = normalize(_document_text(path))
    assert gold.items
    assert any(i.severity == "critical" for i in gold.items)
    for item in gold.items:
        assert len(item.groups) >= 2, f"{item.id}: use at least two groups"
        for hint in item.hints:
            assert normalize(hint) in doc_text, f"{path.name} {item.id}: hint not in document"
    for cid, anchor in gold.clean:
        assert anchor in doc_text, f"{path.name} {cid}: clean anchor not in document"


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.name)
def test_gold_is_not_trivially_matched_by_bare_quotes(path):
    """Flagging every paragraph with an empty explanation must not reach most of the gold."""
    gold = load_gold(path)
    paras = [p.strip() for p in _document_text(path).splitlines() if len(p.strip()) > 10]
    findings = []
    for i, p in enumerate(paras):
        f = finding(p)
        f["id"] = f"D{i}"
        findings.append(f)
    assert score(report_of(*findings), gold)["recall"]["value"] <= 0.3


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.name)
def test_gold_matches_an_oracle_report(path):
    """Self-check of the rules: a finding quoting the hint and restating the description must match."""
    data = json.loads(path.read_text(encoding="utf-8"))
    findings = []
    for i, d in enumerate(data["defects"]):
        f = finding(d["blocks_hint"][0], d["description"], d["severity"])
        f["id"] = f"O{i}"
        findings.append(f)
    result = score(report_of(*findings), parse_gold(data))
    assert result["recall"]["value"] == 1.0, [r["id"] for r in result["items"] if not r["matched"]]
    assert result["located"]["ok"] == result["located"]["total"]


def test_private_gold_if_present_loads():
    path = ROOT / "eval" / "gold_private" / "requirements.gold.json"
    if not path.is_file():
        pytest.skip("private gold is not in this checkout")
    gold = load_gold(path)
    assert sum(1 for i in gold.items if i.severity == "critical") == 7
    assert 20 <= len(gold.items) <= 30
    doc_text = normalize((ROOT / "docs" / "requirements.md").read_text(encoding="utf-8"))
    for item in gold.items:
        for hint in item.hints:
            assert normalize(hint) in doc_text, f"{item.id}: hint not in document"
