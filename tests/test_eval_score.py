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
         "match": {"where": [["чтение порога"], ["сохранени"]], "what": [["операции чтения"], ["re:\\bнет\\b"]]},
         "blocks_hint": ["Сохранение порога"]},
        {"id": "T2", "severity": "major", "criterion": "complete", "description": "порог по умолчанию",
         "match": {"where": [["по умолчанию"]], "what": [["по умолчанию"], ["порог"], ["не задано"]]}},
        {"id": "T3", "severity": "major", "criterion": "set_complete", "description": "пустой раздел тестов",
         "match": {"where": [["заполняются тестировщиками"]], "what": [["заглушк"]]}},
        {"id": "T4", "severity": "critical", "criterion": "set_consistent", "description": "200 против 400",
         "match": {"where": [["re:\\b200\\b"]], "what": [["re:\\b200\\b"], ["re:\\b400\\b"]]}},
        {"id": "T5", "severity": "minor", "criterion": "ranked", "description": "нет приоритетов",
         "match": {"where": [["требовани"]], "what": [["приоритет"]]}},
    ],
    "clean": [{"id": "C1", "anchor": "в течение минуты после получения события"}],
}

# the old format: every group is searched in the whole haystack (quotes included)
LEGACY_GOLD = {
    "document": "requirements.md",
    "defects": [
        {"id": "T1", "severity": "critical", "match": {"all_of": [["чтение порога"], ["сохранени"]]}},
        {"id": "T2", "severity": "major", "match": {"all_of": [["по умолчанию"], ["порог"]]}},
        {"id": "T3", "severity": "major", "match": {"all_of": [["заполняются тестировщиками"], ["заглушк"]]}},
        {"id": "T4", "severity": "critical", "match": {"all_of": [["re:\\b200\\b"], ["re:\\b400\\b"]]}},
    ],
}


def finding(text: str, explanation: str = "", severity: str = "major") -> dict:
    return {"id": "D1", "severity": severity, "criterion": "complete", "requirement_ids": [],
            "evidence": [{"quote": text}], "explanation": explanation}


def report_of(*findings: dict) -> dict:
    return {"run": {}, "requirements": [], "findings": list(findings), "set_findings": []}


def one_group_gold(*groups: list[str]) -> object:
    """Legacy format: all groups are searched in the whole haystack."""
    return parse_gold({"defects": [{"id": "X", "severity": "major", "match": {"all_of": list(groups)}}]})


def gold_of(*items: dict) -> object:
    """Items are {"id", "where", "what", optional "hint", "severity"}."""
    return parse_gold({"defects": [
        {"id": i["id"], "severity": i.get("severity", "major"),
         "match": {"where": i.get("where", []), "what": i.get("what", [])},
         "blocks_hint": i.get("hint", [])} for i in items]})


def finding_with(fid: str, quote: str, explanation: str, requirement_ids: list[str] | None = None) -> dict:
    f = finding(quote, explanation)
    f["id"] = fid
    f["requirement_ids"] = requirement_ids or []
    return f


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


def test_requirement_title_is_part_of_where_but_not_of_what(report):
    # "передача push" is only in the title of R-003 that D-001 refers to
    assert score(report, gold_of({"id": "X", "where": [["передача push"]]}))["recall"]["found"] == 1
    assert score(report, gold_of({"id": "X", "what": [["передача push"]]}))["recall"]["found"] == 0


# ---------------------------------------------------------------- where / what split


def test_what_is_not_searched_in_quotes():
    gold = gold_of({"id": "X", "where": [["порог"]], "what": [["противореч"]]})
    # the quote has the defect word, the finding's own text has not
    off_topic = finding("Противоречие: порог", "Не задан формат даты.")
    assert score(report_of(off_topic), gold)["recall"]["found"] == 0
    on_topic = finding("порог", "Здесь противоречие между двумя пунктами.")
    assert score(report_of(on_topic), gold)["recall"]["found"] == 1


def test_where_may_match_the_quote():
    gold = gold_of({"id": "X", "where": [["заполняются тестировщиками"]], "what": [["заглушк"]]})
    assert score(report_of(finding("заполняются тестировщиками", "Там заглушка.")), gold)["recall"]["found"] == 1
    assert score(report_of(finding("другой абзац", "Там заглушка.")), gold)["recall"]["found"] == 0


def test_what_uses_title_explanation_and_question_only():
    gold = gold_of({"id": "X", "what": [["ключ дедупликации"]]})
    f = finding("ключ дедупликации", "")
    f["recommendation"] = "Указать ключ дедупликации"
    assert score(report_of(f), gold)["recall"]["found"] == 0
    f = finding("q", "")
    f["clarifying_question"] = "Какой ключ дедупликации?"
    assert score(report_of(f), gold)["recall"]["found"] == 1
    f = finding("q", "")
    f["title"] = "Нет ключ дедупликации"
    assert score(report_of(f), gold)["recall"]["found"] == 1


def test_off_topic_finding_quoting_the_same_text_is_not_credited():
    """The scenario from the review: a finding on another topic quotes the gold paragraph."""
    gold = gold_of(
        {"id": "G15", "where": [["повторная доставка"]], "what": [["ключ", "окно"], ["идемпотент", "дедуплик"]]})
    coverage = finding_with("S-016", "Повторная доставка одного события процессинга",
                            "Для требований нет ни одного критерия приёмки.")
    real = finding_with("D-010", "Повторная доставка одного события процессинга",
                        "Не указан ключ идемпотентности и окно дедупликации.")
    assert score(report_of(coverage), gold)["recall"]["found"] == 0
    res = score(report_of(coverage, real), gold)
    assert by_id(res)["G15"]["finding_ids"] == ["D-010"]
    assert [u["id"] for u in res["unmatched_findings"]] == ["S-016"]


# ---------------------------------------------------------------- crediting limits


def _three_items() -> object:
    return gold_of(
        {"id": "A", "what": [["альфа"]]},
        {"id": "B", "what": [["альфа"], ["бета"]]},
        {"id": "C", "what": [["альфа"], ["бета"], ["гамма"]]},
    )


def test_a_finding_is_credited_to_at_most_two_items_the_best_ones():
    res = score(report_of(finding_with("D-1", "q", "альфа бета гамма")), _three_items())
    rows = by_id(res)
    assert [i for i, r in rows.items() if r["matched"]] == ["B", "C"]  # A is the weakest match
    assert rows["A"]["uncredited_finding_ids"] == ["D-1"]
    assert res["recall"]["found"] == 2
    # the finding is still a true positive for precision
    assert res["precision_lower_bound"] == {"matched": 1, "total": 1, "value": 1.0}


def test_hint_in_the_quote_makes_a_match_stronger():
    gold = gold_of({"id": "A", "what": [["альфа"]], "hint": ["особый абзац"]},
                   {"id": "B", "what": [["альфа"]]}, {"id": "C", "what": [["альфа"]]})
    rows = by_id(score(report_of(finding_with("D-1", "Особый абзац", "альфа")), gold))
    assert rows["A"]["matched"] is True
    assert sum(1 for r in rows.values() if r["matched"]) == 2


def test_a_finding_that_references_many_requirements_is_credited_to_one_item():
    reqs8 = [f"R-{i:03d}" for i in range(8)]
    res8 = score(report_of(finding_with("S-1", "q", "альфа бета", reqs8)), _three_items())
    assert sum(1 for r in res8["items"] if r["matched"]) == 2
    res9 = score(report_of(finding_with("S-1", "q", "альфа бета", [*reqs8, "R-999"])), _three_items())
    assert [r["id"] for r in res9["items"] if r["matched"]] == ["B"]


def test_cap_is_per_finding_not_per_item():
    rep = report_of(finding_with("D-1", "q", "альфа"), finding_with("D-2", "q", "альфа"))
    rows = by_id(score(rep, gold_of({"id": "A", "what": [["альфа"]]})))
    assert rows["A"]["finding_ids"] == ["D-1", "D-2"]


def test_legacy_all_of_still_searches_quotes(report):
    result = score(report, parse_gold(LEGACY_GOLD))
    assert [r["id"] for r in result["items"] if r["matched"]] == ["T1", "T2", "T3"]


# ---------------------------------------------------------------- metrics on the sample report


def test_metrics_on_sample_report(report):
    result = score(report, parse_gold(TINY_GOLD))
    items = by_id(result)
    assert [i for i, r in items.items() if r["matched"]] == ["T1", "T2", "T3"]
    assert items["T1"]["finding_ids"] == ["S-001"]
    assert items["T2"]["finding_ids"] == ["D-002"]
    assert items["T3"]["finding_ids"] == ["S-002"]
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
    {"defects": [{"id": "A", "severity": "major", "match": {}}]},
    {"defects": [{"id": "A", "severity": "major", "match": {"where": [], "what": []}}]},
    {"defects": [{"id": "A", "severity": "major", "match": {"what": [[]]}}]},
    {"defects": [{"id": "A", "severity": "major", "match": {"where": [["x"]], "what": ["y"]}}]},
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
        assert item.where, f"{item.id}: where (location) is required"
        assert item.what, f"{item.id}: what (the defect concept) is required"
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
def test_gold_is_not_matched_by_an_off_topic_finding_quoting_the_defect_text(path):
    """A finding that quotes the defective paragraph but talks about nothing in particular earns no credit."""
    data = json.loads(path.read_text(encoding="utf-8"))
    findings = []
    for i, d in enumerate(data["defects"]):
        f = finding(d["blocks_hint"][0], "Формулировка корректна, замечаний к оформлению нет.")
        f["id"] = f"O{i}"
        findings.append(f)
    assert score(report_of(*findings), parse_gold(data))["recall"]["found"] == 0


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
