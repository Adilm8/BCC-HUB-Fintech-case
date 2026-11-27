import json

import pytest

from eval import ablate
from tests.fakes import FakeLLM
from tests.test_pipeline_run import raw_doc, settings
from validator.pipeline import run_pipeline


def test_variants_resolve_to_flags_and_toggles():
    assert ablate.resolve("full") == ([], {})
    assert ablate.resolve("rules_only") == (["--rules-only"], {})
    assert ablate.resolve("rules_only+no_critic") == (["--rules-only"], {"EXP_NO_REVIEW": "1"})
    assert ablate.resolve("full+EXP_X") == ([], {"EXP_X": "1"})
    with pytest.raises(ValueError):
        ablate.resolve("bogus")


def test_no_review_toggle_skips_the_critic(monkeypatch):
    monkeypatch.setenv("EXP_NO_REVIEW", "1")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    assert "review" not in [s for s, _ in f.calls]
    assert rep.run.status.value == "complete"


def test_ablation_runs_each_variant_and_writes_a_table(tmp_path, monkeypatch):
    gold = tmp_path / "g.json"
    gold.write_text(json.dumps({"document": "rooms.md", "defects": [
        {"id": "G1", "severity": "major", "criterion": "verifiable", "description": "d",
         "match": {"where": [["быстро"]]}}]}), encoding="utf-8")
    doc = tmp_path / "rooms.md"
    doc.write_text("# Т\n\n## Ф\n\n- **BR-3.** Система быстро подтверждает бронирование.\n", encoding="utf-8")
    seen = []

    def fake_run(d, out_dir, flags, env, profile):
        seen.append((flags, env))
        rep = run_pipeline(d, settings(), client=FakeLLM())
        (out_dir / "report.json").write_text(rep.model_dump_json(), encoding="utf-8")
        return 0

    monkeypatch.setattr(ablate, "run_variant", fake_run)
    out = tmp_path / "out"
    assert ablate.main([str(doc), "--variants", "full,no_critic", "--runs", "2", "--gold", str(gold),
                        "--out", str(out)]) == 0
    assert len(seen) == 4 and seen[2][1] == {"EXP_NO_REVIEW": "1"}
    md = (out / "ablate.md").read_text(encoding="utf-8")
    assert "| full | 2 |" in md and "| no_critic | 2 |" in md
    data = json.loads((out / "ablate.json").read_text(encoding="utf-8"))
    assert len(data["runs"]["full"]) == 2 and data["runs"]["full"][0]["recall"] is not None
