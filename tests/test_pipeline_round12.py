from tests.fakes import FakeLLM
from tests.test_pipeline_run import raw_doc, settings
from validator.pipeline import run_pipeline


def test_scenario_writer_adds_one_call_and_its_findings(monkeypatch):
    base = FakeLLM()
    rep0 = run_pipeline(raw_doc(), settings(), client=base)
    assert "scenarios" not in [s for s, _ in base.calls]
    monkeypatch.setenv("EXP_SCENARIOS", "1")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    stages = [s for s, _ in f.calls]
    assert stages.count("scenarios") == 1
    prompt = next(p for s, p in f.calls if s == "scenarios")
    assert "12–16 КОНКРЕТНЫХ сценариев" in prompt
    # gemini is sequential: the writer replaces one check batch
    assert stages.count("check") <= max(1, [s for s, _ in base.calls].count("check"))
    assert rep.run.llm_calls <= 9
    texts = [s.explanation for s in rep.set_findings]
    assert any("разный результат для одного случая на границе" in t for t in texts)
    assert len(rep.set_findings) >= len(rep0.set_findings)


def test_scenarios_become_set_findings_contradictions_first():
    from validator.pipeline import experiments as exp
    from validator.pipeline.prompts.io import LLMScenarios

    def sc(outcome, expl, n=2, sev="critical"):
        return {"scenario": "s", "readings": [{"requirement_id": f"R-{i}", "quote": f"q{i}", "prescribes": f"p{i}"}
                                              for i in range(n)],
                "outcome": outcome, "explanation": expl, "severity": sev, "recommendation": "", "clarifying_question": "",
                "answer_options": []}

    data = LLMScenarios.model_validate({"scenarios": [sc("undefined", "gap", 1), sc("defined", ""),
                                                      sc("contradictory", "clash"), sc("contradictory", "one", 1)]})
    out = exp.scenarios_to_set(data).findings
    assert [f.explanation for f in out] == ["clash", "one", "gap"]
    assert out[0].set_check == "contradiction" and len(out[0].evidence) == 2 and "R-0: p0" in out[0].scenario
    # without two places it is a gap, and a gap is never critical
    assert out[1].set_check == "missing_scenario" and out[1].severity == "major"
    assert out[2].criterion == "set_complete"


def test_split_s1_makes_two_narrower_calls(monkeypatch):
    base = FakeLLM()
    rep0 = run_pipeline(raw_doc(), settings(), client=base)
    monkeypatch.setenv("EXP_SPLIT_S1", "1")
    f = FakeLLM()
    rep = run_pipeline(raw_doc(), settings(), client=f)
    sets = [p for s, p in f.calls if s == "set"]
    assert len(sets) == len([1 for s, _ in base.calls if s == "set"]) + 1
    rules = [p for p in sets if "Задание S1a" in p]
    contracts = [p for p in sets if "Задание S1b: сверь контракты" in p]
    assert len(rules) == 1 and len(contracts) == 1
    assert not any("Задание S1: найди противоречия" in p for p in sets)
    assert rep.run.llm_calls <= 10 and rep.run.status == rep0.run.status
    assert len(rep.coverage) == len(rep0.coverage)


def test_rule_hints_reach_check_and_set_prompts_only(monkeypatch):
    doc = raw_doc()
    f = FakeLLM()
    run_pipeline(doc, settings(), client=f)
    assert not any("<rule_findings>" in p for _, p in f.calls)
    monkeypatch.setenv("EXP_RULE_HINTS", "1")
    f = FakeLLM()
    run_pipeline(doc, settings(), client=f)
    with_hints = {s for s, p in f.calls if "<rule_findings>" in p}
    assert with_hints == {"check", "set"}
    check = next(p for s, p in f.calls if s == "check")
    block = check.split("<rule_findings>\n", 1)[1].split("\n</rule_findings>", 1)[0]
    assert block.startswith("- ") and " | " in block
    assert check.index("<rule_findings>") < check.rindex("\n\nЗадание")


def _review_all(verdict, reason, counter=""):
    import re

    from tests.fakes import section

    def responder(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": verdict, "merge_into": ids[0] if verdict == "merge" else "",
                               "severity": "major", "drop_reason": reason, "counter_quote": counter,
                               "improved_question": ""} for i in ids],
                "missed": [], "fix_first": [], "verdict_rationale": "x"}
    return responder


def _minor_check(prompt, schema):
    from tests.fakes import default_check

    out = default_check(prompt, schema)
    for r in out["results"]:
        for f in r["findings"]:
            f["severity"] = "minor"
    return out


def test_cautious_critic_keeps_opinions_and_logs_what_is_dropped(monkeypatch):
    def fake(review):
        return FakeLLM({"review": review, "check": _minor_check})

    def total(rep):
        return len(rep.findings) + len(rep.set_findings)

    plain = run_pipeline(raw_doc(), settings(), client=fake(_review_all("drop", "generic")))
    assert plain.debug is not None and plain.debug.critic_dropped
    assert {d.reason for d in plain.debug.critic_dropped} <= {"generic", "unconfirmed"}
    assert all(d.candidate_id.startswith("C") and d.source for d in plain.debug.critic_dropped)

    monkeypatch.setenv("EXP_CAUTIOUS_CRITIC", "1")
    f = fake(_review_all("drop", "generic"))
    cautious = run_pipeline(raw_doc(), settings(), client=f)
    assert "осторожный критик" in next(p for s, p in f.calls if s == "review")
    assert total(cautious) > total(plain)
    assert not [d for d in cautious.debug.critic_dropped if d.stage == "review"]

    # an unproven document_specifies drop and a merge of different findings are refused too
    for review in (_review_all("drop", "document_specifies", "этого текста нет в документе"),
                   _review_all("merge", "duplicate")):
        rep = run_pipeline(raw_doc(), settings(), client=fake(review))
        assert not [d for d in rep.debug.critic_dropped if d.stage == "review"]
        assert total(rep) == total(cautious)

    # a verified counter-quote still removes a model finding, never a rule finding
    proof = "Сотрудник бронирует переговорную на срок от 15 до 240 минут включительно."
    rep = run_pipeline(raw_doc(), settings(), client=fake(_review_all("drop", "document_specifies",
                                                                  proof)))
    gone = [d for d in rep.debug.critic_dropped if d.stage == "review"]
    assert gone and all(d.reason == "document_specifies" and d.source != "rule" for d in gone)
    assert any(f.source == "rule" for f in rep.findings + rep.set_findings)
