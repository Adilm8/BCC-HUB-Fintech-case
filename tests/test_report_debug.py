import re

from tests.fakes import FakeLLM, default_check, section
from tests.test_pipeline_run import raw_doc, settings
from validator.pipeline import run_pipeline


def _minor_check(prompt, schema):
    out = default_check(prompt, schema)
    for r in out["results"]:
        for f in r["findings"]:
            f["severity"] = "minor"
    return out


def _review_all(verdict, reason, counter=""):
    def responder(prompt, schema):
        ids = re.findall(r"^(C\d+) \|", section(prompt, "candidates"), re.MULTILINE)
        return {"decisions": [{"finding_id": i, "verdict": verdict, "merge_into": "", "severity": "major",
                               "drop_reason": reason, "counter_quote": counter, "improved_question": ""}
                              for i in ids],
                "missed": [], "fix_first": [], "verdict_rationale": "x"}
    return responder


def test_report_json_lists_the_candidates_the_critic_dropped():
    rep = run_pipeline(raw_doc(), settings(), client=FakeLLM({"review": _review_all("drop", "generic"),
                                                              "check": _minor_check}))
    assert rep.debug is not None and rep.debug.critic_dropped
    assert {d.reason for d in rep.debug.critic_dropped} <= {"generic", "unconfirmed"}
    assert all(d.candidate_id.startswith("C") and d.source and d.criterion for d in rep.debug.critic_dropped)


def test_a_verified_counter_quote_drop_is_logged_with_the_quote():
    proof = "Сотрудник бронирует переговорную на срок от 15 до 240 минут включительно."
    rep = run_pipeline(raw_doc(), settings(), client=FakeLLM({"review": _review_all("drop", "document_specifies",
                                                                                     proof)}))
    gone = [d for d in rep.debug.critic_dropped if d.stage == "review"]
    assert gone and all(d.reason == "document_specifies" and d.counter_quote == proof for d in gone)
