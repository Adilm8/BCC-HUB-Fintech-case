from tests.fakes import FakeLLM
from tests.test_pipeline_run import raw_doc, settings
from validator.config import estimate_cost
from validator.pipeline import run_pipeline
from validator.report.markdown import render_markdown


def test_cost_is_estimated_only_for_a_priced_model():
    cost, note = estimate_cost({"claude-sonnet-5-5": (1_000_000, 100_000)}, "anthropic")
    assert cost == 3.0 and "$2 / $10" in note
    assert estimate_cost({"gemini-3.5-flash": (5000, 900)}, "gemini") == (
        None, "бесплатный тариф; на платном ключе — по прайс-листу провайдера")
    assert estimate_cost({"x": (5, 5)}, "other")[0] is None
    assert estimate_cost({}, "gemini") == (None, "модель не вызывалась")


def test_passport_shows_tokens_and_cost():
    rep = run_pipeline(raw_doc(), settings(provider="anthropic"), client=FakeLLM(model="claude-sonnet-5-5"))
    assert rep.run.input_tokens == sum(s.input_tokens for s in rep.run.stages) > 0
    assert rep.run.cost_usd is not None and rep.run.cost_usd > 0
    md = render_markdown(rep)
    assert f"| Токены (вход / выход) | {rep.run.input_tokens} / {rep.run.output_tokens} |" in md
    assert "| Стоимость | ≈ $" in md
    free = run_pipeline(raw_doc(), settings(), client=FakeLLM(model="gemini-3.5-flash"))
    assert free.run.cost_usd is None
