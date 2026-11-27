from __future__ import annotations

from validator.models import Document, InventoryItem, Requirement, SkippedBlock
from validator.pipeline.prompts.io import LLMSetCheck
from validator.pipeline.prompts.rubric import system_prompt
from validator.pipeline.prompts.setcheck import (
    build_merged_prompt,
    build_pairs_prompt,
    build_s1_prompt,
    build_s2_prompt,
)
from validator.pipeline.stages import StageContext
from validator.pipeline.structured import StructuredResult


def run_s1(ctx: StageContext, doc: Document, reqs: list[Requirement],
           inventory: list[InventoryItem], *, second_pass: bool = False, part: str = "",
           author_answers: str = "") -> StructuredResult:
    """`part`: "" — the whole S1; "rules" or "contracts" — one half of a split S1."""
    from validator.pipeline import experiments as exp
    from validator.pipeline.prompts.experimental import S1_CONTRACT_TASK, S1_RULES_TASK

    pairs = ""
    if exp.enabled("EXP_PAIR_ENUM") and part != "contracts":
        pairs = exp.pairs_text(exp.pair_candidates(reqs), reverse=second_pass)
    task = {"rules": S1_RULES_TASK, "contracts": S1_CONTRACT_TASK}.get(part)
    prompt = build_s1_prompt(doc, reqs, inventory, include_document=ctx.include_document(doc.char_count),
                             pairs=pairs, second_pass=second_pass, task=task, author_answers=author_answers)
    return ctx.call("set_check", LLMSetCheck, prompt, system_prompt())


def run_s2(ctx: StageContext, doc: Document, reqs: list[Requirement],
           skipped: list[SkippedBlock], entities: str = "", *,
           author_answers: str = "") -> StructuredResult:
    prompt = build_s2_prompt(doc, reqs, skipped, include_document=ctx.include_document(doc.char_count),
                             entities=entities, author_answers=author_answers)
    return ctx.call("set_check", LLMSetCheck, prompt, system_prompt())


def run_pairs(ctx: StageContext, doc: Document, reqs: list[Requirement], *,
              author_answers: str = "") -> StructuredResult:
    prompt = build_pairs_prompt(doc, reqs, include_document=ctx.include_document(doc.char_count),
                                author_answers=author_answers)
    return ctx.call("set_check", LLMSetCheck, prompt, system_prompt())


def run_scenarios(ctx: StageContext, doc: Document, reqs: list[Requirement]) -> StructuredResult:
    """Scenario writer: concrete cases replayed against the text; returned as an ordinary set check."""
    from dataclasses import replace

    from validator.pipeline import experiments as exp
    from validator.pipeline.prompts.experimental import build_scenario_prompt
    from validator.pipeline.prompts.io import LLMScenarios

    prompt = build_scenario_prompt(doc, reqs, include_document=ctx.include_document(doc.char_count))
    res = ctx.call("set_check", LLMScenarios, prompt, system_prompt())
    return replace(res, data=exp.scenarios_to_set(res.data))


def run_entities(ctx: StageContext, doc: Document) -> StructuredResult:
    from validator.pipeline.experiments import ENTITY_TASK, LLMEntities
    from validator.pipeline.prompts.render import render_document
    from validator.pipeline.prompts.rubric import EXTRACT_SYSTEM

    return ctx.call("extract", LLMEntities, render_document(doc) + "\n\n" + ENTITY_TASK, EXTRACT_SYSTEM)


def run_merged(ctx: StageContext, doc: Document, reqs: list[Requirement], inventory: list[InventoryItem],
               skipped: list[SkippedBlock], *, author_answers: str = "") -> StructuredResult:
    prompt = build_merged_prompt(doc, reqs, inventory, skipped, include_document=ctx.include_document(doc.char_count),
                                 author_answers=author_answers)
    return ctx.call("set_check", LLMSetCheck, prompt, system_prompt())
