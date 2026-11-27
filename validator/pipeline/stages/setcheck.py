from __future__ import annotations

from validator.models import Document, InventoryItem, Requirement, SkippedBlock
from validator.pipeline.prompts.io import LLMSetCheck
from validator.pipeline.prompts.rubric import system_prompt
from validator.pipeline.prompts.setcheck import build_merged_prompt, build_s1_prompt, build_s2_prompt
from validator.pipeline.stages import StageContext
from validator.pipeline.structured import StructuredResult


def run_s1(ctx: StageContext, doc: Document, reqs: list[Requirement],
           inventory: list[InventoryItem]) -> StructuredResult:
    prompt = build_s1_prompt(doc, reqs, inventory, include_document=ctx.include_document(doc.char_count))
    return ctx.call("set_check", LLMSetCheck, prompt, system_prompt())


def run_s2(ctx: StageContext, doc: Document, reqs: list[Requirement],
           skipped: list[SkippedBlock]) -> StructuredResult:
    prompt = build_s2_prompt(doc, reqs, skipped, include_document=ctx.include_document(doc.char_count))
    return ctx.call("set_check", LLMSetCheck, prompt, system_prompt())


def run_merged(ctx: StageContext, doc: Document, reqs: list[Requirement], inventory: list[InventoryItem],
               skipped: list[SkippedBlock]) -> StructuredResult:
    prompt = build_merged_prompt(doc, reqs, inventory, skipped, include_document=ctx.include_document(doc.char_count))
    return ctx.call("set_check", LLMSetCheck, prompt, system_prompt())
