from __future__ import annotations

from validator.models import Document
from validator.pipeline.prompts.io import LLMReview
from validator.pipeline.prompts.review import build_review_prompt
from validator.pipeline.prompts.rubric import system_prompt
from validator.pipeline.stages import StageContext
from validator.pipeline.structured import StructuredResult


def run_review(ctx: StageContext, doc: Document, candidates_text: str) -> StructuredResult:
    prompt = build_review_prompt(doc, candidates_text, include_document=ctx.include_document(doc.char_count))
    return ctx.call("review", LLMReview, prompt, system_prompt())
