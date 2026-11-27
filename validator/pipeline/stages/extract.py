from __future__ import annotations

from validator.models import Document
from validator.pipeline.prompts.extract import build_extract_prompt
from validator.pipeline.prompts.io import LLMExtraction
from validator.pipeline.prompts.rubric import EXTRACT_SYSTEM
from validator.pipeline.stages import StageContext
from validator.pipeline.structured import StructuredResult


def run_extract(ctx: StageContext, doc: Document) -> StructuredResult:
    return ctx.call("extract", LLMExtraction, build_extract_prompt(doc), EXTRACT_SYSTEM)
