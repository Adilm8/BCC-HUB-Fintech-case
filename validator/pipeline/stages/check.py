from __future__ import annotations

import math

from validator.config import Profile
from validator.models import Document, Hint, Requirement
from validator.pipeline.prompts.check import build_check_prompt
from validator.pipeline.prompts.io import LLMCheck
from validator.pipeline.prompts.rubric import system_prompt
from validator.pipeline.stages import StageContext
from validator.pipeline.structured import StructuredResult


def make_batches(reqs: list[Requirement], profile: Profile) -> list[list[Requirement]]:
    """Consecutive batches; each cut moves to the nearest section boundary around the target size."""
    if not reqs:
        return []
    if profile.batch_size <= 0 or len(reqs) <= profile.batch_size:
        return [reqs]
    n = min(math.ceil(len(reqs) / profile.batch_size), profile.max_batches)
    if n <= 1:
        return [reqs]
    size = math.ceil(len(reqs) / n)
    cuts: list[int] = []
    start = 0
    for _ in range(n - 1):
        target = start + size
        if target >= len(reqs):
            break
        window = max(1, size // 3)
        best = target
        for d in range(window + 1):
            for cand in (target - d, target + d):
                if start < cand < len(reqs) and reqs[cand].section != reqs[cand - 1].section:
                    best = cand
                    break
            else:
                continue
            break
        cuts.append(best)
        start = best
    out, prev = [], 0
    for c in cuts:
        out.append(reqs[prev:c])
        prev = c
    out.append(reqs[prev:])
    return [b for b in out if b]


def run_check_batch(ctx: StageContext, doc: Document, batch: list[Requirement], all_reqs: list[Requirement],
                    hints: list[Hint]) -> StructuredResult:
    prompt = build_check_prompt(doc, batch, all_reqs, hints, include_document=ctx.include_document(doc.char_count))
    return ctx.call("check", LLMCheck, prompt, system_prompt())
