"""Flat answer schemas for each model role. Field order matters: evidence first, labels last."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

KindLit = Literal["functional", "nonfunctional", "data", "interface", "integration", "ui", "business_goal",
                  "process", "constraint", "acceptance_criterion", "other"]
SeverityLit = Literal["critical", "major", "minor"]
ReqCriterionLit = Literal["necessary", "appropriate", "unambiguous", "complete", "singular", "feasible",
                          "verifiable", "correct", "conforming", "traceable", "set_consistent"]
SetCriterionLit = Literal["set_complete", "set_consistent", "set_feasible", "set_comprehensible",
                          "set_validatable", "ranked", "modifiable", "traceable", "appropriate"]
AddresseeLit = Literal["business", "analyst", "architect", "security", "qa"]
SetCheckLit = Literal["contradiction", "cross_part", "acceptance_mismatch", "acceptance_gap", "missing_requirement",
                      "missing_scenario", "empty_section", "terminology", "duplication", "traceability", "other"]


class LLMExtractedReq(BaseModel):
    block_ids: list[str]
    doc_id: str
    kind: KindLit
    title: str
    quote: str


class LLMSkipped(BaseModel):
    block_ids: list[str]
    reason: Literal["meta", "template", "empty", "other"]


class LLMExtraction(BaseModel):
    requirements: list[LLMExtractedReq]
    skipped: list[LLMSkipped]


class LLMReqFinding(BaseModel):
    quote: str
    explanation: str
    criterion: ReqCriterionLit
    severity: SeverityLit
    recommendation: str
    clarifying_question: str
    addressee: AddresseeLit = "analyst"  # who can answer the question
    answer_options: list[str]
    related_requirement_ids: list[str]


class LLMReqResult(BaseModel):
    requirement_id: str
    checklist: str  # short pass over the completeness checklist, written before the findings
    findings: list[LLMReqFinding]


class LLMCheck(BaseModel):
    results: list[LLMReqResult]


class LLMEvidence(BaseModel):
    requirement_id: str
    quote: str


class LLMSetFinding(BaseModel):
    evidence: list[LLMEvidence]
    scenario: str
    explanation: str
    set_check: SetCheckLit
    criterion: SetCriterionLit
    severity: SeverityLit
    recommendation: str
    clarifying_question: str
    addressee: AddresseeLit = "analyst"  # who can answer the question
    answer_options: list[str]


class LLMCoverage(BaseModel):
    requirement_id: str
    acceptance_ids: list[str]
    status: Literal["covered", "partial", "none"]
    note: str


class LLMLedgerItem(BaseModel):
    concept: str
    mentions: str  # every place the concept appears, with what each place says


class LLMSetCheck(BaseModel):
    ledger: list[LLMLedgerItem]
    findings: list[LLMSetFinding]
    coverage: list[LLMCoverage]


class LLMDecision(BaseModel):
    finding_id: str
    verdict: Literal["keep", "drop", "merge"]
    merge_into: str
    severity: SeverityLit
    drop_reason: Literal["none", "document_specifies", "generic", "duplicate", "out_of_scope"]
    counter_quote: str
    improved_question: str


class LLMReview(BaseModel):
    decisions: list[LLMDecision]
    missed: list[LLMSetFinding]
    fix_first: list[str]
    verdict_rationale: str
