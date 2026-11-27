"""Data contracts shared by every module: document, requirements, findings, report."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------- enums


class Criterion(StrEnum):
    # ISO/IEC/IEEE 29148:2018 5.2.5, individual requirement
    NECESSARY = "necessary"
    APPROPRIATE = "appropriate"
    UNAMBIGUOUS = "unambiguous"
    COMPLETE = "complete"
    SINGULAR = "singular"
    FEASIBLE = "feasible"
    VERIFIABLE = "verifiable"
    CORRECT = "correct"
    CONFORMING = "conforming"
    # ISO/IEC/IEEE 29148:2018 5.2.6, set of requirements
    SET_COMPLETE = "set_complete"
    SET_CONSISTENT = "set_consistent"
    SET_FEASIBLE = "set_feasible"
    SET_COMPREHENSIBLE = "set_comprehensible"
    SET_VALIDATABLE = "set_validatable"
    # IEEE 830-1998 4.3, not covered above
    RANKED = "ranked"
    MODIFIABLE = "modifiable"
    TRACEABLE = "traceable"


class Severity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


SEVERITY_RANK = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2}


class DefectClass(StrEnum):
    UNVERIFIABLE = "unverifiable"
    CONTRADICTORY = "contradictory"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"
    OTHER = "other"


class ReqKind(StrEnum):
    FUNCTIONAL = "functional"
    NONFUNCTIONAL = "nonfunctional"
    DATA = "data"
    INTERFACE = "interface"
    INTEGRATION = "integration"
    UI = "ui"
    BUSINESS_GOAL = "business_goal"
    PROCESS = "process"
    CONSTRAINT = "constraint"
    ACCEPTANCE_CRITERION = "acceptance_criterion"
    OTHER = "other"


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    CODE = "code"
    OTHER = "other"


class SetCheck(StrEnum):
    CONTRADICTION = "contradiction"
    CROSS_PART = "cross_part"
    ACCEPTANCE_MISMATCH = "acceptance_mismatch"
    ACCEPTANCE_GAP = "acceptance_gap"
    MISSING_REQUIREMENT = "missing_requirement"
    MISSING_SCENARIO = "missing_scenario"
    EMPTY_SECTION = "empty_section"
    TERMINOLOGY = "terminology"
    DUPLICATION = "duplication"
    TRACEABILITY = "traceability"
    OTHER = "other"


class QuoteStatus(StrEnum):
    VERIFIED = "verified"
    REPAIRED = "repaired"
    ANCHORED = "anchored"


class FindingSource(StrEnum):
    LLM = "llm"
    RULE = "rule"
    REVIEW = "review"


class StageKey(StrEnum):
    LOAD = "load"
    STRUCTURE = "structure"
    PRECHECKS = "prechecks"
    EXTRACT = "extract"
    CHECK = "check"
    SET_CHECK = "set_check"
    REVIEW = "review"
    ASSEMBLE = "assemble"


class StageState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


class RunStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    RULES_ONLY = "rules_only"


class Readiness(StrEnum):
    YES = "yes"
    CONDITIONAL = "conditional"
    NO = "no"


class Addressee(StrEnum):
    """Role that can answer a clarifying question."""

    BUSINESS = "business"
    ANALYST = "analyst"
    ARCHITECT = "architect"
    SECURITY = "security"
    QA = "qa"


class FindingDeltaStatus(StrEnum):
    """How a previous finding changed after applying author answers."""

    CLOSED = "closed"
    REMAINED = "remained"
    CHANGED = "changed"


_CONTRADICTORY_CHECKS = {SetCheck.CONTRADICTION, SetCheck.CROSS_PART, SetCheck.ACCEPTANCE_MISMATCH}
_INCOMPLETE_CHECKS = {SetCheck.MISSING_REQUIREMENT, SetCheck.MISSING_SCENARIO, SetCheck.EMPTY_SECTION,
                      SetCheck.ACCEPTANCE_GAP}


def defect_class_for(criterion: Criterion | str, set_check: SetCheck | str | None = None) -> DefectClass:
    """Map a criterion (and optional set check) to the organisers' defect classes."""
    c = Criterion(criterion)
    if c == Criterion.SET_COMPLETE:
        return DefectClass.INCOMPLETE  # something is missing, whatever check found it
    if set_check is not None:
        sc = SetCheck(set_check)
        if sc in _CONTRADICTORY_CHECKS:
            return DefectClass.CONTRADICTORY
        if sc in _INCOMPLETE_CHECKS:
            return DefectClass.INCOMPLETE
        if c == Criterion.SET_CONSISTENT:
            # a consistency criterion without a contradiction check is not a proven contradiction
            return DefectClass.AMBIGUOUS if sc == SetCheck.TERMINOLOGY else DefectClass.OTHER
    if c in (Criterion.VERIFIABLE, Criterion.SET_VALIDATABLE):
        return DefectClass.UNVERIFIABLE
    if c == Criterion.SET_CONSISTENT:
        return DefectClass.CONTRADICTORY
    if c in (Criterion.COMPLETE, Criterion.SET_COMPLETE):
        return DefectClass.INCOMPLETE
    if c in (Criterion.UNAMBIGUOUS, Criterion.SET_COMPREHENSIBLE):
        return DefectClass.AMBIGUOUS
    return DefectClass.OTHER


# ---------------------------------------------------------------- document


class RawDocument(BaseModel):
    path: str
    name: str
    title: str
    format: Literal["md", "txt", "pdf", "docx"]
    text: str
    warnings: list[str] = Field(default_factory=list)
    sha256: str | None = None  # of the original file bytes; None for text passed in directly


class Section(BaseModel):
    id: str
    title: str
    level: int
    path: list[str]
    parent_id: str | None = None
    part: str
    block_ids: list[str] = Field(default_factory=list)


class Block(BaseModel):
    id: str
    kind: BlockKind
    text: str
    raw: str
    section_id: str
    section_path: list[str]
    part: str
    line_start: int
    line_end: int
    doc_id: str | None = None
    cells: list[str] | None = None


class Document(BaseModel):
    path: str
    name: str
    title: str
    format: str
    sha256: str
    char_count: int
    text: str
    sections: list[Section]
    blocks: list[Block]
    parts: list[str]
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- requirements and findings


class Requirement(BaseModel):
    id: str
    doc_id: str | None = None
    id_assigned: bool
    kind: ReqKind
    part: str
    section: str
    block_ids: list[str]
    title: str
    quote: str
    check_status: Literal["checked", "not_checked"] = "not_checked"
    source: Literal["llm", "rule"] = "llm"


class Evidence(BaseModel):
    quote: str
    section: str
    requirement_id: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    quote_status: QuoteStatus


class BaseFinding(BaseModel):
    id: str
    criterion: Criterion
    severity: Severity
    defect_class: DefectClass
    evidence: list[Evidence] = Field(min_length=1)
    explanation: str
    recommendation: str
    clarifying_question: str
    answer_options: list[str] = Field(default_factory=list)
    option_consequences: list[str] = Field(default_factory=list)
    source: FindingSource
    review_note: str | None = None
    addressee: Addressee | None = None
    related_notes: list[str] = Field(default_factory=list)  # gist of findings merged into this one

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quote(self) -> str:
        return self.evidence[0].quote

    @computed_field  # type: ignore[prop-decorator]
    @property
    def section(self) -> str:
        return self.evidence[0].section


class Finding(BaseFinding):
    requirement_ids: list[str] = Field(min_length=1)
    remark: bool = False  # a possible gap noted in doubt, not a confirmed defect: always minor, no question


class SetFinding(BaseFinding):
    set_check: SetCheck
    requirement_ids: list[str] = Field(default_factory=list)
    cross_part: bool = False


# ---------------------------------------------------------------- report


class Question(BaseModel):
    id: str
    text: str
    options: list[str] = Field(default_factory=list)
    option_consequences: list[str] = Field(default_factory=list)
    finding_ids: list[str]
    requirement_ids: list[str] = Field(default_factory=list)
    section: str
    blocking: bool
    addressee: Addressee | None = None


class CoverageRow(BaseModel):
    requirement_id: str
    acceptance_ids: list[str] = Field(default_factory=list)
    status: Literal["covered", "partial", "none"]
    note: str = ""


class SkippedBlock(BaseModel):
    block_id: str
    reason: Literal["meta", "template", "empty", "other"]
    text: str


class Summary(BaseModel):
    requirements_total: int
    requirements_by_kind: dict[str, int]
    requirements_with_findings: int
    requirements_not_checked: int
    findings_total: int
    findings_by_severity: dict[str, int]
    findings_by_criterion: dict[str, int]
    critical_ids: list[str]
    critical_by_class: dict[str, list[str]]
    questions_total: int
    dropped_unverified_quotes: int
    dropped_by_review: int
    requirements_in_set_findings_only: int = 0  # touched only by set-level findings
    merged_duplicates: int = 0  # candidates merged into other findings
    requirements_clean: int = 0  # checked, no own findings, not touched by set findings
    candidates_total: int = 0  # every candidate that entered deduplication and review
    candidates_by_source: dict[str, int] = Field(default_factory=dict)
    dropped_unconfirmed: int = 0  # anchored candidates the critic did not confirm
    blocks_total: int = 0
    blocks_in_requirements: int = 0
    blocks_skipped: int = 0  # service, template and empty blocks
    blocks_structural: int = 0  # headings and table headers
    blocks_other: int = 0  # short fragments and code examples not assigned to a requirement
    questions_in_findings: int = 0  # clarifying questions printed in finding cards
    remarks_total: int = 0  # possible gaps noted in doubt (minor findings flagged `remark`)


class FixItem(BaseModel):
    finding_id: str
    title: str


class Verdict(BaseModel):
    development: Readiness
    testing: Readiness
    headline: str
    rationale: str = ""
    fix_first: list[FixItem] = Field(default_factory=list)


class StageStatus(BaseModel):
    key: StageKey
    title: str
    state: StageState
    detail: str = ""
    model: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class RunInfo(BaseModel):
    document_title: str
    document_name: str
    document_format: str
    document_sha256: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    provider: str
    models_used: list[str]
    agent_version: str
    profile: str
    llm_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # estimate from the list price; None when the price is unknown or free
    cost_note: str = ""
    status: RunStatus
    stages: list[StageStatus]
    standards: list[str]
    warnings: list[str] = Field(default_factory=list)


class DroppedCandidate(BaseModel):
    """A candidate finding removed after the model stages; for analysis only, not rendered."""

    candidate_id: str
    stage: Literal["review", "unconfirmed"]
    reason: str  # document_specifies | generic | out_of_scope | merged | unconfirmed
    merge_into: str = ""
    counter_quote: str = ""
    source: str
    kind: str  # requirement or the set check
    criterion: str
    severity: str
    requirement_ids: list[str] = Field(default_factory=list)
    quote: str = ""
    explanation: str = ""


class ReportDebug(BaseModel):
    critic_dropped: list[DroppedCandidate] = Field(default_factory=list)


class AuthorAnswer(BaseModel):
    question_id: str
    question_text: str = ""
    answer: str
    finding_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)


class FindingDelta(BaseModel):
    previous_id: str
    status: FindingDeltaStatus
    previous_severity: Severity
    current_id: str | None = None
    current_severity: Severity | None = None
    summary: str = ""
    answered_by: list[str] = Field(default_factory=list)
    note: str = ""


class QuestionEffectRow(BaseModel):
    question_id: str
    question_text: str
    answer: str
    finding_ids: list[str] = Field(default_factory=list)
    closed_ids: list[str] = Field(default_factory=list)
    changed_ids: list[str] = Field(default_factory=list)
    remained_ids: list[str] = Field(default_factory=list)
    effect: str


class RefinementSummary(BaseModel):
    previous_report_name: str = ""
    answers_total: int = 0
    previous_findings_by_severity: dict[str, int] = Field(default_factory=dict)
    current_findings_by_severity: dict[str, int] = Field(default_factory=dict)
    previous_findings_total: int = 0
    current_findings_total: int = 0
    previous_verdict_development: Readiness
    previous_verdict_testing: Readiness
    current_verdict_development: Readiness
    current_verdict_testing: Readiness
    previous_verdict_headline: str = ""
    current_verdict_headline: str = ""
    finding_deltas: list[FindingDelta] = Field(default_factory=list)
    question_effects: list[QuestionEffectRow] = Field(default_factory=list)


class Report(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run: RunInfo
    requirements: list[Requirement]
    skipped_blocks: list[SkippedBlock]
    findings: list[Finding]
    set_findings: list[SetFinding]
    coverage: list[CoverageRow]
    questions: list[Question]
    summary: Summary
    verdict: Verdict
    debug: ReportDebug | None = None  # analysis data, not part of the rendered report
    refinement: RefinementSummary | None = None


# ---------------------------------------------------------------- exchange between modules


class Hint(BaseModel):
    rule: str
    block_id: str
    match: str
    message: str


class RuleFinding(BaseModel):
    rule: str
    block_ids: list[str]
    criterion: Criterion
    severity: Severity
    set_check: SetCheck | None = None
    quote: str
    explanation: str
    recommendation: str
    clarifying_question: str


class InventoryItem(BaseModel):
    block_id: str
    kind: Literal["quantity", "code", "identifier"]
    value: str
    unit: str = ""
    text: str = ""


class PrecheckResult(BaseModel):
    hints: list[Hint] = Field(default_factory=list)
    findings: list[RuleFinding] = Field(default_factory=list)
    inventory: list[InventoryItem] = Field(default_factory=list)


class StageEvent(BaseModel):
    stage: StageKey
    state: StageState
    message: str = ""
    done: int | None = None
    total: int | None = None


STANDARDS = ["ISO/IEC/IEEE 29148:2018", "IEEE 830-1998"]

STAGE_TITLES: dict[StageKey, str] = {
    StageKey.LOAD: "Чтение документа",
    StageKey.STRUCTURE: "Разбор структуры",
    StageKey.PRECHECKS: "Проверки по правилам",
    StageKey.EXTRACT: "Нарезка требований",
    StageKey.CHECK: "Проверка каждого требования",
    StageKey.SET_CHECK: "Проверка набора требований",
    StageKey.REVIEW: "Критический пересмотр",
    StageKey.ASSEMBLE: "Сборка отчёта",
}
