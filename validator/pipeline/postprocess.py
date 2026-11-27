"""Deterministic processing: requirements assembly, candidate validation, filters, dedup, numbering."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from validator.models import (
    SEVERITY_RANK,
    BlockKind,
    Criterion,
    Document,
    Evidence,
    Finding,
    FindingSource,
    QuoteStatus,
    ReqKind,
    Requirement,
    RuleFinding,
    SetCheck,
    SetFinding,
    Severity,
    SkippedBlock,
    defect_class_for,
)
from validator.pipeline.prompts.io import LLMExtraction, LLMReqResult, LLMReview, LLMSetFinding
from validator.pipeline.quotes import QuoteIndex, normalize, quotes_overlap, word_set

SEP = " › "
MINOR_ONLY = {Criterion.CONFORMING, Criterion.SINGULAR, Criterion.RANKED, Criterion.MODIFIABLE}
CONTRADICTION_CHECKS = {SetCheck.CONTRADICTION, SetCheck.CROSS_PART, SetCheck.ACCEPTANCE_MISMATCH}
# higher first: used when two findings about the same place are merged
CRITERION_PRIORITY = [
    Criterion.SET_CONSISTENT, Criterion.CORRECT, Criterion.COMPLETE, Criterion.SET_COMPLETE, Criterion.VERIFIABLE,
    Criterion.UNAMBIGUOUS, Criterion.FEASIBLE, Criterion.SET_VALIDATABLE, Criterion.SET_FEASIBLE,
    Criterion.APPROPRIATE, Criterion.NECESSARY, Criterion.SET_COMPREHENSIBLE, Criterion.TRACEABLE,
    Criterion.SINGULAR, Criterion.CONFORMING, Criterion.RANKED, Criterion.MODIFIABLE,
]
GENERIC_PATTERNS = [
    r"^требование недостаточно (детализировано|конкретно|четко|чётко)\.?$",
    r"^необходимо уточнить\.?$",
    r"^требует уточнения\.?$",
    r"^неясно\.?$",
]
_ROW_SKIP_KINDS = {BlockKind.HEADING, BlockKind.TABLE_HEADER}
_BLOCK_ID = re.compile(r"\bB\d{3,4}\b")


def section_label(path: list[str], doc_title: str = "") -> str:
    p = list(path)
    if len(p) > 1 and doc_title and p[0] == doc_title:
        p = p[1:]
    return SEP.join(p) if p else (doc_title or "—")


# ---------------------------------------------------------------- requirements


@dataclass
class Extracted:
    requirements: list[Requirement]
    skipped: list[SkippedBlock]
    added_by_code: int = 0


def build_requirements(doc: Document, extraction: LLMExtraction | None, index: QuoteIndex) -> Extracted:
    blocks = {b.id: b for b in doc.blocks}
    raw: list[tuple[int, Requirement]] = []
    used: set[str] = set()
    skipped: list[SkippedBlock] = []
    if extraction is not None:
        for s in extraction.skipped:
            for bid in s.block_ids:
                if bid in blocks and bid not in used:
                    skipped.append(SkippedBlock(block_id=bid, reason=s.reason, text=blocks[bid].text[:300]))
                    used.add(bid)
        for item in extraction.requirements:
            bids = [b for b in dict.fromkeys(item.block_ids) if b in blocks]
            if not bids and item.quote:
                m = index.find(item.quote)
                bids = m.block_ids if m else []
            if not bids:
                continue
            bids.sort(key=index.block_order)
            text = " ".join(blocks[b].text for b in bids)
            quote = text
            if item.quote.strip():
                m = index.find(item.quote, bids)
                if m and set(m.block_ids) & set(bids):
                    quote = m.text
            doc_id = _doc_id(item.doc_id, [blocks[b] for b in bids])
            first = blocks[bids[0]]
            raw.append((index.block_order(bids[0]), Requirement(
                id="", doc_id=doc_id, id_assigned=doc_id is None, kind=ReqKind(item.kind),
                part=first.part, section=section_label(first.section_path, doc.title), block_ids=bids,
                title=(item.title.strip() or text)[:160], quote=quote, source="llm",
            )))
            used.update(bids)
            # requirements whose blocks were also marked skipped stay requirements
            skipped = [s for s in skipped if s.block_id not in bids]
    added = 0
    for b in doc.blocks:
        if b.id in used or b.kind in _ROW_SKIP_KINDS or len(b.text) < 20:
            continue
        if extraction is not None and b.kind == BlockKind.CODE:
            continue
        raw.append((index.block_order(b.id), Requirement(
            id="", doc_id=b.doc_id, id_assigned=b.doc_id is None, kind=ReqKind.OTHER, part=b.part,
            section=section_label(b.section_path, doc.title), block_ids=[b.id], title=b.text[:160], quote=b.text,
            source="rule",
        )))
        added += 1
    raw.sort(key=lambda t: t[0])
    reqs = []
    for i, (_, r) in enumerate(raw, 1):
        reqs.append(r.model_copy(update={"id": f"R-{i:03d}"}))
    skipped.sort(key=lambda s: index.block_order(s.block_id))
    return Extracted(reqs, skipped, added)


def _doc_id(candidate: str, blocks) -> str | None:
    cand = (candidate or "").strip().strip("*.: ")
    if cand:
        nc = normalize(cand)
        for b in blocks:
            if nc and nc in normalize(b.text[: max(len(cand) + 20, 40)]):
                return cand
    for b in blocks:
        if b.doc_id:
            return b.doc_id
    return None


# ---------------------------------------------------------------- candidates


@dataclass
class Candidate:
    criterion: Criterion
    severity: Severity
    evidence: list[Evidence]
    explanation: str
    recommendation: str
    question: str
    options: list[str]
    requirement_ids: list[str]
    source: FindingSource
    set_check: SetCheck | None = None  # None: finding about a single requirement
    scenario: str = ""
    cid: str = ""
    review_note: str | None = None
    needs_confirmation: bool = False  # anchored quote: kept only if the critic confirms
    confirmed: bool = False
    final_id: str = ""
    merged_from: list[str] = field(default_factory=list)

    @property
    def is_set(self) -> bool:
        return self.set_check is not None

    @property
    def block_ids(self) -> set[str]:
        return {b for e in self.evidence for b in e.block_ids}

    @property
    def two_quote_contradiction(self) -> bool:
        verified = [e for e in self.evidence if e.quote_status != QuoteStatus.ANCHORED]
        return (self.set_check in CONTRADICTION_CHECKS or self.criterion == Criterion.SET_CONSISTENT) and len(
            verified) >= 2


@dataclass
class Stats:
    dropped_unverified_quotes: int = 0
    dropped_generic: int = 0
    dropped_unknown_ids: int = 0
    dropped_by_review: int = 0
    dropped_schema_items: int = 0
    merged: int = 0


class CandidateBuilder:
    def __init__(self, doc: Document, reqs: list[Requirement], index: QuoteIndex, stats: Stats):
        self.doc = doc
        self.reqs = {r.id: r for r in reqs}
        self.by_doc_id = {normalize(r.doc_id): r.id for r in reqs if r.doc_id}
        self.index = index
        self.stats = stats
        self.blocks = {b.id: b for b in doc.blocks}
        self._block_req: dict[str, str] = {}
        for r in reqs:
            for b in r.block_ids:
                self._block_req.setdefault(b, r.id)

    # ------------------------------------------------------------ helpers

    def resolve_req(self, rid: str) -> str | None:
        rid = (rid or "").strip()
        if rid in self.reqs:
            return rid
        m = re.fullmatch(r"R-?(\d+)", rid, re.IGNORECASE)
        if m:
            cand = f"R-{int(m.group(1)):03d}"
            if cand in self.reqs:
                return cand
        return self.by_doc_id.get(normalize(rid)) if rid else None

    def section_of(self, block_ids: list[str]) -> str:
        for b in block_ids:
            if b in self.blocks:
                return section_label(self.blocks[b].section_path, self.doc.title)
        return "—"

    def evidence(self, quote: str, rid: str | None) -> Evidence | None:
        prefer = self.reqs[rid].block_ids if rid else None
        m = self.index.find(quote, prefer)
        if m is None:
            return None
        if rid is None:
            rid = next((self._block_req[b] for b in m.block_ids if b in self._block_req), None)
        return Evidence(quote=m.text, section=self.section_of(m.block_ids), requirement_id=rid,
                        block_ids=m.block_ids, quote_status=m.status)

    def anchored(self, rid: str) -> Evidence:
        r = self.reqs[rid]
        return Evidence(quote=r.quote[:300], section=r.section, requirement_id=rid, block_ids=r.block_ids,
                        quote_status=QuoteStatus.ANCHORED)

    def requirement_evidence(self, rid: str) -> Evidence:
        """Requirement text as evidence: it is document text, so it is verified."""
        r = self.reqs[rid]
        return Evidence(quote=r.quote[:300], section=r.section, requirement_id=rid, block_ids=r.block_ids,
                        quote_status=QuoteStatus.VERIFIED)

    def clean_text(self, text: str) -> str:
        """Internal block ids never reach the report: replace them with requirement ids or drop them."""
        def repl(m: re.Match) -> str:
            rid = self._block_req.get(m.group(0))
            return rid or ""

        out = _BLOCK_ID.sub(repl, text or "")
        out = re.sub(r"\(\s*[,;]?\s*\)", "", out)
        out = re.sub(r"\s{2,}", " ", out)
        return out.replace(" ,", ",").replace(" .", ".").strip()

    @staticmethod
    def _question(q: str) -> str:
        q = (q or "").strip()
        if q and not q.endswith("?"):
            q = q.rstrip(".!") + "?"
        return q

    def _generic(self, explanation: str) -> bool:
        e = explanation.strip()
        if len(e) < 30:
            return True
        low = normalize(e)
        return any(re.match(p, low) for p in GENERIC_PATTERNS)

    def _section_by_title(self, title: str):
        nt = normalize(title or "")
        if not nt:
            return None
        return next((s for s in self.doc.sections if normalize(s.title) == nt), None) or next(
            (s for s in self.doc.sections if nt in normalize(s.title)), None)

    def _section_blocks(self, title: str) -> list[str]:
        sec = self._section_by_title(title)
        return [b for b in sec.block_ids if b in self.blocks] if sec else []

    # ------------------------------------------------------------ sources

    def from_check(self, results: list[LLMReqResult]) -> tuple[list[Candidate], set[str]]:
        out: list[Candidate] = []
        seen: set[str] = set()
        for res in results:
            rid = self.resolve_req(res.requirement_id)
            if rid is None:
                # try to bind each finding by its quote
                pass
            else:
                seen.add(rid)
            for f in res.findings:
                frid = rid
                ev = self.evidence(f.quote, frid)
                if frid is None:
                    if ev is None or ev.requirement_id is None:
                        self.stats.dropped_unknown_ids += 1
                        continue
                    frid = ev.requirement_id
                needs_conf = False
                if ev is None:
                    ev = self.anchored(frid)
                    needs_conf = True
                if self._generic(f.explanation):
                    self.stats.dropped_generic += 1
                    continue
                crit = Criterion(f.criterion)
                related = [r for r in (self.resolve_req(x) for x in f.related_requirement_ids) if r and r != frid]
                sev = Severity(f.severity)
                if crit == Criterion.SET_CONSISTENT and related:
                    evidence = [ev] + [self.requirement_evidence(r) for r in related[:2]]
                    out.append(Candidate(crit, sev, evidence, f.explanation.strip(), f.recommendation.strip(),
                                         self._question(f.clarifying_question), f.answer_options[:4],
                                         [frid, *related], FindingSource.LLM, set_check=SetCheck.CONTRADICTION,
                                         needs_confirmation=needs_conf))
                    continue
                if crit == Criterion.SET_CONSISTENT:
                    crit = Criterion.CORRECT
                out.append(Candidate(crit, sev, [ev], f.explanation.strip(), f.recommendation.strip(),
                                     self._question(f.clarifying_question), f.answer_options[:4], [frid],
                                     FindingSource.LLM, needs_confirmation=needs_conf))
        return out, seen

    def from_set(self, findings: list[LLMSetFinding], source: FindingSource = FindingSource.LLM) -> list[Candidate]:
        out = []
        for f in findings:
            evidence: list[Evidence] = []
            req_ids: list[str] = []
            for e in f.evidence:
                rid = self.resolve_req(e.requirement_id) if e.requirement_id else None
                ev = self.evidence(e.quote, rid)
                if ev is None:
                    continue
                if any(normalize(x.quote) == normalize(ev.quote) for x in evidence):
                    continue
                evidence.append(ev)
                if ev.requirement_id and ev.requirement_id not in req_ids:
                    req_ids.append(ev.requirement_id)
            if not evidence:
                self.stats.dropped_unverified_quotes += 1
                continue
            if self._generic(f.explanation):
                self.stats.dropped_generic += 1
                continue
            check = SetCheck(f.set_check)
            crit = Criterion(f.criterion)
            sev = Severity(f.severity)
            explanation = f.explanation.strip()
            if check in CONTRADICTION_CHECKS:
                # a contradiction without two places or a scenario is not proven
                if (len(evidence) < 2 or not f.scenario.strip()) and sev == Severity.CRITICAL:
                    sev = Severity.MAJOR
                if crit not in (Criterion.SET_CONSISTENT,):
                    crit = Criterion.SET_CONSISTENT
            if f.scenario.strip():
                explanation = explanation.rstrip() + " Пример: " + f.scenario.strip()
            parts = {self.blocks[b].part for e in evidence for b in e.block_ids if b in self.blocks}
            cross = len(parts) > 1
            if cross and check == SetCheck.CONTRADICTION:
                check = SetCheck.CROSS_PART
            c = Candidate(crit, sev, evidence, explanation, f.recommendation.strip(),
                          self._question(f.clarifying_question), f.answer_options[:4], req_ids, source,
                          set_check=check, scenario=f.scenario.strip())
            out.append(c)
        return out

    def from_rules(self, findings: list[RuleFinding]) -> list[Candidate]:
        out = []
        for f in findings:
            prefer = [b for b in f.block_ids if b in self.blocks]
            if not prefer:
                prefer = self._section_blocks(f.quote)
            m = self.index.find(f.quote, prefer) if f.quote else None
            if m is None and prefer:
                b = self.blocks[prefer[0]]
                m_ev = Evidence(quote=b.text[:300], section=self.section_of(prefer), block_ids=prefer[:1],
                                requirement_id=self._block_req.get(prefer[0]), quote_status=QuoteStatus.VERIFIED)
            elif m is not None:
                m_ev = Evidence(quote=m.text, section=self.section_of(m.block_ids), block_ids=m.block_ids,
                                requirement_id=next((self._block_req[b] for b in m.block_ids if b in self._block_req),
                                                    None), quote_status=m.status)
            else:
                sec = self._section_by_title(f.quote)
                if sec is None:
                    self.stats.dropped_unverified_quotes += 1
                    continue
                # a section title is document text even when the section has no blocks of its own
                m_ev = Evidence(quote=sec.title, section=section_label(sec.path, self.doc.title), block_ids=[],
                                requirement_id=None, quote_status=QuoteStatus.VERIFIED)
            rids = [m_ev.requirement_id] if m_ev.requirement_id else []
            check = f.set_check
            if check is None and not rids:
                check = SetCheck.OTHER
            out.append(Candidate(f.criterion, f.severity, [m_ev], f.explanation, f.recommendation,
                                 self._question(f.clarifying_question), [], rids, FindingSource.RULE,
                                 set_check=check))
        return out


# ---------------------------------------------------------------- guards, dedup


def clean_texts(cands: list[Candidate], builder: CandidateBuilder) -> None:
    for c in cands:
        c.explanation = builder.clean_text(c.explanation)
        c.recommendation = builder.clean_text(c.recommendation)
        c.question = builder.clean_text(c.question)
        c.options = [builder.clean_text(o) for o in c.options]


def apply_guards(cands: list[Candidate]) -> None:
    for c in cands:
        if c.criterion in MINOR_ONLY:
            c.severity = Severity.MINOR
        anchored = all(e.quote_status == QuoteStatus.ANCHORED for e in c.evidence)
        if anchored and c.severity == Severity.CRITICAL:
            c.severity = Severity.MAJOR


def _crit_rank(c: Criterion) -> int:
    return CRITERION_PRIORITY.index(c) if c in CRITERION_PRIORITY else len(CRITERION_PRIORITY)


def _same_place(a: Candidate, b: Candidate) -> bool:
    if not (a.block_ids & b.block_ids):
        return False
    for ea in a.evidence:
        for eb in b.evidence:
            if set(ea.block_ids) & set(eb.block_ids) and quotes_overlap(ea.quote, eb.quote):
                if a.is_set and b.is_set and a.set_check != b.set_check and not (
                        a.set_check in CONTRADICTION_CHECKS and b.set_check in CONTRADICTION_CHECKS):
                    continue
                return True
    return False


def _related_explanations(a: Candidate, b: Candidate) -> bool:
    """Two findings on overlapping quotes merge only when they talk about the same thing."""
    if a.criterion == b.criterion and not (a.is_set and b.is_set):
        return True
    wa, wb = word_set(a.explanation), word_set(b.explanation)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.35


def merge_into(winner: Candidate, loser: Candidate) -> None:
    if SEVERITY_RANK[loser.severity] < SEVERITY_RANK[winner.severity]:
        winner.severity = loser.severity
    for e in loser.evidence:
        if not any(normalize(e.quote) == normalize(x.quote) for x in winner.evidence):
            winner.evidence.append(e)
    for r in loser.requirement_ids:
        if r not in winner.requirement_ids:
            winner.requirement_ids.append(r)
    if not winner.question and loser.question:
        winner.question = loser.question
        winner.options = loser.options
    winner.confirmed = winner.confirmed or loser.confirmed
    winner.needs_confirmation = winner.needs_confirmation and loser.needs_confirmation
    winner.merged_from.append(loser.cid)


def _priority(c: Candidate) -> tuple:
    return (0 if c.is_set and c.set_check in CONTRADICTION_CHECKS else 1, SEVERITY_RANK[c.severity],
            _crit_rank(c.criterion), 0 if c.source == FindingSource.LLM else 1, -len(c.evidence))


def dedup(cands: list[Candidate], stats: Stats) -> list[Candidate]:
    ordered = sorted(cands, key=_priority)
    kept: list[Candidate] = []
    for c in ordered:
        target = next((k for k in kept if _same_place(k, c) and _related_explanations(k, c)), None)
        if target is None:
            kept.append(c)
            continue
        # a requirement finding merged with a set finding becomes a set finding
        if c.is_set and not target.is_set:
            target.set_check = c.set_check
            target.criterion = c.criterion
        merge_into(target, c)
        stats.merged += 1
    return kept


# ---------------------------------------------------------------- review


def assign_candidate_ids(cands: list[Candidate]) -> None:
    for i, c in enumerate(cands, 1):
        c.cid = f"C{i:02d}"


def candidates_for_review(cands: list[Candidate], limit_chars: int = 60000) -> str:
    lines = []
    for c in cands:
        quotes = " || ".join(f"«{e.quote[:200]}» [{e.quote_status.value}]" for e in c.evidence[:3])
        kind = c.set_check.value if c.set_check else "requirement"
        lines.append(
            f"{c.cid} | {','.join(c.requirement_ids) or '—'} | {kind} | {c.criterion.value} | {c.severity.value} | "
            f"{quotes} | {c.explanation[:400]} | вопрос: {c.question[:200]}"
        )
    text = "\n".join(lines)
    return text[:limit_chars]


@dataclass
class ReviewOutcome:
    fix_first: list[str]
    rationale: str
    missed: list[Candidate]


def apply_review(cands: list[Candidate], review: LLMReview, builder: CandidateBuilder,
                 stats: Stats) -> tuple[list[Candidate], ReviewOutcome]:
    by_id = {c.cid: c for c in cands}
    dropped: set[str] = set()
    for d in review.decisions:
        c = by_id.get(d.finding_id.strip())
        if c is None or c.cid in dropped:
            continue
        verdict = d.verdict
        if verdict == "keep":
            c.confirmed = True
            new_sev = Severity(d.severity)
            if not (c.two_quote_contradiction and SEVERITY_RANK[new_sev] > SEVERITY_RANK[c.severity]):
                c.severity = new_sev
        elif verdict in ("merge",) or d.drop_reason == "duplicate":
            target = by_id.get(d.merge_into.strip())
            if target is None or target is c or target.cid in dropped:
                c.confirmed = True
                continue
            if c.two_quote_contradiction and not _same_place(target, c):
                # a proven contradiction is merged only with a finding about the same place
                c.confirmed = True
                continue
            if c.is_set and not target.is_set:
                # keep the set-level finding: fold the requirement finding into it
                merge_into(c, target)
                dropped.add(target.cid)
                c.confirmed = True
                stats.merged += 1
                continue
            merge_into(target, c)
            dropped.add(c.cid)
            stats.merged += 1
        elif verdict == "drop":
            if d.drop_reason == "document_specifies":
                if d.counter_quote.strip() and builder.index.find(d.counter_quote) is not None:
                    dropped.add(c.cid)
                    stats.dropped_by_review += 1
                else:
                    c.confirmed = True  # unproven drop: the finding stays
                    c.review_note = "критик предложил удалить без подтверждающей цитаты; находка оставлена"
            elif d.drop_reason in ("generic", "out_of_scope"):
                if c.two_quote_contradiction:
                    c.confirmed = True
                    continue
                dropped.add(c.cid)
                stats.dropped_by_review += 1
            else:
                c.confirmed = True
        if d.improved_question.strip() and c.cid not in dropped:
            c.question = CandidateBuilder._question(d.improved_question)
    kept = [c for c in cands if c.cid not in dropped]
    missed = builder.from_set(review.missed[:5], FindingSource.REVIEW)
    for m in missed:
        m.confirmed = True
    fix = [x.strip() for x in review.fix_first if x.strip() in by_id and x.strip() not in dropped]
    return kept, ReviewOutcome(fix[:7], review.verdict_rationale.strip(), missed)


def drop_unconfirmed_anchored(cands: list[Candidate], reviewed: bool, stats: Stats) -> list[Candidate]:
    """Findings with only an anchored quote stay only if the critic confirmed them."""
    out = []
    for c in cands:
        if c.needs_confirmation and reviewed and not c.confirmed:
            stats.dropped_unverified_quotes += 1
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------- numbering


MAX_EVIDENCE = 4


def compact_evidence(evidence: list[Evidence], limit: int = MAX_EVIDENCE) -> list[Evidence]:
    """Drop quotes contained in another kept quote; keep at most `limit`, preferring new blocks."""
    kept: list[Evidence] = []
    for e in evidence:
        ne = normalize(e.quote)
        clash = next((i for i, k in enumerate(kept) if ne in normalize(k.quote) or normalize(k.quote) in ne), None)
        if clash is None:
            kept.append(e)
        elif len(ne) < len(normalize(kept[clash].quote)) and kept[clash].quote_status == e.quote_status:
            kept[clash] = e  # the shorter quote points at the defect more precisely
    if len(kept) <= limit:
        return kept
    out = kept[:2]
    seen = {b for e in out for b in e.block_ids}
    for e in kept[2:]:
        if len(out) >= limit:
            break
        if not set(e.block_ids) <= seen:
            out.append(e)
            seen.update(e.block_ids)
    for e in kept[2:]:
        if len(out) >= limit:
            break
        if e not in out:
            out.append(e)
    return out


def finalize(cands: list[Candidate], reqs: list[Requirement]) -> tuple[list[Finding], list[SetFinding]]:
    order = {r.id: i for i, r in enumerate(reqs)}
    for c in cands:
        c.evidence = compact_evidence(c.evidence)

    def req_pos(c: Candidate) -> int:
        return min((order.get(r, 10**6) for r in c.requirement_ids), default=10**6)

    singles = sorted((c for c in cands if not c.is_set and c.requirement_ids),
                     key=lambda c: (req_pos(c), SEVERITY_RANK[c.severity], _crit_rank(c.criterion)))
    sets = sorted((c for c in cands if c.is_set or not c.requirement_ids),
                  key=lambda c: (SEVERITY_RANK[c.severity], 0 if c.set_check in CONTRADICTION_CHECKS else 1,
                                 req_pos(c)))
    findings: list[Finding] = []
    set_findings: list[SetFinding] = []
    for i, c in enumerate(singles, 1):
        c.final_id = f"D-{i:03d}"
        findings.append(Finding(
            id=c.final_id, criterion=c.criterion, severity=c.severity, defect_class=defect_class_for(c.criterion),
            evidence=c.evidence, explanation=c.explanation, recommendation=c.recommendation,
            clarifying_question=c.question, answer_options=c.options, source=c.source, review_note=c.review_note,
            requirement_ids=c.requirement_ids,
        ))
    for i, c in enumerate(sets, 1):
        c.final_id = f"S-{i:03d}"
        check = c.set_check or SetCheck.OTHER
        parts = {e.section.split(SEP)[0] for e in c.evidence}
        set_findings.append(SetFinding(
            id=c.final_id, criterion=c.criterion, severity=c.severity,
            defect_class=defect_class_for(c.criterion, check), evidence=c.evidence, explanation=c.explanation,
            recommendation=c.recommendation, clarifying_question=c.question, answer_options=c.options,
            source=c.source, review_note=c.review_note, set_check=check, requirement_ids=c.requirement_ids,
            cross_part=check == SetCheck.CROSS_PART or len(parts) > 1 and check in CONTRADICTION_CHECKS,
        ))
    return findings, set_findings


def kind_counts(reqs: list[Requirement]) -> dict[str, int]:
    return dict(Counter(r.kind.value for r in reqs))
