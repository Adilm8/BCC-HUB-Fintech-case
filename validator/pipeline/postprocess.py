"""Deterministic processing: requirements assembly, candidate validation, filters, dedup, numbering."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from validator.models import (
    SEVERITY_RANK,
    AuthorAnswer,
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
# deterministic facts about the document: the critic may lower them but not delete them as "generic"
FACT_RULES = {"missing_req_ids", "no_ranking", "duplicate_id", "numbering_gap", "dangling_reference",
              "units_mismatch", "idempotency_without_key", "flag_with_mandatory_params", "one_line_process"}
_BLOCK_ID = re.compile(r"\bB\d{3,4}\b")
_CAND_ID = re.compile(r"\b(?:см\.?\s+)?C\d{2,3}\b", re.IGNORECASE)
_DUPLICATE_NOTE = re.compile(r"(дублир\w*|уже уч[её]т\w*|не добавля\w*)", re.IGNORECASE)
_HTTP_CODE = re.compile(r"\b[1-5]\d\d\b")
_OPEN_QUESTION = re.compile(r"^(как|какой|какая|какое|какие|каким|какую|каких|что|чем|когда|где|кто|сколько|"
                            r"почему|зачем|в каких|в каком|на каком|по какой|по каким|откуда|куда)\b",
                            re.IGNORECASE)
_YES_NO = re.compile(r"^(да|нет)\b", re.IGNORECASE)
_META_KEYS = re.compile(
    r"^(наименование|название|продукт|автор|author|developer|разработчик|владелец|owner|business owner|"
    r"product owner|аналитик|задача|jira|тип работы|влияние на|версия|дата|статус документа|gitlab|"
    r"repository|репозиторий|framework|фреймворк|проект|подразделение|фио|подпись|согласован)(?!\w)", re.IGNORECASE)


_ABBREVIATIONS = {"п", "пп", "ст", "рис", "см", "т.е", "т.д", "т.п", "др", "г", "гг", "напр", "т", "ч", "стр", "e.g",
                  "i.e", "vs", "no"}


def first_sentence(text: str, limit: int = 220) -> str:
    """First sentence; never splits inside quotes, after a single letter («Часть A.») or inside numbers."""
    t = " ".join((text or "").split())
    depth = 0
    ascii_open = False
    for i, ch in enumerate(t):
        if ch in "«“„":
            depth += 1
        elif ch in "»”":
            depth = max(0, depth - 1)
        elif ch == '"':
            ascii_open = not ascii_open
        elif ch in ".!?" and depth == 0 and not ascii_open:
            if i + 1 < len(t) and t[i + 1] != " ":
                continue
            start = t.rfind(" ", 0, i) + 1
            token = t[start:i].strip("(«\"").lower()
            if len(token) <= 1 or token in _ABBREVIATIONS or re.fullmatch(r"[\d.,]+", token):
                continue
            nxt = t[i + 2: i + 3]
            if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "«\"("):
                continue
            return short_text(t[: i + 1], limit)
    return short_text(t, limit)


def _sentence(text: str) -> str:
    s = first_sentence(text).strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s if s.endswith((".", "!", "?", "…")) else s + "."


def short_text(text: str, limit: int = 120) -> str:
    """Cut at a word boundary and mark the cut with an ellipsis."""
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:—-") + "…"


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
                if bid in blocks and _is_normative(blocks[bid]):
                    continue  # a binding statement is never service text, whatever the model said
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
            sep = "; " if all(blocks[b].kind == BlockKind.TABLE_ROW for b in bids) else " "
            text = sep.join(blocks[b].text for b in bids)
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
                title=short_text(item.title.strip() or quote), quote=quote, source="llm",
            )))
            used.update(bids)
            # requirements whose blocks were also marked skipped stay requirements
            skipped = [s for s in skipped if s.block_id not in bids]
    # document-info rows and the preamble are service text, whatever the model said about them
    meta_ids = {b.id for b in doc.blocks if _is_meta_block(b, doc.parts)}
    if meta_ids:
        kept_raw = []
        for pos, r in raw:
            if set(r.block_ids) <= meta_ids:
                for bid in r.block_ids:
                    if all(s.block_id != bid for s in skipped):
                        skipped.append(SkippedBlock(block_id=bid, reason="meta", text=blocks[bid].text[:300]))
                continue
            kept_raw.append((pos, r))
        raw = kept_raw
        used.update(meta_ids)
        for bid in meta_ids:
            if all(s.block_id != bid for s in skipped):
                skipped.append(SkippedBlock(block_id=bid, reason="meta", text=blocks[bid].text[:300]))
    added = 0
    for b in doc.blocks:
        if b.id in used or b.kind in _ROW_SKIP_KINDS or len(b.text) < 20:
            continue
        if extraction is not None and b.kind == BlockKind.CODE:
            continue
        raw.append((index.block_order(b.id), Requirement(
            id="", doc_id=b.doc_id, id_assigned=b.doc_id is None, kind=ReqKind.OTHER, part=b.part,
            section=section_label(b.section_path, doc.title), block_ids=[b.id], title=short_text(b.text), quote=b.text,
            source="rule",
        )))
        added += 1
    raw = _code_blocks_to_own_section(raw, blocks, index, doc)
    raw = _coarsen_table_rows(raw, blocks, index, doc)
    raw.sort(key=lambda t: t[0])
    # two different requirements never share a title: fall back to each one's own quote
    seen_titles: dict[str, int] = {}
    for _, r in raw:
        seen_titles[normalize(r.title)] = seen_titles.get(normalize(r.title), 0) + 1
    for i, (pos, r) in enumerate(raw):
        if seen_titles[normalize(r.title)] > 1 and normalize(r.quote) != normalize(r.title):
            raw[i] = (pos, r.model_copy(update={"title": short_text(r.quote)}))
    reqs = []
    for i, (_, r) in enumerate(raw, 1):
        reqs.append(r.model_copy(update={"id": f"R-{i:03d}"}))
    skipped.sort(key=lambda s: index.block_order(s.block_id))
    return Extracted(reqs, skipped, added)


_NORMATIVE = re.compile(r"(должн|обязан|\bshall\b|\bmust\b)", re.IGNORECASE)


_REQUIREMENT_HEADING = re.compile(r"требован|requirement|правил|rule", re.IGNORECASE)


def _is_normative(b) -> bool:
    """A block that binds the system: a modal verb, or a list item or paragraph under a requirements heading."""
    if b.kind in (BlockKind.HEADING, BlockKind.TABLE_HEADER, BlockKind.CODE):
        return False
    if _NORMATIVE.search(b.text):
        return True
    heading = b.section_path[-1] if b.section_path else ""
    return (b.kind in (BlockKind.LIST_ITEM, BlockKind.PARAGRAPH) and bool(_REQUIREMENT_HEADING.search(heading))
            and len(b.text) >= 20)


def _is_meta_block(b, parts: list[str] | None = None) -> bool:
    """A key-value row of document attributes, or a preamble paragraph before the first part."""
    if b.kind == BlockKind.TABLE_ROW and b.cells and len(b.cells) == 2 and _META_KEYS.match(b.cells[0].strip()):
        return True
    outside_parts = not b.part or (parts is not None and parts and b.part not in parts)
    return (b.kind == BlockKind.PARAGRAPH and outside_parts and len(b.section_path) <= 1
            and not _NORMATIVE.search(b.text))


def _code_blocks_to_own_section(raw, blocks, index, doc):
    """An example (code block) belongs to the heading it sits under, not to a requirement of another section."""
    moved: list[tuple[str, str]] = []  # (block id, section id)
    out = []
    for pos, r in raw:
        home = blocks[r.block_ids[0]].section_id
        keep = [b for b in r.block_ids if not (blocks[b].kind == BlockKind.CODE and blocks[b].section_id != home)]
        gone = [b for b in r.block_ids if b not in keep]
        if gone and keep:
            moved += [(b, blocks[b].section_id) for b in gone]
            text = " ".join(blocks[b].text for b in keep)
            quote = r.quote if normalize(r.quote) in normalize(text) else text
            r = r.model_copy(update={"block_ids": keep, "quote": quote})
        out.append((pos, r))
    for bid, sec in moved:
        host = next((i for i, (_, r) in enumerate(out) if blocks[r.block_ids[0]].section_id == sec), None)
        if host is not None:
            pos, r = out[host]
            ids = sorted([*r.block_ids, bid], key=index.block_order)
            out[host] = (pos, r.model_copy(update={"block_ids": ids, "quote": " ".join(blocks[b].text for b in ids)}))
        else:
            b = blocks[bid]
            above = [i for i, (_, r) in enumerate(out)
                     if blocks[r.block_ids[0]].part == b.part and index.block_order(r.block_ids[0]) < index.block_order(bid)]
            if above:
                host = max(above, key=lambda i: index.block_order(out[i][1].block_ids[-1]))
                pos, r = out[host]
                ids = sorted([*r.block_ids, bid], key=index.block_order)
                out[host] = (pos, r.model_copy(update={"block_ids": ids}))
            else:
                out.append((index.block_order(bid), Requirement(
                    id="", doc_id=None, id_assigned=True, kind=ReqKind.INTERFACE, part=b.part,
                    section=section_label(b.section_path, doc.title), block_ids=[bid],
                    title=short_text(f"Пример: {b.section_path[-1] if b.section_path else b.text}"), quote=b.text,
                    source="llm")))
    return out


_COARSE_KINDS = {ReqKind.INTERFACE, ReqKind.DATA}


def _table_key(bid: str, blocks, index) -> tuple[str, int] | None:
    """The table a row belongs to: its section plus the position of the first row of the run of table rows."""
    b = blocks[bid]
    if b.kind != BlockKind.TABLE_ROW:
        return None
    order = index.block_order(bid)
    start = order
    ids = index.block_ids
    while start > 0:
        prev = blocks[ids[start - 1]]
        if prev.kind not in (BlockKind.TABLE_ROW, BlockKind.TABLE_HEADER) or prev.section_id != b.section_id:
            break
        start -= 1
    return b.section_id, start


def _coarsen_table_rows(raw, blocks, index, doc):
    """Interface or data requirements that are single rows of the same table become one requirement per table.

    Done in code, whatever granularity the model chose; acceptance criteria and rules stay one row each."""
    groups: dict[tuple[str, int], list[int]] = {}
    for i, (_, r) in enumerate(raw):
        if r.kind not in _COARSE_KINDS:
            continue
        keys = {_table_key(b, blocks, index) for b in r.block_ids}
        if len(keys) != 1 or None in keys:
            continue
        groups.setdefault(next(iter(keys)), []).append(i)
    drop: set[int] = set()
    for (sec_id, _), members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda i: raw[i][0])
        ids = sorted({b for i in members for b in raw[i][1].block_ids}, key=index.block_order)
        first_pos, first = raw[members[0]]
        sec = next((s for s in doc.sections if s.id == sec_id), None)
        names = [blocks[b].cells[0] if blocks[b].cells else blocks[b].text.split(" | ")[0] for b in ids]
        title = short_text(f"Таблица «{sec.title if sec else first.section}»: " + ", ".join(n for n in names if n))
        doc_ids = [raw[i][1].doc_id for i in members if raw[i][1].doc_id]
        merged = first.model_copy(update={
            "block_ids": ids, "quote": "; ".join(blocks[b].text for b in ids), "title": title,
            "doc_id": doc_ids[0] if doc_ids else first.doc_id, "id_assigned": not doc_ids and first.id_assigned,
        })
        raw[members[0]] = (first_pos, merged)
        drop.update(members[1:])
    return [x for i, x in enumerate(raw) if i not in drop]


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
    addressee: str = ""
    protected: bool = False  # severity fixed by a deterministic rule: the critic may not lower or drop it
    no_merge: bool = False  # aggregated findings keep their own requirement list
    keep: bool = False  # a deterministic fact (missing ids, ranking): the critic may not drop it as generic
    rule: str = ""
    related: list[str] = field(default_factory=list)
    option_consequences: list[str] = field(default_factory=list)
    previous_ids: list[str] = field(default_factory=list)
    remark: bool = False  # a possible gap noted in doubt: always minor, no question

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
    merged_before_review: int = 0
    dropped_unconfirmed: int = 0
    drop_log: list = field(default_factory=list)  # DroppedCandidate rows for the report's debug section


def log_drop(stats: Stats, c: Candidate, stage: str, reason: str, *, merge_into: str = "",
             counter_quote: str = "") -> None:
    from validator.models import DroppedCandidate

    stats.drop_log.append(DroppedCandidate(
        candidate_id=c.cid, stage=stage, reason=reason, merge_into=merge_into, counter_quote=counter_quote[:300],
        source=c.source.value, kind=c.set_check.value if c.set_check else "requirement",
        criterion=c.criterion.value, severity=c.severity.value, requirement_ids=list(c.requirement_ids),
        quote=c.evidence[0].quote[:200] if c.evidence else "", explanation=c.explanation[:300]))


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
        if m.section_id is not None:
            # a heading: the evidence points at its section, not at a requirement
            sec = next(s for s in self.doc.sections if s.id == m.section_id)
            return Evidence(quote=m.text, section=section_label(sec.path, self.doc.title), requirement_id=None,
                            block_ids=m.block_ids, quote_status=m.status)
        # the requirement is the one that owns the matched text, not the one the model named
        if rid is None or not set(m.block_ids) & set(self.reqs[rid].block_ids):
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
        out = _CAND_ID.sub("", out)
        if self.doc.title:
            out = out.replace(f"«{self.doc.title} › ", "«").replace(f"{self.doc.title} › ", "")
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
                opts, cons = align_options_and_consequences(f.answer_options, f.option_consequences)
                if crit == Criterion.SET_CONSISTENT and related:
                    evidence = [ev] + [self.requirement_evidence(r) for r in related[:2]]
                    out.append(Candidate(crit, sev, evidence, f.explanation.strip(), f.recommendation.strip(),
                                         self._question(f.clarifying_question), opts,
                                         [frid, *related], FindingSource.LLM, set_check=SetCheck.CONTRADICTION,
                                         needs_confirmation=needs_conf, addressee=f.addressee,
                                         option_consequences=cons))
                    continue
                if crit == Criterion.SET_CONSISTENT:
                    crit = Criterion.CORRECT
                out.append(Candidate(crit, sev, [ev], f.explanation.strip(), f.recommendation.strip(),
                                     self._question(f.clarifying_question), opts,
                                     [frid], FindingSource.LLM, needs_confirmation=needs_conf,
                                     addressee=f.addressee, option_consequences=cons))
        return out, seen

    def from_remarks(self, results) -> list[Candidate]:
        """Remarks with a quote found in the document; anything else is dropped like an unverified finding."""
        out: list[Candidate] = []
        for res in results:
            rid = self.resolve_req(res.requirement_id)
            for r in getattr(res, "remarks", [])[:2]:
                ev = self.evidence(r.quote, rid) if r.quote.strip() else None
                if ev is None or not ev.requirement_id or not r.text.strip():
                    self.stats.dropped_unverified_quotes += 1
                    continue
                out.append(Candidate(Criterion(r.criterion), Severity.MINOR, [ev], r.text.strip(), "", "", [],
                                     [ev.requirement_id], FindingSource.LLM, remark=True))
        return out

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
            opts, cons = align_options_and_consequences(f.answer_options, f.option_consequences)
            c = Candidate(crit, sev, evidence, explanation, f.recommendation.strip(),
                          self._question(f.clarifying_question), opts, req_ids,
                          source, set_check=check, scenario=f.scenario.strip(), addressee=f.addressee,
                          option_consequences=cons)
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
                                 set_check=check, rule=f.rule, keep=f.rule in FACT_RULES,
                                 protected=f.severity == Severity.CRITICAL))
        return out


def candidates_from_previous_findings(findings: list[Finding]) -> list[Candidate]:
    """Rebuild review candidates from a previous report's per-requirement findings."""
    out: list[Candidate] = []
    for f in findings:
        out.append(Candidate(
            criterion=f.criterion,
            severity=f.severity,
            evidence=[e.model_copy(deep=True) for e in f.evidence],
            explanation=f.explanation,
            recommendation=f.recommendation,
            question=f.clarifying_question,
            options=list(f.answer_options),
            requirement_ids=list(f.requirement_ids),
            source=f.source,
            review_note=f.review_note,
            confirmed=True,
            addressee=f.addressee.value if f.addressee else "",
            protected=f.source == FindingSource.RULE and f.severity == Severity.CRITICAL,
            keep=f.source == FindingSource.RULE,
            related=list(f.related_notes),
            option_consequences=list(f.option_consequences),
            previous_ids=[f.id],
        ))
    return out


# ---------------------------------------------------------------- guards, dedup


def clean_texts(cands: list[Candidate], builder: CandidateBuilder) -> None:
    for c in cands:
        c.explanation = builder.clean_text(c.explanation)
        c.recommendation = builder.clean_text(c.recommendation)
        c.question = builder.clean_text(c.question)
        c.options = [builder.clean_text(o) for o in c.options]
        c.option_consequences = [builder.clean_text(x) for x in c.option_consequences]


def align_options_and_consequences(
    options: list[str],
    consequences: list[str] | None = None,
    limit: int = 4,
) -> tuple[list[str], list[str]]:
    """Distinct answer options paired with their consequences; fewer than two options yield empty lists."""
    cons_in = list(consequences or [])
    out_opts: list[str] = []
    out_cons: list[str] = []
    for idx, raw_opt in enumerate(options):
        opt = (raw_opt or "").strip()
        if opt and all(normalize(opt) != normalize(x) and not quotes_overlap(opt, x, 0.85) for x in out_opts):
            out_opts.append(opt)
            raw_cons = cons_in[idx] if idx < len(cons_in) else ""
            out_cons.append((raw_cons or "").strip())
    if len(out_opts) < 2:
        return [], []
    return out_opts[:limit], out_cons[:limit]


def distinct_options(options: list[str], limit: int = 4) -> list[str]:
    """Answer options that actually differ; fewer than two distinct options are no choice at all."""
    opts, _ = align_options_and_consequences(options, None, limit)
    return opts


def _status_code_conflict(c: Candidate, parts: dict[str, str]) -> bool:
    """Two verified places of one document part prescribe different HTTP status codes."""
    if c.set_check not in CONTRADICTION_CHECKS:
        return False
    verified = [e for e in c.evidence if e.quote_status != QuoteStatus.ANCHORED]

    def codes(e: Evidence) -> frozenset[str]:
        # an example body carries no code itself: the code is in the heading it sits under
        found = _HTTP_CODE.findall(e.quote)
        if not found and e.section:
            found = _HTTP_CODE.findall(e.section.split(SEP)[-1])
        return frozenset(found)

    # each place must prescribe exactly one code; quotes listing many codes are about schemas, not a conflict
    coded = [codes(e) for e in verified]
    single = [s for s in coded if len(s) == 1]
    if len(single) < 2 or len(set(single)) < 2:
        return False
    owners = {parts.get(b, "") for e in verified for b in e.block_ids}
    return len(owners) == 1


def _exclusive_places(c: Candidate) -> bool:
    """A contradiction needs at least two verified quotes from different blocks."""
    verified = [e for e in c.evidence if e.quote_status != QuoteStatus.ANCHORED]
    seen: list[set[str]] = []
    for e in verified:
        blocks = set(e.block_ids)
        if blocks and all(not (blocks & s) for s in seen):
            seen.append(blocks)
    return len(seen) >= 2


def apply_guards(cands: list[Candidate], parts: dict[str, str] | None = None) -> None:
    for c in cands:
        if c.set_check in CONTRADICTION_CHECKS and c.source != FindingSource.RULE and not _exclusive_places(c):
            c.set_check = SetCheck.OTHER  # one place cannot contradict itself: a gap or an ambiguity
        if c.criterion in MINOR_ONLY:
            c.severity = Severity.MINOR
        anchored = all(e.quote_status == QuoteStatus.ANCHORED for e in c.evidence)
        if anchored and c.severity == Severity.CRITICAL:
            c.severity = Severity.MAJOR
        if parts is not None and not anchored and _status_code_conflict(c, parts):
            c.severity = Severity.CRITICAL


def _crit_rank(c: Criterion) -> int:
    return CRITERION_PRIORITY.index(c) if c in CRITERION_PRIORITY else len(CRITERION_PRIORITY)


def _same_place(a: Candidate, b: Candidate) -> bool:
    if not (a.block_ids & b.block_ids):
        return False
    for ea in a.evidence:
        for eb in b.evidence:
            if set(ea.block_ids) & set(eb.block_ids) and quotes_overlap(ea.quote, eb.quote):
                if a.is_set and b.is_set and a.set_check != b.set_check and not (
                        a.set_check in CONTRADICTION_CHECKS and b.set_check in CONTRADICTION_CHECKS) and not (
                        SetCheck.OTHER in (a.set_check, b.set_check)):
                    continue
                return True
    return False


def _related_explanations(a: Candidate, b: Candidate) -> bool:
    """Two findings on overlapping quotes merge only when they talk about the same thing."""
    if a.criterion == b.criterion and not (a.is_set and b.is_set):
        return True
    # set findings compare by stems (inflected forms); anything involving a requirement finding stays strict
    sets = a.is_set and b.is_set
    wa = stem_set(a.explanation) if sets else word_set(a.explanation)
    wb = stem_set(b.explanation) if sets else word_set(b.explanation)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= (0.35 if sets else 0.45)


def stem_set(text: str) -> set[str]:
    """Crude stems (word prefixes) so that inflected Russian forms compare equal."""
    return {w[:5] for w in word_set(text) if len(w) > 3}


_GAP_CHECKS = {SetCheck.EMPTY_SECTION, SetCheck.MISSING_REQUIREMENT}


def _rule_twin(a: Candidate, b: Candidate) -> bool:
    """A rule finding and a model finding about the same placeholder or empty block."""
    if (a.source == FindingSource.RULE) == (b.source == FindingSource.RULE):
        return False
    rule, other = (a, b) if a.source == FindingSource.RULE else (b, a)
    if rule.set_check not in _GAP_CHECKS or rule.no_merge or not other.is_set:
        return False
    return bool(rule.block_ids & other.block_ids) and other.criterion in (
        Criterion.SET_COMPLETE, Criterion.COMPLETE, Criterion.SET_VALIDATABLE)


_OWN_GAP_CRITERIA = {Criterion.COMPLETE, Criterion.UNAMBIGUOUS, Criterion.VERIFIABLE}


def _keeps_own_gap(a: Candidate, b: Candidate) -> bool:
    """A requirement's own gap (an unhandled branch, an undefined term) is not absorbed by a set contradiction."""
    for x, y in ((a, b), (b, a)):
        if not x.is_set and x.criterion in _OWN_GAP_CRITERIA and y.is_set and y.set_check in CONTRADICTION_CHECKS:
            return True
    return False


def _loosely_related(a: Candidate, b: Candidate) -> bool:
    """The critic called them duplicates: refuse only when they share almost no content words."""
    wa, wb = stem_set(a.explanation), stem_set(b.explanation)
    if not wa or not wb:
        return True
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.2


def _same_quotes(a: Candidate, b: Candidate) -> bool:
    """Two findings built on exactly the same set of quotes (two or more) are one finding."""
    qa = {normalize(e.quote) for e in a.evidence}
    qb = {normalize(e.quote) for e in b.evidence}
    return len(qa) >= 2 and qa == qb


def _set_overlap(a: Candidate, b: Candidate) -> bool:
    """Two set findings quoting mostly the same blocks describe one defect."""
    if not (a.is_set and b.is_set) or a.no_merge or b.no_merge:
        return False
    if a.two_quote_contradiction and b.two_quote_contradiction:
        return False  # distinct contradictions may share a place; quote overlap decides for them
    inter = a.block_ids & b.block_ids
    small = min(len(a.block_ids), len(b.block_ids)) or 1
    return len(inter) >= 2 and len(inter) / small >= 0.5




def merge_into(winner: Candidate, loser: Candidate) -> None:
    if SEVERITY_RANK[loser.severity] < SEVERITY_RANK[winner.severity]:
        winner.severity = loser.severity
    # keep the gist of a merged finding that says something different, as a separate related note
    wa, wb = stem_set(winner.explanation), stem_set(loser.explanation)
    if wb and len(winner.related) < 3 and len(wa & wb) / len(wb) < 0.5 and not _DUPLICATE_NOTE.search(
            loser.explanation):
        gist = _sentence(loser.explanation)
        known = [winner.explanation, *winner.related]
        if len(gist) > 25 and all(not quotes_overlap(gist, k, 0.5) for k in known):
            winner.related.append(gist)
    for note in loser.related:
        if len(winner.related) < 3 and note not in winner.related:
            winner.related.append(note)
    winner.protected = winner.protected or loser.protected
    if not winner.addressee:
        winner.addressee = loser.addressee
    for e in loser.evidence:
        if not any(normalize(e.quote) == normalize(x.quote) for x in winner.evidence):
            winner.evidence.append(e)
    # an empty section belongs to no requirement: do not import foreign requirement ids into it
    if not (winner.set_check == SetCheck.EMPTY_SECTION and winner.source == FindingSource.RULE):
        for r in loser.requirement_ids:
            if r not in winner.requirement_ids:
                winner.requirement_ids.append(r)
    if not winner.question and loser.question:
        winner.question = loser.question
        winner.options = list(loser.options)
        winner.option_consequences = list(loser.option_consequences)
    winner.confirmed = winner.confirmed or loser.confirmed
    winner.needs_confirmation = winner.needs_confirmation and loser.needs_confirmation
    winner.merged_from.append(loser.cid)
    for pid in loser.previous_ids:
        if pid not in winner.previous_ids:
            winner.previous_ids.append(pid)


def _priority(c: Candidate) -> tuple:
    return (0 if c.is_set and c.set_check in CONTRADICTION_CHECKS else 1, SEVERITY_RANK[c.severity],
            _crit_rank(c.criterion), 0 if c.source == FindingSource.LLM else 1, -len(c.evidence))


def dedup(cands: list[Candidate], stats: Stats) -> list[Candidate]:
    ordered = sorted(cands, key=_priority)
    kept: list[Candidate] = []
    for c in ordered:
        target = next((k for k in kept if not (k.no_merge or c.no_merge) and (
            _same_quotes(k, c) or (not _keeps_own_gap(k, c) and (
                (_same_place(k, c) and _related_explanations(k, c))
                or (_set_overlap(k, c) and _related_explanations(k, c)) or _rule_twin(k, c)))
        )), None)
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
        from validator.pipeline.prompts.render import safe

        quotes = " || ".join(f"«{safe(e.quote[:200])}» [{e.quote_status.value}]" for e in c.evidence[:3])
        kind = c.set_check.value if c.set_check else "requirement"
        prev_tag = f" | пред. ID: {','.join(c.previous_ids)}" if c.previous_ids else ""
        lines.append(
            f"{c.cid} | {','.join(c.requirement_ids) or '—'} | {kind} | {c.criterion.value} | {c.severity.value} | "
            f"{quotes} | {safe(c.explanation[:400])} | вопрос: {safe(c.question[:200])}{prev_tag}"
        )
    text = "\n".join(lines)
    return text[:limit_chars]


_QUESTION_ID_RE = re.compile(r"\bQ-\d{3}\b", re.IGNORECASE)


def _matching_answer_ids(counter_quote: str, author_answers: list[AuthorAnswer]) -> list[str]:
    """Return question IDs whose author answer is substantiated by `counter_quote`."""
    raw = (counter_quote or "").strip()
    if not raw or not author_answers:
        return []
    cq = normalize(raw)
    explicit_qids = {m.upper() for m in _QUESTION_ID_RE.findall(raw)}
    matched: list[str] = []
    for ans in author_answers:
        na = normalize(ans.answer)
        if not na:
            continue
        if (
            ans.question_id.upper() in explicit_qids
            or cq == na
            or (len(cq) >= 6 and cq in na)
            or (len(na) >= 4 and na in cq)
            or quotes_overlap(raw, ans.answer, 0.55)
        ) and ans.question_id not in matched:
            matched.append(ans.question_id)
    return matched


@dataclass
class ReviewOutcome:
    fix_first: list[str]
    rationale: str
    missed: list[Candidate]
    closed_by_answer: dict[str, list[str]] = field(default_factory=dict)


def apply_review(cands: list[Candidate], review: LLMReview, builder: CandidateBuilder,
                 stats: Stats, *,
                 author_answers: list[AuthorAnswer] | None = None) -> tuple[list[Candidate], ReviewOutcome]:
    by_id = {c.cid: c for c in cands}
    dropped: set[str] = set()
    closed_by_answer: dict[str, list[str]] = {}
    answers = author_answers or []
    for d in review.decisions:
        c = by_id.get(d.finding_id.strip())
        if c is None or c.cid in dropped:
            continue
        verdict = d.verdict
        matched_qids = _matching_answer_ids(d.counter_quote, answers)
        lowers = SEVERITY_RANK[Severity(d.severity)] > SEVERITY_RANK[c.severity]
        if c.protected and (verdict != "keep" or lowers) and not matched_qids:
            c.confirmed = True  # a rule-fixed critical gap is neither dropped nor lowered by the critic
            continue
        if verdict == "keep":
            c.confirmed = True
            new_sev = Severity(d.severity)
            if not (c.two_quote_contradiction and lowers and not matched_qids):
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
            if c.no_merge or target.no_merge:
                c.confirmed = True  # aggregated findings keep their own requirement list
                continue
            if (c.criterion != target.criterion and not _loosely_related(target, c)) or _keeps_own_gap(c, target):
                # different criteria and different content, or a requirement's own gap: two defects
                c.confirmed = True
                continue
            if c.is_set and not target.is_set:
                # keep the set-level finding: fold the requirement finding into it
                merge_into(c, target)
                log_drop(stats, target, "review", "merged", merge_into=c.cid)
                dropped.add(target.cid)
                c.confirmed = True
                stats.merged += 1
                continue
            merge_into(target, c)
            log_drop(stats, c, "review", "merged", merge_into=target.cid)
            dropped.add(c.cid)
            stats.merged += 1
        elif verdict == "drop":
            if d.drop_reason == "answered":
                if matched_qids:
                    log_drop(stats, c, "review", "answered", counter_quote=d.counter_quote)
                    dropped.add(c.cid)
                    stats.dropped_by_review += 1
                    for pid in c.previous_ids:
                        closed_by_answer[pid] = list(matched_qids)
                else:
                    c.review_note = (
                        "критик предложил удалить со ссылкой на ответ автора без подтверждающего фрагмента; "
                        "находка оставлена"
                    )
            elif d.drop_reason == "document_specifies":
                in_doc = bool(d.counter_quote.strip()) and builder.index is not None and (
                    builder.index.find(d.counter_quote) is not None
                )
                if in_doc or matched_qids:
                    log_drop(stats, c, "review", "document_specifies", counter_quote=d.counter_quote)
                    dropped.add(c.cid)
                    stats.dropped_by_review += 1
                    if matched_qids:
                        for pid in c.previous_ids:
                            closed_by_answer[pid] = list(matched_qids)
                else:
                    # unproven drop: the finding stays, but an anchored one is not confirmed by it
                    c.review_note = "критик предложил удалить без подтверждающей цитаты; находка оставлена"
            elif d.drop_reason in ("generic", "out_of_scope"):
                if c.two_quote_contradiction or c.keep or c.source == FindingSource.RULE:
                    c.confirmed = True  # deterministic findings are removed only with a quote that disproves them
                    continue
                anchored = all(e.quote_status == QuoteStatus.ANCHORED for e in c.evidence)
                if c.severity != Severity.MINOR and not anchored and c.source != FindingSource.RULE:
                    # a verified major finding is not removed without proof: it stays as minor
                    c.severity = Severity.MINOR
                    c.confirmed = True
                    c.review_note = "критик счёл находку несущественной; критичность понижена до незначительной"
                    continue
                log_drop(stats, c, "review", d.drop_reason)
                dropped.add(c.cid)
                stats.dropped_by_review += 1
            else:
                c.confirmed = True
        if d.improved_question.strip() and c.cid not in dropped:
            c.question = CandidateBuilder._question(d.improved_question)
            # yes/no options do not fit a rewritten open question
            if _OPEN_QUESTION.match(c.question) and any(_YES_NO.match(o) for o in c.options):
                c.options = []
                c.option_consequences = []
    kept = [c for c in cands if c.cid not in dropped]
    # the critic adds only critical defects, never notes about duplicates
    missed = [m for m in builder.from_set(review.missed[:5], FindingSource.REVIEW)
              if m.severity == Severity.CRITICAL and not _DUPLICATE_NOTE.search(m.explanation)
              and not _CAND_ID.search(m.explanation + m.recommendation)]
    for m in missed:
        m.confirmed = True
    fix = [x.strip() for x in review.fix_first if x.strip() in by_id and x.strip() not in dropped]
    return kept, ReviewOutcome(fix[:7], review.verdict_rationale.strip(), missed, closed_by_answer)


_FOLD_CODE = re.compile(r"(?<![\d.,])([1-5]\d\d)(?!\d)")
_FOLD_METHOD = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b")
_FOLD_WRITE = re.compile(r"сохран|измен|модиф|запис|замен", re.IGNORECASE)
_FOLD_CRITERIA = {Criterion.CORRECT, Criterion.SET_CONSISTENT, Criterion.UNAMBIGUOUS}
_HTTP_CODES = {"200", "201", "204", "400", "401", "403", "404", "409", "422", "500"}


def _own_text(c: Candidate) -> str:
    return f"{c.explanation} {c.question}"


def _codes(c: Candidate, *, with_quotes: bool = False) -> set[str]:
    text = _own_text(c)
    if with_quotes:
        text += " " + " ".join(f"{e.quote} {e.section}" for e in c.evidence)
    return {x for x in _FOLD_CODE.findall(text) if x in _HTTP_CODES}


def _fold_twin(rule: Candidate, other: Candidate) -> bool:
    """The model finding describes the same interface defect as the linter finding (same place, same codes or
    the same method used for a write)."""
    if not (rule.block_ids & other.block_ids) or SEVERITY_RANK[other.severity] > SEVERITY_RANK[Severity.MAJOR]:
        return False
    rc, rm = _codes(rule), set(_FOLD_METHOD.findall(_own_text(rule)))
    if len(rc) >= 2:
        return len(rc & _codes(other, with_quotes=True)) >= 2 and len(rc & _codes(other)) >= 1
    if rm:
        return bool(rm & set(_FOLD_METHOD.findall(_own_text(other)))) and bool(_FOLD_WRITE.search(_own_text(other)))
    if len(rc) == 1:
        return bool(rc & _codes(other))
    return False


def fold_rule_twins(cands: list[Candidate]) -> tuple[list[Candidate], int]:
    """After the critic: a linter card absorbs at most one model twin about the same interface defect.

    The linter card keeps its text, options and severity floor (raised to the twin's if higher); the twin's
    explanation becomes a related note, its quotes and requirements are added. Pure: returns a new list."""
    removed: set[int] = set()
    folded = 0
    for rule in cands:
        if rule.source != FindingSource.RULE or not rule.rule.startswith("api_") or rule.criterion not in _FOLD_CRITERIA:
            continue
        twins = [o for o in cands if o.source != FindingSource.RULE and id(o) not in removed and _fold_twin(rule, o)]
        if not twins:
            continue
        twin = min(twins, key=lambda o: (SEVERITY_RANK[o.severity], 0 if o.is_set else 1, -len(o.evidence)))
        if SEVERITY_RANK[twin.severity] < SEVERITY_RANK[rule.severity]:
            rule.severity = twin.severity
        note = _sentence(twin.explanation) if twin.explanation else ""
        if note and note not in rule.related:
            rule.related = [note, *rule.related][:4]
        seen = {normalize(e.quote) for e in rule.evidence}
        rule.evidence += [e for e in twin.evidence if normalize(e.quote) not in seen][:3]
        for r in twin.requirement_ids:
            if r not in rule.requirement_ids:
                rule.requirement_ids.append(r)
        for pid in twin.previous_ids:
            if pid not in rule.previous_ids:
                rule.previous_ids.append(pid)
        if twin.set_check is not None and rule.set_check is None:
            rule.set_check = twin.set_check
        rule.confirmed = True
        removed.add(id(twin))
        folded += 1
    return [c for c in cands if id(c) not in removed], folded


def place_remarks(remarks: list[Candidate], kept: list[Candidate]) -> list[Candidate]:
    """Remarks about places no kept finding of the requirement itself covers, one per place."""
    out: list[Candidate] = []
    for r in remarks:
        if any(r.block_ids & c.block_ids for c in kept
               if not c.is_set and set(c.requirement_ids) & set(r.requirement_ids)):
            continue
        if any(r.block_ids & o.block_ids and r.criterion == o.criterion for o in out):
            continue
        r.severity = Severity.MINOR
        out.append(r)
    return out


def drop_unconfirmed_anchored(cands: list[Candidate], reviewed: bool, stats: Stats) -> list[Candidate]:
    """Findings with only an anchored quote stay only if the critic confirmed them."""
    out = []
    for c in cands:
        if c.needs_confirmation and not c.confirmed:
            stats.dropped_unconfirmed += 1
            log_drop(stats, c, "unconfirmed", "unconfirmed")
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------- numbering


MAX_EVIDENCE = 4
ADDRESSEES = {"business", "analyst", "architect", "security", "qa"}
_SECURITY_WORDS = re.compile(r"(безопасн|персональн|маскир|токен|утечк|журнал\w* .*карт|ИБ\b|шифров)", re.IGNORECASE)


RULE_OPTIONS: dict[str, list[str]] = {
    "placeholder": ["Значение будет указано в документе до начала разработки",
                    "Параметр не применим: отметить «не применимо» с причиной"],
    "empty_cell": ["Ячейки будут заполнены в документе", "Строка лишняя и будет удалена"],
    "empty_section": ["Раздел будет заполнен до передачи в разработку",
                      "Раздел не применим к задаче: исключить его с пояснением"],
    "unfilled_template_table": ["Автор заполнит чек-лист до передачи в разработку",
                                "Чек-лист не используется для этой задачи: удалить его из документа"],
    "missing_req_ids": ["Ввести сквозную нумерацию всех требований",
                        "Нумеровать только функциональные требования и критерии приёмки"],
    "no_ranking": ["Указать приоритет каждого требования (обязательно / желательно)",
                   "Все требования обязательны: зафиксировать это одной фразой"],
    "duplicate_id": ["Перенумеровать повторяющиеся идентификаторы", "Это одно требование: объединить записи"],
    "numbering_gap": ["Пропуск намеренный: требование удалено", "Требование потеряно: восстановить его"],
    "dangling_reference": ["Добавить недостающий раздел или требование", "Исправить ссылку на существующий пункт"],
    "api_get_write_side_effect": ["Изменение выполняется методом POST или PUT с параметрами в теле запроса",
                                  "Оставить GET и обосновать, почему изменение состояния допустимо"],
    "api_status_code_conflict": ["Оставить код из перечня статусов и исправить текст подраздела",
                                 "Оставить код из текста подраздела и добавить его в перечень статусов"],
    "api_undeclared_status_code": ["Добавить код в перечень статусов с описанием и примером",
                                   "Заменить код в тексте на один из объявленных"],
    "api_success_code_error_body": ["Ошибка возвращается с кодом ошибки из перечня и единым телом ошибки",
                                    "Ошибка возвращается с кодом успеха и признаком ошибки: исправить перечень"],
    "api_status_heading_mismatch": ["Исправить заголовок подраздела под описанный код",
                                    "Исправить описание под код из заголовка"],
    "api_req_resp_type_mismatch": ["Привести поле к одному типу в запросе и ответе",
                                   "Различие намеренное: описать преобразование типа"],
    "api_body_content_type_unused": ["Убрать Content-Type для запроса без тела", "Описать тело запроса"],
    "api_get_with_body": ["Убрать тело из запроса GET", "Сменить метод на POST или PUT"],
    "units_mismatch": ["Хранить и сравнивать в одной единице, указать её в требовании",
                       "Оставить разные единицы и описать правило пересчёта и округления"],
    "idempotency_without_key": ["Повтор распознаётся по идентификатору операции или события; указать окно хранения",
                                "Повторная доставка невозможна по контракту источника: зафиксировать это"],
    "flag_with_mandatory_params": ["При выключенном признаке остальные параметры необязательны и не проверяются",
                                   "Параметры обязательны всегда: описать, что с ними делать при выключении"],
    "one_line_process": ["Описать участников, шаги и ветки отказов в отдельной схеме процесса",
                         "Схема иллюстративная: сослаться на функциональные требования как на основной источник"],
}
RULE_OPTION_CONSEQUENCES: dict[str, list[str]] = {
    "placeholder": [
        "Заглушка в {loc} заменяется конкретным значением; пробел полноты закрывается.",
        "В {loc} фиксируется пометка «не применимо» с причиной; замечание о заглушке снимается.",
    ],
    "empty_cell": [
        "Пустые ячейки таблицы в {loc} заполняются фактическими значениями; пробел полноты снимается.",
        "Лишняя строка таблицы в {loc} удаляется; замечание о пустых ячейках закрывается.",
    ],
    "empty_section": [
        "В {loc} добавляются недостающие требования или данные; дефект пустого раздела закрывается.",
        "Раздел исключается или помечается как неприменимый с пояснением; замечание по {dat} снимается.",
    ],
    "unfilled_template_table": [
        "В таблице {gen} проставляются фактические отметки вместо шаблона; замечание закрывается.",
        "Неиспользуемый шаблонный чек-лист удаляется из {gen}; замечание снимается.",
    ],
    "missing_req_ids": [
        "Всем требованиям документа присваиваются уникальные ID; замечание по трассируемости закрывается.",
        "Нумерация вводится для функциональных требований и критериев приёмки; базовые связи трассируются.",
    ],
    "no_ranking": [
        "Каждому требованию назначается приоритет; замечание по ранжированию снимается.",
        "В документе фиксируется, что все требования обязательны; замечание по ранжированию закрывается.",
    ],
    "duplicate_id": [
        "Повторяющиеся идентификаторы в {loc} получают уникальные номера; конфликт трассируемости снимается.",
        "Дублирующиеся записи в {loc} объединяются в одно требование; дублирование ID закрывается.",
    ],
    "numbering_gap": [
        "В {loc} фиксируется намеренный пропуск номера; замечание о нарушении нумерации снимается.",
        "Потерянное требование восстанавливается в {loc}; разрыв нумерации закрывается.",
    ],
    "dangling_reference": [
        "Отсутствующий раздел или требование добавляется в документ; битая ссылка из {gen} закрывается.",
        "Ссылка в {loc} исправляется на существующий пункт; замечание по трассируемости снимается.",
    ],
    "api_get_write_side_effect": [
        "Метод в {loc} меняется на POST/PUT с параметрами в теле запроса; нарушение семантики HTTP закрывается.",
        "Метод GET в {loc} сохраняется с явным обоснованием; замечание переводится в принятое ограничение.",
    ],
    "api_status_code_conflict": [
        "Текст подраздела в {loc} приводится к коду из перечня статусов; противоречие кодов закрывается.",
        "Код из подраздела добавляется в перечень статусов в {loc}; противоречие кодов снимается.",
    ],
    "api_undeclared_status_code": [
        "Необъявленный код добавляется в перечень статусов {gen} с описанием и примером; расхождение закрывается.",
        "Упоминание кода в {loc} заменяется на один из объявленных статусов; расхождение снимается.",
    ],
    "api_success_code_error_body": [
        "Ошибки в {loc} переносятся на код ошибки с единым телом; противоречие статусов закрывается.",
        "Перечень статусов в {loc} обновляется под код успеха с признаком ошибки; противоречие снимается.",
    ],
    "api_status_heading_mismatch": [
        "Заголовок подраздела в {loc} приводится к описанному коду ответа; неоднозначность снимается.",
        "Описание и пример в {loc} приводятся к коду из заголовка; неоднозначность закрывается.",
    ],
    "api_req_resp_type_mismatch": [
        "Тип поля в запросе и ответе {gen} унифицируется; противоречие типов данных закрывается.",
        "В {loc} документируется правило преобразования типа поля; расхождение снимается.",
    ],
    "api_body_content_type_unused": [
        "Избыточный заголовок Content-Type удаляется из {gen}; расхождение описания закрывается.",
        "В {loc} добавляется описание тела запроса под указанный Content-Type; противоречие снимается.",
    ],
    "api_get_with_body": [
        "Тело запроса убирается из GET в {loc} с переносом параметров в query/path; дефект закрывается.",
        "Метод в {loc} меняется на POST или PUT под описанное тело запроса; нарушение снимается.",
    ],
    "units_mismatch": [
        "Для {gen} закрепляется единая единица измерения во всех разделах; конфликт единиц закрывается.",
        "В {loc} добавляются коэффициент пересчёта и правило округления; противоречие единиц снимается.",
    ],
    "idempotency_without_key": [
        "В {loc} фиксируются ключ дедупликации и окно его хранения; пробел идемпотентности закрывается.",
        "В {loc} фиксируется гарантия однократной доставки от источника; замечание по дедупликации снимается.",
    ],
    "flag_with_mandatory_params": [
        "В {loc} уточняется, что при выключенном флаге параметры необязательны и не проверяются; пробел закрывается.",
        "В {loc} описывается обработка обязательных параметров при выключенном флаге; пробел сценария снимается.",
    ],
    "one_line_process": [
        "Схема процесса в {loc} дополняется ветками отказов и альтернативными исходами; пробел закрывается.",
        "В {loc} фиксируется обзорный статус схемы со ссылкой на функциональные правила; замечание снимается.",
    ],
}
_TEST_STUB_CONSEQUENCES = [
    "В {loc} добавляются тестовые сценарии до начала разработки; пробел по тестированию закрывается.",
    "В {loc} добавляется ссылка на отдельный документ QA; пробел по тестированию снимается.",
]
# who answers questions of the deterministic gap rules
RULE_ADDRESSEE = {"units_mismatch": "analyst", "idempotency_without_key": "architect", "boundary_tests_missing": "qa",
                  "non_singular": "analyst", "undefined_term": "analyst", "integration_without_failure": "architect",
                  "flag_with_mandatory_params": "analyst", "one_line_process": "business"}


def _target_forms(c: Candidate) -> dict[str, str]:
    if c.requirement_ids:
        if len(c.requirement_ids) == 1:
            rid = c.requirement_ids[0]
            return {"loc": f"требовании {rid}", "gen": f"требования {rid}", "dat": f"требованию {rid}"}
        rids = ", ".join(c.requirement_ids[:3])
        return {"loc": f"требованиях {rids}", "gen": f"требований {rids}", "dat": f"требованиям {rids}"}
    if c.evidence and c.evidence[0].section and c.evidence[0].section != "—":
        sec = c.evidence[0].section
        return {"loc": f"разделе «{sec}»", "gen": f"раздела «{sec}»", "dat": f"разделу «{sec}»"}
    return {"loc": "документе", "gen": "документа", "dat": "документу"}


def rule_options_with_consequences(c: Candidate) -> tuple[list[str], list[str]]:
    where = " ".join([c.explanation, *(e.section for e in c.evidence)])
    forms = _target_forms(c)
    if c.rule in ("placeholder", "empty_section") and re.search(r"тестир", where, re.IGNORECASE):
        opts = ["Тестовые сценарии будут описаны в документе до начала разработки",
                "Тестирование ведётся по отдельному документу QA: дать ссылку на него"]
        cons = [t.format(**forms) for t in _TEST_STUB_CONSEQUENCES]
        return opts, cons
    if c.rule in RULE_OPTIONS:
        opts = list(RULE_OPTIONS[c.rule])
        cons = [t.format(**forms) for t in RULE_OPTION_CONSEQUENCES.get(c.rule, [])]
        return opts, cons
    if c.set_check == SetCheck.EMPTY_SECTION:
        opts = list(RULE_OPTIONS["empty_section"])
        cons = [t.format(**forms) for t in RULE_OPTION_CONSEQUENCES["empty_section"]]
        return opts, cons
    return [], []


def rule_options(c: Candidate) -> list[str]:
    return rule_options_with_consequences(c)[0]


def fill_missing_consequences(c: Candidate) -> list[str]:
    """Ensure every answer option has a short non-empty consequence aligned by index."""
    if not c.options:
        return []
    loc = _target_forms(c)["loc"]
    out: list[str] = []
    for idx, opt in enumerate(c.options):
        existing = c.option_consequences[idx].strip() if idx < len(c.option_consequences) else ""
        if existing:
            out.append(existing)
        else:
            out.append(f"Фиксирует вариант «{short_text(opt, 70)}» в {loc} и снимает связанный дефект.")
    return out


def infer_addressee(c: Candidate, reqs: dict[str, Requirement]) -> str:
    kinds_here = {reqs[r].kind for r in c.requirement_ids if r in reqs}
    if kinds_here and kinds_here <= {ReqKind.UI}:
        return "business"  # screen texts and behaviour are decided by the product owner or UX
    if c.addressee in ADDRESSEES:
        return c.addressee
    text = f"{c.explanation} {c.question}"
    if c.source == FindingSource.RULE:
        if c.rule in RULE_ADDRESSEE:
            return RULE_ADDRESSEE[c.rule]
        if c.rule.startswith("api_"):
            return "analyst"  # interface contract questions go to the API owner
        where = " ".join([text, *(e.section for e in c.evidence), *(e.quote for e in c.evidence)])
        if re.search(r"тестир|приёмк|приемк", where, re.IGNORECASE):
            return "qa"
        if re.search(r"\burl\b|swagger|endpoint|адрес|контур", where, re.IGNORECASE):
            return "architect"
        # templates, checklists, approvals, versions, numbering and priorities belong to the document's owner
        return "business"
    if _SECURITY_WORDS.search(text):
        return "security"
    if c.set_check in (SetCheck.ACCEPTANCE_GAP, SetCheck.ACCEPTANCE_MISMATCH, SetCheck.MISSING_SCENARIO):
        return "qa"
    kinds = {reqs[r].kind for r in c.requirement_ids if r in reqs}
    if kinds & {ReqKind.INTERFACE, ReqKind.DATA}:
        return "analyst"
    if kinds & {ReqKind.INTEGRATION, ReqKind.NONFUNCTIONAL}:
        return "architect"
    if kinds & {ReqKind.ACCEPTANCE_CRITERION}:
        return "qa"
    if kinds & {ReqKind.BUSINESS_GOAL, ReqKind.PROCESS, ReqKind.UI, ReqKind.FUNCTIONAL}:
        return "business"
    return "analyst"


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
    req_map = {r.id: r for r in reqs}
    for c in cands:
        c.evidence = compact_evidence(c.evidence)
        if c.set_check == SetCheck.EMPTY_SECTION and c.source == FindingSource.RULE:
            c.requirement_ids = []
            c.evidence = [e.model_copy(update={"requirement_id": None}) for e in c.evidence]
        c.addressee = infer_addressee(c, req_map)
        c.options, c.option_consequences = align_options_and_consequences(c.options, c.option_consequences)
        if not c.options and c.source == FindingSource.RULE:
            c.options, c.option_consequences = rule_options_with_consequences(c)
        if c.options:
            c.option_consequences = fill_missing_consequences(c)

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
            clarifying_question=c.question, answer_options=c.options, option_consequences=c.option_consequences,
            source=c.source, review_note=c.review_note,
            requirement_ids=c.requirement_ids, addressee=c.addressee, related_notes=c.related, remark=c.remark,
        ))
    for i, c in enumerate(sets, 1):
        c.final_id = f"S-{i:03d}"
        check = c.set_check or SetCheck.OTHER
        parts = {e.section.split(SEP)[0] for e in c.evidence}
        set_findings.append(SetFinding(
            id=c.final_id, criterion=c.criterion, severity=c.severity,
            defect_class=defect_class_for(c.criterion, check), evidence=c.evidence, explanation=c.explanation,
            recommendation=c.recommendation, clarifying_question=c.question, answer_options=c.options,
            option_consequences=c.option_consequences,
            source=c.source, review_note=c.review_note, set_check=check, requirement_ids=c.requirement_ids,
            cross_part=check == SetCheck.CROSS_PART or len(parts) > 1, addressee=c.addressee,
            related_notes=c.related,
        ))
    return findings, set_findings


def kind_counts(reqs: list[Requirement]) -> dict[str, int]:
    return dict(Counter(r.kind.value for r in reqs))
