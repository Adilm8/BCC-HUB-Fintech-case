"""Quote verification: every quote in the report must come from the document text."""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from validator.models import Document, QuoteStatus

_TRANSLATE = str.maketrans({
    "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‟": '"', "'": '"', "‘": '"', "’": '"', "`": "",
    "–": "-", "—": "-", "−": "-", "‒": "-", "‑": "-",
    "ё": "е", "Ё": "е",
    "*": "", "|": " ",
})
_ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.)\s*")
FUZZY_THRESHOLD = 0.9
SHORT_QUOTE = 12  # shorter quotes are accepted only when unique within the referenced blocks


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalized text plus, for every normalized char, the index of its source char."""
    out: list[str] = []
    origin: list[int] = []
    for idx, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).translate(_TRANSLATE):
            if c.isspace():
                if not out or out[-1] == " ":
                    continue
                out.append(" ")
            else:
                out.append(c.lower())
            origin.append(idx)
    while out and out[-1] == " ":
        out.pop()
        origin.pop()
    return "".join(out), origin


def normalize(text: str) -> str:
    return normalize_with_map(text)[0]


@dataclass
class QuoteMatch:
    text: str  # real document fragment
    block_ids: list[str]
    status: QuoteStatus
    score: float = 1.0


class QuoteIndex:
    """Exact → normalized → fuzzy search of a quote over document blocks."""

    def __init__(self, doc: Document):
        self.doc = doc
        self.block_ids = [b.id for b in doc.blocks]
        self._pos = {b.id: i for i, b in enumerate(doc.blocks)}
        parts: list[str] = []
        self._starts: list[int] = []  # start of each block in the combined original text
        offset = 0
        for b in doc.blocks:
            self._starts.append(offset)
            parts.append(b.text)
            offset += len(b.text) + 1
        self.combined = "\n".join(parts)
        self.norm, self._map = normalize_with_map(self.combined)
        # normalized start of each block
        self._nstarts = [bisect.bisect_left(self._map, s) for s in self._starts]
        self._block_norm = [normalize(b.text) for b in doc.blocks]

    # ------------------------------------------------------------ helpers

    def block_order(self, block_id: str) -> int:
        return self._pos.get(block_id, 10**9)

    def has_block(self, block_id: str) -> bool:
        return block_id in self._pos

    def _blocks_for_span(self, nstart: int, nend: int) -> list[str]:
        first = bisect.bisect_right(self._nstarts, nstart) - 1
        last = bisect.bisect_right(self._nstarts, max(nstart, nend - 1)) - 1
        return self.block_ids[max(first, 0): max(last, 0) + 1]

    def _original(self, nstart: int, nend: int) -> str:
        ostart = self._map[nstart]
        oend = self._map[nend - 1] + 1
        return self.combined[ostart:oend].replace("\n", " ").strip()

    def _norm_range(self, block_ids: list[str] | None) -> tuple[int, int] | None:
        """Normalized span covering the given blocks and their direct neighbours."""
        if not block_ids:
            return None
        idx = sorted(self._pos[b] for b in block_ids if b in self._pos)
        if not idx:
            return None
        lo = max(idx[0] - 1, 0)
        hi = min(idx[-1] + 1, len(self.block_ids) - 1)
        end = self._nstarts[hi + 1] if hi + 1 < len(self._nstarts) else len(self.norm)
        return self._nstarts[lo], end

    def _occurrences(self, needle: str, lo: int = 0, hi: int | None = None) -> list[int]:
        hi = len(self.norm) if hi is None else hi
        found, start = [], lo
        while True:
            i = self.norm.find(needle, start, hi)
            if i < 0:
                return found
            found.append(i)
            start = i + 1

    # ------------------------------------------------------------ search

    def find(self, quote: str, prefer_blocks: list[str] | None = None, *, fuzzy: bool = True) -> QuoteMatch | None:
        if not quote or not quote.strip():
            return None
        fragments = [f for f in _ELLIPSIS.split(quote.strip()) if normalize(f)]
        if not fragments:
            return None
        if len(fragments) > 1:
            return self._find_fragments(fragments, prefer_blocks)
        return self._find_one(fragments[0], prefer_blocks, fuzzy=fuzzy)

    def _find_one(self, quote: str, prefer: list[str] | None, *, fuzzy: bool) -> QuoteMatch | None:
        nq = normalize(quote).strip(" .,;:")
        if not nq:
            return None
        rng = self._norm_range(prefer)
        local = self._occurrences(nq, *rng) if rng else []
        if len(nq) < SHORT_QUOTE:
            hits = local if rng else self._occurrences(nq)
            if len(hits) != 1:
                return None
            return self._match(hits[0], len(nq), QuoteStatus.VERIFIED)
        if local:
            return self._match(local[0], len(nq), QuoteStatus.VERIFIED)
        hits = self._occurrences(nq)
        if hits:
            return self._match(hits[0], len(nq), QuoteStatus.VERIFIED)
        if fuzzy:
            return self._fuzzy(nq, prefer)
        return None

    def _match(self, nstart: int, length: int, status: QuoteStatus, score: float = 1.0) -> QuoteMatch:
        nend = nstart + length
        return QuoteMatch(self._original(nstart, nend), self._blocks_for_span(nstart, nend), status, score)

    def _find_fragments(self, fragments: list[str], prefer: list[str] | None) -> QuoteMatch | None:
        norm_frags = [normalize(f).strip(" .,;:") for f in fragments]
        if any(len(f) < 3 for f in norm_frags):
            norm_frags = [f for f in norm_frags if len(f) >= 3]
            if not norm_frags:
                return None
        rng = self._norm_range(prefer)
        for lo, hi in ([rng] if rng else []) + [(0, len(self.norm))]:
            pos, texts, blocks = lo, [], []
            for f in norm_frags:
                i = self.norm.find(f, pos, hi)
                if i < 0:
                    break
                texts.append(self._original(i, i + len(f)))
                for b in self._blocks_for_span(i, i + len(f)):
                    if b not in blocks:
                        blocks.append(b)
                pos = i + len(f)
            else:
                if sum(len(f) for f in norm_frags) < SHORT_QUOTE:
                    return None
                return QuoteMatch(" … ".join(texts), blocks, QuoteStatus.VERIFIED)
        return None

    def _fuzzy(self, nq: str, prefer: list[str] | None) -> QuoteMatch | None:
        candidates: list[int] = []
        if prefer:
            idx = sorted(self._pos[b] for b in prefer if b in self._pos)
            if idx:
                candidates = list(range(max(idx[0] - 1, 0), min(idx[-1] + 2, len(self.block_ids))))
        best: tuple[float, int, int] | None = None
        for pool in (candidates, range(len(self.block_ids))):
            for bi in pool:
                # a window over this block and the next one (quotes may span adjacent blocks)
                start = self._nstarts[bi]
                end_bi = min(bi + 2, len(self._nstarts))
                end = self._nstarts[end_bi] if end_bi < len(self._nstarts) else len(self.norm)
                hit = self._best_window(nq, start, end)
                if hit and (best is None or hit[0] > best[0]):
                    best = hit
            if best and best[0] >= FUZZY_THRESHOLD:
                break
        if best is None or best[0] < FUZZY_THRESHOLD:
            return None
        score, s, e = best
        return self._match(s, e - s, QuoteStatus.REPAIRED, score)

    def _best_window(self, nq: str, start: int, end: int) -> tuple[float, int, int] | None:
        text = self.norm[start:end]
        n = len(nq)
        if len(text) < n * 0.7:
            return None
        sm = SequenceMatcher(autojunk=False)
        sm.set_seq2(nq)
        best: tuple[float, int, int] | None = None
        step = max(1, n // 8)
        last = max(len(text) - n, 0)
        positions = list(range(0, last + 1, step))
        if positions[-1] != last:
            positions.append(last)
        for p in positions:
            window = text[p: p + n]
            sm.set_seq1(window)
            if sm.real_quick_ratio() < FUZZY_THRESHOLD - 0.05 or sm.quick_ratio() < FUZZY_THRESHOLD - 0.05:
                continue
            r = sm.ratio()
            if best is None or r > best[0]:
                best = (r, p, p + n)
        if best is None:
            return None
        # refine boundaries around the best window
        _, p0, _ = best
        for dp in range(-step, step + 1):
            for dl in (-3, -2, -1, 0, 1, 2, 3):
                p = p0 + dp
                q_end = p + n + dl
                if p < 0 or q_end > len(text) or q_end <= p:
                    continue
                sm.set_seq1(text[p:q_end])
                r = sm.ratio()
                if r > best[0]:
                    best = (r, p, q_end)
        r, p, q = best
        # snap to word boundaries
        while p > 0 and text[p - 1].isalnum() and text[p].isalnum():
            p -= 1
        while q < len(text) and text[q - 1].isalnum() and text[q].isalnum():
            q += 1
        return r, start + p, start + q

    def contains(self, quote: str) -> bool:
        return self.find(quote, fuzzy=True) is not None


def word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", normalize(text)) if len(w) > 2}


def quotes_overlap(a: str, b: str, threshold: float = 0.6) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    wa, wb = word_set(a), word_set(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold
