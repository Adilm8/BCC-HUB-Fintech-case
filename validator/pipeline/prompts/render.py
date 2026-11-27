"""How the document and requirement tables are shown to the model."""

from __future__ import annotations

from collections.abc import Iterable

from validator.models import BlockKind, Document, InventoryItem, Requirement

SEP = " › "


def safe(text: str) -> str:
    """Document text never closes or opens a prompt tag: angle brackets become look-alike characters."""
    return (text or "").replace("<", "‹").replace(">", "›")


def render_document(doc: Document, block_ids: set[str] | None = None) -> str:
    """Section headings (with their ids) and numbered blocks, in document order.

    With block_ids, only those blocks are shown, but every heading is kept so the model sees the structure."""
    by_section: dict[str, list] = {}
    for b in doc.blocks:
        by_section.setdefault(b.section_id, []).append(b)
    lines = [f"<document title=\"{safe(doc.title)}\">"]
    seen: set[str] = set()

    def emit_blocks(blocks: list) -> None:
        for b in blocks:
            if b.kind == BlockKind.HEADING:
                continue  # the section line below already shows the heading
            if block_ids is not None and b.id not in block_ids:
                continue
            prefix = {BlockKind.LIST_ITEM: "- ", BlockKind.TABLE_HEADER: "| ", BlockKind.TABLE_ROW: "| "}.get(b.kind, "")
            text = b.text if b.kind != BlockKind.CODE else b.text.replace("\n", " ")
            lines.append(f"[{b.id}] {prefix}{safe(text)}")

    for s in doc.sections:
        if s.level > 0 and s.title:
            lines.append(f"{'#' * min(max(s.level, 1), 6)} {safe(s.title)}  ({s.id})")
        emit_blocks(by_section.get(s.id, []))
        seen.add(s.id)
    emit_blocks([b for b in doc.blocks if b.section_id not in seen])
    lines.append("</document>")
    return "\n".join(lines)


def requirement_table(reqs: Iterable[Requirement], *, with_text: bool = True, limit: int = 400) -> str:
    rows = []
    for r in reqs:
        text = r.quote if with_text else r.title
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        rows.append(f"{r.id} | {safe(r.doc_id or '—')} | {r.kind.value} | {safe(r.part or '—')} | {safe(r.section)} | "
                     f"блоки {','.join(r.block_ids)} | {safe(text)}")
    return "\n".join(rows)


def inventory_table(items: Iterable[InventoryItem], limit: int = 150) -> str:
    rows = []
    for i, it in enumerate(items):
        if i >= limit:
            rows.append("…")
            break
        rows.append(f"{it.block_id} | {it.kind} | {it.value} {it.unit}".rstrip())
    return "\n".join(rows)
