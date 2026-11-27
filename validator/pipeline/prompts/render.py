"""How the document and requirement tables are shown to the model."""

from __future__ import annotations

from collections.abc import Iterable

from validator.models import BlockKind, Document, InventoryItem, Requirement

SEP = " › "


def render_document(doc: Document, block_ids: set[str] | None = None) -> str:
    """Numbered blocks; headings marked with '#'. With block_ids, only those blocks plus all headings."""
    levels = {s.id: s.level for s in doc.sections}
    lines = [f"<document title=\"{doc.title}\">"]
    for b in doc.blocks:
        if b.kind == BlockKind.HEADING:
            lines.append(f"[{b.id}] {'#' * max(levels.get(b.section_id, 1), 1)} {b.text}")
        elif block_ids is None or b.id in block_ids:
            prefix = {BlockKind.LIST_ITEM: "- ", BlockKind.TABLE_HEADER: "| ", BlockKind.TABLE_ROW: "| "}.get(b.kind, "")
            text = b.text if b.kind != BlockKind.CODE else b.text.replace("\n", " ")
            lines.append(f"[{b.id}] {prefix}{text}")
    lines.append("</document>")
    return "\n".join(lines)


def requirement_table(reqs: Iterable[Requirement], *, with_text: bool = True, limit: int = 400) -> str:
    rows = []
    for r in reqs:
        text = r.quote if with_text else r.title
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        rows.append(f"{r.id} | {r.doc_id or '—'} | {r.kind.value} | {r.part or '—'} | {r.section} | "
                     f"блоки {','.join(r.block_ids)} | {text}")
    return "\n".join(rows)


def inventory_table(items: Iterable[InventoryItem], limit: int = 150) -> str:
    rows = []
    for i, it in enumerate(items):
        if i >= limit:
            rows.append("…")
            break
        rows.append(f"{it.block_id} | {it.kind} | {it.value} {it.unit}".rstrip())
    return "\n".join(rows)
