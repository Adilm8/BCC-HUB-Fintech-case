"""The domain checklist block: hints from a domain profile (validator/domains), never findings."""

from __future__ import annotations

from validator.pipeline.prompts.render import safe

DOMAIN_NOTE = """Пункты <domain_checklist> — подсказки, а не нарушения. Находку оформляй, только если сам документ \
не задаёт проверяемого пункта там, где он применим, и с дословной цитатой из документа. Пункт чек-листа сам \
по себе находкой не является и не цитируется."""


def render_domain_block(profile, max_chars: int = 7000) -> str:
    lines: list[str] = []
    used = 0
    for item in profile.items:
        line = f"- {safe(item.topic)}: {safe(item.check)}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return ("<domain_checklist>\nДоменный чек-лист (подсказки, не нарушения):\n" + "\n".join(lines)
            + "\n</domain_checklist>\n" + DOMAIN_NOTE)
