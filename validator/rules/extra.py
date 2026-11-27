"""Optional deterministic rules, off by default (each behind its own switch in validator/rules/__init__.py).

non_singular                one requirement item bundles three or more independent rules
undefined_term              an abbreviation or code used in requirements is defined nowhere in the document
integration_without_failure a call to an external system with no timeout, unavailability or retry behaviour
"""

from __future__ import annotations

import re

from validator.models import Block, BlockKind, Criterion, Document, RuleFinding, SetCheck, Severity
from validator.rules.gaps import _make_finding, _quote_for_span

MAX_LISTED = 6
_TEXT_KINDS = {BlockKind.PARAGRAPH, BlockKind.LIST_ITEM, BlockKind.TABLE_ROW}
_ACCEPTANCE = re.compile(r"(критери\w*\s+при[её]мк|acceptance|тест-?кейс)", re.IGNORECASE)
_META = re.compile(r"(глоссари|термин|определени|сокращени|история\s+изменен|согласован|версии|"
                   r"лист\s+изменен|содержани|оглавлени)", re.IGNORECASE)


def _path(doc: Document, block: Block) -> list[str]:
    sec = next((s for s in doc.sections if s.id == block.section_id), None)
    return sec.path if sec else block.section_path


def _requirement_blocks(doc: Document) -> list[Block]:
    """Text blocks that state rules: not acceptance rows, glossaries or service sections."""
    out = []
    for b in doc.blocks:
        if b.kind not in _TEXT_KINDS:
            continue
        path = _path(doc, b)
        if any(_ACCEPTANCE.search(t) or _META.search(t) for t in path):
            continue
        out.append(b)
    return out


def _short(text: str, limit: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------- non_singular

_SEGMENT_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-ZА-ЯЁ0-9«(])|;\s+")
_OBLIGATION = re.compile(r"(?<!\w)(должн\w*|обязан\w*|необходимо|требуется|следует|не\s+допуска\w*|запрещ\w*|"
                         r"допуска\w*|разреш\w*|может|могут|не\s+может)(?!\w)", re.IGNORECASE)
_CONDITION = re.compile(r"(?<!\w)(если|при\s+\w+|когда|в\s+случае)(?!\w)", re.IGNORECASE)
_VERB = re.compile(r"(?<!\w)[а-яё]{3,}(ет|ит|ют|ят|ется|ится|ются|ятся)(?!\w)", re.IGNORECASE)
_JOINED = re.compile(r",?\s+и\s+(?=(?:[а-яё]+\s+){0,3}(?:должн|обязан|необходимо|[а-яё]{3,}(?:ет|ит|ют|ят)\b))",
                     re.IGNORECASE)
MIN_RULES = 4


def _rule_segments(text: str) -> list[str]:
    parts: list[str] = []
    for seg in _SEGMENT_SPLIT.split(text):
        parts.extend(p for p in _JOINED.split(seg) if p and p.strip())
    return [p for p in parts if _OBLIGATION.search(p) or (_CONDITION.search(p) and _VERB.search(p))
            or _VERB.search(p)]


def check_non_singular(doc: Document) -> list[RuleFinding]:
    scored = []
    for b in _requirement_blocks(doc):
        if b.kind == BlockKind.TABLE_ROW or len(b.text) < 120:
            continue
        n = len(_rule_segments(b.text))
        if n >= MIN_RULES:
            scored.append((n, b))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1].line_start))
    worst = scored[:MAX_LISTED]
    listed = "; ".join(f"«{_short(b.text)}» ({n} правил)" for n, b in worst)
    first = worst[0][1]
    return [_make_finding(
        rule="non_singular", block_ids=[b.id for _, b in worst], criterion=Criterion.SINGULAR,
        severity=Severity.MINOR, set_check=SetCheck.OTHER, quote=_quote_for_span(first.text),
        explanation=(f"Требования объединяют по несколько независимых правил в одном пункте: {listed}. "
                     "Их нельзя проверить, оценить и проследить по отдельности."),
        recommendation="Разделить каждый такой пункт на отдельные требования: одно условие и одно обязательство.",
        clarifying_question="Можно ли разделить эти пункты на отдельные требования с собственными номерами?",
        answer_options=["Пункты будут разделены на отдельные требования", "Правила связаны и остаются вместе"],
    )]


# ---------------------------------------------------------------- undefined_term

KNOWN_TERMS = {
    "API", "HTTP", "HTTPS", "JSON", "XML", "YAML", "HTML", "CSS", "SMS", "PUSH", "ID", "URL", "URI", "UTC", "GMT",
    "РФ", "РК", "ИИН", "БИН", "ИНН", "КПП", "ОГРН", "СНИЛС", "ФИО", "KZT", "RUB", "USD", "EUR", "NFR", "ТЗ", "ПО",
    "ОС", "ИТ", "IT", "UI", "UX", "PDF", "CSV", "SQL", "REST", "SOAP", "TLS", "SSL", "JWT", "UUID", "ISO", "IEC",
    "IEEE", "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OK", "IP", "DNS", "TCP", "SDK", "CRM", "ERP", "QA",
    "MVP", "KPI", "PIN", "TBD", "ДД", "ММ", "ГГ", "ГГГГ", "ЧЧ", "МБ", "ГБ", "КБ", "ТБ", "MB", "GB", "KB", "TB",
    "ИП", "ООО", "АО", "ТОО", "ЕС", "США", "МСК", "НДС", "PNG", "JPG", "JPEG", "GIF", "SVG", "ZIP", "UTF",
    "ASCII", "CPU", "RAM", "OAuth", "WWW", "FAQ", "PC", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
    "TTL", "SSH", "XLS", "XLSX", "DOCX", "OTP", "POS", "PAN", "CVV", "CVV2", "CVC", "MCC", "RPS", "TPS", "VPN",
    "CDN", "IBAN", "BIC", "БИК", "ЦБ", "ФНС", "НДФЛ", "ИС", "БД", "СУБД", "ИБ", "AML", "KYC", "P95", "P99",
    "АБС", "ДБО", "BFF", "DLQ", "COF", "SLA", "SSO", "MFA", "2FA", "OCR", "ETL", "BI",
}
_QUARTER = re.compile(r"^[QH][1-4]$")
_TOKEN = re.compile(r"(?<![\w\-/.$`])([A-ZА-ЯЁ][A-ZА-ЯЁa-zа-яё]{0,5}\d{0,2})(?![\w\-/`])")


def _is_abbreviation(tok: str) -> bool:
    if _QUARTER.match(tok):
        return False
    if re.fullmatch(r"[A-Z]{4,}", tok):
        return False  # a status, an enum value or a keyword rather than an abbreviation
    if re.fullmatch(r"[A-Z]{2,}[a-z]+", tok):
        return False  # a product or standard name written in its own case (a name, not an abbreviation)
    upper = sum(1 for c in tok if c.isupper())
    digits = sum(1 for c in tok if c.isdigit())
    letters = len(tok) - digits
    return 2 <= len(tok) <= 6 and letters >= 1 and (upper >= 2 or (upper >= 1 and digits >= 1)) and \
        not tok[1:].islower()


def _defined(tok: str, text: str, doc: Document) -> bool:
    t = re.escape(tok)
    if re.search(rf"(?<![\w-]){t}\s*(?:\)\s*)?(?:—|–|-\s|:|=|\(|это\b|означает|обозначает|расшифр)", text):
        return True
    if re.search(rf"\(\s*{t}\s*\)", text) or re.search(rf"[«\"]{t}[»\"]\s*(?:—|–|:)", text):
        return True
    for b in doc.blocks:
        if b.kind == BlockKind.TABLE_ROW and b.cells and b.cells[0].strip().strip("*`").startswith(tok):
            return True
    return False


def check_undefined_terms(doc: Document) -> list[RuleFinding]:
    text = "\n".join(b.text for b in doc.blocks)
    in_code = {m for b in doc.blocks if b.kind == BlockKind.CODE for m in _TOKEN.findall(b.text)}
    in_code |= set(re.findall(r"`([^`]{2,40})`", "\n".join(b.raw for b in doc.blocks)))
    first_use: dict[str, Block] = {}
    blocks = _requirement_blocks(doc)
    labelled = [b for b in blocks if b.doc_id]
    for b in labelled or blocks:  # numbered requirements when the document has them, not its passport table
        for m in _TOKEN.finditer(b.text):
            tok = m.group(1)
            around = b.text[max(0, m.start() - 12):m.end() + 12]
            if re.fullmatch(r"[A-Z]+", tok) and len(re.findall(r"(?<![\w])[A-Z]{2,}(?![\w])", around)) > 1:
                continue  # part of a keyword phrase (a query, a code), not a term
            if tok in KNOWN_TERMS or tok.upper() in KNOWN_TERMS or tok in in_code or not _is_abbreviation(tok):
                continue
            first_use.setdefault(tok, b)
    undefined = [(tok, b) for tok, b in first_use.items() if not _defined(tok, text, doc)]
    if not undefined:
        return []
    shown = undefined[:MAX_LISTED]
    listed = "; ".join(f"{tok} («{_short(b.text, 70)}»)" for tok, b in shown)
    more = f" и ещё {len(undefined) - MAX_LISTED}" if len(undefined) > MAX_LISTED else ""
    tok, first = shown[0]
    idx = first.text.find(tok)
    return [_make_finding(
        rule="undefined_term", block_ids=list(dict.fromkeys(b.id for _, b in shown)),
        criterion=Criterion.UNAMBIGUOUS, severity=Severity.MINOR, set_check=SetCheck.TERMINOLOGY,
        quote=_quote_for_span(first.text, idx, idx + len(tok)),
        explanation=(f"Сокращения и коды используются в требованиях, но нигде в документе не определены: "
                     f"{listed}{more}. Читатель вынужден угадывать их смысл."),
        recommendation="Добавить расшифровку каждого сокращения при первом упоминании или в глоссарий.",
        clarifying_question="Что означает каждое из этих сокращений?",
        answer_options=["Расшифровки будут добавлены в глоссарий", "Сокращения заменим полными названиями"],
    )]


# ---------------------------------------------------------------- integration_without_failure

_INTEGRATION_SECTION = re.compile(r"(интеграц|взаимодейств|внешн\w*\s+(?:систем|сервис)|вызов|обмен\s+данн|\bapi\b)",
                                  re.IGNORECASE)
_CALL = re.compile(r"(вызыва\w*|запрашива\w*|запрос\w*\s+(?:в|к)\s|обраща\w*|отправля\w*\s+запрос|"
                   r"получа\w*\s+(?:от|из)\s|переда[её]т?\w*\s+(?:в|во)\s|интеграц\w*|синхрониз\w*)", re.IGNORECASE)
_FAILURE = re.compile(r"(тайм-?аут|timeout|недоступ\w*|не\s+отвеча\w*|не\s+ответ\w*|повтор\w*|retry|ошиб\w*|сбо[йеия]\w*|"
                      r"отказ\w*|резервн\w*|деградац\w*)", re.IGNORECASE)
_ERROR_SECTION = re.compile(r"(ошиб|исключени|сбо[йеи]|отказ|error|недоступ)", re.IGNORECASE)
_SPECIFIC_FAILURE = re.compile(r"(тайм-?аут|timeout|недоступ\w*|не\s+отвеча\w*|повтор\w*|retry)", re.IGNORECASE)


def check_integration_failures(doc: Document) -> list[RuleFinding]:
    # an error-handling section that names timeouts, unavailability or retries covers every call
    for b in doc.blocks:
        if any(_ERROR_SECTION.search(t) for t in _path(doc, b)) and _SPECIFIC_FAILURE.search(b.text):
            return []
    section_text: dict[str, str] = {}
    for b in doc.blocks:
        section_text[b.section_id] = section_text.get(b.section_id, "") + "\n" + b.text
    bare = []
    for b in _requirement_blocks(doc):
        if b.kind == BlockKind.TABLE_ROW:
            continue  # field and parameter tables describe data, not the call
        if not any(_INTEGRATION_SECTION.search(t) for t in _path(doc, b)) or not _CALL.search(b.text):
            continue
        if _FAILURE.search(b.text) or _FAILURE.search(section_text.get(b.section_id, "")):
            continue
        bare.append(b)
    if not bare:
        return []
    shown = bare[:MAX_LISTED]
    listed = "; ".join(f"«{_short(b.text, 80)}»" for b in shown)
    more = f" и ещё {len(bare) - MAX_LISTED}" if len(bare) > MAX_LISTED else ""
    m = _CALL.search(shown[0].text)
    return [_make_finding(
        rule="integration_without_failure", block_ids=[b.id for b in shown], criterion=Criterion.SET_COMPLETE,
        severity=Severity.MAJOR, set_check=SetCheck.MISSING_SCENARIO,
        quote=_quote_for_span(shown[0].text, m.start(), m.end()),
        explanation=(f"Обращения к внешним системам описаны без ветки отказа: {listed}{more}. Не сказано, что "
                     "делать при тайм-ауте, недоступности или ошибке ответа и нужны ли повторы."),
        recommendation="Для каждого обращения задать тайм-аут, поведение при недоступности и ошибке, правила повтора.",
        clarifying_question="Что делает система, если внешняя система не отвечает вовремя или возвращает ошибку?",
        answer_options=["Повторить запрос по заданным правилам, затем сообщить об ошибке",
                        "Сразу сообщить об ошибке без повтора"],
    )]
