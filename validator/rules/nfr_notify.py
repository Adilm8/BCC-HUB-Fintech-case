"""Rules nfr_unmeasurable, notification_content and mixed_levels: generic and deterministic."""

from __future__ import annotations

import re

from validator.models import BlockKind, Criterion, Document, RuleFinding, SetCheck, Severity
from validator.rules.extra import _requirement_blocks, _short
from validator.rules.gaps import _make_finding, _quote_for_span

MAX_NFR = 5
MAX_NOTIFY = 3

# ---------------------------------------------------------------- nfr_unmeasurable

_UNIT = r"(?:мс|миллисекунд\w*|секунд\w*|сек\.?|с|минут\w*|мин\.?|час\w*|ч)(?!\w)"
_TIME_BOUND = re.compile(
    rf"(в\s+течение\s+\d[\d\s,.]*\s*{_UNIT}|не\s+(?:более|дольше|позднее)\s+\d[\d\s,.]*\s*{_UNIT}|"
    rf"(?<!\w)за\s+\d[\d\s,.]*\s*{_UNIT}|(?<!\w)до\s+\d[\d\s,.]*\s*{_UNIT}|(?<![\w-])SLA(?![\w-])|"
    r"доступност\w*[^.]{0,60}?\d+(?:[.,]\d+)?\s*%|(?<!\w)быстр(?:о|ый|ая|ое)(?!\w)|мгновенн\w*|"
    # a unit without a number: «в течение минуты», «за одну секунду», «ежеминутно»
    r"в\s+течение\s+(?:одн\w+\s+)?(?:минут\w*|секунд\w*|час\w*|сут\w*)|"
    r"(?<!\w)за\s+(?:одн\w+\s+)?(?:минуту|секунду|час)(?!\w)|ежеминутн\w*|ежесекундн\w*)",
    re.IGNORECASE)
_PERCENTILE = re.compile(r"((?<![\w])p\s?\d{2}(?:[.,]\d)?(?!\w)|перцентил\w*|процентил\w*|медиан\w*|"
                         r"в\s+\d{2}(?:[.,]\d+)?\s*%\s+(?:случаев|запросов|операций)|\d{2}(?:[.,]\d+)?\s*%\s+"
                         r"(?:запросов|операций|случаев|вызовов))", re.IGNORECASE)
_INTERVAL = re.compile(r"((?<!\w)от\s+(?:момента|получени|поступлени|отправк|нажати|создани|регистраци|начал)\w*"
                       r"[^.]{0,120}?\s+до\s+|с\s+момента\s+[^.]{0,120}?\s+до\s+|между\s+[^.]{0,80}?\s+и\s+|"
                       r"(?<!\w)до\s+(?:момента|получени|доставк|отображени|ответ|отправк|зачислени)\w*|"
                       r"(?<!\w)(?:в|за)\s+(?:календарн\w+\s+|отчётн\w+\s+|отчетн\w+\s+)?(?:месяц|квартал|год|сутк|недел)\w*)",
                       re.IGNORECASE)
_LOAD = re.compile(r"((?<![\w-])(?:RPS|TPS|QPS)(?![\w-])|запрос\w*\s+в\s+секунд\w*|операци\w*\s+в\s+секунд\w*|"
                   r"при\s+нагрузк\w*|нагрузк\w*\s+до|одновременн\w*|пиков\w*|в\s+час\s+пик)", re.IGNORECASE)
# an end point that can mean either handing over for sending or delivery to the person
_HANDOVER = re.compile(r"(на\s+отправку|переда[её]т\w*|передава\w*|отправля\w*|направля\w*|поставит\w*\s+в\s+очередь)",
                       re.IGNORECASE)
_NFR_SECTION = re.compile(r"(нефункцион|производительн|nfr|качеств|доступност|надёжн|надежн|sla)", re.IGNORECASE)
# validity, retention and timeouts are parameters, not performance
_NOT_PERFORMANCE = re.compile(r"(хран\w*|архив\w*|действ\w*|истека\w*|блокир\w*|тайм-?аут\w*|timeout|ожидани\w*|"
                              r"повтор\w*\s+через|(?<!\w)дн(?:ей|я|и)(?!\w))", re.IGNORECASE)
_AVAILABILITY = re.compile(r"доступност\w*", re.IGNORECASE)


def _nfr_missing(text: str) -> list[str]:
    missing = []
    if not _PERCENTILE.search(text):
        missing.append("доля запросов (перцентиль)")
    if not _INTERVAL.search(text):
        has_start = re.search(r"(?<!\w)(?:после|с\s+момента)\s+\w+", text, re.IGNORECASE)
        missing.append("конец измеряемого интервала" if has_start else "начало и конец измеряемого интервала")
    if not _LOAD.search(text):
        missing.append("нагрузка и условия измерения")
    return missing


def check_nfr_unmeasurable(doc: Document) -> list[RuleFinding]:
    out: list[RuleFinding] = []
    for b in _requirement_blocks(doc):
        if b.kind == BlockKind.TABLE_ROW and not b.doc_id:
            continue  # parameter and field tables
        m = _TIME_BOUND.search(b.text)
        if m is None:
            continue
        # quality requirements only: a time inside a functional rule is a business deadline, not a metric
        quality = _NFR_SECTION.search(" / ".join(b.section_path)) or re.match(r"\W*(?:NFR|НФТ|NF)\W", b.text)
        if not quality or _NOT_PERFORMANCE.search(b.text):
            continue
        if _AVAILABILITY.search(b.text) and _INTERVAL.search(b.text):
            continue  # availability over a stated period is measurable
        missing = _nfr_missing(b.text)
        if len(missing) < 2 or "доля запросов (перцентиль)" not in missing:
            continue  # a stated percentile with a device or load condition is measurable enough
        end_note = ""
        if any("конец измеряемого интервала" in x for x in missing) and _HANDOVER.search(b.text):
            end_note = (" Конец интервала нужно уточнить: передача на отправку внешнему сервису или доставка "
                        "получателю — это разные моменты.")
        out.append(_make_finding(
            rule="nfr_unmeasurable", block_ids=[b.id], criterion=Criterion.VERIFIABLE, severity=Severity.MAJOR,
            set_check=None, quote=_quote_for_span(b.text, m.start(), m.end()),
            explanation=(f"Требование к времени или доступности нельзя однозначно измерить: не заданы "
                         f"{', '.join(missing)}.{end_note} Разные стенды и тестировщики получат разные "
                         "результаты."),
            recommendation=("Указать, от какого события до какого измеряется время, для какой доли запросов "
                            "(например, p95) и при какой нагрузке."),
            clarifying_question="Как измеряется это время: от какого события до какого, для какой доли запросов и "
                                "при какой нагрузке?",
            answer_options=["От получения запроса до передачи на отправку, p95, при расчётной пиковой нагрузке",
                            "От события до доставки клиенту, p99, при средней нагрузке"],
        ))
        if len(out) >= MAX_NFR:
            break
    return out


# ---------------------------------------------------------------- notification_content

_CHANNEL = r"(?:push|пуш|уведомлени\w*|sms|смс|e-?mail|письм\w*|сообщени\w*)"
_SEND = re.compile(rf"((?:отправл|отправит|отправ|направл|направит|присыла|рассыла|высыла|доставля)\w*[^.]{{0,40}}?"
                   rf"{_CHANNEL}|{_CHANNEL}[^.]{{0,40}}?(?:отправл|направл|присыла|доставля|высыла)\w*|"
                   rf"(?:клиент|пользовател|держател)\w*\s+(?:получа\w*|приход\w*)[^.]{{0,20}}?{_CHANNEL})",
                   re.IGNORECASE)
_TEMPLATE = re.compile(rf"(шаблон\w*(?:\s+\w+){{0,3}}\s+{_CHANNEL}|{_CHANNEL}[^.\n]{{0,60}}?(?:шаблон\w*|текст\w*|"
                       rf"«[^»]{{10,}}»|содерж\w*|включа\w*|вида|поля)|текст\w*\s+{_CHANNEL}|"
                       rf"содержани\w*\s+{_CHANNEL}|формат\w*\s+{_CHANNEL}|состав\w*\s+{_CHANNEL})", re.IGNORECASE)
_PII = re.compile(r"(номер\w*\s+карт\w*|цифр\w*\s+карт\w*|PAN|сумм\w*|баланс\w*|остат\w*|ФИО|имени|фамили\w*|"
                  r"телефон\w*|паспорт\w*|ИИН|адрес\w*|счёт\w*|счет\w*)", re.IGNORECASE)
# the place itself names the content: "push с указанием суммы", "уведомление о переводе карты"
_PLACE_CONTENT = re.compile(rf"{_CHANNEL}(?:[\s-]+(?:уведомлени\w*|сообщени\w*))?\s+(?:с|со|о|об|содержащ\w*|"
                            rf"включающ\w*)(?!\w)", re.IGNORECASE)
_TO_PERSON = re.compile(r"(клиент|пользовател|держател|получател|абонент|заёмщик|заемщик|родител|сотрудник)",
                        re.IGNORECASE)
# system-to-system callbacks and one-time codes are not messages whose content the document must give
_NOT_A_MESSAGE = re.compile(r"(webhook|вебхук|(?:sms|смс)[\s-]*код\w*|код\w*\s+(?:по|в|через)\s+(?:sms|смс)|"
                            r"(?<![\w-])OTP(?![\w-]))", re.IGNORECASE)
_ERROR_MESSAGE = re.compile(r"(сообщени\w*\s+об\s+ошибк|текст\s+ошибк|message\b|код\w*\s+ответ)", re.IGNORECASE)


def check_notification_content(doc: Document) -> list[RuleFinding]:
    full = "\n".join(b.text for b in doc.blocks)
    if _TEMPLATE.search(full):
        return []  # the document gives a template or text for its messages somewhere
    out: list[RuleFinding] = []
    for b in _requirement_blocks(doc):
        m = _SEND.search(b.text)
        if m is None or _ERROR_MESSAGE.search(b.text) or _PLACE_CONTENT.search(b.text):
            continue
        if _NOT_A_MESSAGE.search(b.text) or not _TO_PERSON.search(b.text):
            continue
        pii = sorted({x.group(1).lower() for x in _PII.finditer(b.text)})
        if pii:
            explanation = (f"Отправка сообщения клиенту («{_short(b.text, 70)}») описана без шаблона и текста, а "
                           f"в сообщение попадают данные ({', '.join(pii[:4])}) без правила маскирования и формата.")
        else:
            explanation = (f"Отправка сообщения клиенту («{_short(b.text, 70)}») описана без содержания: в документе "
                           "нет шаблона, текста или состава полей сообщения.")
        out.append(_make_finding(
            rule="notification_content", block_ids=[b.id], criterion=Criterion.COMPLETE, severity=Severity.MAJOR,
            set_check=None, quote=_quote_for_span(b.text, m.start(), m.end()), explanation=explanation,
            recommendation="Привести шаблон сообщения: текст, подставляемые поля, их формат и маскирование.",
            clarifying_question="Какой текст и какие поля содержит это сообщение и как маскируются данные клиента?",
            answer_options=["Шаблон с маскированием: последние 4 цифры карты, сумма без остатка",
                            "Сообщение без персональных и платёжных данных, только ссылка в приложение"],
        ))
        if len(out) >= MAX_NOTIFY:
            break
    return out


# ---------------------------------------------------------------- mixed_levels

MAX_MIXED = 3
_API_SECTION = re.compile(r"((?<![\w])api(?![\w])|метод\w*|endpoint|эндпоинт\w*|контроллер\w*|(?<![\w])rest(?![\w])|"
                          r"параметр\w*\s+запрос\w*|спецификаци\w*\s+(?:интерфейс|сервис|метод)|запрос\w*\s+и\s+ответ)",
                          re.IGNORECASE)
_HTTP_LINE = re.compile(r"(?<![\w])(GET|POST|PUT|PATCH|DELETE)\s+/", re.IGNORECASE)
_BUSINESS = re.compile(r"(накопленн\w*|расход\w*|сч[её]тчик\w*|(?:за|в)\s+(?:календарн\w+\s+)?месяц\w*|уведомлени\w*|"
                       r"начислени\w*|расч[её]т\w*|процент\w*|пен[яиюей]|итогов\w*|остат\w*|лимит\w*\s+(?:за|в)\s+)",
                       re.IGNORECASE)
_RULE_VERB = re.compile(r"(сохраня\w*|пересчитыва\w*|начисля\w*|сбрасыва\w*|не\s+изменя\w*|обнуля\w*|учитыва\w*|"
                        r"списыва\w*|увеличива\w*|уменьша\w*|отправля\w*|не\s+отправля\w*|не\s+учитыва\w*)",
                        re.IGNORECASE)
_CONTRACT = re.compile(r"(`[^`]+`|(?<![\w])[a-z]+[A-Z]\w*|(?<![\w])[a-z]+_[a-z_]+|(?<![\w])(?:поле|поля|параметр\w*|"
                       r"тел[оа]\s+(?:запрос|ответ)\w*|код\w*\s+ответ\w*|заголов\w*|в\s+ответ\w*|запрос\w*)(?![\w]))")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZА-ЯЁ«(])")


def _api_part(doc: Document) -> set[str]:
    """Sections of an interface specification: by their heading, or by an HTTP method line inside."""
    ids = {s.id for s in doc.sections if any(_API_SECTION.search(t) for t in s.path)}
    for b in doc.blocks:
        if _HTTP_LINE.search(b.text):
            ids.add(b.section_id)
    return ids


def check_mixed_levels(doc: Document) -> list[RuleFinding]:
    api = _api_part(doc)
    if not api:
        return []
    hits: list[tuple[object, str, int]] = []
    for b in doc.blocks:
        if b.section_id not in api or b.kind in (BlockKind.CODE, BlockKind.HEADING, BlockKind.TABLE_HEADER):
            continue
        pos = 0
        for sent in _SENTENCE.split(b.text):
            start = b.text.find(sent, pos)
            pos = max(pos, start + len(sent))
            if _BUSINESS.search(sent) and _RULE_VERB.search(sent) and not _CONTRACT.search(sent):
                hits.append((b, sent.strip(), start))
                break
        if len(hits) >= MAX_MIXED:
            break
    if not hits:
        return []
    listed = "; ".join(f"«{_short(s, 90)}»" for _, s, _ in hits)
    first, sent, start = hits[0]
    return [_make_finding(
        rule="mixed_levels", block_ids=[b.id for b, _, _ in hits], criterion=Criterion.APPROPRIATE,
        severity=Severity.MAJOR, set_check=SetCheck.OTHER,
        quote=_quote_for_span(first.text, max(0, start), max(0, start) + len(sent)),
        explanation=(f"В спецификации интерфейса записаны бизнес-правила другой обработки, а не контракт метода: "
                     f"{listed}. Уровни описания смешаны: неясно, какой сервис отвечает за правило и как метод его "
                     "обеспечивает."),
        recommendation="Перенести правило в бизнес-требования сервиса, который его выполняет, а в спецификации "
                       "метода оставить поля, проверки и ответы.",
        clarifying_question="Какой сервис выполняет это правило и должно ли оно быть в спецификации метода?",
        answer_options=["Перенести правило в бизнес-требования сервиса, который его выполняет",
                        "Оставить и описать, как метод это обеспечивает (поля, проверки, ответы)"],
    )]
