"""Prompts for the set inspector: S1 (consistency, parts) and S2 (acceptance criteria, completeness)."""

from __future__ import annotations

from validator.models import Document, InventoryItem, ReqKind, Requirement, SkippedBlock
from validator.pipeline.prompts.examples import SET_EXAMPLES
from validator.pipeline.prompts.render import inventory_table, render_document, requirement_table

S1_TASK = """Задание S1: найди противоречия внутри набора и между частями документа.
Шаг 1: заполни ledger — реестр общих понятий. Для каждого значения, единицы, даты и временной базы, \
лимита на период, метода и кода ответа, обязательного поля и флага, показываемой пользователю величины, \
а также «кто что читает и записывает» в mentions перечисли ВСЕ места документа, где оно упоминается, и что \
каждое место о нём говорит. Не больше 20 понятий, самые важные для поведения системы.
Шаг 2: сравни места попарно. Обязательно проверь:
- флаги и переключатели, которые одна часть определяет, а другая игнорирует;
- операции, которых требует одна часть (чтение, сохранение, отображение, выключение), но другая часть \
(спецификация интерфейса, интеграция) не предоставляет — это пробел: set_check missing_requirement, \
criterion set_complete (не противоречие);
- проверки, которых требуют бизнес-правила, но нет в логике интерфейса;
- одинаковые величины с разными значениями, единицами, форматами или типами в разных местах;
- коды ответов и результаты, которые по-разному описаны в логике, перечне статусов и примерах;
- назначение, которое описано по-разному в разных полях одной спецификации;
- бизнес-логику или правило другой системы или сервиса, записанное внутри спецификации интерфейса (смешение \
уровней: интерфейс одного сервиса описывает поведение, за которое отвечает другой) — находка set_check other, \
criterion appropriate, severity major;
- для каждого флага или переключателя: где задаётся, где читается и где влияет на поведение; обязательны ли \
остальные параметры, когда он выключен;
- для каждой величины, которую видит пользователь или использует другая система: какая операция или \
компонент её предоставляет; нет источника — находка;
- бизнес-условия допуска (статус, тип, права объекта) из одной части: проверяются ли они в интерфейсе другой;
- одна величина в разных единицах в разных местах без правила пересчёта и округления;
- момент события в цели против основы расчёта в правилах: цель требует реакции «в момент» действия, а правило \
считает только завершённые, подтверждённые или проведённые операции;
- валюта, курс пересчёта и комиссии учитываемых операций, если объект или лимит задан в одной валюте;
- лимит или ограничение на период против правила, которое сбрасывает или обходит его при изменении настройки \
или состояния: что происходит второй раз за тот же период;
- перечень хранимых данных против полей и флагов, которые определены в других частях: что не хранится;
- заголовки и подразделы, которые смешивают разные коды ответа или помещают описание одного кода внутрь другого;
- код ответа, под заголовком которого приведён пример, против кода, который для этой ситуации называют перечень \
статусов и логика метода (например, тело с признаком ошибки под заголовком кода успеха, тогда как перечень \
относит неверные параметры к коду ошибки): находка должна описывать именно этот конфликт — «под заголовком \
кода X приведён пример ошибки, а перечень и логика требуют кода Y», с цитатами заголовка примера и строки \
перечня; не подменяй его спором о составе полей;
- условия в описаниях процесса и взаимодействия систем (схемы шагов, последовательности вызовов) против \
функциональных правил: учтены ли флаги и переключатели, условие перехода («с ниже на выше», «не чаще раза \
в период»), исключения; шаг без условия, которое требует правило, — находка.
- пары правил, которые ограничивают одно и то же событие или величину (лимит частоты за период, сброс или \
исключение при изменении настройки или состояния, пороги, временная база): для КАЖДОЙ такой пары составь \
сценарий «событие уже произошло в периоде → настройку изменили → условие наступает снова» и сравни, что \
требует каждое правило; разный результат — противоречие (contradiction), обычно critical;
- разные названия одного компонента, сервиса или понятия в разных местах (set_check terminology, \
criterion set_comprehensible).
Правила:
- contradiction и cross_part — только когда два места требуют несовместимого. Если одна часть просто не \
учитывает флаг, условие или операцию другой, это пробел: set_check missing_requirement, criterion set_complete.
- Каждая находка о противоречии содержит минимум две цитаты из разных мест в evidence и конкретный сценарий \
в scenario: входные данные → что следует из места X → что следует из места Y. Нет сценария — нет находки.
- set_check cross_part — если места в разных частях документа; contradiction — внутри одной части; \
terminology — разные термины для одного понятия.
- Не сверяй требования с критериями приёмки: это делает другой этап.
- coverage верни пустым списком.
- Не больше 12 находок, самые существенные."""

S2_TASK = """Задание S2: сверь требования с критериями приёмки и оцени полноту набора.
1. Для каждой строки критериев приёмки примени текст требований буквально к её исходным условиям и действию \
и сравни полученный результат с ожидаемым результатом строки. Расхождение — находка acceptance_mismatch \
(criterion set_consistent) с двумя цитатами (требование и строка критерия) и сценарием в scenario.
2. coverage: для каждого требования типов functional, nonfunctional, integration, data перечисли \
ID покрывающих критериев приёмки (R-…) и статус covered | partial | none; в note — чего не хватает. \
Требования типа interface (параметры, примеры, схемы) в coverage не включай.
3. Непокрытые сценарии (missing_scenario): границы диапазонов, отказы и ошибки, значения по умолчанию, \
повторы и отмены, которых нет ни в требованиях, ни в критериях. Объединяй однотипные пробелы в одну находку.
4. Отсутствующие классы требований (missing_requirement): по обобщённому чек-листу полноты ниже — \
только то, что действительно нужно этому документу.
5. Пустые разделы и заглушки (empty_section), незаполненные шаблоны.
Для находки-пропуска цитата — место, где пропуск виден (заголовок пустого раздела, соседнее требование, фраза, \
из которой следует потребность). scenario можно оставить пустым, кроме acceptance_mismatch.
ledger верни пустым списком. Не больше 12 находок.

Обобщённый чек-лист полноты:
- функциональные: предусловия, границы, отказы, значения по умолчанию, повтор и отмена;
- нефункциональные: производительность с метрикой, доступность, безопасность, аудит, хранение и удаление данных;
- интерфейсы: операции под каждую потребность, схемы запроса и ответа, коды ошибок, идемпотентность, права;
- процессы и взаимодействие систем: участники, шаги, ветки отказа и условия (флаги, условия перехода, лимиты), \
согласованные с функциональными правилами; описание одной строкой без них — пробел (missing_requirement);
- документ: версия, приоритеты требований, согласование, откат."""

PAIRS_TASK = """Задание S1b: проверь попарно правила, которые ограничивают одно и то же событие или величину.
Шаг 1: в ledger выпиши каждое такое событие или величину (concept) и в mentions — все правила, которые его \
ограничивают: лимиты частоты за период, сброс или исключение при изменении настройки или состояния, пороги и \
условия перехода, временная база, исключения из учёта.
Шаг 2: для КАЖДОЙ пары правил одного события составь конкретный сценарий: событие уже произошло в периоде → \
изменили настройку или состояние → условие наступает снова в том же периоде. Что требует первое правило, что \
второе? Если результаты различаются — находка contradiction (criterion set_consistent, обычно critical) с двумя \
цитатами и сценарием в scenario. Если правило не говорит, что делать в этой ситуации, — missing_scenario \
(criterion set_complete, major).
Не повторяй дефекты интерфейса (методы, коды ответов): их проверяют другие этапы. coverage верни пустым \
списком. Не больше 6 находок."""

MERGED_TASK = """Выполни оба задания в одном ответе: сначала S1, затем S2. Общий лимит — 16 находок."""


def _common(doc: Document, reqs: list[Requirement], include_document: bool) -> list[str]:
    parts = []
    if include_document:
        parts.append(render_document(doc))
    parts.append("<requirements>\n" + requirement_table(reqs, with_text=not include_document) + "\n</requirements>")
    return parts


def build_s1_prompt(doc: Document, reqs: list[Requirement], inventory: list[InventoryItem], *,
                    include_document: bool = True, merged_with_s2: str | None = None, pairs: str = "",
                    second_pass: bool = False) -> str:
    parts = _common(doc, list(reversed(reqs)) if second_pass else reqs, include_document)
    parts.append("<parts>\n" + "\n".join(f"- {p}" for p in doc.parts) + "\n</parts>" if doc.parts else
                 "<parts>документ из одной части</parts>")
    if inventory:
        parts.append("<inventory>\n" + inventory_table(inventory) + "\n</inventory>")
    if pairs:
        parts.append("<pairs>\n" + pairs + "\n</pairs>")
    parts.append(SET_EXAMPLES)
    task = S1_TASK
    if pairs:
        from validator.pipeline.experiments import PAIRS_INSTRUCTION

        task += "\n" + PAIRS_INSTRUCTION
    if second_pass:
        from validator.pipeline.experiments import SECOND_PASS_NOTE

        task += "\n" + SECOND_PASS_NOTE
    parts.append(task)
    if merged_with_s2:
        parts.append(merged_with_s2)
    return "\n\n".join(parts)


def build_s2_prompt(doc: Document, reqs: list[Requirement], skipped: list[SkippedBlock], *,
                    include_document: bool = True, entities: str = "") -> str:
    parts = _common(doc, reqs, include_document)
    acc = [r for r in reqs if r.kind == ReqKind.ACCEPTANCE_CRITERION]
    parts.append("<acceptance_criteria>\n" + (requirement_table(acc) if acc else "в документе нет критериев приёмки")
                 + "\n</acceptance_criteria>")
    if skipped:
        parts.append("<skipped_blocks>\n" + "\n".join(f"{s.block_id} | {s.reason} | {s.text[:150]}" for s in skipped[:80])
                     + "\n</skipped_blocks>")
    if entities:
        from validator.pipeline.experiments import ENTITY_CHECKLIST

        parts.append("<entities>\n" + entities + "\n</entities>")
        parts.append(SET_EXAMPLES)
        parts.append(S2_TASK + "\n\n" + ENTITY_CHECKLIST)
        return "\n\n".join(parts)
    parts.append(SET_EXAMPLES)
    parts.append(S2_TASK)
    return "\n\n".join(parts)


def build_pairs_prompt(doc: Document, reqs: list[Requirement], *, include_document: bool = True) -> str:
    rule_kinds = {ReqKind.FUNCTIONAL, ReqKind.BUSINESS_GOAL, ReqKind.PROCESS, ReqKind.DATA, ReqKind.INTEGRATION,
                  ReqKind.NONFUNCTIONAL, ReqKind.ACCEPTANCE_CRITERION, ReqKind.CONSTRAINT, ReqKind.OTHER}
    rules = [r for r in reqs if r.kind in rule_kinds] or reqs
    parts = _common(doc, rules, include_document)
    parts.append(SET_EXAMPLES)
    parts.append(PAIRS_TASK)
    return "\n\n".join(parts)


def build_merged_prompt(doc: Document, reqs: list[Requirement], inventory: list[InventoryItem],
                        skipped: list[SkippedBlock], *, include_document: bool = True) -> str:
    s2 = build_s2_prompt(doc, reqs, skipped, include_document=False).split("<acceptance_criteria>", 1)[1]
    s1 = build_s1_prompt(doc, reqs, inventory, include_document=include_document)
    s1 = s1.replace("coverage верни пустым списком.", "")
    return s1 + "\n\n<acceptance_criteria>" + s2 + "\n\n" + MERGED_TASK
