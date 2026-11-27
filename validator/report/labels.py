"""Russian display labels and standard clause references shared by reports and prompts."""

from __future__ import annotations

from dataclasses import dataclass

from validator.models import (
    BlockKind,
    Criterion,
    DefectClass,
    FindingDeltaStatus,
    FindingSource,
    QuoteStatus,
    Readiness,
    ReqKind,
    RunStatus,
    SetCheck,
    Severity,
    StageState,
)


@dataclass(frozen=True)
class CriterionMeta:
    """Metadata and standard clause references for a single quality criterion."""

    code: Criterion
    label: str
    standard_ref: str
    ieee830_ref: str
    definition: str
    violation_signs: str
    typical_question: str


CRITERION_META: dict[Criterion, CriterionMeta] = {
    Criterion.NECESSARY: CriterionMeta(
        code=Criterion.NECESSARY,
        label="Необходимость",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — необходимость",
        ieee830_ref="",
        definition=(
            "Требование описывает возможность или ограничение, без которого система "
            "не удовлетворит потребность заинтересованной стороны."
        ),
        violation_signs=(
            "Не связано с целями документа, дублирует другое требование или "
            "сформулировано как необязательное пожелание."
        ),
        typical_question="Какую бизнес-цель или пользовательский сценарий обеспечивает это требование?",
    ),
    Criterion.APPROPRIATE: CriterionMeta(
        code=Criterion.APPROPRIATE,
        label="Уместность уровня",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — уместность (без проектных решений)",
        ieee830_ref="",
        definition=(
            "Уровень детализации соответствует уровню документа; требование описывает, "
            "что должна делать система, а не внутреннюю реализацию."
        ),
        violation_signs=(
            "В бизнес- или функциональном требовании без обоснования зафиксированы "
            "конкретная технология, структура хранения или имя внутреннего компонента."
        ),
        typical_question="Является ли указанное техническое решение обязательным ограничением или примером реализации?",
    ),
    Criterion.UNAMBIGUOUS: CriterionMeta(
        code=Criterion.UNAMBIGUOUS,
        label="Однозначность",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — однозначность",
        ieee830_ref="IEEE 830-1998, п. 4.3.2",
        definition=(
            "Требование допускает ровно одну трактовку для всех читателей; термины "
            "определены и используются единообразно."
        ),
        violation_signs=(
            "Несколько допустимых трактовок, неочевидные местоимения, связка «и/или», "
            "неуточнённый порядок условий или неопределённый термин."
        ),
        typical_question="Какая именно из возможных трактовок этого требования должна быть реализована?",
    ),
    Criterion.COMPLETE: CriterionMeta(
        code=Criterion.COMPLETE,
        label="Полнота",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — полнота",
        ieee830_ref="IEEE 830-1998, п. 4.3.3",
        definition=(
            "Требование содержит достаточно сведений для реализации и проверки: субъект, "
            "действие, объект, условия, границы и обработку ошибок."
        ),
        violation_signs=(
            "Не указаны граничные значения, значение по умолчанию, единицы измерения, "
            "поведение при ошибке, альтернативные ветки или присутствуют заглушки."
        ),
        typical_question="Каково ожидаемое поведение системы в граничных и ошибочных случаях?",
    ),
    Criterion.SINGULAR: CriterionMeta(
        code=Criterion.SINGULAR,
        label="Единичность",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — единичность",
        ieee830_ref="IEEE 830-1998, п. 4.3.7",
        definition="Одна запись описывает одну независимую возможность, условие или ограничение.",
        violation_signs=(
            "В одном пункте объединены несколько самостоятельных действий, функций или "
            "проверяемых результатов, которые логично тестировать раздельно."
        ),
        typical_question="Следует ли разделить эту запись на отдельные самостоятельные требования?",
    ),
    Criterion.FEASIBLE: CriterionMeta(
        code=Criterion.FEASIBLE,
        label="Выполнимость",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — выполнимость",
        ieee830_ref="",
        definition=(
            "Требование реализуемо в рамках технических, сетевых, временных, "
            "бюджетных и регуляторных ограничений."
        ),
        violation_signs=(
            "Абсолютные гарантии (100% без сбоев, мгновенно, неограниченный объём) "
            "либо противоречие физическим и системным ограничениям."
        ),
        typical_question="Какие реалистичные допустимые пределы и допущения применимы к этому требованию?",
    ),
    Criterion.VERIFIABLE: CriterionMeta(
        code=Criterion.VERIFIABLE,
        label="Проверяемость",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — проверяемость",
        ieee830_ref="IEEE 830-1998, п. 4.3.6",
        definition=(
            "Существует конечный и объективный способ подтвердить выполнение требования "
            "(тестированием, анализом, инспекцией или демонстрацией)."
        ),
        violation_signs=(
            "Нет измеримого значения, единицы измерения или условий проверки; используются "
            "субъективные оценки («быстро», «удобно», «надёжно»)."
        ),
        typical_question="По какому измеримому критерию и при каких условиях будет проверяться выполнение?",
    ),
    Criterion.CORRECT: CriterionMeta(
        code=Criterion.CORRECT,
        label="Корректность",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — корректность",
        ieee830_ref="IEEE 830-1998, п. 4.3.1",
        definition=(
            "Требование точно отражает потребность заказчика и не противоречит "
            "нормам протоколов, форматов, единиц измерения и безопасности."
        ),
        violation_signs=(
            "Фактическая ошибка, нарушение семантики протокола или формата, неверные "
            "единицы измерения или конфликт примера с формулировкой."
        ),
        typical_question="Подтверждаете ли вы указанное поведение или в формулировке допущена фактическая ошибка?",
    ),
    Criterion.CONFORMING: CriterionMeta(
        code=Criterion.CONFORMING,
        label="Соответствие правилам и шаблону",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.5 — соответствие шаблону",
        ieee830_ref="",
        definition=(
            "Требование оформлено по принятому шаблону с обязывающей модальностью "
            "и единообразной структурой."
        ),
        violation_signs=(
            "Слабая модальность («может», «желательно», «по возможности»), "
            "отсутствие субъекта действия или нарушение структуры документа."
        ),
        typical_question="Является ли это требование обязательным к реализации и какой субъект выполняет действие?",
    ),
    Criterion.SET_COMPLETE: CriterionMeta(
        code=Criterion.SET_COMPLETE,
        label="Полнота набора",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.6 — полнота набора",
        ieee830_ref="IEEE 830-1998, п. 4.3.3",
        definition=(
            "Набор требований покрывает все необходимые функции, интерфейсы, данные, "
            "нефункциональные требования, права доступа и сценарии ошибок без пробелов."
        ),
        violation_signs=(
            "Пустые разделы, незаполненные шаблоны, заглушки, отсутствующие классы "
            "требований, непокрытые сценарии или отсутствие критериев приёмки."
        ),
        typical_question="Какие недостающие требования или сценарии необходимо добавить в документ?",
    ),
    Criterion.SET_CONSISTENT: CriterionMeta(
        code=Criterion.SET_CONSISTENT,
        label="Согласованность набора",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.6 — согласованность набора",
        ieee830_ref="IEEE 830-1998, п. 4.3.4",
        definition=(
            "Требования внутри документа и между его частями не противоречат друг другу "
            "по значениям, логике, условиям, операциям и терминологии."
        ),
        violation_signs=(
            "Разные значения одного параметра, несовместимые условия или статусы, "
            "расхождение между частями документа или между требованием и критерием приёмки."
        ),
        typical_question="Какая из противоречащих формулировок является актуальной и подлежит реализации?",
    ),
    Criterion.SET_FEASIBLE: CriterionMeta(
        code=Criterion.SET_FEASIBLE,
        label="Выполнимость набора",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.6 — выполнимость набора",
        ieee830_ref="",
        definition=(
            "Совокупность всех требований реализуема совместно без взаимно исключающих "
            "ограничений."
        ),
        violation_signs=(
            "Конкурирующие требования, которые по отдельности реализуемы, но вместе "
            "несовместимы по архитектуре, безопасности или ресурсам."
        ),
        typical_question="Какой приоритет или компромисс следует выбрать между конкурирующими требованиями?",
    ),
    Criterion.SET_COMPREHENSIBLE: CriterionMeta(
        code=Criterion.SET_COMPREHENSIBLE,
        label="Понятность набора",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.6 — понятность набора",
        ieee830_ref="IEEE 830-1998, п. 4.3.2",
        definition=(
            "Документ в целом понятен команде разработки и тестирования: термины "
            "и аббревиатуры определены, связи между разделами прозрачны."
        ),
        violation_signs=(
            "Нерасшифрованные аббревиатуры, расхождение терминов для одной сущности, "
            "неясная структура документа."
        ),
        typical_question="Какое единое определение следует закрепить за указанным термином или аббревиатурой?",
    ),
    Criterion.SET_VALIDATABLE: CriterionMeta(
        code=Criterion.SET_VALIDATABLE,
        label="Валидируемость набора",
        standard_ref="ISO/IEC/IEEE 29148:2018, п. 5.2.6 — валидируемость набора",
        ieee830_ref="",
        definition=(
            "Можно подтвердить, что выполнение всего набора требований обеспечит "
            "достижение заявленных бизнес-целей и критериев приёмки."
        ),
        violation_signs=(
            "Отсутствуют бизнес-цели или критерии приёмки, либо ключевые требования "
            "не покрыты приёмочными сценариями."
        ),
        typical_question="По каким приёмочным критериям заказчик подтвердит достижение бизнес-цели?",
    ),
    Criterion.RANKED: CriterionMeta(
        code=Criterion.RANKED,
        label="Ранжирование по важности",
        standard_ref="IEEE 830-1998, п. 4.3.5 — ранжирование по важности и стабильности",
        ieee830_ref="IEEE 830-1998, п. 4.3.5",
        definition="Для требований указаны приоритет (важность) и ожидаемая стабильность.",
        violation_signs="В документе отсутствует приоритизация требований.",
        typical_question="Какие требования являются обязательными для первого релиза, а какие второстепенными?",
    ),
    Criterion.MODIFIABLE: CriterionMeta(
        code=Criterion.MODIFIABLE,
        label="Модифицируемость",
        standard_ref="IEEE 830-1998, п. 4.3.7 — модифицируемость",
        ieee830_ref="IEEE 830-1998, п. 4.3.7",
        definition=(
            "Структура документа позволяет вносить изменения согласованно, без скрытого "
            "дублирования и перепутанных разделов."
        ),
        violation_signs="Дублирование одних и тех же правил в нескольких местах документа.",
        typical_question="Какой раздел следует считать единственным источником истины для этого правила?",
    ),
    Criterion.TRACEABLE: CriterionMeta(
        code=Criterion.TRACEABLE,
        label="Трассируемость",
        standard_ref="IEEE 830-1998, п. 4.3.8 — трассируемость",
        ieee830_ref="IEEE 830-1998, п. 4.3.8",
        definition=(
            "Каждое требование имеет уникальный идентификатор и корректные перекрёстные "
            "ссылки, прослеживается его источник."
        ),
        violation_signs=(
            "Отсутствуют или дублируются идентификаторы, есть пропуски в нумерации "
            "либо ссылки на несуществующие пункты и разделы."
        ),
        typical_question="Какие уникальные идентификаторы и корректные ссылки должны быть закреплены в документе?",
    ),
}

CRITERION_LABELS: dict[Criterion, str] = {c: m.label for c, m in CRITERION_META.items()}
CRITERION_REFS: dict[Criterion, str] = {c: m.standard_ref for c, m in CRITERION_META.items()}

SEVERITY_LABELS: dict[Severity, str] = {
    Severity.CRITICAL: "Критичный",
    Severity.MAJOR: "Существенный",
    Severity.MINOR: "Незначительный",
}

SEVERITY_PLURAL_LABELS: dict[Severity, str] = {
    Severity.CRITICAL: "Критичные",
    Severity.MAJOR: "Существенные",
    Severity.MINOR: "Незначительные",
}

DEFECT_CLASS_ORDER: tuple[DefectClass, ...] = (
    DefectClass.CONTRADICTORY,
    DefectClass.INCOMPLETE,
    DefectClass.UNVERIFIABLE,
    DefectClass.AMBIGUOUS,
    DefectClass.OTHER,
)

DEFECT_CLASS_LABELS: dict[DefectClass, str] = {
    DefectClass.CONTRADICTORY: "Противоречивые",
    DefectClass.INCOMPLETE: "Неполные",
    DefectClass.UNVERIFIABLE: "Непроверяемые",
    DefectClass.AMBIGUOUS: "Неоднозначные",
    DefectClass.OTHER: "Прочие",
}

REQ_KIND_ORDER: tuple[ReqKind, ...] = (
    ReqKind.BUSINESS_GOAL,
    ReqKind.FUNCTIONAL,
    ReqKind.NONFUNCTIONAL,
    ReqKind.DATA,
    ReqKind.INTERFACE,
    ReqKind.INTEGRATION,
    ReqKind.UI,
    ReqKind.PROCESS,
    ReqKind.CONSTRAINT,
    ReqKind.ACCEPTANCE_CRITERION,
    ReqKind.OTHER,
)

REQ_KIND_LABELS: dict[ReqKind, str] = {
    ReqKind.FUNCTIONAL: "Функциональное",
    ReqKind.NONFUNCTIONAL: "Нефункциональное",
    ReqKind.DATA: "Требование к данным",
    ReqKind.INTERFACE: "Интерфейс / API",
    ReqKind.INTEGRATION: "Интеграция",
    ReqKind.UI: "Интерфейс пользователя (UI)",
    ReqKind.BUSINESS_GOAL: "Бизнес-цель",
    ReqKind.PROCESS: "Бизнес-процесс",
    ReqKind.CONSTRAINT: "Ограничение",
    ReqKind.ACCEPTANCE_CRITERION: "Критерий приёмки",
    ReqKind.OTHER: "Прочее",
}

BLOCK_KIND_LABELS: dict[BlockKind, str] = {
    BlockKind.HEADING: "Заголовок",
    BlockKind.PARAGRAPH: "Абзац",
    BlockKind.LIST_ITEM: "Пункт списка",
    BlockKind.TABLE_HEADER: "Заголовок таблицы",
    BlockKind.TABLE_ROW: "Строка таблицы",
    BlockKind.CODE: "Блок кода",
    BlockKind.OTHER: "Прочее",
}

SET_CHECK_LABELS: dict[SetCheck, str] = {
    SetCheck.CONTRADICTION: "Противоречие",
    SetCheck.CROSS_PART: "Расхождение между частями документа",
    SetCheck.ACCEPTANCE_MISMATCH: "Расхождение с критерием приёмки",
    SetCheck.ACCEPTANCE_GAP: "Пробел в покрытии критериями приёмки",
    SetCheck.MISSING_REQUIREMENT: "Отсутствующее требование",
    SetCheck.MISSING_SCENARIO: "Отсутствующий сценарий",
    SetCheck.EMPTY_SECTION: "Пустой или незаполненный раздел",
    SetCheck.TERMINOLOGY: "Расхождение терминологии",
    SetCheck.DUPLICATION: "Дублирование",
    SetCheck.TRACEABILITY: "Нарушение трассируемости",
    SetCheck.OTHER: "Прочее",
}

QUOTE_STATUS_LABELS: dict[QuoteStatus, str] = {
    QuoteStatus.VERIFIED: "подтверждена дословно",
    QuoteStatus.REPAIRED: "уточнена по тексту документа",
    QuoteStatus.ANCHORED: "привязана к тексту требования",
}

FINDING_SOURCE_LABELS: dict[FindingSource, str] = {
    FindingSource.LLM: "семантический анализ",
    FindingSource.RULE: "детерминированное правило",
    FindingSource.REVIEW: "критический пересмотр",
}

STAGE_STATE_LABELS: dict[StageState, str] = {
    StageState.PENDING: "ожидает",
    StageState.RUNNING: "выполняется",
    StageState.DONE: "выполнен",
    StageState.FAILED: "ошибка",
    StageState.SKIPPED: "пропущен",
    StageState.CACHED: "из кэша",
}

RUN_STATUS_LABELS: dict[RunStatus, str] = {
    RunStatus.COMPLETE: "Полный",
    RunStatus.PARTIAL: "Частичный",
    RunStatus.RULES_ONLY: "Только проверки по правилам",
}

READINESS_LABELS: dict[Readiness, str] = {
    Readiness.YES: "Да",
    Readiness.CONDITIONAL: "Условно (после уточнений)",
    Readiness.NO: "Нет",
}

COVERAGE_STATUS_LABELS: dict[str, str] = {
    "covered": "Покрыто",
    "partial": "Частично",
    "none": "Не покрыто",
}

SKIPPED_REASON_LABELS: dict[str, str] = {
    "meta": "служебная информация",
    "template": "шаблон",
    "empty": "пустой блок",
    "other": "прочее",
}


def criterion_label(criterion: Criterion | str) -> str:
    """Return short Russian name for a criterion."""
    try:
        return CRITERION_LABELS[Criterion(criterion)]
    except ValueError:
        return str(criterion)


def criterion_ref(criterion: Criterion | str) -> str:
    """Return full Russian standard clause citation for a criterion."""
    try:
        return CRITERION_REFS[Criterion(criterion)]
    except ValueError:
        return str(criterion)


def severity_label(severity: Severity | str) -> str:
    """Return Russian label for a severity level."""
    try:
        return SEVERITY_LABELS[Severity(severity)]
    except ValueError:
        return str(severity)


def defect_class_label(defect_class: DefectClass | str) -> str:
    """Return Russian plural label for a defect class."""
    try:
        return DEFECT_CLASS_LABELS[DefectClass(defect_class)]
    except ValueError:
        return str(defect_class)


def req_kind_label(kind: ReqKind | str) -> str:
    """Return Russian label for a requirement kind."""
    try:
        return REQ_KIND_LABELS[ReqKind(kind)]
    except ValueError:
        return str(kind)


def set_check_label(set_check: SetCheck | str) -> str:
    """Return Russian label for a set-level check type."""
    try:
        return SET_CHECK_LABELS[SetCheck(set_check)]
    except ValueError:
        return str(set_check)


FINDING_DELTA_STATUS_LABELS: dict[FindingDeltaStatus, str] = {
    FindingDeltaStatus.CLOSED: "закрыто ответом",
    FindingDeltaStatus.REMAINED: "осталось",
    FindingDeltaStatus.CHANGED: "изменилось",
}

ADDRESSEE_LABELS: dict[str, str] = {
    "business": "Бизнес / Product Owner",
    "analyst": "Системный аналитик / API",
    "architect": "Архитектор / интеграции",
    "security": "Информационная безопасность",
    "qa": "QA",
}


def finding_delta_status_label(status: FindingDeltaStatus | str) -> str:
    """Return Russian label for a previous-finding delta status."""
    try:
        return FINDING_DELTA_STATUS_LABELS[FindingDeltaStatus(status)]
    except ValueError:
        return str(status)


def addressee_label(addressee: object) -> str:
    key = getattr(addressee, "value", addressee)
    return ADDRESSEE_LABELS.get(str(key), "Автор ТЗ") if key else "Автор ТЗ"
