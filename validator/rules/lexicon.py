"""Russian and English lexical dictionaries and word-boundary regexes for requirement prechecks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Common Russian inflection suffixes to match morphology while avoiding unrelated stems.
_RU_ADJ = r"(?:ый|ая|ое|ые|ого|ому|ым|ом|ой|ую|ых|ыми|о)"
_RU_ADJ_SOFT = r"(?:ий|ая|ее|ие|его|ему|им|ем|ей|ую|их|ими)"
_RU_ADJ_VELAR = r"(?:ий|ая|ое|ие|ого|ому|им|ом|ой|ую|их|ими|о)"
_RU_ADJ_OJ = r"(?:ой|ая|ое|ые|ого|ому|ым|ом|ую|ых|ыми|о)"


def compile_term_pattern(terms: Sequence[str], *, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    """Compile a sequence of regex terms into a single word-boundary aware pattern.

    Uses Unicode-aware ``(?<!\\w)`` / ``(?!\\w)`` lookarounds instead of ``\\b`` so
    abbreviations ending in punctuation (such as ``и т.д.`` or ``etc.``) match reliably.
    """
    ordered = sorted(terms, key=len, reverse=True)
    body = "|".join(f"(?:{term})" for term in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", flags)


# ---------------------------------------------------------------- hint dictionaries

VAGUE_WORDS_RU: tuple[str, ...] = (
    rf"быстр{_RU_ADJ}",
    rf"удобн{_RU_ADJ}",
    rf"интуитивн{_RU_ADJ}",
    rf"понятн{_RU_ADJ}",
    rf"прост{_RU_ADJ_OJ}",
    rf"дружелюбн{_RU_ADJ}",
    rf"красив{_RU_ADJ}",
    rf"современн{_RU_ADJ}",
    rf"над[её]жн{_RU_ADJ}",
    rf"эффективн{_RU_ADJ}",
    rf"оптимальн{_RU_ADJ}",
    rf"достаточн{_RU_ADJ}",
    rf"адекватн{_RU_ADJ}",
    rf"значительн{_RU_ADJ}",
    rf"существенн{_RU_ADJ}",
    rf"приемлем{_RU_ADJ}",
    rf"разумн{_RU_ADJ}",
    rf"качественн{_RU_ADJ}",
    rf"своевременн{_RU_ADJ}",
    rf"оперативн{_RU_ADJ}",
    rf"мгновенн{_RU_ADJ}",
    rf"моментальн{_RU_ADJ}",
    rf"бесконечн{_RU_ADJ}",
    rf"неограниченн{_RU_ADJ}",
    rf"минимальн{_RU_ADJ}",
    rf"масштабируем{_RU_ADJ}",
    rf"гибк{_RU_ADJ_VELAR}",
    r"небольш(?:ой|ая|ое|ие|ого|ому|им|ом|ую|их|ими)",
    rf"соответствующ{_RU_ADJ_SOFT}",
    rf"надлежащ{_RU_ADJ_SOFT}",
    rf"хорош(?:{_RU_ADJ_SOFT}|о)",
    r"некотор(?:ый|ая|ое|ые|ого|ому|ым|ом|ой|ую|ых|ыми)",
    r"несколько",
    r"в\s+реальном\s+времени",
    r"высок(?:ая|ой|ую)\s+производительност[ьи]",
    r"почти\s+всегда",
    r"обычно",
    r"в\s+основном",
    r"часто",
    r"регулярно",
    r"периодически",
)

VAGUE_WORDS_EN: tuple[str, ...] = (
    r"fast",
    r"quick(?:ly)?",
    r"rapid(?:ly)?",
    r"prompt(?:ly)?",
    r"timely",
    r"instant(?:ly|aneous(?:ly)?)?",
    r"real[- ]time",
    r"user[- ]friendly",
    r"intuitive(?:ly)?",
    r"easy",
    r"easily",
    r"simple",
    r"simply",
    r"convenient(?:ly)?",
    r"modern",
    r"attractive",
    r"reliable",
    r"reliably",
    r"efficient(?:ly)?",
    r"effective(?:ly)?",
    r"optimal(?:ly)?",
    r"robust",
    r"scalable",
    r"flexible",
    r"seamless(?:ly)?",
    r"adequate(?:ly)?",
    r"sufficient(?:ly)?",
    r"appropriate(?:ly)?",
    r"reasonable",
    r"reasonably",
    r"acceptable",
    r"proper(?:ly)?",
    r"suitable",
    r"significant(?:ly)?",
    r"substantial(?:ly)?",
    r"minimal(?:ly)?",
    r"high\s+performance",
    r"low\s+latency",
    r"usually",
    r"normally",
    r"generally",
    r"typically",
    r"mostly",
    r"frequently",
    r"often",
    r"several",
    r"various",
    r"unlimited",
    r"infinite(?:ly)?",
)

WEAK_MODALITY_RU: tuple[str, ...] = (
    r"по\s+возможности",
    r"долж(?:ен|на|но|ны)\s+стараться",
    r"стараться",
    r"стремиться(?:\s+к)?",
    r"может(?:\s+быть)?",
    r"могут",
    r"желательно",
    r"рекомендуется",
    r"рекомендовано",
    r"было\s+бы\s+хорошо",
    r"опционально",
    r"необязательно",
    r"на\s+будущее",
    r"как\s+бонус",
    r"планируется",
    r"предполагается",
    r"допускается",
)

WEAK_MODALITY_EN: tuple[str, ...] = (
    r"may",
    r"might",
    r"could",
    r"should\s+try",
    r"try\s+to",
    r"strive\s+to",
    r"aim\s+to",
    r"desirable",
    r"optional(?:ly)?",
    r"recommended",
    r"nice\s+to\s+have",
    r"ideally",
    r"preferably",
    r"where\s+possible",
    r"as\s+far\s+as\s+possible",
    r"best\s+effort",
)

OPEN_ENDED_LISTS_RU: tuple[str, ...] = (
    r"и\s+т\.\s*д\.?",
    r"и\s+т\.\s*п\.?",
    r"и\s+др\.?",
    r"и\s+пр\.?",
    r"и\s+так\s+далее",
    r"и\s+тому\s+подобн(?:ое|ые|ых)",
    r"и\s+проч(?:ее|ие|их|\.)",
    r"и\s+другие",
    r"и\s+любые\s+другие",
    r"и\s+подобн(?:ое|ые|ых)",
    r"в\s+том\s+числе,?\s+но\s+не\s+ограничиваясь",
    r"но\s+не\s+ограничиваясь",
    r"не\s+ограничиваясь",
)

OPEN_ENDED_LISTS_EN: tuple[str, ...] = (
    r"etc\.?",
    r"and\s+so\s+on",
    r"and\s+so\s+forth",
    r"and\s+the\s+like",
    r"and\s+others",
    r"among\s+others",
    r"and\s+more",
    r"including,?\s+but\s+not\s+limited\s+to",
    r"but\s+not\s+limited\s+to",
    r"not\s+limited\s+to",
)

LOOPHOLES_RU: tuple[str, ...] = (
    r"при\s+необходимости",
    r"в\s+случае\s+необходимости",
    r"по\s+мере\s+необходимости",
    r"если\s+возможно",
    r"если\s+применимо",
    r"если\s+требуется",
    r"если\s+целесообразно",
    r"если\s+позволяют",
    r"при\s+наличии\s+возможности",
    r"насколько\s+возможно",
    r"по\s+усмотрению",
    r"на\s+усмотрение",
    r"в\s+разумных\s+пределах",
    r"как\s+правило",
    r"в\s+общем\s+случае",
    r"в\s+установленном\s+порядке",
)

LOOPHOLES_EN: tuple[str, ...] = (
    r"if\s+necessary",
    r"if\s+possible",
    r"if\s+applicable",
    r"if\s+needed",
    r"if\s+required",
    r"if\s+feasible",
    r"if\s+appropriate",
    r"as\s+necessary",
    r"as\s+needed",
    r"as\s+applicable",
    r"as\s+appropriate",
    r"when\s+necessary",
    r"when\s+applicable",
    r"when\s+possible",
    r"where\s+applicable",
    r"where\s+appropriate",
    r"at\s+(?:the\s+)?discretion",
    r"within\s+reason",
    r"as\s+a\s+rule",
    r"to\s+the\s+extent\s+possible",
)

SUPERLATIVES_COMPARATIVES_RU: tuple[str, ...] = (
    rf"наилучш{_RU_ADJ_SOFT}",
    rf"наивысш{_RU_ADJ_SOFT}",
    rf"наименьш{_RU_ADJ_SOFT}",
    rf"наибольш{_RU_ADJ_SOFT}",
    rf"высочайш{_RU_ADJ_SOFT}",
    rf"лучш{_RU_ADJ_SOFT}",
    r"наиболее",
    r"наименее",
    r"максимально(?:\s+возможн\w*)?",
    (
        r"сам(?:ый|ая|ое|ые|ого|ому|ым|ом|ой|ую|ых|ыми)\s+"
        r"(?:быстр|лучш|высок|низк|прост|удобн|над[её]жн|эффективн|современн|больш|мал)\w*"
    ),
    r"быстрее",
    r"удобнее",
    r"проще",
    r"над[её]жнее",
    r"эффективнее",
    r"качественнее",
    r"точнее",
    r"лучше",
    r"более\s+(?:быстр|удобн|прост|над[её]жн|эффективн|точн|качественн|высок|низк)\w*",
    r"выше\s+качества",
)

SUPERLATIVES_COMPARATIVES_EN: tuple[str, ...] = (
    r"best",
    r"fastest",
    r"quickest",
    r"easiest",
    r"simplest",
    r"highest",
    r"lowest",
    r"maximally",
    r"as\s+fast\s+as\s+possible",
    r"as\s+soon\s+as\s+possible",
    r"as\s+quickly\s+as\s+possible",
    r"asap",
    r"most\s+(?:efficient|reliable|effective|accurate|convenient|modern|optimal|suitable|appropriate)",
    r"faster",
    r"quicker",
    r"better",
    r"easier",
    r"simpler",
    r"more\s+(?:efficient|reliable|effective|accurate|convenient|precise|secure)",
)

AND_OR_RU: tuple[str, ...] = (r"и\s*/\s*или",)
AND_OR_EN: tuple[str, ...] = (r"and\s*/\s*or",)


@dataclass(frozen=True)
class LexiconCategory:
    """A single lexical hint category with compiled bilingual matcher."""

    rule: str
    label_ru: str
    pattern: re.Pattern[str]
    terms_ru: tuple[str, ...]
    terms_en: tuple[str, ...]


HINT_CATEGORIES: tuple[LexiconCategory, ...] = (
    LexiconCategory(
        rule="and_or",
        label_ru="Неоднозначная связка «и/или»",
        pattern=compile_term_pattern(AND_OR_RU + AND_OR_EN),
        terms_ru=AND_OR_RU,
        terms_en=AND_OR_EN,
    ),
    LexiconCategory(
        rule="open_ended_list",
        label_ru="Открытый перечень без закрытого списка вариантов",
        pattern=compile_term_pattern(OPEN_ENDED_LISTS_RU + OPEN_ENDED_LISTS_EN),
        terms_ru=OPEN_ENDED_LISTS_RU,
        terms_en=OPEN_ENDED_LISTS_EN,
    ),
    LexiconCategory(
        rule="loophole",
        label_ru="Условная лазейка, позволяющая не выполнять требование",
        pattern=compile_term_pattern(LOOPHOLES_RU + LOOPHOLES_EN),
        terms_ru=LOOPHOLES_RU,
        terms_en=LOOPHOLES_EN,
    ),
    LexiconCategory(
        rule="weak_modality",
        label_ru="Слабая модальность вместо обязывающей формулировки",
        pattern=compile_term_pattern(WEAK_MODALITY_RU + WEAK_MODALITY_EN),
        terms_ru=WEAK_MODALITY_RU,
        terms_en=WEAK_MODALITY_EN,
    ),
    LexiconCategory(
        rule="superlative_comparative",
        label_ru="Превосходная или сравнительная степень без базы сравнения",
        pattern=compile_term_pattern(SUPERLATIVES_COMPARATIVES_RU + SUPERLATIVES_COMPARATIVES_EN),
        terms_ru=SUPERLATIVES_COMPARATIVES_RU,
        terms_en=SUPERLATIVES_COMPARATIVES_EN,
    ),
    LexiconCategory(
        rule="vague_word",
        label_ru="Расплывчатая или субъективная формулировка без измеримого критерия",
        pattern=compile_term_pattern(VAGUE_WORDS_RU + VAGUE_WORDS_EN),
        terms_ru=VAGUE_WORDS_RU,
        terms_en=VAGUE_WORDS_EN,
    ),
)


# ---------------------------------------------------------------- placeholders

PLACEHOLDER_INLINE_RU: tuple[str, ...] = (
    r"заполня(?:ется|ются|ть)",
    r"уточня(?:ется|ются)",
    (
        r"будет\s+(?:определен[оаы]?|уточнен[оаы]?|добавлен[оаы]?|заполнен[оаы]?|"
        r"описан[оаы]?|предоставлен[оаы]?|разработан[оаы]?|согласован[оаы]?)(?:\s+позже)?"
    ),
    r"подлежит\s+(?:уточнению|заполнению|определению|согласованию)",
    r"требует(?:ся)?\s+(?:уточнени[яе]|заполнени[яе]|доработк[иа]|определени[яе])",
    r"на\s+стадии\s+(?:разработки|согласования|уточнения)",
    r"в\s+процессе\s+(?:разработки|заполнения|уточнения)",
    r"в\s+разработке",
    r"не\s+заполнен[оаы]?",
    r"отметки\s+не\s+проставлены",
    r"да\s*/\s*нет(?:\s*/\s*не\s+требуется)?",
)

PLACEHOLDER_INLINE_EN: tuple[str, ...] = (
    r"TBD",
    r"TBA",
    r"TBC",
    r"TODO",
    r"FIXME",
    r"WIP",
    r"to\s+be\s+(?:determined|defined|confirmed|decided|specified|filled|added|provided)",
    r"work\s+in\s+progress",
    r"coming\s+soon",
    r"placeholder",
    r"yes\s*/\s*no(?:\s*/\s*(?:n/a|not\s+required))?",
)

PLACEHOLDER_INLINE_RE: re.Pattern[str] = compile_term_pattern(
    PLACEHOLDER_INLINE_RU + PLACEHOLDER_INLINE_EN
)

# Standalone cell or block values that act as unfilled placeholders (e.g. "—", "...", "???").
PLACEHOLDER_VALUE_RE: re.Pattern[str] = re.compile(
    r"^(?:"
    r"[—–-]{1,3}"
    r"|\.{3}|…"
    r"|\?{1,3}"
    r"|<\s*(?:TBD|TBA|TODO|заполнить|указать|вставить|значение|текст|описание|placeholder)[^>]*>"
    r"|\[\s*(?:TBD|TBA|TODO|заполнить|указать|вставить|значение|текст|описание|placeholder)[^\]]*\]"
    r")$",
    re.IGNORECASE,
)

# Labeled dash placeholders inside a cell or line, e.g. "Prod: —" or "URL: -".
PLACEHOLDER_LABELED_RE: re.Pattern[str] = re.compile(
    r"(?<!\w)[A-Za-zА-Яа-яЁё0-9_-]{2,40}\s*:\s*(?:[—–-]|\?{1,3}|\.{3}|…)(?=\s*(?:[.,;)\]]|$))",
    re.IGNORECASE,
)

# Unselected "/"-separated options in template tables (e.g. "Да / Нет", "Yes / No / N/A").
TEMPLATE_OPTION_CELL_RE: re.Pattern[str] = re.compile(
    r"^(?!https?://)(?!/)[^/]{1,40}(?:\s*/\s*[^/]{1,40}){1,4}$"
)

# ---------------------------------------------------------------- structure & coverage markers

# Normative obligation verbs indicating a requirement statement.
NORMATIVE_STATEMENT_RE: re.Pattern[str] = re.compile(
    r"(?<!\w)(?:долж(?:ен|на|но|ны)|обязан(?:а|о|ы)?|shall|must)(?!\w)",
    re.IGNORECASE,
)

# Section titles that hold requirements (used for missing-ID coverage checks).
REQUIREMENT_SECTION_HEADING_RE: re.Pattern[str] = re.compile(
    r"требован|requirement",
    re.IGNORECASE,
)

# Section titles whose absence or stub status is critical for development/testing.
CRITICAL_SECTION_HEADING_RE: re.Pattern[str] = re.compile(
    r"требован|тестирован|при[её]мк|requirement|test|acceptance",
    re.IGNORECASE,
)

# Priority and importance markers (IEEE 830-1998 4.3.5 ranking).
PRIORITY_MARKER_RE: re.Pattern[str] = re.compile(
    r"(?<!\w)(?:"
    r"приоритет(?:а|у|ом|е|ы|ов|ам|ами|ах|н\w*)?"
    r"|критичност[ьи]"
    r"|важност[ьи]"
    r"|ранжирован\w*"
    r"|высокий|средний|низкий"
    r"|priority|priorities|prioritiz\w*|importance|criticality|moscow"
    r"|must|should|could"
    r"|high\s+priority|medium\s+priority|low\s+priority"
    r")(?!\w)",
    re.IGNORECASE,
)

# Case-sensitive short priority codes (e.g. P0..P3, П1..П3) checked separately from lowercase words.
PRIORITY_CODE_RE: re.Pattern[str] = re.compile(r"(?<!\w)(?:P[0-3]|П[1-3])(?!\w)")

# Declared data types commonly found in parameter definition tables (EN and RU).
DECLARED_TYPE_RE: re.Pattern[str] = re.compile(
    r"^(?:"
    r"string|str|text|varchar(?:\(\d+\))?|char(?:\(\d+\))?"
    r"|integer|int(?:8|16|32|64)?|uint(?:8|16|32|64)?|long|bigint|smallint|short"
    r"|number|numeric(?:\(\d+(?:,\s*\d+)?\))?|decimal(?:\(\d+(?:,\s*\d+)?\))?|float(?:32|64)?|double"
    r"|boolean|bool"
    r"|date|datetime|timestamp|time|uuid|guid|uri|url|email"
    r"|object|json|map|dict|record|struct|enum|binary|bytes|byte\[\]|file|any"
    r"|array(?:\s*[\[<(][A-Za-zА-Яа-яЁё0-9_,\s]+[\]>\)])?|list(?:\s*[\[<(][A-Za-zА-Яа-яЁё0-9_,\s]+[\]>\)])?"
    r"|(?:string|integer|int|number|boolean|object|uuid)\[\]"
    r"|строк[аи]|строков(?:ый|ой|ая|ое)|текст"
    r"|цел(?:ое|ый)(?:\s+число)?|числ(?:о|овой|овое)|дробн(?:ое|ый)(?:\s+число)?"
    r"|вещественн(?:ое|ый)(?:\s+число)?|десятичн(?:ое|ый)(?:\s+число)?"
    r"|булев(?:о|ое|ый|ая)?|логическ(?:ий|ое|ая)"
    r"|дата(?:\s+и\s+время)?|время|массив(?:\s+[а-яё]+)?|список(?:\s+[а-яё]+)?|объект|словарь|перечисление|файл"
    r")(?:\s*\?|\s*\|\s*null)?$",
    re.IGNORECASE,
)
