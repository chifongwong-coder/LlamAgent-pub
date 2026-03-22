"""
Skill matcher: tokenization, word normalization, and LLM prompt templates.

Provides lightweight NLP utilities for B-level tag matching and
LLM prompt templates for disambiguation (B-level 2+ candidates)
and fallback (C-level full metadata scan).

Zero external NLP dependencies:
- English: generate normalized word variants (plural/tense), then exact set intersection
- Chinese: handled via substring matching at the index level
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------
# Word normalization (English only, zero dependencies)
# ------------------------------------------------------------------


def normalize_word(word: str) -> set[str]:
    """
    Generate candidate base forms for an English word.

    Handles common inflections: plural (-s, -es, -ies),
    past tense (-ed), progressive (-ing), with doubled-consonant
    and silent-e restoration.

    Words shorter than 4 characters return as-is to avoid over-normalization.

    Args:
        word: Lowercase English word.

    Returns:
        Set of candidate forms (always includes the original).
    """
    word = word.lower()
    if len(word) < 4:
        return {word}

    forms = {word}

    # Plural: -ies -> -y, -es -> -e / strip, -s -> strip
    if word.endswith("ies") and len(word) > 4:
        forms.add(word[:-3] + "y")  # cities -> city
    if word.endswith("es") and len(word) > 3:
        forms.add(word[:-1])  # databases -> database
        forms.add(word[:-2])  # processes -> process
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        forms.add(word[:-1])  # migrations -> migration, tools -> tool

    # Past tense / past participle: -ed
    if word.endswith("ed") and not word.endswith("eed") and len(word) > 4:
        base = word[:-2]
        forms.add(base)  # migrated -> migrat
        forms.add(base + "e")  # migrated -> migrate
        if len(base) >= 2 and base[-1] == base[-2]:
            forms.add(base[:-1])  # stopped -> stop

    # Progressive: -ing
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        forms.add(base)  # deploying -> deploy
        forms.add(base + "e")  # migrating -> migrate
        if len(base) >= 2 and base[-1] == base[-2]:
            forms.add(base[:-1])  # running -> run

    return forms


# ------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------

# Split on non-alphanumeric, keep English/digit words and Chinese character blocks
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def tokenize_query(query: str) -> set[str]:
    """
    Tokenize a query string for tag matching.

    English words are lowercased and expanded into normalized variants.
    Chinese characters are kept as-is (matching is done via substring at the index level).

    Args:
        query: User query string.

    Returns:
        Set of normalized tokens (includes all variant forms).
    """
    tokens = set()
    for match in _TOKEN_PATTERN.finditer(query):
        word = match.group().lower()
        if word.isascii():
            tokens.update(normalize_word(word))
        else:
            tokens.add(word)
    return tokens


# ------------------------------------------------------------------
# LLM prompt templates
# ------------------------------------------------------------------

DISAMBIGUATE_SYSTEM = """\
You are a skill selection assistant. Given the user's query and a list of \
candidate skills, select the most relevant skill(s) that should be activated \
to guide the agent's response.

Rules:
- Select only skills that are clearly relevant to the query.
- If none are relevant, return an empty list.
- Return JSON only: {"selected": ["skill-name-1", "skill-name-2"]}
"""

DISAMBIGUATE_USER_TEMPLATE = """\
User query: {query}

Candidate skills:
{candidates}

Which skill(s) should be activated? Return JSON: {{"selected": [...]}}"""

FALLBACK_SYSTEM = """\
You are a skill selection assistant. Given the user's query and the complete \
list of available skills, determine if any skill is relevant.

Rules:
- Select only skills that are clearly relevant to the query.
- If none are relevant, return an empty list.
- Return JSON only: {"selected": ["skill-name-1", "skill-name-2"]}
"""

FALLBACK_USER_TEMPLATE = """\
User query: {query}

Available skills:
{skills}

Which skill(s), if any, should be activated? Return JSON: {{"selected": [...]}}"""


def format_skill_list(skills: list) -> str:
    """
    Format a list of SkillMeta objects into a numbered text list.

    Multiline descriptions are collapsed to a single line to avoid
    ambiguous formatting in numbered lists.

    Args:
        skills: List of SkillMeta objects.

    Returns:
        Formatted string with numbered skill entries.
    """
    lines = []
    for i, meta in enumerate(skills, 1):
        desc = " ".join(meta.description.split())
        lines.append(f"{i}. {meta.name}: {desc}")
    return "\n".join(lines)
