from __future__ import annotations

import asyncpg

from .constants import QUERY_ALIASES, TYPE_BOOSTS
from .models import QueryProfile
from .profile import _focus_boost
from .text import _keyword_overlap


def _score_memory(
    row: asyncpg.Record,
    query_profile: QueryProfile,
    session_id: str,
) -> float:
    lexical = float(row["lexical_score"] or 0.0)
    vector = float(row["vector_score"] or 0.0)
    alias = QUERY_ALIASES.get((row["category"], row["key"]), "")
    overlap = _keyword_overlap(
        query_profile.overlap_terms,
        f"{row['category']} {row['key']} {row['value']} {row['evidence']} {alias}",
    )
    if lexical <= 0 and overlap <= 0 and vector <= 0.12:
        return 0.0
    type_boost = TYPE_BOOSTS.get(row["memory_type"], 1.0)
    same_session = 0.25 if row["source_session"] == session_id else 0.0
    stable_memory = 0.2 if row["category"] in {"personal_context", "creative_style"} else 0.0
    focus_boost = _focus_boost(
        category=str(row["category"]),
        key=str(row["key"]),
        query_profile=query_profile,
    )
    return (
        (lexical * 2.0)
        + (vector * 1.4)
        + (overlap * 1.3)
        + type_boost
        + same_session
        + stable_memory
        + focus_boost
    )


def _score_turn(
    row: asyncpg.Record,
    query_profile: QueryProfile,
    session_id: str,
) -> float:
    lexical = float(row["lexical_score"] or 0.0)
    vector = float(row["vector_score"] or 0.0)
    overlap = _keyword_overlap(query_profile.overlap_terms, row["content_text"])
    if lexical <= 0 and overlap <= 0 and vector <= 0.14:
        return 0.0
    same_session = 0.2 if row["session_id"] == session_id else 0.0
    return (lexical * 1.8) + (vector * 1.1) + (overlap * 1.2) + same_session
