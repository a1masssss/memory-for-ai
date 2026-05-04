from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import asyncpg

from .schemas import Citation, RecallResponse


SECTION_TITLES = {
    "personal_context": "Known Facts About This User",
    "creative_style": "Stable Creative Profile",
    "camera_language": "Camera And Composition Preferences",
    "motion_language": "Motion Preferences",
    "lighting": "Lighting Preferences",
    "negative_constraints": "Negative Prompt Memory",
    "generation_feedback": "Relevant Prior Generation Feedback",
    "character_continuity": "Continuity Anchors",
    "communication": "Communication Preferences",
    "opinions": "Known Opinions",
}

TYPE_BOOSTS = {
    "fact": 1.5,
    "preference": 1.4,
    "constraint": 1.35,
    "style_profile": 1.35,
    "generation_feedback": 1.15,
    "continuity_anchor": 1.25,
    "event": 1.0,
    "opinion": 1.0,
}

QUERY_ALIASES = {
    ("personal_context", "current_location"): "location city live lives where moved based",
    ("personal_context", "employment"): "employment work works job company role joined",
    ("personal_context", "pet"): "pet dog cat animal name named",
    ("personal_context", "dietary_preference"): "diet food vegetarian vegan eats",
    ("personal_context", "allergy"): "allergy allergic avoid food constraint",
    ("personal_context", "family"): "family child son daughter kid",
    ("communication", "answer_style"): "communication answer style concise direct verbose",
    ("creative_style", "visual_style"): "visual style aesthetic look generation video",
    ("camera_language", "camera_direction"): "camera composition framing shot angle video",
    ("motion_language", "motion_style"): "motion movement pacing speed video",
    ("lighting", "lighting_style"): "lighting light color mood video",
    ("negative_constraints", "avoid_glossy_polish"): "avoid negative constraint glossy polish",
}


async def build_recall_response(
    connection: asyncpg.Connection,
    *,
    query: str,
    session_id: str,
    user_id: str | None,
    max_tokens: int,
) -> RecallResponse:
    rows = await connection.fetch(
        """
        SELECT
            id,
            memory_type,
            category,
            key,
            value,
            evidence,
            source_session,
            source_turn,
            updated_at,
            CASE
                WHEN length($1) > 0
                THEN ts_rank_cd(search_vector, websearch_to_tsquery('english', $1))
                ELSE 0
            END AS lexical_score
        FROM memories
        WHERE
            active = true
            AND (
                source_session = $2
                OR ($3::text IS NOT NULL AND user_id = $3)
            )
        ORDER BY updated_at DESC
        LIMIT 80
        """,
        query,
        session_id,
        user_id,
    )

    ranked = sorted(
        ((_score(row, query, session_id), row) for row in rows),
        key=lambda item: item[0],
        reverse=True,
    )
    ranked = [(score, row) for score, row in ranked if score > 0.4]
    if not ranked:
        return RecallResponse(context="", citations=[])

    context, used = _assemble_context(ranked, max_tokens=max_tokens)
    citations = [
        Citation(
            turn_id=str(row["source_turn"]),
            score=round(score, 3),
            snippet=_snippet(f"{row['key']}: {row['value']}"),
        )
        for score, row in used
    ]
    return RecallResponse(context=context, citations=citations)


def _score(row: asyncpg.Record, query: str, session_id: str) -> float:
    lexical = float(row["lexical_score"] or 0.0)
    alias = QUERY_ALIASES.get((row["category"], row["key"]), "")
    overlap = _keyword_overlap(
        query,
        f"{row['category']} {row['key']} {row['value']} {row['evidence']} {alias}",
    )
    if lexical <= 0 and overlap <= 0:
        return 0.0
    type_boost = TYPE_BOOSTS.get(row["memory_type"], 1.0)
    same_session = 0.25 if row["source_session"] == session_id else 0.0
    stable_memory = 0.2 if row["category"] in {"personal_context", "creative_style"} else 0.0
    return (lexical * 2.0) + (overlap * 1.3) + type_boost + same_session + stable_memory


def _assemble_context(
    ranked: list[tuple[float, asyncpg.Record]],
    *,
    max_tokens: int,
) -> tuple[str, list[tuple[float, asyncpg.Record]]]:
    grouped: dict[str, list[tuple[float, asyncpg.Record]]] = defaultdict(list)
    for score, row in ranked:
        grouped[row["category"]].append((score, row))

    lines: list[str] = []
    used: list[tuple[float, asyncpg.Record]] = []
    budget = max(16, max_tokens)

    for category in _section_order(grouped):
        title = SECTION_TITLES.get(category, category.replace("_", " ").title())
        section_lines = [f"## {title}"]
        for score, row in grouped[category][:6]:
            section_lines.append(f"- {row['value']}")
            used.append((score, row))

        candidate_lines = lines + section_lines + [""]
        if _approx_tokens("\n".join(candidate_lines)) > budget:
            break
        lines = candidate_lines

    return "\n".join(lines).strip(), used


def _section_order(grouped: dict[str, list[tuple[float, asyncpg.Record]]]) -> list[str]:
    preferred = [
        "personal_context",
        "creative_style",
        "negative_constraints",
        "camera_language",
        "motion_language",
        "lighting",
        "character_continuity",
        "communication",
        "opinions",
        "generation_feedback",
    ]
    remaining = [category for category in grouped if category not in preferred]
    return [category for category in preferred if category in grouped] + sorted(remaining)


def _keyword_overlap(query: str, document: str) -> float:
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    document_terms = _terms(document)
    return len(query_terms & document_terms) / len(query_terms)


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 2
    }


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))


def _snippet(text: Any) -> str:
    value = str(text).strip()
    return value if len(value) <= 220 else value[:217].rstrip() + "..."
