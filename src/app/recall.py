from __future__ import annotations

from dataclasses import dataclass
import re
from collections import defaultdict
from typing import Any

import asyncpg

from .embeddings import embed_text, vector_literal
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

RECENT_CONTEXT_TITLE = "Relevant From Recent Conversations"
MAX_MEMORIES_PER_SECTION = 6
MAX_HOP_ANCHORS = 8
HOP_RELATION_BOOSTS = {
    "same_profile": 0.45,
    "co_mentioned": 0.3,
    "opinion_arc": 0.2,
    "opinion_refinement": 0.28,
    "opinion_correction": 0.35,
}

STOPWORDS = {
    "what",
    "which",
    "this",
    "that",
    "these",
    "those",
    "user",
    "assistant",
    "tool",
    "their",
    "them",
    "they",
    "there",
    "about",
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

QUERY_INTENT_RULES = (
    {
        "pattern": re.compile(
            r"\b(where|live|based|located|location|city|home base|call home|hometown)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("location", "city", "live", "moved", "based", "home"),
        "keys": {("personal_context", "current_location")},
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(work|job|company|employer|role|joined|career|day job)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("employment", "work", "job", "company", "role", "joined"),
        "keys": {("personal_context", "employment")},
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(food|diet|dietary|eat|eats|allerg|restriction|vegetarian|vegan|shellfish)\b",
            flags=re.IGNORECASE,
        ),
        "terms": (
            "diet",
            "dietary",
            "vegetarian",
            "vegan",
            "allergy",
            "allergic",
            "restriction",
            "food",
        ),
        "keys": {
            ("personal_context", "dietary_preference"),
            ("personal_context", "allergy"),
        },
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(reply|respond|response|communication|tone|chatty|brief|concise|direct|wordy|verbose)\b",
            flags=re.IGNORECASE,
        ),
        "terms": (
            "communication",
            "answer",
            "style",
            "concise",
            "direct",
            "brief",
            "verbose",
        ),
        "keys": {("communication", "answer_style")},
        "categories": {"communication"},
    },
    {
        "pattern": re.compile(
            r"\b(style|styled|aesthetic|look|visual|vibe|feel|fashioned)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("visual", "style", "aesthetic", "look", "creative", "video"),
        "keys": {("creative_style", "visual_style")},
        "categories": {"creative_style"},
    },
    {
        "pattern": re.compile(
            r"\b(avoid|negative|constraint|glossy|artifacts?|wrong|failed|shouldn't)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("avoid", "negative", "constraint", "artifacts", "feedback"),
        "keys": set(),
        "categories": {"negative_constraints", "generation_feedback"},
    },
    {
        "pattern": re.compile(
            r"\b(camera|framing|frame|shot|angle|composition)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("camera", "composition", "framing", "shot", "angle"),
        "keys": {("camera_language", "camera_direction")},
        "categories": {"camera_language"},
    },
    {
        "pattern": re.compile(
            r"\b(motion|movement|pace|pacing|energy|slow motion|fast paced)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("motion", "movement", "pacing", "speed", "energy"),
        "keys": {("motion_language", "motion_style")},
        "categories": {"motion_language"},
    },
    {
        "pattern": re.compile(
            r"\b(light|lighting|lit|mood|neon|daylight|golden hour)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("lighting", "light", "mood", "daylight", "neon"),
        "keys": {("lighting", "lighting_style")},
        "categories": {"lighting"},
    },
    {
        "pattern": re.compile(
            r"\b(pet|dog|cat|animal|owner|named)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("pet", "dog", "cat", "animal", "named"),
        "keys": {("personal_context", "pet")},
        "categories": {"personal_context"},
    },
)


@dataclass(frozen=True)
class QueryProfile:
    raw_query: str
    retrieval_query: str
    overlap_terms: set[str]
    target_keys: set[tuple[str, str]]
    target_categories: set[str]


async def build_recall_response(
    connection: asyncpg.Connection,
    *,
    query: str,
    session_id: str,
    user_id: str | None,
    max_tokens: int,
) -> RecallResponse:
    query_profile = _build_query_profile(query)
    query_embedding = vector_literal(await embed_text(query_profile.raw_query))
    memory_rows = await connection.fetch(
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
            END AS lexical_score,
            CASE
                WHEN embedding IS NOT NULL
                THEN (1 - (embedding <=> $4::vector))
                ELSE 0
            END AS vector_score
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
        query_profile.retrieval_query,
        session_id,
        user_id,
        query_embedding,
    )
    direct_ranked_memories = sorted(
        (
            (_score_memory(row, query_profile, session_id), row)
            for row in memory_rows
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    direct_ranked_memories = [
        (score, row) for score, row in direct_ranked_memories if score > 0.4
    ]
    linked_ranked_memories = await _expand_linked_memories(
        connection,
        ranked_memories=direct_ranked_memories,
        query_profile=query_profile,
        query_embedding=query_embedding,
        session_id=session_id,
        user_id=user_id,
    )
    ranked_memories = _merge_ranked_memories(
        direct_ranked_memories,
        linked_ranked_memories,
    )

    turn_rows = await connection.fetch(
        """
        SELECT
            id,
            content_text,
            session_id,
            timestamp,
            CASE
                WHEN length($1) > 0
                THEN ts_rank_cd(search_vector, websearch_to_tsquery('english', $1))
                ELSE 0
            END AS lexical_score,
            CASE
                WHEN embedding IS NOT NULL
                THEN (1 - (embedding <=> $4::vector))
                ELSE 0
            END AS vector_score
        FROM turns
        WHERE
            session_id = $2
            OR ($3::text IS NOT NULL AND user_id = $3)
        ORDER BY timestamp DESC
        LIMIT 24
        """,
        query_profile.retrieval_query,
        session_id,
        user_id,
        query_embedding,
    )

    ranked_turns = sorted(
        ((_score_turn(row, query_profile, session_id), row) for row in turn_rows),
        key=lambda item: item[0],
        reverse=True,
    )
    ranked_turns = [(score, row) for score, row in ranked_turns if score > 0.2]

    if not ranked_memories and not ranked_turns:
        return RecallResponse(context="", citations=[])

    context, used_memories, used_turns = _assemble_context(
        ranked_memories,
        ranked_turns,
        query_profile=query_profile,
        max_tokens=max_tokens,
    )
    citations = [
        Citation(
            turn_id=str(row["source_turn"]),
            score=round(score, 3),
            snippet=_snippet(f"{row['key']}: {row['value']}"),
        )
        for score, row in used_memories
    ]
    citations.extend(
        Citation(
            turn_id=str(row["id"]),
            score=round(score, 3),
            snippet=_snippet(row["content_text"]),
        )
        for score, row in used_turns
    )
    return RecallResponse(context=context, citations=citations)


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


async def _expand_linked_memories(
    connection: asyncpg.Connection,
    *,
    ranked_memories: list[tuple[float, asyncpg.Record]],
    query_profile: QueryProfile,
    query_embedding: str,
    session_id: str,
    user_id: str | None,
) -> list[tuple[float, asyncpg.Record]]:
    if not ranked_memories:
        return []

    anchor_scores = {
        str(row["id"]): score
        for score, row in ranked_memories[:MAX_HOP_ANCHORS]
    }
    linked_rows = await connection.fetch(
        """
        SELECT
            ml.source_memory_id,
            ml.relation,
            ml.weight,
            m.id,
            m.memory_type,
            m.category,
            m.key,
            m.value,
            m.evidence,
            m.source_session,
            m.source_turn,
            m.updated_at,
            CASE
                WHEN length($2) > 0
                THEN ts_rank_cd(m.search_vector, websearch_to_tsquery('english', $2))
                ELSE 0
            END AS lexical_score,
            CASE
                WHEN m.embedding IS NOT NULL
                THEN (1 - (m.embedding <=> $5::vector))
                ELSE 0
            END AS vector_score
        FROM memory_links AS ml
        JOIN memories AS m
            ON m.id = ml.target_memory_id
        WHERE
            ml.source_memory_id = ANY($1::uuid[])
            AND m.active = true
            AND (
                m.source_session = $3
                OR ($4::text IS NOT NULL AND m.user_id = $4)
            )
        ORDER BY m.updated_at DESC
        LIMIT 120
        """,
        list(anchor_scores.keys()),
        query_profile.retrieval_query,
        session_id,
        user_id,
        query_embedding,
    )

    direct_ids = {str(row["id"]) for _, row in ranked_memories}
    expanded: dict[str, tuple[float, asyncpg.Record]] = {}
    for row in linked_rows:
        target_id = str(row["id"])
        if target_id in direct_ids:
            continue

        anchor_score = anchor_scores.get(str(row["source_memory_id"]), 0.0)
        if anchor_score <= 0:
            continue

        lexical = float(row["lexical_score"] or 0.0)
        vector = float(row["vector_score"] or 0.0)
        alias = QUERY_ALIASES.get((row["category"], row["key"]), "")
        overlap = _keyword_overlap(
            query_profile.overlap_terms,
            f"{row['category']} {row['key']} {row['value']} {row['evidence']} {alias}",
        )
        relation_boost = HOP_RELATION_BOOSTS.get(str(row["relation"]), 0.2)
        stable_memory = 0.15 if row["category"] == "personal_context" else 0.0
        focus_boost = _focus_boost(
            category=str(row["category"]),
            key=str(row["key"]),
            query_profile=query_profile,
        )
        propagated = (
            (anchor_score * 0.5)
            + (lexical * 1.0)
            + (vector * 1.0)
            + (overlap * 0.7)
            + (float(row["weight"] or 1.0) * relation_boost)
            + stable_memory
            + focus_boost
        )
        if propagated <= 0.75:
            continue

        existing = expanded.get(target_id)
        if existing is None or propagated > existing[0]:
            expanded[target_id] = (propagated, row)

    return sorted(expanded.values(), key=lambda item: item[0], reverse=True)


def _merge_ranked_memories(
    direct_ranked: list[tuple[float, asyncpg.Record]],
    linked_ranked: list[tuple[float, asyncpg.Record]],
) -> list[tuple[float, asyncpg.Record]]:
    merged: dict[str, tuple[float, asyncpg.Record]] = {}
    for score, row in direct_ranked + linked_ranked:
        memory_id = str(row["id"])
        current = merged.get(memory_id)
        if current is None or score > current[0]:
            merged[memory_id] = (score, row)
    return sorted(merged.values(), key=lambda item: item[0], reverse=True)


def _assemble_context(
    ranked_memories: list[tuple[float, asyncpg.Record]],
    ranked_turns: list[tuple[float, asyncpg.Record]],
    *,
    query_profile: QueryProfile,
    max_tokens: int,
) -> tuple[
    str,
    list[tuple[float, asyncpg.Record]],
    list[tuple[float, asyncpg.Record]],
]:
    grouped, used_memories, used_tokens = _select_ranked_memories(
        ranked_memories,
        budget=max(16, max_tokens),
    )
    lines = _render_memory_sections(grouped)
    used_turns: list[tuple[float, asyncpg.Record]] = []
    budget = max(16, max_tokens)

    used_turn_ids = {str(row["source_turn"]) for _, row in used_memories}
    recent_lines: list[str] = []
    for score, row in ranked_turns:
        if str(row["id"]) in used_turn_ids:
            continue
        if _turn_is_redundant_with_selected_memories(
            row["content_text"],
            used_memories=used_memories,
            query_profile=query_profile,
        ):
            continue
        turn_summary = _format_recent_turn(row["content_text"])
        item_cost = _approx_tokens(f"- {turn_summary}")
        if not recent_lines:
            item_cost += _approx_tokens(f"## {RECENT_CONTEXT_TITLE}") + 1
        if used_tokens + item_cost > budget:
            continue
        if not recent_lines:
            recent_lines.append(f"## {RECENT_CONTEXT_TITLE}")
        recent_lines.append(f"- {turn_summary}")
        used_turns.append((score, row))
        used_tokens += item_cost

    if recent_lines:
        lines.extend(recent_lines)
        lines.append("")

    return "\n".join(lines).strip(), used_memories, used_turns


def _select_ranked_memories(
    ranked_memories: list[tuple[float, asyncpg.Record]],
    *,
    budget: int,
) -> tuple[
    dict[str, list[tuple[float, asyncpg.Record]]],
    list[tuple[float, asyncpg.Record]],
    int,
]:
    grouped: dict[str, list[tuple[float, asyncpg.Record]]] = defaultdict(list)
    used_memories: list[tuple[float, asyncpg.Record]] = []
    used_tokens = 0

    for score, row in ranked_memories:
        category = str(row["category"])
        if len(grouped[category]) >= MAX_MEMORIES_PER_SECTION:
            continue

        title = SECTION_TITLES.get(category, category.replace("_", " ").title())
        item_cost = _approx_tokens(f"- {row['value']}")
        if not grouped[category]:
            item_cost += _approx_tokens(f"## {title}") + 1

        if used_tokens + item_cost > budget:
            continue

        grouped[category].append((score, row))
        used_memories.append((score, row))
        used_tokens += item_cost

    return grouped, used_memories, used_tokens


def _render_memory_sections(
    grouped: dict[str, list[tuple[float, asyncpg.Record]]],
) -> list[str]:
    lines: list[str] = []
    for category in _section_order(grouped):
        title = SECTION_TITLES.get(category, category.replace("_", " ").title())
        lines.append(f"## {title}")
        for _, row in grouped[category]:
            lines.append(f"- {row['value']}")
        lines.append("")
    return lines


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


def _turn_is_redundant_with_selected_memories(
    content_text: str,
    *,
    used_memories: list[tuple[float, asyncpg.Record]],
    query_profile: QueryProfile,
) -> bool:
    if not used_memories:
        return False

    turn_terms = _terms(content_text)
    if not turn_terms:
        return False

    for _, row in used_memories:
        category = str(row["category"])
        key = str(row["key"])
        if (
            (category, key) not in query_profile.target_keys
            and category not in query_profile.target_categories
        ):
            continue
        alias = QUERY_ALIASES.get((category, key), "")
        slot_terms = _terms(f"{category} {key} {alias}")
        if turn_terms & slot_terms:
            return True
    return False


def _build_query_profile(query: str) -> QueryProfile:
    expanded_terms: list[str] = []
    seen_terms: set[str] = set()
    target_keys: set[tuple[str, str]] = set()
    target_categories: set[str] = set()

    for rule in QUERY_INTENT_RULES:
        if not rule["pattern"].search(query):
            continue
        for term in rule["terms"]:
            if term not in seen_terms:
                seen_terms.add(term)
                expanded_terms.append(term)
        target_keys.update(rule["keys"])
        target_categories.update(rule["categories"])

    retrieval_query = query
    if expanded_terms:
        retrieval_query = f"{query} {' '.join(expanded_terms)}"

    return QueryProfile(
        raw_query=query,
        retrieval_query=retrieval_query,
        overlap_terms=_terms(retrieval_query),
        target_keys=target_keys,
        target_categories=target_categories,
    )


def _focus_boost(
    *,
    category: str,
    key: str,
    query_profile: QueryProfile,
) -> float:
    if (category, key) in query_profile.target_keys:
        return 0.55
    if category in query_profile.target_categories:
        return 0.2
    return 0.0


def _keyword_overlap(query_terms_or_text: set[str] | str, document: str) -> float:
    query_terms = (
        query_terms_or_text
        if isinstance(query_terms_or_text, set)
        else _terms(query_terms_or_text)
    )
    if not query_terms:
        return 0.0
    document_terms = _terms(document)
    return len(query_terms & document_terms) / len(query_terms)


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 2 and term not in STOPWORDS
    }


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))


def _snippet(text: Any) -> str:
    value = str(text).strip()
    return value if len(value) <= 220 else value[:217].rstrip() + "..."


def _format_recent_turn(content_text: str) -> str:
    normalized = re.sub(r"\s+", " ", content_text).strip()
    return normalized if len(normalized) <= 280 else normalized[:277].rstrip() + "..."
