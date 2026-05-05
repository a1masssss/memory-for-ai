from __future__ import annotations

import asyncpg

from .constants import HOP_RELATION_BOOSTS, MAX_HOP_ANCHORS, QUERY_ALIASES
from .models import QueryProfile
from .profile import _focus_boost
from .text import _keyword_overlap


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
