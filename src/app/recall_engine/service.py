from __future__ import annotations

import asyncpg

from ..embeddings import embed_text, vector_literal
from ..schemas import Citation, RecallResponse
from .context import _assemble_context
from .links import _expand_linked_memories, _merge_ranked_memories
from .profile import _build_query_profile
from .scoring import _score_memory, _score_turn
from .text import _snippet


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
