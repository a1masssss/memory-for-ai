from __future__ import annotations

import json
from typing import Any

import asyncpg

from .embeddings import vector_literal
from .memory_types import ExtractedMemory


MUTABLE_MEMORY_KEYS = {
    ("personal_context", "name"),
    ("personal_context", "current_location"),
    ("personal_context", "employment"),
    ("personal_context", "current_role"),
    ("personal_context", "dietary_preference"),
    ("communication", "answer_style"),
    ("creative_style", "visual_style"),
    ("camera_language", "camera_direction"),
    ("motion_language", "motion_style"),
    ("lighting", "lighting_style"),
}


async def save_extracted_memories(
    connection: asyncpg.Connection,
    *,
    memories: list[ExtractedMemory],
    memory_embeddings: list[list[float]] | None,
    user_id: str | None,
    session_id: str,
    turn_id: str,
) -> None:
    inserted_memories: list[tuple[str, ExtractedMemory]] = []
    for index, memory in enumerate(memories):
        supersedes = await _supersede_existing(
            connection,
            memory=memory,
            user_id=user_id,
            session_id=session_id,
        )
        if supersedes == "duplicate":
            continue
        embedding_literal = None
        if memory_embeddings is not None and index < len(memory_embeddings):
            embedding_literal = vector_literal(memory_embeddings[index])

        row = await connection.fetchrow(
            """
            INSERT INTO memories (
                user_id,
                memory_type,
                category,
                key,
                value,
                attributes,
                confidence,
                evidence,
                source_session,
                source_turn,
                supersedes,
                embedding
            )
            VALUES ($1, $2::memory_kind, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12::vector)
            RETURNING id
            """,
            user_id,
            memory.memory_type,
            memory.category,
            memory.key,
            memory.value,
            json.dumps(memory.attributes),
            memory.confidence,
            memory.evidence,
            session_id,
            turn_id,
            supersedes,
            embedding_literal,
        )
        if row is not None:
            inserted_memories.append((str(row["id"]), memory))

    if inserted_memories:
        await _create_memory_links(
            connection,
            inserted_memories=inserted_memories,
            user_id=user_id,
            session_id=session_id,
        )


async def _supersede_existing(
    connection: asyncpg.Connection,
    *,
    memory: ExtractedMemory,
    user_id: str | None,
    session_id: str,
) -> str | None:
    if (
        (memory.category, memory.key) not in MUTABLE_MEMORY_KEYS
        and not _is_explicit_opinion_correction(memory)
    ):
        return None

    row = await connection.fetchrow(
        """
        SELECT id, value
        FROM memories
        WHERE
            active = true
            AND category = $1
            AND key = $2
            AND (
                ($3::text IS NOT NULL AND user_id = $3)
                OR ($3::text IS NULL AND source_session = $4)
            )
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        memory.category,
        memory.key,
        user_id,
        session_id,
    )
    if row is None:
        return None

    if _normalize(row["value"]) == _normalize(memory.value):
        return "duplicate"

    await connection.execute(
        """
        UPDATE memories
        SET active = false, updated_at = now()
        WHERE id = $1
        """,
        row["id"],
    )
    return str(row["id"])


def _normalize(value: Any) -> str:
    return str(value).strip().lower()


def _is_explicit_opinion_correction(memory: ExtractedMemory) -> bool:
    return (
        memory.category == "opinions"
        and memory.attributes.get("revision_kind") == "correction"
    )


def memory_embedding_text(memory: ExtractedMemory) -> str:
    return " ".join(
        part
        for part in (
            memory.category.replace("_", " "),
            memory.key.replace("_", " "),
            memory.value,
            memory.evidence,
        )
        if part
    )


async def _create_memory_links(
    connection: asyncpg.Connection,
    *,
    inserted_memories: list[tuple[str, ExtractedMemory]],
    user_id: str | None,
    session_id: str,
) -> None:
    if len(inserted_memories) > 1:
        for source_id, _ in inserted_memories:
            for target_id, _ in inserted_memories:
                if source_id == target_id:
                    continue
                await _insert_link(
                    connection,
                    source_id=source_id,
                    target_id=target_id,
                    relation="co_mentioned",
                    weight=1.0,
                )

    await _link_opinion_history(
        connection,
        inserted_memories=inserted_memories,
        user_id=user_id,
        session_id=session_id,
    )

    for source_id, memory in inserted_memories:
        if memory.category != "personal_context":
            continue
        related_rows = await connection.fetch(
            """
            SELECT id
            FROM memories
            WHERE
                active = true
                AND category = 'personal_context'
                AND id <> $1::uuid
                AND (
                    ($2::text IS NOT NULL AND user_id = $2)
                    OR ($2::text IS NULL AND source_session = $3)
                )
            """,
            source_id,
            user_id,
            session_id,
        )
        for row in related_rows:
            target_id = str(row["id"])
            await _insert_link(
                connection,
                source_id=source_id,
                target_id=target_id,
                relation="same_profile",
                weight=0.9,
            )
            await _insert_link(
                connection,
                source_id=target_id,
                target_id=source_id,
                relation="same_profile",
                weight=0.9,
            )


async def _insert_link(
    connection: asyncpg.Connection,
    *,
    source_id: str,
    target_id: str,
    relation: str,
    weight: float,
) -> None:
    await connection.execute(
        """
        INSERT INTO memory_links (
            source_memory_id,
            target_memory_id,
            relation,
            weight
        )
        VALUES ($1::uuid, $2::uuid, $3, $4)
        ON CONFLICT (source_memory_id, target_memory_id, relation) DO NOTHING
        """,
        source_id,
        target_id,
        relation,
        weight,
    )


async def _link_opinion_history(
    connection: asyncpg.Connection,
    *,
    inserted_memories: list[tuple[str, ExtractedMemory]],
    user_id: str | None,
    session_id: str,
) -> None:
    for source_id, memory in inserted_memories:
        if memory.category != "opinions":
            continue

        prior_rows = await connection.fetch(
            """
            SELECT id
            FROM memories
            WHERE
                category = 'opinions'
                AND key = $1
                AND id <> $2::uuid
                AND (
                    ($3::text IS NOT NULL AND user_id = $3)
                    OR ($3::text IS NULL AND source_session = $4)
                )
            ORDER BY updated_at DESC
            LIMIT 3
            """,
            memory.key,
            source_id,
            user_id,
            session_id,
        )
        relation = _opinion_relation(memory)
        for row in prior_rows:
            await _insert_link(
                connection,
                source_id=str(row["id"]),
                target_id=source_id,
                relation=relation,
                weight=1.0,
            )


def _opinion_relation(memory: ExtractedMemory) -> str:
    revision_kind = str(memory.attributes.get("revision_kind") or "").strip()
    if revision_kind == "correction":
        return "opinion_correction"
    if revision_kind in {"refinement", "conditional"}:
        return "opinion_refinement"
    return "opinion_arc"
