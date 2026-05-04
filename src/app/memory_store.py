from __future__ import annotations

import json
from typing import Any

import asyncpg

from .extractor import ExtractedMemory


MUTABLE_MEMORY_KEYS = {
    ("personal_context", "current_location"),
    ("personal_context", "employment"),
    ("creative_style", "visual_style"),
    ("camera_language", "camera_direction"),
    ("motion_language", "motion_style"),
    ("lighting", "lighting_style"),
}


async def save_extracted_memories(
    connection: asyncpg.Connection,
    *,
    memories: list[ExtractedMemory],
    user_id: str | None,
    session_id: str,
    turn_id: str,
) -> None:
    for memory in memories:
        supersedes = await _supersede_existing(
            connection,
            memory=memory,
            user_id=user_id,
            session_id=session_id,
        )
        if supersedes == "duplicate":
            continue

        await connection.execute(
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
                supersedes
            )
            VALUES ($1, $2::memory_kind, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
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
        )


async def _supersede_existing(
    connection: asyncpg.Connection,
    *,
    memory: ExtractedMemory,
    user_id: str | None,
    session_id: str,
) -> str | None:
    if (memory.category, memory.key) not in MUTABLE_MEMORY_KEYS:
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
