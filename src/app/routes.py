from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Response, status

from . import db
from .extractor import extract_memories
from .memory_store import save_extracted_memories
from .recall import build_recall_response
from .schemas import (
    RecallRequest,
    RecallResponse,
    SearchRequest,
    SearchResponse,
    TurnCreate,
    TurnCreated,
    UserMemoriesResponse,
)


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/turns", response_model=TurnCreated, status_code=status.HTTP_201_CREATED)
async def create_turn(payload: TurnCreate) -> TurnCreated:
    content_text = "\n".join(
        f"{message.role}: {message.content}" for message in payload.messages
    )
    extracted_memories = extract_memories(payload.messages)
    async with db.transaction() as connection:
        row = await connection.fetchrow(
            """
            INSERT INTO turns (
                session_id,
                user_id,
                messages,
                timestamp,
                metadata,
                content_text
            )
            VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6)
            RETURNING id
            """,
            payload.session_id,
            payload.user_id,
            json.dumps([message.model_dump() for message in payload.messages]),
            payload.timestamp,
            json.dumps(payload.metadata),
            content_text,
        )
        if row is None:
            raise RuntimeError("Failed to create turn")
        turn_id = str(row["id"])
        await save_extracted_memories(
            connection,
            memories=extracted_memories,
            user_id=payload.user_id,
            session_id=payload.session_id,
            turn_id=turn_id,
        )
    return TurnCreated(id=turn_id)


@router.post("/recall", response_model=RecallResponse)
async def recall(payload: RecallRequest) -> RecallResponse:
    async with db.pool().acquire() as connection:
        return await build_recall_response(
            connection,
            query=payload.query,
            session_id=payload.session_id,
            user_id=payload.user_id,
            max_tokens=payload.max_tokens,
        )


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    rows = await db.fetch(
        """
        SELECT
            content_text AS content,
            ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS score,
            session_id,
            timestamp,
            metadata
        FROM turns
        WHERE
            ($2::text IS NULL OR session_id = $2)
            AND ($3::text IS NULL OR user_id = $3)
            AND search_vector @@ websearch_to_tsquery('english', $1)
        ORDER BY score DESC, timestamp DESC
        LIMIT $4
        """,
        payload.query,
        payload.session_id,
        payload.user_id,
        payload.limit,
    )
    return SearchResponse(
        results=[
            {
                "content": row["content"],
                "score": float(row["score"] or 0.0),
                "session_id": row["session_id"],
                "timestamp": row["timestamp"],
                "metadata": _json_dict(row["metadata"]),
            }
            for row in rows
        ]
    )


@router.get("/users/{user_id}/memories", response_model=UserMemoriesResponse)
async def get_user_memories(user_id: str) -> UserMemoriesResponse:
    rows = await db.fetch(
        """
        SELECT
            id,
            memory_type AS type,
            key,
            value,
            confidence,
            source_session,
            source_turn,
            created_at,
            updated_at,
            supersedes,
            active
        FROM memories
        WHERE user_id = $1
        ORDER BY active DESC, updated_at DESC
        """,
        user_id,
    )
    return UserMemoriesResponse(
        memories=[
            {
                "id": str(row["id"]),
                "type": row["type"],
                "key": row["key"],
                "value": row["value"],
                "confidence": float(row["confidence"]),
                "source_session": row["source_session"],
                "source_turn": str(row["source_turn"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "supersedes": str(row["supersedes"]) if row["supersedes"] else None,
                "active": row["active"],
            }
            for row in rows
        ]
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str) -> Response:
    await db.execute("DELETE FROM turns WHERE session_id = $1", session_id)
    await db.execute("DELETE FROM memories WHERE source_session = $1", session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str) -> Response:
    await db.execute("DELETE FROM turns WHERE user_id = $1", user_id)
    await db.execute("DELETE FROM memories WHERE user_id = $1", user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}
