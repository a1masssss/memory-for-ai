from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


MAX_ID_LENGTH = 256
MAX_MESSAGE_CONTENT_LENGTH = 20_000
MAX_MESSAGES_PER_TURN = 50
MAX_QUERY_LENGTH = 2_000

MessageRole = Literal["user", "assistant", "tool"]
MemoryType = Literal[
    "fact",
    "preference",
    "opinion",
    "event",
    "constraint",
    "style_profile",
    "generation_feedback",
    "continuity_anchor",
]


class Message(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CONTENT_LENGTH)
    name: str | None = Field(default=None, max_length=128)


class TurnCreate(BaseModel):
    session_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    user_id: str | None = Field(default=None, max_length=MAX_ID_LENGTH)
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES_PER_TURN)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnCreated(BaseModel):
    id: str


class RecallRequest(BaseModel):
    query: str = Field(max_length=MAX_QUERY_LENGTH)
    session_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    user_id: str | None = Field(default=None, max_length=MAX_ID_LENGTH)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation]


class SearchRequest(BaseModel):
    query: str = Field(max_length=MAX_QUERY_LENGTH)
    session_id: str | None = Field(default=None, max_length=MAX_ID_LENGTH)
    user_id: str | None = Field(default=None, max_length=MAX_ID_LENGTH)
    limit: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: datetime
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class UserMemory(BaseModel):
    id: str
    type: MemoryType
    key: str
    value: str
    confidence: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_session: str
    source_turn: str
    created_at: datetime
    updated_at: datetime
    supersedes: str | None
    active: bool


class UserMemoriesResponse(BaseModel):
    memories: list[UserMemory]
