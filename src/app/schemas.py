from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    content: str = Field(min_length=1)
    name: str | None = None


class TurnCreate(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    messages: list[Message] = Field(min_length=1)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnCreated(BaseModel):
    id: str


class RecallRequest(BaseModel):
    query: str
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation]


class SearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None
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
    source_session: str
    source_turn: str
    created_at: datetime
    updated_at: datetime
    supersedes: str | None
    active: bool


class UserMemoriesResponse(BaseModel):
    memories: list[UserMemory]
