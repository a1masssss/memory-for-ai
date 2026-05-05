from __future__ import annotations

import asyncio

from app import extractor
from app.memory_types import ExtractedMemory
from app.schemas import Message


class StubSettings:
    openai_extraction_enabled = True
    openai_api_key = "test-key"
    openai_model = "gpt-5-mini"
    openai_base_url = "https://api.openai.com/v1"
    openai_timeout_seconds = 20.0


def test_extract_memories_prefers_llm_output_without_rule_merge(monkeypatch) -> None:
    monkeypatch.setattr(extractor, "get_settings", lambda: StubSettings())
    monkeypatch.setattr(
        extractor,
        "extract_with_openai",
        _async_return(
            [
                ExtractedMemory(
                    memory_type="fact",
                    category="personal_context",
                    key="employment",
                    value="Notion",
                    evidence="I work at Notion.",
                    confidence=0.95,
                    attributes={"source": "openai"},
                )
            ]
        ),
    )
    monkeypatch.setattr(
        extractor,
        "extract_rule_based_memories",
        lambda messages: [
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="employment",
                value="Rule Corp",
                evidence="rule fallback",
                confidence=0.5,
            )
        ],
    )

    memories = asyncio.run(
        extractor.extract_memories(
            [Message(role="user", content="I work at Notion.")]
        )
    )

    assert len(memories) == 1
    assert memories[0].value == "Notion"
    assert memories[0].attributes == {"source": "openai"}


def test_extract_memories_uses_rules_only_when_llm_fails(monkeypatch) -> None:
    monkeypatch.setattr(extractor, "get_settings", lambda: StubSettings())
    monkeypatch.setattr(extractor, "extract_with_openai", _async_return(None))
    monkeypatch.setattr(
        extractor,
        "extract_rule_based_memories",
        lambda messages: [
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="current_location",
                value="Berlin",
                evidence="I moved to Berlin.",
                confidence=0.8,
            )
        ],
    )

    memories = asyncio.run(
        extractor.extract_memories(
            [Message(role="user", content="I moved to Berlin.")]
        )
    )

    assert len(memories) == 1
    assert memories[0].value == "Berlin"


def test_extract_memories_dedupes_duplicate_llm_memories(monkeypatch) -> None:
    monkeypatch.setattr(extractor, "get_settings", lambda: StubSettings())
    monkeypatch.setattr(
        extractor,
        "extract_with_openai",
        _async_return(
            [
                ExtractedMemory(
                    memory_type="fact",
                    category="personal_context",
                    key="employment",
                    value="Notion",
                    evidence="I work at Notion.",
                    confidence=0.95,
                    attributes={"source": "openai"},
                ),
                ExtractedMemory(
                    memory_type="fact",
                    category="personal_context",
                    key="employment",
                    value="Notion",
                    evidence="I work at Notion.",
                    confidence=0.93,
                    attributes={"source": "openai"},
                ),
            ]
        ),
    )

    memories = asyncio.run(
        extractor.extract_memories(
            [Message(role="user", content="I work at Notion.")]
        )
    )

    assert len(memories) == 1
    assert memories[0].value == "Notion"


def test_rule_extractor_ignores_non_user_messages() -> None:
    memories = extractor.extract_rule_based_memories(
        [
            Message(role="assistant", content="The user works at Notion in Berlin."),
            Message(role="tool", content="employment=Notion"),
        ]
    )

    assert memories == []


def _async_return(value):
    async def _inner(*args, **kwargs):  # noqa: ANN001, ANN202
        return value

    return _inner
