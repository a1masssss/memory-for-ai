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


def test_rule_extractor_handles_broader_generic_personal_facts() -> None:
    memories = extractor.extract_rule_based_memories(
        [
            Message(
                role="user",
                content=(
                    "My name is Alex. I'm a PM at Notion, based out of Berlin. "
                    "I have a cat named Mochi. My son's name is Leo. "
                    "I have a peanut allergy and don't eat meat. "
                    "Please give me detailed step-by-step answers."
                ),
            )
        ]
    )

    values = {memory.value for memory in memories}
    keys = {(memory.category, memory.key) for memory in memories}

    assert "Alex" in values
    assert "Notion" in values
    assert "PM" in values
    assert "Berlin" in values
    assert "Cat named Mochi" in values
    assert "Has a son named Leo" in values
    assert "Allergic to peanuts" in values
    assert "Vegetarian" in values
    assert "Prefers detailed, step-by-step answers" in values
    assert ("personal_context", "current_role") in keys


def test_rule_extractor_promotes_preparation_and_travel_to_events() -> None:
    memories = extractor.extract_rule_based_memories(
        [
            Message(
                role="user",
                content=(
                    "I'm preparing for a system design interview next month. "
                    "I'm also planning a trip to Tokyo this summer."
                ),
            )
        ]
    )

    values = {memory.value for memory in memories}
    event_keys = {
        memory.key for memory in memories if memory.category == "project_goal"
    }

    assert "Preparing for a system design interview" in values
    assert "Planning a trip to Tokyo" in values
    assert {"upcoming_focus", "travel_plan"} <= event_keys


def test_rule_extractor_does_not_confuse_interview_phrasing_for_employment() -> None:
    memories = extractor.extract_rule_based_memories(
        [
            Message(
                role="user",
                content=(
                    "I'm preparing for a system design interview at a FAANG company "
                    "next month."
                ),
            )
        ]
    )

    keys = {(memory.category, memory.key) for memory in memories}
    values = {memory.value for memory in memories}

    assert ("personal_context", "employment") not in keys
    assert ("personal_context", "current_role") not in keys
    assert "Preparing for a system design interview at a FAANG company" in values


def _async_return(value):
    async def _inner(*args, **kwargs):  # noqa: ANN001, ANN202
        return value

    return _inner
