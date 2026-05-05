from __future__ import annotations

import asyncio
import json
from typing import Any

from app.openai_extractor import extract_with_openai
from app.schemas import Message


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_extractor_parses_structured_output(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert timeout == 20.0
        body = json.loads(request.data.decode("utf-8"))
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        return FakeHTTPResponse(
            {
                "output_text": json.dumps(
                    {
                        "memories": [
                            {
                                "memory_type": "fact",
                                "category": "personal_context",
                                "key": "employment",
                                "value": "Works at Notion",
                                "evidence": "I just joined Notion as a PM.",
                                "confidence": 0.92,
                                "attributes": {
                                    "revision_kind": "correction",
                                    "topic": "employment",
                                },
                            }
                        ]
                    }
                )
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    memories = asyncio.run(
        extract_with_openai(
            messages=[
                Message(role="user", content="I just joined Notion as a PM.")
            ],
            api_key="test-key",
            model="gpt-5-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=20.0,
        )
    )

    assert len(memories) == 1
    assert memories[0].memory_type == "fact"
    assert memories[0].category == "personal_context"
    assert memories[0].key == "employment"
    assert memories[0].value == "Works at Notion"
    assert memories[0].attributes == {
        "source": "openai",
        "revision_kind": "correction",
        "topic": "employment",
    }


def test_openai_extractor_skips_invalid_memories(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        return FakeHTTPResponse(
            {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "memories": [
                                            {
                                                "memory_type": "invalid",
                                                "category": "personal_context",
                                                "key": "employment",
                                                "value": "Works at Notion",
                                                "evidence": "I joined Notion.",
                                                "confidence": 0.9,
                                            }
                                        ]
                                    }
                                ),
                            }
                        ]
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    memories = asyncio.run(
        extract_with_openai(
            messages=[Message(role="user", content="I joined Notion.")],
            api_key="test-key",
            model="gpt-5-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=20.0,
        )
    )

    assert memories == []


def test_openai_extractor_returns_none_on_invalid_json(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        return FakeHTTPResponse({"output_text": "{not valid json"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    memories = asyncio.run(
        extract_with_openai(
            messages=[Message(role="user", content="I joined Notion.")],
            api_key="test-key",
            model="gpt-5-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=20.0,
        )
    )

    assert memories is None


def test_openai_extractor_filters_non_scalar_attributes(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001
        return FakeHTTPResponse(
            {
                "output_text": json.dumps(
                    {
                        "memories": [
                            {
                                "memory_type": "opinion",
                                "category": "opinions",
                                "key": "typescript",
                                "value": "Likes TypeScript",
                                "evidence": "I love TypeScript.",
                                "confidence": 0.88,
                                "attributes": {
                                    "Topic Name": "TypeScript",
                                    "stance": "positive",
                                    "weight": 2,
                                    "nested": {"bad": "value"},
                                    "items": ["bad"],
                                },
                            }
                        ]
                    }
                )
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    memories = asyncio.run(
        extract_with_openai(
            messages=[Message(role="user", content="I love TypeScript.")],
            api_key="test-key",
            model="gpt-5-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=20.0,
        )
    )

    assert memories is not None
    assert memories[0].attributes == {
        "source": "openai",
        "topic_name": "TypeScript",
        "stance": "positive",
        "weight": 2.0,
    }
