from __future__ import annotations

from datetime import UTC, datetime

import httpx


def test_health(client: httpx.Client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_turn_roundtrip_extracts_and_recalls_video_memory(
    client: httpx.Client,
) -> None:
    user_id = "test-contract-video"
    client.delete(f"/users/{user_id}")

    response = client.post(
        "/turns",
        json={
            "session_id": "contract-video-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I want this fashion video to feel raw and documentary, not glossy. Use slow-motion close-ups.",
                },
                {
                    "role": "assistant",
                    "content": "I will keep it natural and restrained.",
                },
            ],
            "timestamp": datetime(2025, 3, 15, 10, 30, tzinfo=UTC).isoformat(),
            "metadata": {"test": "contract"},
        },
    )

    assert response.status_code == 201
    assert response.json()["id"]

    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    values = {memory["value"] for memory in memories.json()["memories"]}
    assert "raw documentary realism" in values
    assert "avoid glossy polish" in values
    assert "slow-motion movement" in values

    recall = client.post(
        "/recall",
        json={
            "query": "How should the next fashion video be styled?",
            "session_id": "contract-video-2",
            "user_id": user_id,
            "max_tokens": 256,
        },
    )
    assert recall.status_code == 200
    body = recall.json()
    assert "raw documentary realism" in body["context"]
    assert "avoid glossy polish" in body["context"]
    assert body["citations"]


def test_concurrent_users_do_not_bleed(client: httpx.Client) -> None:
    client.delete("/users/test-user-raw")
    client.delete("/users/test-user-anime")

    client.post(
        "/turns",
        json={
            "session_id": "isolation-raw",
            "user_id": "test-user-raw",
            "messages": [
                {
                    "role": "user",
                    "content": "For my videos, prefer raw documentary style and soft daylight.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "isolation-anime",
            "user_id": "test-user-anime",
            "messages": [
                {
                    "role": "user",
                    "content": "For my videos, prefer anime visuals with neon lighting.",
                }
            ],
            "timestamp": "2025-03-15T10:31:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    recall = client.post(
        "/recall",
        json={
            "query": "What style should the video use?",
            "session_id": "isolation-new",
            "user_id": "test-user-raw",
            "max_tokens": 256,
        },
    )

    assert recall.status_code == 200
    context = recall.json()["context"]
    assert "raw documentary realism" in context
    assert "anime-inspired visuals" not in context
    assert "neon lighting" not in context


def test_style_can_move_from_documentary_to_cinematic(client: httpx.Client) -> None:
    user_id = "test-style-to-cinematic"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "style-to-cinematic-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I want this fashion video to feel raw and documentary, not glossy.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "style-to-cinematic-2",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Actually let us move away from documentary. I want everything cinematic now.",
                }
            ],
            "timestamp": "2025-03-16T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    memories = client.get(f"/users/{user_id}/memories").json()["memories"]
    cinematic = [
        memory
        for memory in memories
        if memory["key"] == "visual_style" and memory["value"] == "cinematic visuals"
    ]
    documentary = [
        memory
        for memory in memories
        if memory["key"] == "visual_style"
        and memory["value"] == "raw documentary realism"
    ]

    assert cinematic and cinematic[0]["active"] is True
    assert documentary and documentary[0]["active"] is False

    recall = client.post(
        "/recall",
        json={
            "query": "What visual style should we use now?",
            "session_id": "style-to-cinematic-3",
            "user_id": user_id,
            "max_tokens": 256,
        },
    )
    context = recall.json()["context"]
    assert "cinematic visuals" in context
    assert "raw documentary realism" not in context


def test_generic_facts_evolve_and_are_recalled(client: httpx.Client) -> None:
    user_id = "test-generic-evolution"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "generic-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I work at Stripe as an engineer. I just moved to Berlin from NYC last month. I am vegetarian and allergic to shellfish. I was walking Biscuit this morning.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "generic-2",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I just joined Notion as a PM.",
                }
            ],
            "timestamp": "2025-03-16T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    memories = client.get(f"/users/{user_id}/memories").json()["memories"]
    employment = [
        memory for memory in memories if memory["key"] == "employment"
    ]
    values = {memory["value"] for memory in memories}

    assert "Berlin" in values
    assert "Vegetarian" in values
    assert "Allergic to shellfish" in values
    assert "Dog named Biscuit" in values
    assert any(memory["value"] == "Notion" and memory["active"] for memory in employment)
    assert any(not memory["active"] and memory["value"] == "Stripe" for memory in employment)

    recall = client.post(
        "/recall",
        json={
            "query": "Where does this user live and where do they work?",
            "session_id": "generic-3",
            "user_id": user_id,
            "max_tokens": 512,
        },
    )
    context = recall.json()["context"]
    assert "Berlin" in context
    assert "Notion" in context
    assert "Stripe" not in context


def test_search_returns_structured_memory_results(client: httpx.Client) -> None:
    user_id = "test-search-memory"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "search-memory-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "For brand videos, prefer raw documentary style with handheld camera feel.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    response = client.post(
        "/search",
        json={
            "query": "documentary style",
            "session_id": None,
            "user_id": user_id,
            "limit": 5,
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert any(result["content"] == "raw documentary realism" for result in results)
    assert any(result["metadata"].get("kind") == "memory" for result in results)


def test_unicode_turn_does_not_crash(client: httpx.Client) -> None:
    user_id = "test-unicode"
    client.delete(f"/users/{user_id}")

    response = client.post(
        "/turns",
        json={
            "session_id": "unicode-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "For the launch video, use neon lighting, subtle motion, and keep the mood calm ✨ Привет.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {"unicode": "✓"},
        },
    )

    assert response.status_code == 201


def test_unanswered_specific_query_returns_empty_context(client: httpx.Client) -> None:
    user_id = "test-noise-resistance"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "noise-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I just moved to Berlin from NYC last month.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    response = client.post(
        "/recall",
        json={
            "query": "What is this user's favorite color?",
            "session_id": "noise-2",
            "user_id": user_id,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"context": "", "citations": []}


def test_cold_recall_returns_empty_context(client: httpx.Client) -> None:
    client.delete("/users/test-cold-user")

    response = client.post(
        "/recall",
        json={
            "query": "What does this user like?",
            "session_id": "cold-session",
            "user_id": "test-cold-user",
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"context": "", "citations": []}


def test_malformed_turn_input_returns_4xx(client: httpx.Client) -> None:
    missing_fields = client.post("/turns", json={"session_id": "bad"})
    assert missing_fields.status_code == 422

    bad_json = client.post(
        "/turns",
        content="{bad json",
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= bad_json.status_code < 500
