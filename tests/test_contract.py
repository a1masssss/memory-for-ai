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


def test_recall_handles_paraphrased_location_and_employment_queries(
    client: httpx.Client,
) -> None:
    user_id = "test-paraphrase-facts"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "paraphrase-facts-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I just moved to Berlin from NYC and now work at Notion as a PM.",
                }
            ],
            "timestamp": "2025-03-18T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    location_recall = client.post(
        "/recall",
        json={
            "query": "What city does this user call home these days?",
            "session_id": "paraphrase-facts-2",
            "user_id": user_id,
            "max_tokens": 128,
        },
    )
    assert location_recall.status_code == 200
    assert "Berlin" in location_recall.json()["context"]

    employment_recall = client.post(
        "/recall",
        json={
            "query": "Which company are they at now?",
            "session_id": "paraphrase-facts-3",
            "user_id": user_id,
            "max_tokens": 128,
        },
    )
    assert employment_recall.status_code == 200
    assert "Notion" in employment_recall.json()["context"]


def test_recall_handles_paraphrased_food_and_reply_style_queries(
    client: httpx.Client,
) -> None:
    user_id = "test-paraphrase-preferences"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "paraphrase-preferences-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I am vegetarian, allergic to shellfish, and please keep answers concise and direct.",
                }
            ],
            "timestamp": "2025-03-18T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    food_recall = client.post(
        "/recall",
        json={
            "query": "Any food restrictions I should keep in mind?",
            "session_id": "paraphrase-preferences-2",
            "user_id": user_id,
            "max_tokens": 128,
        },
    )
    assert food_recall.status_code == 200
    food_context = food_recall.json()["context"]
    assert "Vegetarian" in food_context
    assert "Allergic to shellfish" in food_context

    style_recall = client.post(
        "/recall",
        json={
            "query": "How chatty should my replies be?",
            "session_id": "paraphrase-preferences-3",
            "user_id": user_id,
            "max_tokens": 128,
        },
    )
    assert style_recall.status_code == 200
    assert "Prefers concise, direct answers" in style_recall.json()["context"]


def test_lowercase_facts_and_style_preferences_are_extracted(client: httpx.Client) -> None:
    user_id = "test-lowercase-facts"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "lowercase-facts-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "i work at notion now, i'm based in berlin these days, and please keep answers brief and direct.",
                }
            ],
            "timestamp": "2025-03-18T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    values = {memory["value"] for memory in memories.json()["memories"]}
    assert "Notion" in values
    assert "Berlin" in values
    assert "Prefers concise, direct answers" in values


def test_fact_correction_supersedes_prior_value(client: httpx.Client) -> None:
    user_id = "test-fact-correction"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "fact-correction-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I live in Berlin.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "fact-correction-2",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Actually I live in Munich now.",
                }
            ],
            "timestamp": "2025-03-16T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    memories = client.get(f"/users/{user_id}/memories").json()["memories"]
    locations = [memory for memory in memories if memory["key"] == "current_location"]
    assert any(memory["value"] == "Munich" and memory["active"] for memory in locations)
    assert any(memory["value"] == "Berlin" and not memory["active"] for memory in locations)
    assert any(
        memory["value"] == "Munich"
        and memory["attributes"].get("revision_kind") == "correction"
        for memory in locations
    )


def test_opinion_history_preserves_arc_and_attributes(client: httpx.Client) -> None:
    user_id = "test-opinion-arc"
    client.delete(f"/users/{user_id}")

    turns = [
        "I love TypeScript.",
        "TypeScript generics are getting annoying.",
        "TypeScript is fine for big projects, but I'd use Python for scripts.",
    ]
    for index, content in enumerate(turns, start=1):
        client.post(
            "/turns",
            json={
                "session_id": f"opinion-arc-{index}",
                "user_id": user_id,
                "messages": [{"role": "user", "content": content}],
                "timestamp": f"2025-03-1{index}T10:30:00Z",
                "metadata": {},
            },
        ).raise_for_status()

    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    opinion_memories = [
        memory
        for memory in memories.json()["memories"]
        if memory["key"] == "typescript"
    ]
    values = {memory["value"] for memory in opinion_memories}
    stances = {memory["attributes"].get("stance") for memory in opinion_memories}

    assert "Likes TypeScript" in values
    assert "TypeScript generics are getting annoying" in values
    assert "TypeScript is fine for big projects" in values
    assert {"positive", "negative", "conditional"} <= stances
    assert all(memory["active"] for memory in opinion_memories)

    recall = client.post(
        "/recall",
        json={
            "query": "What does this user think about TypeScript?",
            "session_id": "opinion-arc-4",
            "user_id": user_id,
            "max_tokens": 512,
        },
    )
    assert recall.status_code == 200
    context = recall.json()["context"]
    assert "Likes TypeScript" in context
    assert "TypeScript generics are getting annoying" in context
    assert "TypeScript is fine for big projects" in context


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


def test_recall_includes_recent_raw_turn_when_extractor_misses_it(
    client: httpx.Client,
) -> None:
    user_id = "test-recent-context"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "recent-context-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I'm preparing for a system design interview at a FAANG company next month.",
                },
                {
                    "role": "assistant",
                    "content": "Got it. We can focus on distributed systems and tradeoffs.",
                },
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    recall = client.post(
        "/recall",
        json={
            "query": "What is the user preparing for?",
            "session_id": "recent-context-2",
            "user_id": user_id,
            "max_tokens": 256,
        },
    )

    assert recall.status_code == 200
    body = recall.json()
    assert "Relevant From Recent Conversations" in body["context"]
    assert "system design interview" in body["context"]
    assert body["citations"]


def test_recall_uses_global_ranking_under_tight_budget(client: httpx.Client) -> None:
    user_id = "test-global-budget"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "global-budget-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I work at Stripe as an engineer. I just moved to Berlin from NYC last month. "
                        "I am vegetarian and allergic to shellfish. My dog is named Biscuit. "
                        "Please give me concise, direct answers."
                    ),
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    recall = client.post(
        "/recall",
        json={
            "query": "How should I answer this user?",
            "session_id": "global-budget-2",
            "user_id": user_id,
            "max_tokens": 20,
        },
    )

    assert recall.status_code == 200
    context = recall.json()["context"]
    assert "Communication Preferences" in context
    assert "Prefers concise, direct answers" in context


def test_recall_can_multi_hop_across_linked_personal_memories(
    client: httpx.Client,
) -> None:
    user_id = "test-multi-hop"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "multi-hop-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I was walking Biscuit this morning.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "multi-hop-2",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I just moved to Berlin from NYC last month.",
                }
            ],
            "timestamp": "2025-03-16T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "multi-hop-3",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I work at Notion as a PM and I am vegetarian.",
                }
            ],
            "timestamp": "2025-03-17T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    recall = client.post(
        "/recall",
        json={
            "query": "Which city does Biscuit's owner call home?",
            "session_id": "multi-hop-4",
            "user_id": user_id,
            "max_tokens": 24,
        },
    )

    assert recall.status_code == 200
    context = recall.json()["context"]
    assert "Dog named Biscuit" in context
    assert "Berlin" in context


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
