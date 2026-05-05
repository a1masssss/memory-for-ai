from __future__ import annotations

from datetime import UTC, datetime

import httpx

from app.schemas import MAX_MESSAGE_CONTENT_LENGTH, MAX_MESSAGES_PER_TURN


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


def test_broader_generic_facts_are_structured_and_recalled(
    client: httpx.Client,
) -> None:
    user_id = "test-broader-generic-facts"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "broader-generic-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "My name is Alex. I'm a PM at Notion, based out of Berlin. "
                        "I have a cat named Mochi. My son's name is Leo. "
                        "I have a peanut allergy and don't eat meat. "
                        "Please give me detailed step-by-step answers."
                    ),
                }
            ],
            "timestamp": "2025-03-18T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    values = {memory["value"] for memory in memories.json()["memories"]}
    assert "Alex" in values
    assert "Notion" in values
    assert "PM" in values
    assert "Berlin" in values
    assert "Cat named Mochi" in values
    assert "Has a son named Leo" in values
    assert "Allergic to peanuts" in values
    assert "Vegetarian" in values
    assert "Prefers detailed, step-by-step answers" in values

    recall = client.post(
        "/recall",
        json={
            "query": (
                "Who is this user, and what should I remember about their role, "
                "family, pet, food restrictions, and answer style?"
            ),
            "session_id": "broader-generic-2",
            "user_id": user_id,
            "max_tokens": 512,
        },
    )
    assert recall.status_code == 200
    context = recall.json()["context"]
    assert "Alex" in context
    assert "PM" in context
    assert "Cat named Mochi" in context
    assert "Allergic to peanuts" in context
    assert "Prefers detailed, step-by-step answers" in context


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


def test_shorthand_correction_phrase_updates_location_and_employment(
    client: httpx.Client,
) -> None:
    user_id = "test-shorthand-correction"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "shorthand-correction-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I live in Berlin and work at Stripe.",
                }
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "shorthand-correction-2",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Sorry, not Berlin - Munich now. Sorry, not Stripe - I joined Notion.",
                }
            ],
            "timestamp": "2025-03-16T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    recall = client.post(
        "/recall",
        json={
            "query": "Where do they live and where do they work now?",
            "session_id": "shorthand-correction-3",
            "user_id": user_id,
            "max_tokens": 256,
        },
    )
    assert recall.status_code == 200
    context = recall.json()["context"]
    assert "Munich" in context
    assert "Notion" in context
    assert "Berlin" not in context
    assert "Stripe" not in context


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


def test_multi_message_turn_with_tool_noise_extracts_only_user_fact(
    client: httpx.Client,
) -> None:
    user_id = "test-tool-noise"
    client.delete(f"/users/{user_id}")

    response = client.post(
        "/turns",
        json={
            "session_id": "tool-noise-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I live in Berlin.",
                },
                {
                    "role": "tool",
                    "name": "lookup",
                    "content": "Potential cities: Paris, London, Madrid",
                },
                {
                    "role": "assistant",
                    "content": "Got it, Berlin noted.",
                },
            ],
            "timestamp": "2025-03-19T09:30:00Z",
            "metadata": {},
        },
    )

    assert response.status_code == 201
    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    values = {memory["value"] for memory in memories.json()["memories"]}
    assert "Berlin" in values
    assert "Paris" not in values
    assert "London" not in values
    assert "Madrid" not in values


def test_search_returns_raw_turn_when_no_structured_memory_matches(
    client: httpx.Client,
) -> None:
    user_id = "test-search-raw-turn"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "search-raw-turn-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I am prepping for a distributed systems interview next week.",
                }
            ],
            "timestamp": "2025-03-19T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    response = client.post(
        "/search",
        json={
            "query": "distributed systems interview",
            "session_id": None,
            "user_id": user_id,
            "limit": 5,
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert any(result["metadata"].get("kind") == "turn" for result in results)
    assert any(
        "distributed systems interview" in result["content"].lower()
        for result in results
    )


def test_search_respects_session_scope_for_same_user(client: httpx.Client) -> None:
    user_id = "test-search-session-scope"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "search-scope-a",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "For one project, prefer neon lighting.",
                }
            ],
            "timestamp": "2025-03-19T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "search-scope-b",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "For the other project, prefer soft daylight.",
                }
            ],
            "timestamp": "2025-03-19T10:31:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    scoped = client.post(
        "/search",
        json={
            "query": "project",
            "session_id": "search-scope-a",
            "user_id": None,
            "limit": 10,
        },
    )
    global_user = client.post(
        "/search",
        json={
            "query": "project",
            "session_id": None,
            "user_id": user_id,
            "limit": 10,
        },
    )

    assert scoped.status_code == 200
    scoped_contents = {result["content"] for result in scoped.json()["results"]}
    assert any("neon lighting" in content.lower() for content in scoped_contents)
    assert not any("soft daylight" in content.lower() for content in scoped_contents)

    assert global_user.status_code == 200
    global_contents = {result["content"] for result in global_user.json()["results"]}
    assert any("neon lighting" in content.lower() for content in global_contents)
    assert any("soft daylight" in content.lower() for content in global_contents)


def test_search_cold_query_returns_empty_results(client: httpx.Client) -> None:
    user_id = "test-search-cold"
    client.delete(f"/users/{user_id}")

    response = client.post(
        "/search",
        json={
            "query": "favorite spaceship",
            "session_id": None,
            "user_id": user_id,
            "limit": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"results": []}


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
                    "content": "The repo codename is blue-heron and the failing shard is allocator-7.",
                },
                {
                    "role": "assistant",
                    "content": "Got it. I will keep those debugging details in view.",
                },
            ],
            "timestamp": "2025-03-15T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    recall = client.post(
        "/recall",
        json={
            "query": "Which codename and failing shard did the user mention?",
            "session_id": "recent-context-2",
            "user_id": user_id,
            "max_tokens": 256,
        },
    )

    assert recall.status_code == 200
    body = recall.json()
    assert "Relevant From Recent Conversations" in body["context"]
    assert "blue-heron" in body["context"]
    assert "allocator-7" in body["context"]
    assert body["citations"]


def test_session_only_memory_does_not_bleed_without_user_id(
    client: httpx.Client,
) -> None:
    client.delete("/sessions/anon-session-a")
    client.delete("/sessions/anon-session-b")

    client.post(
        "/turns",
        json={
            "session_id": "anon-session-a",
            "user_id": None,
            "messages": [
                {
                    "role": "user",
                    "content": "I live in Berlin and work at Notion.",
                }
            ],
            "timestamp": "2025-03-20T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    same_session = client.post(
        "/recall",
        json={
            "query": "Where does this user live?",
            "session_id": "anon-session-a",
            "user_id": None,
            "max_tokens": 128,
        },
    )
    other_session = client.post(
        "/recall",
        json={
            "query": "Where does this user live?",
            "session_id": "anon-session-b",
            "user_id": None,
            "max_tokens": 128,
        },
    )

    assert same_session.status_code == 200
    assert "Berlin" in same_session.json()["context"]
    assert other_session.status_code == 200
    assert other_session.json() == {"context": "", "citations": []}


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


def test_recall_does_not_blow_far_past_small_token_budget(client: httpx.Client) -> None:
    user_id = "test-budget-discipline"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "budget-discipline-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I work at Notion, live in Berlin, have a dog named Biscuit, "
                        "am vegetarian, allergic to shellfish, and prefer concise answers. "
                        "For videos, use raw documentary style, handheld framing, soft daylight, "
                        "slow motion, and avoid glossy polish."
                    ),
                }
            ],
            "timestamp": "2025-03-20T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    recall = client.post(
        "/recall",
        json={
            "query": "Give me the most important things to remember about this user.",
            "session_id": "budget-discipline-2",
            "user_id": user_id,
            "max_tokens": 24,
        },
    )

    assert recall.status_code == 200
    context = recall.json()["context"]
    assert context
    assert len(context.split()) <= 48


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


def test_repeating_same_mutable_fact_does_not_create_duplicate_memory(
    client: httpx.Client,
) -> None:
    user_id = "test-duplicate-mutable-fact"
    client.delete(f"/users/{user_id}")

    payload = {
        "session_id": "duplicate-mutable-fact",
        "user_id": user_id,
        "messages": [{"role": "user", "content": "I work at Notion."}],
        "timestamp": "2025-03-21T10:30:00Z",
        "metadata": {},
    }

    client.post("/turns", json=payload).raise_for_status()
    client.post("/turns", json=payload).raise_for_status()

    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    employment_memories = [
        memory
        for memory in memories.json()["memories"]
        if memory["key"] == "employment"
    ]
    assert len(employment_memories) == 1
    assert employment_memories[0]["value"] == "Notion"
    assert employment_memories[0]["active"] is True


def test_delete_session_only_removes_that_session_data(client: httpx.Client) -> None:
    user_id = "test-delete-session"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "delete-session-a",
            "user_id": user_id,
            "messages": [{"role": "user", "content": "I live in Berlin."}],
            "timestamp": "2025-03-22T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()
    client.post(
        "/turns",
        json={
            "session_id": "delete-session-b",
            "user_id": user_id,
            "messages": [{"role": "user", "content": "I work at Notion."}],
            "timestamp": "2025-03-23T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    deleted = client.delete("/sessions/delete-session-a")
    assert deleted.status_code == 204

    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    memory_values = {memory["value"] for memory in memories.json()["memories"]}
    assert "Berlin" not in memory_values
    assert "Notion" in memory_values

    recall = client.post(
        "/recall",
        json={
            "query": "Where does this user work and live?",
            "session_id": "delete-session-c",
            "user_id": user_id,
            "max_tokens": 256,
        },
    )
    assert recall.status_code == 200
    context = recall.json()["context"]
    assert "Notion" in context
    assert "Berlin" not in context


def test_delete_user_clears_memories_and_turns(client: httpx.Client) -> None:
    user_id = "test-delete-user"
    client.delete(f"/users/{user_id}")

    client.post(
        "/turns",
        json={
            "session_id": "delete-user-1",
            "user_id": user_id,
            "messages": [
                {
                    "role": "user",
                    "content": "I live in Berlin and work at Notion.",
                }
            ],
            "timestamp": "2025-03-24T10:30:00Z",
            "metadata": {},
        },
    ).raise_for_status()

    deleted = client.delete(f"/users/{user_id}")
    assert deleted.status_code == 204

    memories = client.get(f"/users/{user_id}/memories")
    assert memories.status_code == 200
    assert memories.json()["memories"] == []

    recall = client.post(
        "/recall",
        json={
            "query": "Where do they live?",
            "session_id": "delete-user-2",
            "user_id": user_id,
            "max_tokens": 128,
        },
    )
    search = client.post(
        "/search",
        json={
            "query": "Berlin",
            "session_id": None,
            "user_id": user_id,
            "limit": 5,
        },
    )

    assert recall.status_code == 200
    assert recall.json() == {"context": "", "citations": []}
    assert search.status_code == 200
    assert search.json()["results"] == []


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


def test_request_validation_rejects_invalid_limits_and_roles(
    client: httpx.Client,
) -> None:
    bad_role = client.post(
        "/turns",
        json={
            "session_id": "bad-role",
            "user_id": "bad-role-user",
            "messages": [{"role": "system", "content": "not allowed"}],
            "timestamp": "2025-03-25T10:30:00Z",
            "metadata": {},
        },
    )
    too_many_tokens = client.post(
        "/recall",
        json={
            "query": "Where do they live?",
            "session_id": "bad-recall-limit",
            "user_id": "bad-role-user",
            "max_tokens": 9000,
        },
    )
    bad_search_limit = client.post(
        "/search",
        json={
            "query": "Berlin",
            "session_id": None,
            "user_id": "bad-role-user",
            "limit": 100,
        },
    )
    too_many_messages = client.post(
        "/turns",
        json={
            "session_id": "too-many-messages",
            "user_id": "bad-role-user",
            "messages": [
                {"role": "user", "content": "hello"}
                for _ in range(MAX_MESSAGES_PER_TURN + 1)
            ],
            "timestamp": "2025-03-25T10:30:00Z",
            "metadata": {},
        },
    )
    too_large_message = client.post(
        "/turns",
        json={
            "session_id": "too-large-message",
            "user_id": "bad-role-user",
            "messages": [
                {
                    "role": "user",
                    "content": "x" * (MAX_MESSAGE_CONTENT_LENGTH + 1),
                }
            ],
            "timestamp": "2025-03-25T10:30:00Z",
            "metadata": {},
        },
    )

    assert bad_role.status_code == 422
    assert too_many_tokens.status_code == 422
    assert bad_search_limit.status_code == 422
    assert too_many_messages.status_code == 422
    assert too_large_message.status_code == 422
