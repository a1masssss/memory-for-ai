from __future__ import annotations

import json
from pathlib import Path

import httpx


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "recall_quality.json"


def test_recall_quality_fixture(client: httpx.Client) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    total_expected = 0
    matched_expected = 0

    for scenario in fixture["scenarios"]:
        user_id = scenario["user_id"]
        client.delete(f"/users/{user_id}")

        for turn in scenario["turns"]:
            response = client.post(
                "/turns",
                json={
                    "session_id": turn["session_id"],
                    "user_id": user_id,
                    "messages": turn["messages"],
                    "timestamp": turn["timestamp"],
                    "metadata": {"fixture": scenario["name"]},
                },
            )
            assert response.status_code == 201

        for probe in scenario["probes"]:
            response = client.post(
                "/recall",
                json={
                    "query": probe["query"],
                    "session_id": probe["session_id"],
                    "user_id": user_id,
                    "max_tokens": 512,
                },
            )
            assert response.status_code == 200
            context = response.json()["context"].lower()

            for expected in probe["expected_terms"]:
                total_expected += 1
                if expected.lower() in context:
                    matched_expected += 1

            for forbidden in probe["forbidden_terms"]:
                assert forbidden.lower() not in context

    quality = matched_expected / total_expected
    assert quality >= 0.8
