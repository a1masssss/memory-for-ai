from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest


def test_restart_persistence(client: httpx.Client) -> None:
    if os.getenv("ENABLE_DOCKER_RESTART_TEST") != "1":
        pytest.skip("set ENABLE_DOCKER_RESTART_TEST=1 to run Docker restart test")

    user_id = "test-restart-persistence"
    client.delete(f"/users/{user_id}")
    client.post(
        "/turns",
        json={
            "session_id": "restart-1",
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

    subprocess.run(
        ["docker", "compose", "restart", "api"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    _wait_for_health(client)

    recall = client.post(
        "/recall",
        json={
            "query": "Where does this user live?",
            "session_id": "restart-2",
            "user_id": user_id,
            "max_tokens": 256,
        },
    )

    assert recall.status_code == 200
    assert "Berlin" in recall.json()["context"]


def _wait_for_health(client: httpx.Client) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise AssertionError("service did not become healthy after restart")
