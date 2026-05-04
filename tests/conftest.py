from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
import pytest


@pytest.fixture(scope="session")
def service_url() -> str:
    return os.getenv("MEMORY_SERVICE_URL", "http://127.0.0.1:8080").rstrip("/")


@pytest.fixture(scope="session")
def client(service_url: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=service_url, timeout=10.0) as http_client:
        _wait_for_health(http_client)
        yield http_client


def _wait_for_health(client: httpx.Client) -> None:
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError("memory service did not become healthy") from last_error
