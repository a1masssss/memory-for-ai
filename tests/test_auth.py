from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.routes import require_memory_auth


def test_auth_is_optional_when_token_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("MEMORY_AUTH_TOKEN", raising=False)
    get_settings.cache_clear()

    try:
        app = _build_auth_test_app()
        with TestClient(app) as client:
            response = client.get("/protected")

        assert response.status_code == 200
    finally:
        get_settings.cache_clear()


def test_auth_requires_valid_bearer_token_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_AUTH_TOKEN", "super-secret-token")
    get_settings.cache_clear()

    try:
        app = _build_auth_test_app()
        with TestClient(app) as client:
            health = client.get("/health")
            missing = client.get("/protected")
            wrong = client.get(
                "/protected",
                headers={"Authorization": "Bearer wrong-token"},
            )
            correct = client.get(
                "/protected",
                headers={"Authorization": "Bearer super-secret-token"},
            )
            lowercase_scheme = client.get(
                "/protected",
                headers={"Authorization": "bearer super-secret-token"},
            )

        assert health.status_code == 200
        assert missing.status_code == 401
        assert wrong.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"
        assert wrong.headers["www-authenticate"] == "Bearer"
        assert correct.status_code == 200
        assert lowercase_scheme.status_code == 200
    finally:
        get_settings.cache_clear()


def _build_auth_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/protected")
    def protected(_: None = Depends(require_memory_auth)) -> dict[str, str]:
        return {"status": "ok"}

    return app
