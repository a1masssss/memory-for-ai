from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplyApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "memory-service"
    database_url: str = "postgresql://memory:memory@postgres:5432/memory"
    memory_auth_token: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_extraction_enabled: bool = True
    openai_timeout_seconds: float = 20.0


@lru_cache
def get_settings() -> ApplyApiSettings:
    return ApplyApiSettings()
