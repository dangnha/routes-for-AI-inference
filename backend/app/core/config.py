from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Local AI Inference Platform"
    api_v1_prefix: str = "/api/v1"

    # Supabase Postgres is the only database — there is no local fallback.
    database_url: str

    @field_validator("database_url")
    @classmethod
    def _require_postgres(cls, value: str) -> str:
        if not value.startswith("postgresql"):
            raise ValueError(
                "DATABASE_URL must be a Supabase Postgres URL "
                "(postgresql+psycopg://...). A SQLite value here usually means a "
                "stale DATABASE_URL is exported in your shell, overriding .env."
            )
        return value

    # Supabase project (Auth + Storage).
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "ocr-uploads"

    # AI inference service.
    ai_service_url: str = "http://localhost:8001"
    ai_service_api_key: str | None = None

    # URL the AI service uses to POST finished jobs back to this backend.
    ai_callback_base_url: str = "http://localhost:8000"
    # Shared secret required on the inbound webhook (X-Webhook-Secret header).
    webhook_secret: str = "change-me"

    cors_origins: list[str] = ["http://localhost:8501"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
