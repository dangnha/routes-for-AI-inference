from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_service import MODELS_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Shared secret for callers; when set, /models* requires the X-Api-Key header.
    api_key: str | None = None

    # Shared secret sent with the webhook callback (X-Webhook-Secret).
    webhook_secret: str = "change-me"
    # How long the callback POST to the backend may take, in seconds.
    callback_timeout: float = 300.0

    trocr_model_id: str = "microsoft/trocr-base-printed"
    gotocr_model_id: str = "stepfun-ai/GOT-OCR2_0"

    # EasyOCR keeps its weights here (downloaded on first use).
    easyocr_model_dir: str = str(MODELS_DIR / "easyocr")
    easyocr_langs: list[str] = ["en"]

    max_length: int = 256
    # Named INFERENCE_DEVICE (not DEVICE) so a generic DEVICE env var cannot
    # silently override it.
    inference_device: str = "cuda"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
