"""HTTP client for the AI inference service."""
from __future__ import annotations

import httpx

from app.core.config import settings


def _headers() -> dict[str, str]:
    if settings.ai_service_api_key:
        return {"X-Api-Key": settings.ai_service_api_key}
    return {}


async def list_models() -> list[dict[str, str]]:
    url = f"{settings.ai_service_url}/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, headers=_headers())
        response.raise_for_status()
        return response.json()


async def submit_inference(
    model: str,
    content: bytes,
    filename: str,
    task_id: str,
    callback_url: str,
) -> dict:
    """Hand an image to the AI service as an async job; it answers via webhook."""
    url = f"{settings.ai_service_url}/models/{model}/jobs"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            files={"file": (filename, content, "application/octet-stream")},
            data={"task_id": task_id, "callback_url": callback_url},
            headers=_headers(),
        )
        response.raise_for_status()
        return response.json()
