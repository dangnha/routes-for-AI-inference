"""Outbound webhook callback used to report finished inference jobs."""
from __future__ import annotations

import httpx

from ai_service.config import settings


async def post_callback(url: str, payload: dict) -> None:
    """POST a job result to the caller's callback URL."""
    async with httpx.AsyncClient(timeout=settings.callback_timeout) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"X-Webhook-Secret": settings.webhook_secret},
        )
        response.raise_for_status()
