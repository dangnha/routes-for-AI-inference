"""Async inference jobs.

A job is accepted over HTTP (``POST /models/{slug}/jobs``) and executed in a
FastAPI background task. When it finishes, the result — or the error — is POSTed
to the ``callback_url`` supplied by the caller. This webhook pattern replaces the
previous Redis Pub/Sub + SSE streaming stack.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from ai_service.base import OCRProvider
from ai_service.webhooks import post_callback

logger = logging.getLogger("uvicorn.error")

JOBS: dict[str, dict] = {}


def create_job(task_id: str, model: str) -> str:
    job_id = uuid4().hex
    JOBS[job_id] = {
        "job_id": job_id,
        "task_id": task_id,
        "model": model,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


def get_job(job_id: str) -> dict | None:
    return JOBS.get(job_id)


async def run_job(
    job_id: str,
    provider: OCRProvider,
    content: bytes,
    task_id: str,
    callback_url: str,
) -> None:
    job = JOBS.get(job_id)
    if job is not None:
        job["status"] = "running"

    payload = {
        "task_id": task_id,
        "model": provider.slug,
        "status": "succeeded",
        "text": None,
        "confidence": None,
        "error": None,
        "lines": [],
    }

    try:
        result = await asyncio.to_thread(provider.predict, content)
        payload["text"] = result.text
        payload["confidence"] = result.confidence
        payload["lines"] = result.lines
        if job is not None:
            job["status"] = "succeeded"
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "failed"
        payload["error"] = str(exc)[:500]
        if job is not None:
            job["status"] = "failed"
        logger.exception("Inference job %s failed", job_id)

    try:
        await post_callback(callback_url, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Webhook callback to %s failed: %s", callback_url, exc)
