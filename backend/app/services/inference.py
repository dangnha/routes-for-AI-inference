"""Task inference orchestration.

The backend never runs the model itself. It hands the image to the AI inference
service as an async job and waits for the service to report back via the
``/webhooks/inference`` endpoint — the webhook pattern that replaced the old
Redis Pub/Sub + WebSocket/SSE streaming stack. The database is the single source
of truth, so the frontend simply polls the task.
"""
from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import InferenceTask
from app.services.ai_client import submit_inference
from app.services.storage import download_object


def callback_url() -> str:
    base = settings.ai_callback_base_url.rstrip("/")
    return f"{base}{settings.api_v1_prefix}/webhooks/inference"


def _mark_failed(db, task_id: str, error: str) -> None:
    task = db.get(InferenceTask, task_id)
    if task is not None:
        task.status = "failed"
        task.error = error[:500]
        db.commit()


async def dispatch_inference(
    task_id: str, object_key: str, model: str, filename: str
) -> None:
    """Download the image and submit it to the AI service as an async job."""
    db = SessionLocal()
    try:
        content = await asyncio.to_thread(download_object, object_key)

        task = db.get(InferenceTask, task_id)
        if task is None:
            return
        task.status = "running"
        task.progress = 25
        db.commit()

        await submit_inference(
            model=model,
            content=content,
            filename=filename,
            task_id=task_id,
            callback_url=callback_url(),
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        _mark_failed(db, task_id, str(exc))
    finally:
        db.close()


def apply_result(db, payload) -> None:
    """Persist the result the AI service POSTed back to our webhook."""
    task = db.get(InferenceTask, payload.task_id)
    if task is None:
        return
    task.status = payload.status
    task.text = payload.text
    task.confidence = payload.confidence
    task.error = payload.error
    task.lines = payload.lines or None
    if payload.status == "succeeded":
        task.progress = 100
    db.commit()
