"""Router factory: one router per OCR model.

Each model exposes a synchronous ``/infer`` endpoint (handy for direct testing)
and an asynchronous ``/jobs`` endpoint that reports its result back via webhook.
"""
from __future__ import annotations

import asyncio

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from ai_service.base import OCRProvider
from ai_service.config import settings
from ai_service.jobs import create_job, get_job, run_job


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


jobs_router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(verify_api_key)])


@jobs_router.get("/{job_id}")
async def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def build_model_router(provider: OCRProvider) -> APIRouter:
    router = APIRouter(
        prefix=f"/models/{provider.slug}",
        tags=["models"],
        dependencies=[Depends(verify_api_key)],
    )

    @router.post("/infer")
    async def infer(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        try:
            result = await asyncio.to_thread(provider.predict, content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        return {
            "model": provider.slug,
            "text": result.text,
            "confidence": result.confidence,
            "lines": result.lines,
        }

    @router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_inference_job(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        task_id: str = Form(...),
        callback_url: str = Form(...),
    ) -> dict:
        content = await file.read()
        job_id = create_job(task_id, provider.slug)
        background_tasks.add_task(
            run_job, job_id, provider, content, task_id, callback_url
        )
        return {"job_id": job_id, "status": "accepted"}

    return router
