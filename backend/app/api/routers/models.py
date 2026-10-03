"""Proxy the AI service model catalog so the frontend can offer a selector."""
import httpx
from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentUser
from app.schemas import ModelInfo
from app.services.ai_client import list_models

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_available_models(_: CurrentUser) -> list[ModelInfo]:
    try:
        return await list_models()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="AI service unavailable") from exc
