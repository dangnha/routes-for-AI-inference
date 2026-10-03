"""Inbound webhook: the AI inference service reports finished jobs here."""
from fastapi import APIRouter, Header, HTTPException, status

from app.api.deps import DbSession
from app.core.config import settings
from app.schemas import InferenceCallback
from app.services.inference import apply_result

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/inference")
def inference_callback(
    payload: InferenceCallback,
    db: DbSession,
    x_webhook_secret: str | None = Header(default=None),
) -> dict[str, bool]:
    if x_webhook_secret != settings.webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret",
        )
    apply_result(db, payload)
    return {"ok": True}
