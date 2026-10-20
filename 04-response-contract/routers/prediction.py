"""Endpoints. response_model is the output contract of each route."""

from fastapi import APIRouter, Depends, Query, Request

from schemas.prediction import ErrorResponse, IrisFeatures, PredictionResponse
from services.prediction_service import PredictionService

router = APIRouter(tags=["prediction"])


def get_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@router.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict (with a stable response contract)",
    responses={
        400: {"model": ErrorResponse, "description": "Confidence below min_confidence"},
        422: {"model": ErrorResponse, "description": "Request body does not match the schema"},
    },
)
def predict(
    payload: IrisFeatures,
    min_confidence: float = Query(0.0, ge=0, le=1, description="Minimum accepted confidence"),
    service: PredictionService = Depends(get_service),
):
    # The service also returns debug_raw_scores; response_model removes it.
    return service.predict(payload, min_confidence=min_confidence)


@router.get(
    "/models/{model_id}",
    summary="Get model info (404 if it does not exist)",
    responses={404: {"model": ErrorResponse}},
)
def get_model(model_id: str, service: PredictionService = Depends(get_service)):
    return service.get_model(model_id)


@router.get("/debug/boom", summary="Raise an error on purpose to show a 500")
def boom():
    raise RuntimeError("simulated internal error")
