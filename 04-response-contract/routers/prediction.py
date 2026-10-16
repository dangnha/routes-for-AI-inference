"""ROUTER LAYER - declare endpoints and attach the response_model (the output contract)."""

from fastapi import APIRouter, Depends, Query, Request

from schemas.prediction import ErrorResponse, IrisFeatures, PredictionResponse
from services.prediction_service import PredictionService

router = APIRouter(tags=["prediction"])


def get_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@router.post(
    "/predict",
    response_model=PredictionResponse,  # <-- the contract: filters + validates the output
    summary="Predict (with a stable response contract)",
    responses={
        200: {"description": "Prediction succeeded"},
        400: {"model": ErrorResponse, "description": "Confidence below the requested threshold"},
        422: {"model": ErrorResponse, "description": "Request body does not match the schema"},
    },
)
def predict(
    payload: IrisFeatures,
    min_confidence: float = Query(0.0, ge=0, le=1, description="Minimum accepted confidence"),
    service: PredictionService = Depends(get_service),
):
    # Returns a dict that also has `debug_raw_scores`; response_model strips it.
    return service.predict(payload, min_confidence=min_confidence)


@router.get(
    "/models/{model_id}",
    summary="Get model info (404 if it does not exist)",
    responses={404: {"model": ErrorResponse}},
)
def get_model(model_id: str, service: PredictionService = Depends(get_service)):
    return service.get_model(model_id)


@router.get("/debug/boom", summary="Deliberately raise an error to show status 500")
def boom():
    raise RuntimeError("simulated internal error")
