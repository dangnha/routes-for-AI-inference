"""Endpoints: read the request, call the service."""

from fastapi import APIRouter, Depends, Request

from schemas.prediction import IrisFeatures
from services.prediction_service import PredictionService

router = APIRouter(tags=["prediction"])


def get_service(request: Request) -> PredictionService:
    """Return the service built at startup."""
    return request.app.state.prediction_service


@router.get("/info", summary="Model information")
def info(service: PredictionService = Depends(get_service)):
    return service.model_info()


@router.post("/predict", summary="Predict the Iris species")
def predict(payload: IrisFeatures, service: PredictionService = Depends(get_service)):
    return service.predict(payload)
