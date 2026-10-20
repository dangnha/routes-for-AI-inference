"""
Part 5 - Design a stable response contract
response_model fixes the shape of every success response,
and the exception handlers give every error one shape: {"error": {"code", "message"}}.

Run:
    python train_model.py
    uvicorn main:app --reload --port 8004
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models.iris_model import IrisModel
from routers import prediction
from services.errors import LowConfidenceError, ModelNotFoundError
from services.prediction_service import PredictionService

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    iris_model = IrisModel(BASE_DIR / "model.pkl", BASE_DIR / "model_meta.json")
    iris_model.load()
    app.state.prediction_service = PredictionService(iris_model)
    print("[startup] model + service ready")
    yield
    iris_model.unload()
    print("[shutdown] model released")


app = FastAPI(title="Design a Stable Response Contract", version="1.0.0", lifespan=lifespan)

app.include_router(prediction.router)


# Error handlers: map each error to a status code and the same JSON shape.
@app.exception_handler(LowConfidenceError)
async def low_confidence_handler(request: Request, exc: LowConfidenceError):
    return JSONResponse(
        status_code=400,
        content={"error": {"code": "LOW_CONFIDENCE", "message": str(exc)}},
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": {"code": "MODEL_NOT_FOUND", "message": str(exc)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request body is invalid",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    # Do not leak internal details to the client.
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8004, reload=True)
