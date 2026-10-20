"""
Part 3.5 - Organizing source code
Same API as 03-validate-input, split into files by responsibility:

    main.py    -> create the app and register routers
    routers/   -> endpoints, HTTP in and out
    schemas/   -> Pydantic data shapes
    services/  -> business logic
    models/    -> load the model and run predictions

Run:
    python train_model.py
    uvicorn main:app --reload --port 8003
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from models.iris_model import IrisModel
from routers import prediction
from services.prediction_service import PredictionService

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, then build the service around it.
    iris_model = IrisModel(BASE_DIR / "model.pkl", BASE_DIR / "model_meta.json")
    iris_model.load()
    app.state.prediction_service = PredictionService(iris_model)

    print("[startup] model + service ready")
    yield
    iris_model.unload()
    print("[shutdown] model released")


app = FastAPI(
    title="Architecture - routers / schemas / services / models",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(prediction.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8003, reload=True)
