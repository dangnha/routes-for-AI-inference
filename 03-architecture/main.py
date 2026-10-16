"""
Part 4 - Source code architecture
=================================
Problem: as we add models, endpoints and validation, `main.py` grows into a long file that is hard
to read and maintain. We split the code by responsibility (layer).

Pure refactor: this is Part 3's code (validate-input), unchanged in behaviour, just spread across
files. The router Part 3 introduced now lives in `routers/prediction.py`.

    main.py    -> create the app and register routers (no business logic)
    routers/   -> declare endpoints and handle HTTP (request/response)
    schemas/   -> define data structures (Pydantic)
    services/  -> business logic, orchestrate inference
    models/    -> load the model and run predictions

Request flow for POST /predict
------------------------------
    HTTP request
        v
    routers/prediction.py            (read the body, call the service)
        v
    services/prediction_service.py   (schema -> feature vector)
        v
    models/iris_model.py             (loaded at startup, run sklearn predict)

Run (from the architecture/ folder):
    python train_model.py
    uvicorn main:app --reload --port 8003
UI:
    streamlit run app.py
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
    # MODEL LAYER: load the model once.
    iris_model = IrisModel(BASE_DIR / "model.pkl", BASE_DIR / "model_meta.json")
    iris_model.load()

    # SERVICE LAYER: build the service around that model.
    app.state.iris_model = iris_model
    app.state.prediction_service = PredictionService(iris_model)

    print("[startup] model + service ready")
    yield
    iris_model.unload()
    print("[shutdown] model released")


app = FastAPI(
    title="Architecture - routers / schemas / services / models",
    description="Part 4: split the source code into layers.",
    version="1.0.0",
    lifespan=lifespan,
)

# ROUTER LAYER: register endpoint groups. A new model -> include_router another module.
app.include_router(prediction.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8003, reload=True)
