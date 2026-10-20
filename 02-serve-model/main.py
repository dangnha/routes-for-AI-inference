import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
META_PATH = BASE_DIR / "model_meta.json"

# Loaded once at startup, reused by every request.
ml_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH.name} not found. Run `python train_model.py` first.")
    with open(MODEL_PATH, "rb") as f:
        ml_state["model"] = pickle.load(f)
    ml_state["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))
    print("[startup] model loaded once and ready.")
    yield
    # ---- SHUTDOWN ----
    ml_state.clear()
    print("[shutdown] model released.")


app = FastAPI(
    title="Serve a Pretrained Model - Iris Logistic Regression",
    version="1.0.0",
    lifespan=lifespan,
)


# Model logic lives here, the endpoint only handles HTTP.
def run_inference(features: list[float]) -> dict:
    model = ml_state["model"]
    x = np.asarray(features, dtype=float).reshape(1, -1)  # shape (1, 4)

    prediction = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    target_names = ml_state["meta"]["target_names"]

    return {
        "prediction": prediction,
        "label": target_names[prediction],
        "probabilities": {name: round(float(p), 4) for name, p in zip(target_names, probabilities)},
    }


@app.get("/info", summary="Model information")
def info():
    return {
        "model": "LogisticRegression",
        "dataset": "iris",
        "feature_names": ml_state["meta"]["feature_names"],
        "target_names": ml_state["meta"]["target_names"],
    }


@app.post("/predict", summary="Predict the Iris species from 4 features")
def predict(features: list[float]):
    # body: [sepal_length, sepal_width, petal_length, petal_width]
    return run_inference(features)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
