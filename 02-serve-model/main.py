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

# In-memory state: load once, reuse for every request.
ml_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH.name} not found. Run `python train_model.py` first.")
    with open(MODEL_PATH, "rb") as f:
        ml_state["model"] = pickle.load(f)

    if META_PATH.exists():
        ml_state["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        ml_state["meta"] = {"feature_names": [], "target_names": []}

    print("[startup] model loaded once and ready.")
    yield
    # ---- SHUTDOWN ----
    ml_state.clear()
    print("[shutdown] model released.")


app = FastAPI(
    title="Serve a Pretrained Model - Iris Logistic Regression",
    description="Part 2: serve a pretrained scikit-learn model with FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# SERVICE - keep prediction logic out of the endpoint.
# The endpoint only handles HTTP; this function handles "how the model runs".
# ---------------------------------------------------------------------------
def run_inference(features: list[float]) -> dict:
    """Run one prediction: feature vector -> model -> result dict."""
    model = ml_state["model"]

    # 1) Build the input vector in the right shape: (1, 4)
    x = np.asarray(features, dtype=float).reshape(1, -1)

    # 2) Call the model
    prediction = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]

    # 3) Map the class index to a readable label
    target_names = ml_state["meta"].get("target_names") or [str(i) for i in range(len(probabilities))]

    return {
        "prediction": prediction,
        "label": target_names[prediction],
        "probabilities": {name: round(float(p), 4) for name, p in zip(target_names, probabilities)},
    }


@app.get("/info", summary="Model information (feature and class names)")
def info():
    return {
        "model": "LogisticRegression",
        "dataset": "iris",
        "feature_names": ml_state["meta"].get("feature_names", []),
        "target_names": ml_state["meta"].get("target_names", []),
    }


@app.post(
    "/predict",
    summary="Predict the Iris species from 4 features",
    description="Send a JSON array of four numbers: [sepal_length, sepal_width, petal_length, petal_width].",
)
def predict(features: list[float]):
    return run_inference(features)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
