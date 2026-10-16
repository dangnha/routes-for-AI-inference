import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).parent
ml_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(BASE_DIR / "model.pkl", "rb") as f:
        ml_state["model"] = pickle.load(f)
    ml_state["meta"] = json.loads((BASE_DIR / "model_meta.json").read_text(encoding="utf-8"))
    print("[startup] model loaded once")
    yield
    ml_state.clear()


app = FastAPI(
    title="Validate Input with Pydantic",
    description="Part 3: use a Pydantic schema to validate the request body.",
    version="1.0.0",
    lifespan=lifespan,
)


# ===========================================================================
# ROUTER - a group of endpoints. Declared here, included into `app` at the bottom.
# A router keeps related routes together so they can later move to another file.
# ===========================================================================
router = APIRouter(tags=["prediction"])


# ===========================================================================
# REQUEST SCHEMA - the contract for the data clients send.
# Each field has a name, a type and constraints. FastAPI uses this schema to:
#   1) validate automatically (invalid -> 422),
#   2) render the form in Swagger UI.
# ===========================================================================
class IrisFeatures(BaseModel):
    """Request body for /predict. Four positive floats."""

    sepal_length: float = Field(..., gt=0, le=20, description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, le=20, description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, le=20, description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, le=20, description="Petal width (cm)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        }
    )

    def to_vector(self) -> list[float]:
        """Convert the schema into the feature vector the model expects."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


# ===========================================================================
# ENDPOINT - the `payload: IrisFeatures` argument makes FastAPI validate the body.
# If the body is invalid, this function is NEVER called; FastAPI returns 422 first.
# It is registered on the ROUTER, not directly on `app`.
# ===========================================================================
@router.post(
    "/predict",
    summary="Predict the Iris species (with input validation)",
    responses={422: {"description": "Request body does not match the IrisFeatures schema"}},
)
def predict(payload: IrisFeatures):
    model = ml_state["model"]
    x = np.asarray(payload.to_vector(), dtype=float).reshape(1, -1)

    prediction = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    target_names = ml_state["meta"]["target_names"]

    return {
        "prediction": prediction,
        "label": target_names[prediction],
        "probabilities": {name: round(float(p), 4) for name, p in zip(target_names, probabilities)},
    }


# Register the router's routes on the app (one line per group of endpoints).
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
