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


app = FastAPI(title="Validate Input with Pydantic", version="1.0.0", lifespan=lifespan)

# A router groups related endpoints. It is added to the app at the bottom.
router = APIRouter(tags=["prediction"])


# Request schema: invalid bodies get a 422 before predict() runs.
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
        """Feature order the model expects."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


@router.post("/predict", summary="Predict the Iris species (with input validation)")
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


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
