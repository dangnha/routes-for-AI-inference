"""Business logic: turn the request schema into a vector and call the model. No HTTP here."""

from models.iris_model import IrisModel
from schemas.prediction import IrisFeatures


class PredictionService:
    def __init__(self, model: IrisModel):
        self._model = model

    def predict(self, payload: IrisFeatures) -> dict:
        return self._model.predict(payload.to_vector())

    def model_info(self) -> dict:
        return {"model": "LogisticRegression", "dataset": "iris", "target_names": self._model.target_names}
