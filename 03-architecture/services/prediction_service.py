"""SERVICE LAYER — nghiệp vụ: điều phối model, chuyển đổi dữ liệu. Không dính HTTP."""

from models.iris_model import IrisModel
from schemas.prediction import IrisFeatures


class PredictionService:
    """Điều phối một lần dự đoán: schema -> vector -> model."""

    def __init__(self, model: IrisModel):
        self._model = model

    def predict(self, payload: IrisFeatures) -> dict:
        vector = payload.to_vector()
        return self._model.predict(vector)

    def model_info(self) -> dict:
        return {"model": "LogisticRegression", "dataset": "iris", "target_names": self._model.target_names}
