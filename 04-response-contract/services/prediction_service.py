"""SERVICE LAYER - business logic. Maps the inference result to the response contract."""

from models.iris_model import IrisModel
from schemas.prediction import IrisFeatures
from services.errors import LowConfidenceError, ModelNotFoundError


class PredictionService:
    def __init__(self, model: IrisModel):
        self._model = model

    def predict(self, payload: IrisFeatures, min_confidence: float = 0.0) -> dict:
        raw = self._model.predict(payload.to_vector())  # internal inference result
        scores: dict[str, float] = raw["scores"]
        best_label = max(scores, key=scores.get)
        confidence = scores[best_label]

        # The confidence threshold is a BUSINESS rule -> 400 (not 422).
        if confidence < min_confidence:
            raise LowConfidenceError(confidence, min_confidence)

        return {
            "model": self._model.name,
            "predicted_label": raw["class_name"],
            "predicted_class": raw["class_index"],
            "confidence": round(confidence, 4),
            "probabilities": [
                {"label": label, "probability": round(score, 4)} for label, score in scores.items()
            ],
            # Internal field, NOT part of the contract -> removed by response_model.
            "debug_raw_scores": raw["raw_scores"],
        }

    def get_model(self, model_id: str) -> dict:
        if model_id != self._model.name:
            raise ModelNotFoundError(model_id)
        return {
            "id": self._model.name,
            "algorithm": self._model.algorithm,
            "dataset": self._model.dataset,
            "classes": self._model.target_names,
        }
