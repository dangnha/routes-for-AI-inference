"""MODEL LAYER — load model và thực hiện dự đoán. Không biết gì về HTTP."""

import json
import pickle
from pathlib import Path

import numpy as np


class IrisModel:
    """Bọc (wrap) một sklearn model: quản lý load/unload và chạy predict."""

    def __init__(self, model_path: Path, meta_path: Path):
        self._model_path = model_path
        self._meta_path = meta_path
        self._model = None
        self._target_names: list[str] = []

    # -- vòng đời -----------------------------------------------------------
    def load(self) -> None:
        with open(self._model_path, "rb") as f:
            self._model = pickle.load(f)
        self._target_names = json.loads(self._meta_path.read_text(encoding="utf-8"))["target_names"]

    def unload(self) -> None:
        self._model = None
        self._target_names = []

    # -- suy luận -----------------------------------------------------------
    def predict(self, features: list[float]) -> dict:
        """Nhận feature vector -> trả nhãn + xác suất."""
        x = np.asarray(features, dtype=float).reshape(1, -1)
        index = int(self._model.predict(x)[0])
        probabilities = self._model.predict_proba(x)[0]
        return {
            "prediction": index,
            "label": self._target_names[index],
            "probabilities": {
                name: round(float(p), 4) for name, p in zip(self._target_names, probabilities)
            },
        }

    @property
    def target_names(self) -> list[str]:
        return self._target_names
