"""Load the sklearn model and run predictions. Returns the raw inference result."""

import json
import pickle
from pathlib import Path

import numpy as np


class IrisModel:
    name = "iris-logreg"
    dataset = "iris"
    algorithm = "LogisticRegression"

    def __init__(self, model_path: Path, meta_path: Path):
        self._model_path = model_path
        self._meta_path = meta_path
        self._model = None
        self._target_names: list[str] = []

    def load(self) -> None:
        with open(self._model_path, "rb") as f:
            self._model = pickle.load(f)
        self._target_names = json.loads(self._meta_path.read_text(encoding="utf-8"))["target_names"]

    def unload(self) -> None:
        self._model = None
        self._target_names = []

    def predict(self, features: list[float]) -> dict:
        x = np.asarray(features, dtype=float).reshape(1, -1)
        index = int(self._model.predict(x)[0])
        scores = self._model.predict_proba(x)[0]
        return {
            "class_index": index,
            "class_name": self._target_names[index],
            "scores": {n: float(p) for n, p in zip(self._target_names, scores)},
            "raw_scores": self._model.decision_function(x)[0].tolist(),  # internal only
        }

    @property
    def target_names(self) -> list[str]:
        return self._target_names
