"""TRAINING — chạy 1 lần để tạo model.pkl cho Phần 5."""

import json
import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {"feature_names": list(iris.feature_names), "target_names": list(iris.target_names)},
        f,
        ensure_ascii=False,
        indent=2,
    )

print("Model saved to model.pkl")
