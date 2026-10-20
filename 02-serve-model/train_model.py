"""Train the Iris model once and save it to model.pkl. The server only loads this file."""

import json
import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Feature and class names, so the API can return readable labels.
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {"feature_names": list(iris.feature_names), "target_names": list(iris.target_names)},
        f,
        indent=2,
    )

print("Model saved to model.pkl")
