"""
PHẦN 2 — TRAINING (huấn luyện)
==============================
Script này chỉ chạy MỘT LẦN để tạo ra model đã train: `model.pkl`.

Phân biệt TRAINING và INFERENCE
-------------------------------
- TRAINING (file này) : đọc dữ liệu, fit model, lưu ra file. Chạy offline, nặng.
- INFERENCE (main.py) : load model đã train, nhận input, trả dự đoán. Chạy trong server,
  nhẹ và phải nhanh.

Server KHÔNG train lại model mỗi request — nó chỉ load file `model.pkl` một lần.

Chạy:
    python train_model.py
"""

import json

import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Lưu thêm metadata: tên feature và tên nhãn để lúc inference hiển thị cho đúng.
# (Chỉ là tiện ích — model đã đủ để dự đoán.)
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "feature_names": list(iris.feature_names),
            "target_names": list(iris.target_names),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("Model saved to model.pkl")
