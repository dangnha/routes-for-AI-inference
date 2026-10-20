# Part 2 — Serve a Pretrained Model

## 1. Introduction

> A model is not a simple calculation. It must be **loaded**, given input in the right
> **feature format**, and run through **inference**.

In Part 1 the endpoint just called `get_items()`. Here we replace it with a **pretrained ML
model** (Logistic Regression on the Iris dataset) and handle three extra things: load the model,
format the input, and call the model.

**Training vs inference**

| | Training | Inference |
|---|----------|-----------|
| What | read data, `model.fit()` | `model.predict()` on new input |
| When | **offline**, once | **online**, every request |
| Where | `train_model.py` | `main.py` |
| Cost | heavy, slow | must be light and fast |

So the server **never retrains per request**; it **loads `model.pkl` once** at startup.

## 2. Setup guide

```bash
cd 02-serve-model
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py            # creates model.pkl (run once)
uvicorn main:app --reload --port 8001
```

- Swagger UI: <http://127.0.0.1:8001/docs> → `POST /predict` → **Try it out**.

Example request/response:

```jsonc
// Request  [5.1, 3.5, 1.4, 0.2]      (a plain JSON array - no schema yet)
// Response
{ "prediction": 0, "label": "setosa", "probabilities": { "setosa": 0.98, "versicolor": 0.02, "virginica": 0.0 } }
```

## 3. Code structure

```
02-serve-model/
├── train_model.py     # TRAINING: fit + save model.pkl (run once)
├── main.py            # INFERENCE: FastAPI app, load model once, POST /predict
├── requirements.txt
└── model.pkl          # produced by train_model.py
```

Pipeline:

```
Iris dataset ──(train_model.py: fit)──► model.pkl ──(load once at startup)──► server
client [features] ──► np.array (1, 4) ──► predict() / predict_proba() ──► JSON
```

### Key points in `main.py`

1. **Load the model once** in `lifespan` (startup) and keep it in `ml_state`.
2. **Read the input**: `features: list[float]` tells FastAPI the body is a JSON array. Still no
   schema and no validation - Part 3 introduces a Pydantic `BaseModel` for that.
3. **Build the input vector correctly**: scikit-learn wants a 2D array `(n_samples, n_features)`:
   `x = np.asarray(features).reshape(1, -1)`.
4. **Call the model**: `predict()` for the label, `predict_proba()` for the probabilities.
5. **Return JSON** with the label and probabilities.

## 4. Additional information

### The role of a "service"

Notice `run_inference()`: it keeps prediction logic out of the endpoint. In Part 2 this is just a
concept; **Part 3.5** (`03-architecture`) formalizes it into `services/` and `models/` folders.

### Bring your own model

Replace `train_model.py` with any training script, as long as it saves the model the same way,
or adjust `main.py` to load it (e.g. `joblib`, `torch.load`, `transformers`).
