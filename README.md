# AIO2026 · Module 05 — Building Routes for AI Inference using FastAPI

## 1. Introduction

This module answers one question: **how do other people (a UI, a mobile app, a teammate) call my
AI model?** The answer is to turn the model into a **service** exposed through a **REST API** built
with **FastAPI**.

The lesson has five parts. Each folder is self-contained, with its own `README.md`,
`requirements.txt` and a FastAPI app in `main.py`.

| Part | Folder | Topic | Port |
|------|--------|-------|------|
| 1 | [`01-simple-api/`](./01-simple-api) | Turn a plain **Python function** into an API: GET, path and query parameters, JSON body | 8000 |
| 2 | [`02-serve-model/`](./02-serve-model) | Serve a **pretrained ML model** (Logistic Regression / Iris), load it once with `lifespan` | 8001 |
| 3 | [`03-validate-input/`](./03-validate-input) | **Validate input** with a Pydantic schema (HTTP 422) and group routes with `APIRouter` | 8002 |
| 3.5 | [`03-architecture/`](./03-architecture) | **Organize the code** into routers / schemas / services / models, same API as Part 3 | 8003 |
| 4 | [`04-response-contract/`](./04-response-contract) | **Response contract**: `response_model`, one error shape, status codes 200/400/404/422/500 | 8004 |

[`05-multi-model-server/`](./05-multi-model-server) is an OCR project (model registry, image upload,
several models). It is **not part of this lesson** and is kept for later reuse; see its own README.

## 2. Setup guide

Python **3.10+**.

```bash
cd <part-folder>
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py                       # every part except 01, creates model.pkl once
uvicorn main:app --reload --port <port>     # see the table above
```

Then open `http://127.0.0.1:<port>/docs` and try the routes from Swagger UI, `curl` or `requests`.

## 3. Code structure

```
routes-for-AI-inference/
├── 01-simple-api/
│   └── main.py              # get_items(), ITEMS, GET/POST /items, /items/{item_id}, /search, /health
├── 02-serve-model/
│   ├── train_model.py       # fit + save model.pkl (runs once)
│   └── main.py              # lifespan loads the model once, POST /predict, GET /info
├── 03-validate-input/
│   ├── train_model.py
│   └── main.py              # the whole API in one file: Pydantic schema + APIRouter + endpoint
├── 03-architecture/
│   ├── train_model.py
│   ├── main.py              # create app + register routers
│   └── routers/  schemas/  services/  models/
├── 04-response-contract/
│   ├── train_model.py
│   ├── main.py              # app + error handlers
│   └── routers/  schemas/  services/  models/
└── 05-multi-model-server/   # not part of this lesson (OCR project, kept for later)
```

Imports always point one way (`routers → services → models`), which keeps each part easy to change
and test.

## 4. Notes

- Generated artifacts (`model.pkl`, `model_meta.json`) are created by `train_model.py` and are
  git-ignored.
