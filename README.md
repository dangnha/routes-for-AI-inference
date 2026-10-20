# AIO2026 · Module 05 — Serving AI Models with FastAPI

## 1. Introduction

This module answers one question: **how do other people (a UI, a mobile app, a teammate) call my
AI model?** The answer is to turn the model into a **service** exposed through a **REST API** built
with **FastAPI**.

Every folder below is a self-contained part of the lesson with its own `README.md`,
`requirements.txt`, a FastAPI **backend** (`main.py`) and a **Streamlit** client (`app.py`).

| # | Folder | Topic | API port | Streamlit |
|---|--------|-------|----------|-----------|
| 1 | [`simple-api/`](./simple-api) | Turn a plain **Python function** into an API (client–server, what an API is) — no body, no schema, no error handling | 8000 | `app.py` |
| 2 | [`serve-model/`](./serve-model) | Serve a **pretrained ML model** (Logistic Regression / Iris) | 8001 | `app.py` |
| 3 | [`validate-input/`](./validate-input) | **Validate input** with Pydantic schemas + introduce `APIRouter` (HTTP 422) | 8002 | `app.py` |
| 4 | [`architecture/`](./architecture) | **Pure refactor of Part 3**: split the code by layer: routers / schemas / services / models | 8003 | `app.py` |
| 5 | [`response-contract/`](./response-contract) | **Response contract**: `response_model` + status codes 200/400/404/422/500 | 8004 | `app.py` |
| 6 | [`multi-model-server/`](./multi-model-server) | **Multi-model AI server**: OCR (UI) + classification + detection (API) (**final project**) | 8005 | `app.py` (OCR) |

**Story of the lesson** — each part adds exactly one idea:

1. **Simple problem** — we have a Python function. How can others call it? → API, client–server, FastAPI (keep it minimal).
2. **Serve a model** — a model is not a calculation: it must be loaded, fed the right features, and run through inference.
3. **Validate input** — what if the client sends missing/wrong data? → Pydantic schema + automatic 422, and group the endpoints on an `APIRouter`.
4. **Architecture** — the *same* code as Part 3, just split into layers (routers / schemas / services / models).
5. **Response contract** — the model's output is not the API's output; declare the response and use meaningful status codes.
6. **Multi-model server** — many models (lightweight OCR in the UI, plus classification and
   detection on the API), device choice, and lifecycle management.

**All models run locally** (loaded in the FastAPI process with `torch`, `onnxruntime` or
`scikit-learn`). No remote inference API is used.

## 2. Setup guide

### Prerequisites

- Python **3.10+** (3.12 recommended)
- Ability to run two terminals per part (one for the API, one for Streamlit)

### General pattern (apply to each part)

```bash
cd <part-folder>
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then, for parts 2–6 only, build the small classifier model once:

```bash
python train_model.py
```

Start the backend:

```bash
uvicorn main:app --reload --port <port>     # see the table above
```

Start the UI in a second terminal:

```bash
streamlit run app.py
```

Open the API docs at `http://127.0.0.1:<port>/docs` and the UI at <http://localhost:8501>.

### Quick start per part

| Part | Port | Command |
|------|------|---------|
| simple-api | 8000 | `uvicorn main:app --reload --port 8000` |
| serve-model | 8001 | `python train_model.py` then `uvicorn main:app --reload --port 8001` |
| validate-input | 8002 | `python train_model.py` then `uvicorn main:app --reload --port 8002` |
| architecture | 8003 | `python train_model.py` then `uvicorn main:app --reload --port 8003` |
| response-contract | 8004 | `python train_model.py` then `uvicorn main:app --reload --port 8004` |
| multi-model-server | 8005 | `python train_model.py` then `uvicorn main:app --reload --port 8005` |

> To run two Streamlit apps at the same time, give each one a port:
> `streamlit run app.py --server.port 8502`.

### Part 6 (final project) in short

```bash
cd multi-model-server
pip install -r requirements.txt

# Provide a Hugging Face token (public models, but this makes downloads reliable):
#   put `HF_TOKEN=hf_...` in a .env file, or  set HF_TOKEN=hf_...
python download_models.py                    # fetch all weights into model_cache/<model>/
python train_model.py
uvicorn main:app --reload --port 8005      # terminal 1
streamlit run app.py                        # terminal 2
python smoke_test.py                        # terminal 3: tests every backend endpoint
```

In the UI: pick a **device** (CPU or GPU) and an **OCR model**, then upload an image or click a
**sample invoice**. The Streamlit UI is **OCR-only**; the classification and detection endpoints
still exist on the backend — test them from `/docs` or with `smoke_test.py`. Start the backend
with `set MMS_OFFLINE=1` to run fully from the local cache (no network). To demo without any model
weights at all: `set MMS_MOCK=1`.

## 3. Code structure

```
Module 05/code/
├── README.md
├── .gitignore
├── simple-api/
│   ├── main.py              # get_items() + GET /items (basic GET, returns a list)
│   ├── app.py               # Streamlit client
│   └── requirements.txt
├── serve-model/
│   ├── train_model.py       # TRAINING: fit + save model.pkl
│   ├── main.py              # INFERENCE: load model once, POST /predict
│   ├── app.py  requirements.txt
├── validate-input/
│   ├── main.py              # the whole API in ONE file (Pydantic schema + APIRouter + endpoint)
│   ├── train_model.py  app.py  requirements.txt
├── architecture/
│   ├── main.py              # create app + register routers
│   ├── routers/  schemas/  services/  models/
│   ├── train_model.py  app.py  requirements.txt
├── response-contract/
│   ├── main.py              # app + error handlers
│   ├── routers/  schemas/  services/  models/
│   ├── train_model.py  app.py  requirements.txt
└── multi-model-server/      # FINAL PROJECT
    ├── main.py              # app + routers + error handlers
    ├── app.py               # Streamlit UI (OCR only)
    ├── core/                # config, errors, backend check
    ├── routers/             # ocr.py, classification.py, detection.py, system.py
    ├── schemas/             # Pydantic contracts
    ├── services/            # ocr_service, classification_service, detection_service, preprocess
    ├── models/              # registry + wrappers (RapidOCR, TrOCR-small, YOLOS, sklearn)
    ├── download_models.py   # pre-downloads all weights into model_cache/<model>/
    ├── smoke_test.py        # PASS/FAIL check for every backend endpoint
    ├── train_model.py  requirements.txt
```

Generated model caches live in `multi-model-server/model_cache/` (one folder per model, git-ignored).

The dependency direction always goes **outer → inner** (`routers → services → models`), which keeps
each part easy to change and test.

## 4. Additional information

### Default ports

API: 8000–8005 (one per part). Streamlit: 8501 (use `--server.port` to change).

### Part 6 models

| Name | Engine | Device | Size | Notes |
|------|--------|--------|------|-------|
| `rapidocr` | RapidOCR (ONNX) | cpu | ~16 MB (bundled) | Full-page OCR, fast, best for invoices on CPU |
| `trocr-small-printed` | TrOCR small | cpu / cuda | ~246 MB (~62M) | Printed text per line, light transformer |

Detection uses `hustvl/yolos-tiny` (~26 MB). All weights are downloaded once by
`download_models.py` into the project-local `model_cache/` folder (**one folder per model**;
only `safetensors`, no duplicate `.bin`); the server loads them from there and can run with
`MMS_OFFLINE=1`. A Hugging Face token (`HF_TOKEN`, see `multi-model-server/README.md`) makes the
download reliable.

Sample invoices (already wired to buttons in the UI):

1. https://imgv2-2-f.scribdassets.com/img/document/558034587/original/cf20b8ea94/1?v=1
2. https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5YP2LMDSFPIh2c36w9MILmGcfO5vAletMNbkBD0OqBtCciXk7c11yG5Ba&s=10
3. https://d2czqxs5dso3qv.cloudfront.net/images/85f1e89b-dea6-4c9c-aa6e-9051527a67f8.png

### Troubleshooting (Part 6, image models)

1. **`ImportError: operator torchvision::nms does not exist`**
   `torch` and `torchvision` versions do not match. Install a matching pair, e.g.
   `torch 2.5.1` ↔ `torchvision 0.20.1`: `pip install --upgrade torchvision`.
2. **`cannot import name 'ReasoningEffort' from 'mistral_common...'`** (transformers 5.x)
   → `pip install --upgrade mistral-common`.
3. **`torch.load ... require users to upgrade torch to at least v2.6`**
   a model ships `.bin` weights and `torch < 2.6`. The OCR models here ship `safetensors`, so they
   work.
4. **Download fails / hangs** → set a Hugging Face token (`HF_TOKEN`, e.g. in a `.env` file).
   The downloader already disables Xet (`HF_HUB_DISABLE_XET=1`), which is more reliable on
   OneDrive / network drives.
5. **UI shows "server offline"** → start the backend first and check the *API base URL* in the sidebar.

The server never crashes: image endpoints return **503** with a clear message when a backend is
missing, and you can always demo with `MMS_MOCK=1`.

### Notes

- Code comments may be in Vietnamese; the **UI, API docs and READMEs are in English**.
- Generated artifacts (`model.pkl`, `model_meta.json`) are created by `train_model.py` and are
  git-ignored.
