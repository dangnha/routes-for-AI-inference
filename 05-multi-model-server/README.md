# Part 6 — Multi-model AI Server

## 1. Introduction

A real AI server serves **many tasks**: OCR, classification, detection. Some models are heavy,
inputs can be images, and each model needs different compute. This part shows how to run several
models behind one FastAPI app **without turning the API into a mess**.

- **One router per task**: `/ocr`, `/classification`, `/detection`.
- **One wrapper per model** in `models/`, all sharing the same `load / predict / unload` contract.
- **A registry** that manages the model lifecycle (lazy load per device, release on demand).
- **A service per task** that orchestrates preprocess → model → response.
- **A Streamlit UI** (the final project UI) that talks to the local server.

**Everything runs locally.** Models are loaded in the server process with `torch`,
`onnxruntime` or `scikit-learn`. No remote inference API is used.

### What is OCR (as a black box)

```
invoice image ──► [ image encoder ] ──► [ text decoder ] ──► text / structured JSON
                      (CNN / ViT)          (Transformer)
```

You never touch the encoder/decoder. You only: (1) give the model a correctly formatted image,
(2) call it, (3) turn its output into something usable. That is exactly the **service layer** job.

### Available models

The server ships **two OCR models**, both running locally:

| Name | Engine | Kind | Device | Notes |
|------|--------|------|--------|-------|
| `rapidocr` | RapidOCR (ONNX) | document | **cpu** | Full-page detect + read. Fast, best for invoices on CPU. |
| `trocr-base-printed` | TrOCR | line | cpu, cuda | Printed text, ~334M, accurate transformer. |

- **`line` models** need the page split into lines first (horizontal projection in
  `services/preprocess.py`).
- **`document` models** read the whole page at once.
- All weights are downloaded once by `download_models.py` into a **project-local cache**
  (`model_cache/`); the server then loads them from there, so nothing is fetched at request time.
- Detection uses `hustvl/yolos-tiny` (`pytorch`). It is COCO-trained, so invoice photos usually
  return **0 objects** — use a normal photo with people/cars to see boxes.

### Choosing the device

The UI (and the API) lets you pick **CPU or GPU** for OCR:

- `rapidocr` → CPU only (ONNX Runtime).
- `trocr-base-printed` → CPU or cuda (PyTorch).

Only devices that actually exist on the machine are offered (`GET /devices`).

## 2. Setup guide

Follow these steps in order.

### Step 1 — Create and activate a virtual environment

```bash
cd multi-model-server
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **torch / torchvision must match.** If you see
> `ImportError: operator torchvision::nms does not exist`, install a matching pair, e.g.
> `torch 2.5.1` ↔ `torchvision 0.20.1`. See **4. Additional information → Troubleshooting**.

### Step 3 — Download all models (once)

```bash
python download_models.py
```

This fetches the two OCR models and the detector into the **project-local cache** (`model_cache/`).
Re-running it is safe: already-downloaded models are skipped. To prove the server never uses the
network afterwards, start it with `MMS_OFFLINE=1`.

### Step 4 — Train the small classifier (once)

```bash
python train_model.py
```

This creates `model.pkl` and `model_meta.json`.

### Step 5 — Start the API server

```bash
uvicorn main:app --reload --port 8005
```

Check it: open <http://127.0.0.1:8005/docs> (Swagger UI) or <http://127.0.0.1:8005/health>.

### Step 6 — Start the Streamlit UI (in a second terminal)

```bash
streamlit run app.py
```

Streamlit opens <http://localhost:8501>. In the sidebar make sure **API base URL** is
`http://127.0.0.1:8005`.

### Step 7 — Try it

1. Choose a **device** (CPU or GPU) and an **OCR model** in the sidebar.
2. In the **OCR** tab, upload an image or click a **sample invoice**, then press **Run OCR**.
3. Try the **Classification** and **Detection** tabs too.

### Optional — run without downloading model weights

To demo the UI/flow before downloading anything:

```bash
set MMS_MOCK=1
uvicorn main:app --port 8005
```

## 3. Code structure

```
multi-model-server/
├── main.py                      # create app, register routers, error handlers
├── app.py                       # Streamlit UI (client)
├── download_models.py           # pre-downloads all model weights into model_cache/
├── train_model.py               # builds model.pkl for the classifier
├── core/
│   ├── config.py                # model specs, devices, cache dir, env vars
│   ├── errors.py                # domain errors carrying an HTTP status code
│   └── env_check.py             # clear error if the vision backend is broken
├── routers/
│   ├── ocr.py                   # GET /ocr/models, POST /ocr/upload
│   ├── classification.py        # GET /classification/info, POST /classification/predict
│   ├── detection.py             # POST /detection/upload
│   └── system.py                # /health, /devices, /models, /models/{name}/unload
├── schemas/                     # Pydantic request/response contracts
│   ├── common.py                # error contract + system models
│   ├── ocr.py  classification.py  detection.py
├── services/                    # business logic (preprocess -> model -> response)
│   ├── ocr_service.py
│   ├── classification_service.py
│   ├── detection_service.py
│   └── preprocess.py            # decode image, segment lines
├── models/                      # model wrappers + lifecycle registry
│   ├── registry.py              # lazy load per device, unload, status
│   ├── ocr.py                   # RapidOCREngine, TrOCRLineModel
│   ├── classifier.py            # IrisClassifier
│   └── detector.py              # ObjectDetector (YOLOS)
└── requirements.txt
```

Request flow (OCR example):

```
Streamlit UI  --POST /ocr/upload (file, model, device)-->  routers/ocr.py
                                                              |
                                                              v
                                             services/ocr_service.py
                                       (validate device -> preprocess -> call model)
                                                              |
                                                              v
                                    models/registry.py --ensure_loaded(name, device)--> models/ocr.py
```

### API summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness + default device |
| GET | `/devices` | devices available on this machine |
| GET | `/models` | status of every model (and where it is loaded) |
| POST | `/models/{name}/unload` | release a model from memory |
| GET | `/ocr/models` | list OCR models, engines and devices |
| POST | `/ocr/upload` | OCR an uploaded image |
| GET | `/classification/info` | classifier description |
| POST | `/classification/predict` | predict Iris species |
| POST | `/detection/upload` | detect objects in an image |

All errors use one contract:

```json
{ "error": { "code": "INVALID_IMAGE", "message": "Could not decode image" } }
```

| Error code | HTTP | When |
|------------|------|------|
| `UNKNOWN_MODEL` | 404 | model name does not exist |
| `INVALID_DEVICE` | 400 | device not supported / not available |
| `INVALID_IMAGE` | 400 | uploaded file is not a decodable image |
| `IMAGE_TOO_LARGE` | 413 | image over the size limit |
| `MODEL_BACKEND_ERROR` | 503 | backend missing or inference failed |
| `VALIDATION_ERROR` | 422 | request body does not match the schema |

## 4. Additional information

### Sample invoices for testing

1. https://imgv2-2-f.scribdassets.com/img/document/558034587/original/cf20b8ea94/1?v=1
2. https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5YP2LMDSFPIh2c36w9MILmGcfO5vAletMNbkBD0OqBtCciXk7c11yG5Ba&s=10
3. https://d2czqxs5dso3qv.cloudfront.net/images/85f1e89b-dea6-4c9c-aa6e-9051527a67f8.png

They are already wired to buttons in the UI. The Streamlit app downloads the image itself and
uploads the file to the server, so the server never needs to fetch anything from the internet.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MMS_DEVICE` | `auto` | Default device: `cpu` / `cuda` / `mps` |
| `MMS_MOCK` | `0` | `1` = do not download models, return canned results |
| `MMS_OFFLINE` | `0` | `1` = load models from the local cache only (never touch the network) |
| `MMS_MODEL_CACHE` | `model_cache/` | Folder where downloaded weights are stored |
| `MMS_OCR_MODEL` | `rapidocr` | Default OCR model |
| `MMS_MAX_OCR_LINES` | `20` | Max lines for `line` models |
| `MMS_MAX_UPLOAD_MB` | `10` | Max upload size |

### Model lifecycle & resources

- Models load **lazily**, on first use, per device (`name::device`).
- `POST /models/{name}/unload` frees RAM/VRAM (`gc.collect()` + `torch.cuda.empty_cache()`).
- All models are released on shutdown.

### Troubleshooting

1. **`ImportError: operator torchvision::nms does not exist`**
   torch and torchvision versions do not match → `pip install --upgrade torchvision`.
2. **`cannot import name 'ReasoningEffort' from 'mistral_common...'`** (transformers 5.x)
   → `pip install --upgrade mistral-common`.
3. **`torch.load ... require users to upgrade torch to at least v2.6`**
   a model only ships `.bin` weights and `torch < 2.6`. The OCR models used here ship
   `safetensors`, so they are fine.
4. **UI says "server offline"** → start `uvicorn main:app --port 8005` first and check the
   API base URL in the sidebar.

In all cases the server never crashes: the endpoint returns **503** with a clear message, and you
can always demo the UI with `MMS_MOCK=1`.
