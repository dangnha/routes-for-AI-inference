# Local AI Inference Platform

A **fully local** OCR platform built in three tiers: **Streamlit** (UI) →
**FastAPI + Supabase** (backend) → **AI service** (OCR models). Model weights are
downloaded up front into the `models/` directory, so inference needs no cloud, and
the AI service reports results back to the backend through a **webhook**.

---

## Overview

- **Frontend** (`frontend/`) — Streamlit: sign in, pick a bundled example or upload
  your own image, and view results (with bounding boxes for models that detect text
  regions).
- **Backend** (`backend/`) — FastAPI: authenticates via **Supabase Auth**, manages
  tasks, issues **presigned URLs** so clients upload straight to **Supabase
  Storage**, orchestrates inference, and receives results via webhook.
- **AI service** (`ai_service/`) — FastAPI: three OCR models, one router each;
  accepts async jobs and calls the backend back with the result.
- **Supabase is the single data layer**: **Postgres** (tasks), **Auth** (users),
  **Storage** (images). There is no local database.

### OCR models

Three models with **different architectures**, all running locally:

| Slug | Model (HF) | ~Params | Approach |
| --- | --- | --- | --- |
| `trocr` | `microsoft/trocr-base-printed` | 334M | Transformer encoder–decoder, **single line of printed text** |
| `easyocr` | EasyOCR (CRAFT + CRNN) | ~100MB | **Detection + recognition** across multiple lines (receipts, signs) |
| `gotocr` | `stepfun-ai/GOT-OCR2_0` | 580M | General OCR, outputs **Markdown** (layout/tables) |

Models are **lazy-loaded** on first use and run on `INFERENCE_DEVICE` (default
`cuda`). **GOT-OCR2.0 requires a GPU**, because the model's own code always moves
tensors to CUDA.

### Project structure

```text
fastapi/
├── frontend/                    # Tier 1 — Streamlit
│   ├── app.py                   # main Streamlit app
│   ├── client.py                # httpx client for the backend
│   ├── samples.py               # maps each model to its example images
│   ├── samples/                 # bundled examples (printed/multiline/document)
│   └── requirements.txt
├── backend/                     # Tier 2 — FastAPI + Supabase
│   ├── app/
│   │   ├── main.py              # app factory, middleware, health, routers
│   │   ├── models.py            # SQLAlchemy ORM (tasks table only)
│   │   ├── schemas.py           # Pydantic
│   │   ├── core/                # config, database, auth (Supabase), errors
│   │   ├── api/
│   │   │   ├── deps.py          # DI + auth/RBAC guard
│   │   │   └── routers/         # auth, uploads, models, tasks, webhooks
│   │   └── services/            # storage, ai_client, inference
│   └── tests/
├── ai_service/                  # Tier 3 — OCR models
│   ├── main.py
│   ├── base.py                  # OCRProvider + OCRResult
│   ├── config.py
│   ├── routers.py               # one router per model: /infer + /jobs, /jobs/{id}
│   ├── jobs.py                  # job registry + background execution
│   ├── webhooks.py              # POSTs results back to the backend
│   └── providers/               # encdec, trocr, easyocr, gotocr
├── scripts/
│   ├── download_models.py       # download models into models/
│   ├── make_samples.py          # generate the example images
│   └── smoke_infer.py           # run each model on a sample (real output)
├── models/                      # local model cache (not committed)
├── requirements.txt             # backend + ai service
├── pytest.ini
├── .env.example
└── README.md
```

---

## Requirements

- Python 3.12+
- **A Supabase project (required)** — Postgres + Auth + Storage
- NVIDIA GPU (recommended) or CPU — GOT-OCR2.0 runs on GPU only
- ~3 GB of disk for the models (`models/`)

---

## Setup

Follow the steps in order. The three tiers run in three separate terminals (each
with the virtualenv activated).

### 1 — Clone

```bash
git clone <repo-url> && cd fastapi
```

### 2 — Create a virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3 — Install PyTorch (CUDA)

Install the CUDA build **first** so `easyocr` does not pull in the CPU build:

```bash
pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

> On CPU: skip this step and set `INFERENCE_DEVICE=cpu` in step 5.

### 4 — Install dependencies

```bash
pip install -r requirements.txt            # backend + ai service
pip install -r frontend/requirements.txt   # Streamlit UI
```

### 5 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the **required** Supabase values:

- `DATABASE_URL` — Supabase → **Connect** → **Session pooler**; keep the
  `postgresql+psycopg://` prefix and insert the DB password.
- `SUPABASE_URL`, `SUPABASE_ANON_KEY` — used for Auth (project → **Settings → API**).
- `SUPABASE_SERVICE_ROLE_KEY` — used for Storage.

Everything else has sensible defaults for local development.

> **Supabase Auth note:** go to *Authentication → Providers → Email* and turn off
> **Confirm email** so users can sign in right after registering. To make an admin
> account, set `app_metadata` = `{"role": "admin"}` for that user.

### 6 — Download the models (one time)

```bash
python scripts/download_models.py
```

Downloads TrOCR and GOT-OCR2.0 into `models/hf/` and EasyOCR into
`models/easyocr/`. After this, inference runs fully offline.

### 7 — (Optional) Regenerate the example images

The examples already ship in `frontend/samples/`. To regenerate them:

```bash
python scripts/make_samples.py
```

### 8 — Run the AI service (port 8001)

```bash
uvicorn ai_service.main:app --host 0.0.0.0 --port 8001
```

Docs: <http://localhost:8001/docs>

### 9 — Run the backend (port 8000)

```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Docs: <http://localhost:8000/docs>

### 10 — Run the Streamlit frontend (port 8501)

```bash
streamlit run frontend/app.py --server.port 8501
```

Open the app: <http://localhost:8501>

### 11 — Verify

```bash
curl http://localhost:8000/health   # {"status":"ok"}
curl http://localhost:8000/ready    # {"status":"ready","ai_service":"ok"}
```

The backend creates the `inference_tasks` table and the `ocr-uploads` bucket on
startup. Register an account with a real email in the UI, then try an example image.

---

## API

### Backend

Every `/api/v1/*` endpoint (except register/login and the webhook) requires an
`Authorization: Bearer <Supabase access token>` header.

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | — | Liveness probe |
| `GET` | `/ready` | — | Readiness probe (DB + AI service) |
| `POST` | `/api/v1/auth/register` | — | Register via Supabase Auth (email + password) |
| `POST` | `/api/v1/auth/token` | — | Exchange email + password for an access token |
| `POST` | `/api/v1/uploads/presign` | Bearer | Get a presigned URL + object key (Supabase Storage) |
| `GET` | `/api/v1/models` | Bearer | List available models |
| `POST` | `/api/v1/tasks` | Bearer | Create a task from an uploaded object key |
| `GET` | `/api/v1/tasks` | Bearer | List the user's tasks (paginated) |
| `GET` | `/api/v1/tasks/admin/all` | admin | All tasks (RBAC demo) |
| `GET` | `/api/v1/tasks/{task_id}` | Bearer | Task detail (owner or admin) |
| `POST` | `/api/v1/webhooks/inference` | `X-Webhook-Secret` | AI service reports a finished job |

### AI service

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/models` | List models |
| `POST` | `/models/{slug}/infer` | Synchronous inference (JSON) — handy for testing |
| `POST` | `/models/{slug}/jobs` | Accept an async job + `callback_url` |
| `GET` | `/jobs/{job_id}` | Job status |

---

## Environment variables

Loaded from `.env` via `pydantic-settings`.

### Backend

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | *(required)* | Supabase Postgres URL (`postgresql+psycopg://…`) |
| `SUPABASE_URL` | *(required)* | Supabase project URL |
| `SUPABASE_ANON_KEY` | *(required)* | Anon key (Supabase Auth) |
| `SUPABASE_SERVICE_ROLE_KEY` | *(required)* | Service role key (Supabase Storage) |
| `SUPABASE_STORAGE_BUCKET` | `ocr-uploads` | Bucket name |
| `WEBHOOK_SECRET` | `change-me` | Secret for the webhook (`X-Webhook-Secret`) |
| `AI_SERVICE_URL` | `http://localhost:8001` | AI service URL |
| `AI_CALLBACK_BASE_URL` | `http://localhost:8000` | URL the AI service calls back to |

### Frontend

| Variable | Default | Description |
| --- | --- | --- |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL (read by `frontend/app.py`) |

### AI service

| Variable | Default | Description |
| --- | --- | --- |
| `INFERENCE_DEVICE` | `cuda` | `cpu` or `cuda` |
| `WEBHOOK_SECRET` | `change-me` | Secret sent with the callback (must match the backend) |
| `TROCR_MODEL_ID` | `microsoft/trocr-base-printed` | TrOCR checkpoint |
| `GOTOCR_MODEL_ID` | `stepfun-ai/GOT-OCR2_0` | GOT-OCR2.0 checkpoint |
| `EASYOCR_LANGS` | `["en"]` | EasyOCR languages |
| `MAX_LENGTH` | `256` | Max text length (TrOCR) |

---

## Testing

```bash
pytest                 # unit tests (no models / Supabase needed)
pytest -m slow         # integration: needs the AI service running + models downloaded
python scripts/smoke_infer.py    # print each model's real output on a sample image
```
