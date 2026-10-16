# Part 4 — Source Code Architecture

## 1. Introduction

> As we add models, endpoints and validation, `main.py` gets longer and harder to read,
> maintain and change. So we split the code by responsibility.

This part is a **pure refactor of Part 3**: the behaviour is identical, only the file layout
changes. Compare `architecture/main.py` with `validate-input/main.py` - same schema, same endpoint,
same response. The router that Part 3 introduced now lives in its own module under `routers/`.

| Layer | Responsibility | Must NOT |
|-------|----------------|----------|
| `main.py` | assemble the app, `include_router` | hold business logic or call the model |
| `routers/` | read the request, call a service, return a response | run predictions or load models |
| `schemas/` | describe incoming/outgoing data | know about the model |
| `services/` | business logic: schema → vector → model | know about HTTP |
| `models/` | load the model, `predict()` | know about HTTP or schemas |

Dependencies go **one direction**: outer → inner. Changing an inner layer (swap the model) does not
break outer layers (the endpoint stays the same).

## 2. Setup guide

```bash
cd architecture
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --port 8003
```

In a second terminal:

```bash
streamlit run app.py
```

- Swagger UI: <http://127.0.0.1:8003/docs>
- Streamlit UI: <http://localhost:8501>

## 3. Code structure

```
architecture/
├── main.py                       # create app + register routers + model lifecycle
├── routers/prediction.py         # declare endpoints, handle HTTP
├── schemas/prediction.py         # data structures (Pydantic)
├── services/prediction_service.py# business logic, orchestrate inference
├── models/iris_model.py          # load model + run predictions
├── train_model.py
├── app.py                        # Streamlit UI (client)
└── requirements.txt
```

Request flow:

```
POST /predict
   v
routers/prediction.py        -> predict(payload, service=Depends(get_service))
   v
services/prediction_service.py -> payload.to_vector() then call the model
   v
models/iris_model.py         -> sklearn predict() / predict_proba()
   v
JSON response
```

Imports: `main` imports `routers`, `models`, `services`; `routers` imports `schemas`, `services`;
`services` imports `models`, `schemas`; `models` imports nothing from the project.

## 4. Additional information

### `Depends` + `app.state`

The model is loaded **once** in `lifespan` and stored on `app.state`. Routers get it back through
the `get_service()` dependency. So:

- the model is not reloaded per request,
- a router does not need to know how the model was created — only that a service exists.

### Benefits

- Add a model → add a file in `models/` + a new router; **don't touch** existing code.
- Change validation → only `schemas/`.
- Change business logic → only `services/`.
- Each layer is easier to test on its own.
