# Part 3 — Validate Input with Pydantic

## 1. Introduction

> Same "serve a pretrained model" problem, new question: **what happens if the client sends a
> missing feature, the wrong type, or a string instead of a number?**

Right now the API accepts invalid data, and the input-checking logic is mixed with the inference
logic. We need one place to **declare** what valid input looks like, and let the framework check it.

This part also swaps `@app.post(...)` for an **`APIRouter`**: related routes are collected on a
router and the app includes it once. Tiny change here, but it is what makes Part 3.5 possible
(the router moves to its own file).

```
client ──POST /predict──►  JSON body { "sepal_length": 5.1, ... }
                                    │
                                    ▼
                     ┌───────────────────────────────┐
                     │ FastAPI checks it against      │
                     │ IrisFeatures (Pydantic schema) │
                     └───────────────────────────────┘
                          │ valid             │ invalid
                          ▼                   ▼
                  call predict()        return 422 (endpoint NOT called)
```

## 2. Setup guide

```bash
cd 03-validate-input
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --port 8002
```

- Swagger UI: <http://127.0.0.1:8002/docs> — the schema shows all four fields with descriptions.
  Use **Try it out**, delete a field, and **Execute** to see the **422** error.

## 3. Code structure

```
03-validate-input/
├── main.py            # the WHOLE API in one file: schema + model + APIRouter + endpoint
├── train_model.py     # one-off training script (not part of the API)
└── requirements.txt
```

The endpoints are grouped on a router and mounted once:

```python
router = APIRouter(tags=["prediction"])

@router.post("/predict")          # registered on the router...
def predict(payload: IrisFeatures): ...

app.include_router(router)        # ...and included into the app
```

The request schema:

```python
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=20)
    sepal_width:  float = Field(..., gt=0, le=20)
    petal_length: float = Field(..., gt=0, le=20)
    petal_width:  float = Field(..., gt=0, le=20)
```

FastAPI uses this schema to validate **before** the endpoint and to build the Swagger form.

## 4. Additional information

### Error cases → HTTP 422

| Client sends | FastAPI detects | Result |
|--------------|-----------------|--------|
| missing `petal_width` | required field (`...`) missing | **422** |
| `"sepal_length": "abc"` | cannot parse as `float` | **422** |
| `"sepal_length": -5` | violates `gt=0` | **422** |
| four positive numbers | – | **200** + prediction |

The 422 body looks like:

```json
{ "detail": [ { "type": "missing", "loc": ["body", "petal_width"], "msg": "Field required" } ] }
```

### Key point

Validation is **declared**, not hand-written inside the endpoint. The endpoint therefore only ever
receives data that is already guaranteed to be valid, so the inference logic stays clean. Compare
with `serve-model/main.py`, which takes a raw JSON array and gives vaguer errors when input is
missing.
