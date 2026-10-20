# Part 5 — Design a Stable Response Contract

## 1. Introduction

> The model's internal output is not necessarily what the client needs. Returning the raw model
> object makes the API structure hard to control. We need a **standard for the response**.

Example: the model also returns `raw_scores` (raw decision values). The client does not need them,
and if the model changes, the shape of the response would change too. We want to **lock** that shape.

Three things to distinguish:

| | What it is | Where |
|---|------------|-------|
| **Inference result** | what the model returns (internal, may have extra fields) | `models/iris_model.py` |
| **API response** | what we *promise* the client (exact schema) | `schemas/prediction.py` + `response_model` |
| **HTTP status code** | the state of the request | `routers/` + exception handlers |

## 2. Setup guide

```bash
cd 04-response-contract
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload --port 8004
```

- Swagger UI: <http://127.0.0.1:8004/docs> — open `POST /predict` and check the **Responses** section.

## 3. Code structure

```
04-response-contract/
├── main.py                      # app + ERROR HANDLERS (map errors -> status + JSON)
├── routers/prediction.py        # APIRouter: /health, /predict, /models/{id}, /debug/boom
├── schemas/prediction.py        # request schema + response schemas + error schema
├── services/
│   ├── prediction_service.py    # business logic, raises LowConfidenceError
│   └── errors.py                # domain errors
├── models/iris_model.py         # load + predict (returns the internal result)
└── requirements.txt
```

### Request schema vs response schema

```python
# REQUEST - what the client sends
class IrisFeatures(BaseModel): ...

# RESPONSE - what we promise to return
class PredictionResponse(BaseModel):
    model: str
    predicted_label: str
    predicted_class: int
    confidence: float
    probabilities: list[ProbabilityItem]
```

They are separate because input and output serve different purposes.

### What `response_model` controls

```python
@router.post("/predict", response_model=PredictionResponse)
def predict(...):
    return service.predict(...)   # this dict CONTAINS "debug_raw_scores"
```

`response_model=PredictionResponse` will:

- **filter** fields that are not declared (here `debug_raw_scores` is **removed**),
- **type-check** every returned field,
- **document** the response in Swagger.

## 4. Additional information

### HTTP status codes

| Code | Meaning | Scenario in this part |
|------|---------|-----------------------|
| **200** | OK | successful prediction |
| **400** | Bad Request | `min_confidence` not met (**business** error) |
| **404** | Not Found | `/models/{id}` with an unknown id |
| **422** | Unprocessable Entity | body does not match the schema — FastAPI does this |
| **500** | Internal Server Error | unexpected error (`/debug/boom`) |

**400 vs 422**: 422 = data with the *wrong shape* (schema) → framework handles it.
400 = data with the *right shape* but breaking a *business* rule → the service raises it.

### One error contract for everything

```json
{ "error": { "code": "LOW_CONFIDENCE", "message": "..." } }
```

Thanks to the `@app.exception_handler`s, clients can parse the same structure for 400/404/422/500.
