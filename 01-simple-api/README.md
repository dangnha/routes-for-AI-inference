# Part 1 — Simple API

## 1. Introduction

> We can write a Python function that computes something. How can **other people** call our
> function and use it?

Our function is `get_items()`, which returns a list. It runs on our machine, but nobody else can
"reach into" our machine to call it. We need a shared doorway: an **API**.

Key ideas:

- **Client–server**: a client (browser, script, Streamlit app) sends a request; a server
  (FastAPI) runs our Python function and sends back a response.
- **API**: the contract that says what the client sends and what it gets back.
- **FastAPI** turns a Python function into an endpoint: `@app.get("/items")`.
- FastAPI generates **Swagger UI** documentation automatically at `/docs`.

> Keep it simple: this first part has **no request body, no schema and no error handling**.
> The endpoint is a basic **GET** with no parameters: the function returns a list, and the
> response is a plain JSON array. Every later part adds exactly one new idea.

## 2. Setup guide

```bash
cd simple-api
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload --port 8000
```

Start the UI in a second terminal:

```bash
streamlit run app.py
```

Then:

1. Open <http://127.0.0.1:8000/docs> → `GET /items` → **Try it out** → **Execute**.
   This is a request from the browser and the HTTP response (status code + JSON body).
2. Open the Streamlit UI (<http://localhost:8501>) and send a request from there.

Example:

```jsonc
// Request  GET /items
// Response (HTTP 200)
["apple", "banana", "cherry", "durian", "elderberry"]
```

An **HTTP response** has three parts: status code (`200 OK`), headers, and body. This part only
produces `200` — the other status codes (400/422/500) arrive in later parts.

## 3. Code structure

```
simple-api/
├── main.py            # get_items() + FastAPI app + GET /items
├── app.py             # Streamlit UI (client)
└── requirements.txt
```

Request flow: `app.py` → `GET /items` → `main.py` → `get_items()` → JSON response.

## 4. Additional information

### Local vs deploy

| | Local | Deploy |
|---|-------|--------|
| Address | `http://127.0.0.1:8000` (only your machine) | public domain/IP |
| Run with | `uvicorn main:app --reload` | container + reverse proxy / process manager |
| Who can call | only you | any client on the internet |
| Care about | little | HTTPS, auth, rate limits, resource limits |

The Python code is basically the same; only the infrastructure differs.

### Next part

`/items` is still a plain function returning a list. Part 2 replaces `get_items()` with a
**trained ML model**, which adds real work: loading the model and running inference.
