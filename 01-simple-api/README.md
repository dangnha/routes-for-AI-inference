# Part 1 — Simple API

## 1. Introduction

> We can write a Python function that computes something. How can **other people** call our
> function and use it?

Our function is `get_items()`, which returns a list. It runs on our machine, but nobody else can
"reach into" our machine to call it. We need a shared doorway: an **API**.

Key ideas:

- **Client–server**: a client (browser, curl, a Python script) sends a request; a server
  (FastAPI) runs our Python function and sends back a response.
- **API**: the contract that says what the client sends and what it gets back.
- **FastAPI** turns a Python function into an endpoint: `@app.get("/items")`.
- FastAPI generates **Swagger UI** documentation automatically at `/docs`.

Routes in `main.py`:

| Route | Shows | Status |
|-------|-------|--------|
| `GET /items` | a basic GET, returns the list | 200 |
| `GET /items/{item_id}` | path parameter | 200, 404 if the index is out of range, 422 for `/items/abc` |
| `GET /search?q=an&limit=1` | query parameters | 200 |
| `POST /items` with `{"name": "fig"}` | JSON body | 201 |
| `GET /health` | liveness check | 200 |

All routes share one `ITEMS` list, so an item added with `POST /items` shows up in `GET /items`.

## 2. Setup guide

```bash
cd 01-simple-api
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

Then:

1. Open <http://127.0.0.1:8000/docs> → `GET /items` → **Try it out** → **Execute**.
   This is a request from the browser and the HTTP response (status code + JSON body).

Example:

```jsonc
// Request  GET /items
// Response (HTTP 200)
["apple", "banana", "cherry", "durian", "elderberry"]
```

An **HTTP response** has three parts: status code (`200 OK`), headers, and body.

## 3. Code structure

```
01-simple-api/
├── main.py            # get_items(), ITEMS and the routes above
└── requirements.txt
```

Request flow: client → `GET /items` → `main.py` → `get_items()` → JSON response.

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
