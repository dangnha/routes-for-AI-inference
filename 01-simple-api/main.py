from fastapi import FastAPI


# ---------------------------------------------------------------------------
# 1) The plain Python function - this is the logic we want to share.
#    It is just Python, it knows nothing about HTTP.
# ---------------------------------------------------------------------------
def get_items() -> list[str]:
    """Return an arbitrary list of items."""
    return ["apple", "banana", "cherry", "durian", "elderberry"]


# ---------------------------------------------------------------------------
# 2) Expose the function behind an API. `app` is the server object.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Simple API - List",
    description="Part 1: turn a plain Python function into an API with FastAPI.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 3) Endpoint (route): maps a URL + HTTP method to a Python function.
#    A basic GET: no parameters, no request body.
#    GET /items runs get_items() and returns the list as a JSON array.
# ---------------------------------------------------------------------------
@app.get(
    "/items",
    summary="Return a list of items",
    description="A basic GET endpoint: no parameters, no body. Returns a plain JSON array.",
)
def items_endpoint() -> list[str]:
    return get_items()


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
