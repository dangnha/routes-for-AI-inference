from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def get_items() -> list[str]:
    """Return an arbitrary list of items."""
    return ["apple", "banana", "cherry", "durian", "elderberry"]


app = FastAPI(title="Simple API - List", version="1.0.0")

# One list shared by every route, so items added with POST show up in GET.
ITEMS = get_items()


@app.get("/items", summary="Return a list of items")
def items_endpoint() -> list[str]:
    return ITEMS


# Path parameter: /items/2
@app.get("/items/{item_id}", summary="Return one item by its index")
def read_item(item_id: int):
    if not 0 <= item_id < len(ITEMS):
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "item": ITEMS[item_id]}


# Query parameters: /search?q=an&limit=1
@app.get("/search", summary="Search items by text")
def search(q: str, limit: int = 3):
    hits = [i for i in ITEMS if q in i]
    return hits[:limit]


# Request body: {"name": "fig"}
class NewItem(BaseModel):
    name: str


@app.post("/items", status_code=201, summary="Add an item")
def add_item(item: NewItem):
    ITEMS.append(item.name)
    return {"added": item.name, "count": len(ITEMS)}


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
