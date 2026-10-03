import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routers import auth, models, tasks, uploads, webhooks
from app.core.config import settings
from app.core.database import Base, SessionLocal, engine
from app.core.errors import register_exception_handlers
from app.services.storage import ensure_bucket

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)  # Supabase Postgres; use Alembic in production
    try:
        ensure_bucket()
    except Exception as exc:
        logger.warning("Could not ensure storage bucket: %s", exc)
    yield
    engine.dispose()


app = FastAPI(title=settings.app_name, version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{time.perf_counter() - started_at:.6f}"
    return response


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
async def ready() -> dict[str, object]:
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))

    ai_service = "ok"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.ai_service_url}/health")
            if response.status_code != 200:
                ai_service = "unavailable"
    except httpx.HTTPError:
        ai_service = "unavailable"

    return {"status": "ready", "ai_service": ai_service}


app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(uploads.router, prefix=settings.api_v1_prefix)
app.include_router(models.router, prefix=settings.api_v1_prefix)
app.include_router(tasks.router, prefix=settings.api_v1_prefix)
app.include_router(webhooks.router, prefix=settings.api_v1_prefix)
