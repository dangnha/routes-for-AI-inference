from fastapi import Depends, FastAPI

from ai_service.providers import MODELS, list_models
from ai_service.routers import build_model_router, jobs_router, verify_api_key

app = FastAPI(title="Local AI Inference Service", version="3.0.0")


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models", tags=["models"], dependencies=[Depends(verify_api_key)])
async def models() -> list[dict[str, str]]:
    return list_models()


app.include_router(jobs_router)

for _provider in MODELS.values():
    app.include_router(build_model_router(_provider))
