"""Model registry: each OCR provider maps to its own router at /models/{slug}."""
from ai_service.base import OCRProvider
from ai_service.providers.easyocr import EasyOCRProvider
from ai_service.providers.gotocr import GotOCRProvider
from ai_service.providers.trocr import TrOCRProvider

MODELS: dict[str, OCRProvider] = {
    provider.slug: provider
    for provider in (TrOCRProvider(), EasyOCRProvider(), GotOCRProvider())
}


def get_provider(slug: str) -> OCRProvider:
    try:
        return MODELS[slug]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{slug}'") from exc


def list_models() -> list[dict[str, str]]:
    return [
        {"slug": p.slug, "name": p.name, "description": p.description}
        for p in MODELS.values()
    ]
