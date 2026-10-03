"""Common inference contracts for the AI service.

Each OCR model is a provider implementing :meth:`OCRProvider.predict`. Providers
are registered in :mod:`ai_service.providers` and exposed as one router per
model. Long-running jobs are executed by :mod:`ai_service.jobs` and reported
back to the caller through a webhook callback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    # Optional per-detection boxes: [{"box": [[x, y], ...], "text": str, "confidence": float}]
    lines: list[dict] = field(default_factory=list)


class OCRProvider(ABC):
    slug: str
    name: str
    description: str

    @abstractmethod
    def predict(self, content: bytes) -> OCRResult:
        """Return recognized text for the given image bytes."""
