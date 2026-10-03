"""EasyOCR — text detection + recognition pipeline for multi-line text.

Unlike the single-line encoder-decoder models, EasyOCR first detects text
regions (CRAFT) then recognizes each line, so it handles receipts, signs and
scenes with several lines. It reuses the same PyTorch install as the rest of the
service (CUDA when ``DEVICE=cuda``).
"""
from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image

from ai_service.base import OCRProvider, OCRResult
from ai_service.config import settings


class EasyOCRProvider(OCRProvider):
    slug = "easyocr"
    name = "EasyOCR · Multi-line"
    description = "Text detection + recognition; good for receipts, signs and scenes."

    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        with self._lock:
            if self._reader is None:
                import easyocr

                self._reader = easyocr.Reader(
                    settings.easyocr_langs,
                    gpu=settings.inference_device != "cpu",
                    model_storage_directory=settings.easyocr_model_dir,
                    download_enabled=True,
                    verbose=False,
                )

    def predict(self, content: bytes) -> OCRResult:
        if not content:
            raise ValueError("Empty input")

        self._ensure_loaded()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        detections = self._reader.readtext(np.asarray(image))

        lines: list[dict] = []
        confidences: list[float] = []
        for box, text, confidence in detections:
            if not text:
                continue
            lines.append(
                {
                    "box": [[int(x), int(y)] for x, y in box],
                    "text": text,
                    "confidence": round(float(confidence), 4),
                }
            )
            confidences.append(float(confidence))

        full_text = "\n".join(line["text"] for line in lines)
        mean_confidence = (
            round(sum(confidences) / len(confidences), 6) if confidences else 0.0
        )
        return OCRResult(text=full_text, confidence=mean_confidence, lines=lines)
