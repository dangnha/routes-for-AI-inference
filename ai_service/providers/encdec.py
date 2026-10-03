"""Shared logic for Hugging Face vision encoder-decoder OCR models.

TrOCR uses a :class:`~transformers.VisionEncoderDecoderModel` paired with a
:class:`~transformers.TrOCRProcessor`, so loading, preprocessing, prediction and
confidence estimation are shared. Each concrete provider supplies its own
processor class, since ``AutoProcessor`` does not reliably resolve the combined
image+text processor for TrOCR.
"""
from __future__ import annotations

import io
import threading

import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

from ai_service.base import OCRProvider, OCRResult
from ai_service.config import settings


class EncoderDecoderProvider(OCRProvider):
    model_id: str

    def __init__(
        self,
        slug: str,
        name: str,
        description: str,
        model_id: str,
        max_length: int | None = None,
        processor_cls=AutoProcessor,
    ) -> None:
        self.slug = slug
        self.name = name
        self.description = description
        self.model_id = model_id
        self.max_length = max_length or settings.max_length
        self.processor_cls = processor_cls
        self._processor = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                self._processor = self.processor_cls.from_pretrained(self.model_id)
                dtype = torch.float16 if settings.inference_device != "cpu" else torch.float32
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_id, dtype=dtype
                )
                self._model.to(settings.inference_device)
                self._model.eval()

    def _pixel_values(self, content: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        dtype = next(self._model.parameters()).dtype
        return inputs.pixel_values.to(
            device=settings.inference_device, dtype=dtype
        )

    @torch.no_grad()
    def predict(self, content: bytes) -> OCRResult:
        if not content:
            raise ValueError("Empty input")

        self._ensure_loaded()
        pixel_values = self._pixel_values(content)
        generated = self._model.generate(
            pixel_values,
            max_length=self.max_length,
            output_scores=True,
            return_dict_in_generate=True,
        )
        text = self._processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )[0].strip()
        return OCRResult(text=text, confidence=self._mean_confidence(generated.scores))

    @staticmethod
    def _mean_confidence(scores: tuple[torch.Tensor, ...]) -> float:
        if not scores:
            return 0.0
        per_token = [torch.softmax(s.float(), dim=-1).max(dim=-1).values for s in scores]
        return round(float(torch.stack(per_token).mean().item()), 6)
