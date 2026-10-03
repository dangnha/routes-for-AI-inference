"""GOT-OCR2.0 — general OCR that outputs Markdown (layout, tables, formulas).

This model ships custom code on the Hub and is loaded with
``trust_remote_code=True``. It runs on CPU or CUDA depending on ``DEVICE``.
"""
from __future__ import annotations

import io
import threading

from PIL import Image

from ai_service.base import OCRProvider, OCRResult
from ai_service.config import settings


def _patch_legacy_cache_api() -> None:
    """Shim removed transformers APIs used by GOT-OCR2.0's remote code.

    GOT-OCR2.0 predates transformers 4.44 and calls ``DynamicCache.seen_tokens``
    and ``get_max_length()``, both removed since. Re-add them as thin wrappers
    over the current API so the model runs on a modern transformers install.
    """
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None


class GotOCRProvider(OCRProvider):
    slug = "gotocr"
    name = "GOT-OCR2.0 · General OCR"
    description = "General OCR 2.0, outputs Markdown with layout/tables."

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                if settings.inference_device == "cpu":
                    raise ValueError(
                        "GOT-OCR2.0 requires a CUDA device; set INFERENCE_DEVICE=cuda"
                    )
                _patch_legacy_cache_api()
                from transformers import AutoModel, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    settings.gotocr_model_id, trust_remote_code=True
                )
                self._model = AutoModel.from_pretrained(
                    settings.gotocr_model_id,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self._model = self._model.to(settings.inference_device).eval()

    def predict(self, content: bytes) -> OCRResult:
        if not content:
            raise ValueError("Empty input")

        self._ensure_loaded()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        # gradio_input=True makes chat() use the PIL image directly instead of
        # treating the argument as a file path. GOT-OCR's chat() always moves
        # tensors to CUDA, so this provider effectively requires a GPU.
        result = self._model.chat(
            self._tokenizer, image, ocr_type="ocr", gradio_input=True
        )
        return OCRResult(text=str(result).strip(), confidence=1.0)
