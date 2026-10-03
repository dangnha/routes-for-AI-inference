"""TrOCR — transformer encoder-decoder for single-line printed text."""
from transformers import TrOCRProcessor

from ai_service.config import settings
from ai_service.providers.encdec import EncoderDecoderProvider


class TrOCRProvider(EncoderDecoderProvider):
    def __init__(self) -> None:
        super().__init__(
            slug="trocr",
            name="TrOCR · Printed text",
            description="Single-line printed text recognition.",
            model_id=settings.trocr_model_id,
            processor_cls=TrOCRProcessor,
        )
