import pytest
import torch

from ai_service.base import OCRProvider, OCRResult
from ai_service.providers import MODELS, list_models
from ai_service.providers.encdec import EncoderDecoderProvider


def test_registry_has_three_distinct_models():
    assert set(MODELS) == {"trocr", "easyocr", "gotocr"}


def test_list_models_shape():
    for model in list_models():
        assert {"slug", "name", "description"} <= set(model)


def test_mean_confidence_uniform():
    scores = (torch.zeros(1, 10), torch.zeros(1, 10))
    assert EncoderDecoderProvider._mean_confidence(scores) == pytest.approx(0.1)


def test_mean_confidence_empty():
    assert EncoderDecoderProvider._mean_confidence(()) == 0.0


def test_rejects_empty_input():
    provider = object.__new__(EncoderDecoderProvider)
    with pytest.raises(ValueError, match="Empty input"):
        provider.predict(b"")


class _FakeProvider(OCRProvider):
    slug = "fake"
    name = "Fake"
    description = ""

    def predict(self, content: bytes) -> OCRResult:
        return OCRResult(text="hello", confidence=1.0)


def test_provider_result_carries_optional_lines():
    result = _FakeProvider().predict(b"x")
    assert result.text == "hello"
    assert result.lines == []
