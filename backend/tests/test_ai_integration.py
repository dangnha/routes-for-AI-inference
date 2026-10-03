"""Optional integration test against a running AI service (real model output).

Needs the AI service (default :8001, override with ``AI_SERVICE_URL``) and
downloaded models. Run with:

    pytest -m slow
"""
import os
from pathlib import Path

import httpx
import pytest

AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://localhost:8001")
SAMPLES = Path(__file__).resolve().parents[2] / "frontend" / "samples"


def _ai_service_up() -> bool:
    try:
        return httpx.get(f"{AI_SERVICE_URL}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ai_service_up(), reason="AI service not running on :8001"),
]


@pytest.mark.parametrize(
    ("model", "sample"),
    [
        ("trocr", "printed/printed_line.png"),
        ("easyocr", "multiline/address_label.png"),
        ("gotocr", "document/invoice.png"),
    ],
)
def test_model_returns_text(model: str, sample: str):
    path = SAMPLES / sample
    assert path.exists(), f"missing sample {path}"

    with path.open("rb") as handle:
        response = httpx.post(
            f"{AI_SERVICE_URL}/models/{model}/infer",
            files={"file": (path.name, handle, "image/png")},
            timeout=300.0,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == model
    assert body["text"].strip(), f"{model} returned empty text"
