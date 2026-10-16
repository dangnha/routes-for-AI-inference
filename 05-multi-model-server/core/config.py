"""
Central configuration for the multi-model server.

Everything that can change per environment (model ids, device, thresholds, mock mode)
lives here and is read from environment variables, so you never edit code to switch
environments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Local model cache: every downloaded weight lives INSIDE the project so the
# server can run fully offline. `download_models.py` fills this cache; set
# MMS_OFFLINE=1 to forbid any network access at load time.
# ---------------------------------------------------------------------------
MODEL_CACHE_DIR = Path(os.environ.get("MMS_MODEL_CACHE", BASE_DIR / "model_cache"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR / "hf" / "hub"))


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def offline_only() -> bool:
    """True when models must be loaded from the local cache (no network access)."""
    return _flag("MMS_OFFLINE", False)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Device helpers: cpu / cuda (gpu) / mps (apple silicon)
# ---------------------------------------------------------------------------
def cuda_available() -> bool:
    """True if a CUDA GPU is available (torch import is optional)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def available_devices() -> list[str]:
    """Devices this machine can actually run models on."""
    devices = ["cpu"]
    if cuda_available():
        devices.append("cuda")
    return devices


def resolve_device() -> str:
    """Default device: MMS_DEVICE if set, otherwise auto-detect."""
    requested = os.environ.get("MMS_DEVICE", "auto").strip().lower()
    if requested in ("cpu", "cuda", "mps"):
        return requested
    return "cuda" if cuda_available() else "cpu"


@dataclass(frozen=True)
class OCRModelSpec:
    name: str
    engine: str  # "rapidocr" | "trocr"
    hf_id: str | None
    kind: str  # "line" = per text line | "document" = whole page at once
    description: str
    devices: tuple[str, ...]  # devices this model may run on


# The server ships exactly TWO OCR models (both run locally):
#   - rapidocr          : CPU only (ONNX Runtime), reads the whole page at once
#   - trocr-base-printed: CPU or GPU (PyTorch), reads one printed text line at a time
OCR_MODELS: dict[str, OCRModelSpec] = {
    "rapidocr": OCRModelSpec(
        name="rapidocr",
        engine="rapidocr",
        hf_id=None,
        kind="document",
        description="RapidOCR (ONNX Runtime) — fast CPU OCR, detects and reads all text on the page",
        devices=("cpu",),
    ),
    "trocr-base-printed": OCRModelSpec(
        name="trocr-base-printed",
        engine="trocr",
        hf_id="microsoft/trocr-base-printed",
        kind="line",
        description="TrOCR base (printed) — printed text per line, accurate transformer (~334M)",
        devices=("cpu", "cuda"),
    ),
}

DETECTION_HF_ID = "hustvl/yolos-tiny"  # object detection, very light


@dataclass(frozen=True)
class Settings:
    device: str = field(default_factory=resolve_device)
    mock: bool = field(default_factory=lambda: _flag("MMS_MOCK", False))
    default_ocr_model: str = field(default_factory=lambda: os.environ.get("MMS_OCR_MODEL", "rapidocr"))
    max_ocr_lines: int = field(default_factory=lambda: _int("MMS_MAX_OCR_LINES", 20))
    detection_threshold: float = 0.5
    max_upload_mb: int = field(default_factory=lambda: _int("MMS_MAX_UPLOAD_MB", 10))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
