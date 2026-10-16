"""
Central configuration for the multi-model server.

Everything that can change per environment (model ids, device, thresholds, mock mode, the
Hugging Face token) lives here and is read from environment variables or a local `.env`
file, so you never edit code to switch environments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load `KEY=VALUE` lines from a project-local `.env` file (created by you, never committed).

    Real environment variables always win, so `set HF_TOKEN=...` still overrides the file.
    """
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Read `.env` first so HF_TOKEN is in place before huggingface_hub / transformers are imported.
_load_dotenv()

# Xet keeps a second, large shard cache and is fragile on OneDrive / network drives. The classic
# HTTP downloader is smaller and more reliable for the few small models used here.
# Set HF_HUB_DISABLE_XET=0 to opt back in.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---------------------------------------------------------------------------
# Local model cache - one tidy folder per model, easy to inspect and delete:
#
#   model_cache/
#   ├── trocr-small-printed/   # config.json, model.safetensors, tokenizer files ...
#   └── yolos-tiny/            # config.json, model.safetensors, preprocessor_config.json
#
# `download_models.py` fills these folders. RapidOCR ships its (tiny, ~16 MB) ONNX weights
# inside the installed package, so it needs no folder here.
# ---------------------------------------------------------------------------
MODEL_CACHE_DIR = Path(os.environ.get("MMS_MODEL_CACHE", BASE_DIR / "model_cache"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def model_dir(name: str) -> Path:
    """The folder a given model is downloaded into, e.g. `model_cache/trocr-small-printed`."""
    return MODEL_CACHE_DIR / name


def _has_weights(path: Path) -> bool:
    """True when `path` looks like a finished `save_pretrained` folder."""
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def resolve_hf_source(hf_id: str, name: str) -> str:
    """Prefer the pre-downloaded project-local folder; fall back to the Hub id if it is missing."""
    local = model_dir(name)
    return str(local) if _has_weights(local) else hf_id


def hf_token() -> str | None:
    """The Hugging Face token used to download models.

    Resolution order: `HF_TOKEN` env var -> `HUGGING_FACE_HUB_TOKEN` env var -> the token saved
    by `hf auth login` (also readable from the project-local `.env` file).
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001
        return None


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


# The server ships exactly TWO OCR models, both LIGHTWEIGHT and both run locally:
#   - rapidocr           : CPU only (ONNX Runtime), ~16 MB, reads the whole page at once
#   - trocr-small-printed: CPU or GPU (PyTorch), ~62M params (~246 MB), reads one line at a time
OCR_MODELS: dict[str, OCRModelSpec] = {
    "rapidocr": OCRModelSpec(
        name="rapidocr",
        engine="rapidocr",
        hf_id=None,
        kind="document",
        description="RapidOCR (ONNX Runtime) - fast, tiny CPU OCR; detects and reads all text on the page",
        devices=("cpu",),
    ),
    "trocr-small-printed": OCRModelSpec(
        name="trocr-small-printed",
        engine="trocr",
        hf_id="microsoft/trocr-small-printed",
        kind="line",
        description="TrOCR small (printed) - printed text per line, light transformer (~62M)",
        devices=("cpu", "cuda"),
    ),
}

DETECTION_HF_ID = "hustvl/yolos-tiny"  # object detection, very light (~26 MB)


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
