"""
Pre-download every model the multi-model server uses, so the server can run fully offline.

Each model is placed in its own tidy folder under the project-local cache (see
core.config.MODEL_CACHE_DIR), so no weights land in the global Hugging Face cache and the
server never needs the network again:

    model_cache/trocr-small-printed/   TrOCR small (PyTorch)  ~246 MB
    model_cache/yolos-tiny/            object detection       ~26 MB

RapidOCR keeps its tiny (~16 MB) ONNX weights inside the installed package - nothing to fetch.

The Iris classifier is trained locally (no download): run `python train_model.py` once.

Hugging Face token: recommended for reliable downloads and required for gated/private repos.
Put it in a project-local `.env` file (`HF_TOKEN=hf_...`) or set the `HF_TOKEN` environment
variable. See the README, section "Hugging Face token".

Usage:
    python download_models.py                # download everything
    python download_models.py ocr            # only the OCR models
    python download_models.py detection      # only the detector
"""

from __future__ import annotations

import sys
from pathlib import Path

import core.config as config


# Skip the duplicate `pytorch_model.bin` (and other framework formats): transformers loads the
# smaller, safer `model.safetensors` by default.
_IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.ot", "*.tflite"]


def _mask(token: str | None) -> str:
    if not token:
        return "<none>"
    return f"{token[:7]}...{token[-4:]}" if len(token) > 14 else "<set>"


def _folder_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def download_rapidocr() -> None:
    """RapidOCR ships its ONNX weights with the package; instantiate it to validate them."""
    from rapidocr_onnxruntime import RapidOCR

    RapidOCR()
    print("  [ok] rapidocr (ONNX Runtime, cpu, bundled weights)")


def download_hf(hf_id: str, name: str) -> None:
    """Download `hf_id` into `model_cache/<name>/` (safetensors only)."""
    from huggingface_hub import snapshot_download

    target = config.model_dir(name)
    snapshot_download(
        repo_id=hf_id,
        local_dir=target,
        token=config.hf_token(),
        ignore_patterns=_IGNORE,
    )
    print(f"  [ok] {hf_id} -> {target.name}/ ({_folder_size_mb(target):.0f} MB)")


def main() -> None:
    wanted = {arg.lower() for arg in sys.argv[1:]} or {"ocr", "detection"}

    print(f"Model cache : {config.MODEL_CACHE_DIR}")
    print(f"HF token    : {_mask(config.hf_token())}")

    if "ocr" in wanted:
        print("Downloading OCR models...")
        for spec in config.OCR_MODELS.values():
            if spec.engine == "rapidocr":
                download_rapidocr()
            elif spec.engine == "trocr":
                download_hf(spec.hf_id, spec.name)

    if "detection" in wanted:
        print("Downloading detection model...")
        download_hf(config.DETECTION_HF_ID, "yolos-tiny")

    if not (config.BASE_DIR / "model.pkl").exists():
        print("\nNote: the classifier is not built yet -> run `python train_model.py` once.")

    print("\nAll requested models are ready. Start the server with MMS_OFFLINE=1 to run fully offline.")


if __name__ == "__main__":
    main()
