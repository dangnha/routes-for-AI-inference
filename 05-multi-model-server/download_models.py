"""
Pre-download every model the multi-model server uses, so the server runs fully offline.

Everything is written into the project-local cache (see core.config.MODEL_CACHE_DIR), so no
weights land in the global Hugging Face cache and the server never needs the network again.

Models covered:
  - rapidocr           (RapidOCR / ONNX Runtime, CPU)
  - trocr-base-printed (TrOCR, PyTorch)
  - yolos-tiny         (object detection, PyTorch)

The Iris classifier is trained locally (no download): run `python train_model.py` once.

Usage:
    python download_models.py
"""

from __future__ import annotations

import core.config as config  # imported first: it points HF_HOME at the project cache


def download_rapidocr() -> None:
    """RapidOCR ships its ONNX weights with the package; instantiate it to validate them."""
    from rapidocr_onnxruntime import RapidOCR

    RapidOCR()
    print("  [ok] rapidocr (ONNX Runtime, cpu)")


def download_trocr(hf_id: str) -> None:
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    AutoImageProcessor.from_pretrained(hf_id)
    AutoTokenizer.from_pretrained(hf_id)
    VisionEncoderDecoderModel.from_pretrained(hf_id)
    print(f"  [ok] trocr {hf_id}")


def download_detector(hf_id: str) -> None:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    AutoImageProcessor.from_pretrained(hf_id)
    AutoModelForObjectDetection.from_pretrained(hf_id)
    print(f"  [ok] detector {hf_id}")


def main() -> None:
    print(f"Model cache: {config.MODEL_CACHE_DIR}")

    print("Downloading OCR models...")
    for spec in config.OCR_MODELS.values():
        if spec.engine == "rapidocr":
            download_rapidocr()
        elif spec.engine == "trocr":
            download_trocr(spec.hf_id)

    print("Downloading detection model...")
    download_detector(config.DETECTION_HF_ID)

    if not (config.BASE_DIR / "model.pkl").exists():
        print("\nNote: the classifier is not built yet -> run `python train_model.py` once.")

    print("\nAll models are ready. Start the server with MMS_OFFLINE=1 to run fully offline.")


if __name__ == "__main__":
    main()
