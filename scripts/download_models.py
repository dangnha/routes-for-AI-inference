"""Download every model into the project-local ``models/`` cache (one-time).

After this runs, the AI service loads from ``./models`` and never needs network.
Usage:

    python scripts/download_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Must happen before transformers/huggingface_hub import (ai_service/__init__ also sets it).
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))

from ai_service.config import settings  # noqa: E402


def download_trocr() -> None:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    print(f"  TrOCR: {settings.trocr_model_id}")
    TrOCRProcessor.from_pretrained(settings.trocr_model_id)
    VisionEncoderDecoderModel.from_pretrained(settings.trocr_model_id)
    print("  done")


def download_gotocr() -> None:
    from transformers import AutoModel, AutoTokenizer

    print(f"  GOT-OCR2.0: {settings.gotocr_model_id}")
    AutoTokenizer.from_pretrained(settings.gotocr_model_id, trust_remote_code=True)
    AutoModel.from_pretrained(
        settings.gotocr_model_id, trust_remote_code=True, low_cpu_mem_usage=True
    )
    print("  done")


def download_easyocr() -> None:
    import easyocr

    print(f"  EasyOCR: langs={settings.easyocr_langs}")
    easyocr.Reader(
        settings.easyocr_langs,
        gpu=False,
        model_storage_directory=settings.easyocr_model_dir,
        download_enabled=True,
        verbose=True,
    )
    print("  done")


def dir_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1e6:.1f} MB"


def main() -> None:
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    print(f"HF_HOME = {os.environ['HF_HOME']}\n")

    for name, fn in (
        ("trocr", download_trocr),
        ("gotocr", download_gotocr),
        ("easyocr", download_easyocr),
    ):
        print(f"• {name}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {name} failed: {exc}")

    print("\nCache sizes:")
    for sub in ("hf", "easyocr"):
        path = PROJECT_ROOT / "models" / sub
        if path.exists():
            print(f"  models/{sub}: {dir_size(path)}")


if __name__ == "__main__":
    main()
