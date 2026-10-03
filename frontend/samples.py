"""Bundled example images, grouped by the model they suit best."""
from __future__ import annotations

from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# model slug -> (folder, human label)
MODEL_SAMPLES: dict[str, tuple[str, str]] = {
    "trocr": ("printed", "Single-line printed text"),
    "easyocr": ("multiline", "Multi-line: receipts, signs, addresses"),
    "gotocr": ("document", "Full pages with layout and tables"),
}


def sample_files(model: str) -> list[Path]:
    folder, _ = MODEL_SAMPLES.get(model, (None, ""))
    if folder is None:
        return []
    base = SAMPLES_DIR / folder
    return sorted(base.glob("*.png")) if base.exists() else []


def all_samples() -> list[Path]:
    return sorted(SAMPLES_DIR.glob("*/*.png"))


def label(model: str) -> str:
    return MODEL_SAMPLES.get(model, ("", "Examples"))[1]


def pretty_name(path: Path) -> str:
    return path.stem.replace("_", " ").title()
