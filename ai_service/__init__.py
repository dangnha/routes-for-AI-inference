"""AI inference service package.

The Hugging Face cache is pinned to the project-local ``models/hf`` directory
*before* any submodule imports ``transformers``/``huggingface_hub``, so inference
runs fully offline once ``scripts/download_models.py`` has populated the cache.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
