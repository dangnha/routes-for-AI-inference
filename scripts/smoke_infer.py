"""Run each model on its bundled sample and print the real output.

    python scripts/smoke_infer.py            # all models
    python scripts/smoke_infer.py trocr      # one model
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

from ai_service.config import settings  # noqa: E402
from ai_service.providers import MODELS  # noqa: E402
from samples import sample_files  # noqa: E402


def run(slug: str) -> None:
    provider = MODELS[slug]
    files = sample_files(slug)
    if not files:
        print(f"• {slug}: no bundled sample, skipping")
        return

    sample = files[0]
    print(f"• {slug}  ({provider.name})  ← {sample.name}")
    started = time.perf_counter()
    try:
        result = provider.predict(sample.read_bytes())
    except Exception as exc:  # noqa: BLE001
        print(f"  !! failed: {exc}\n")
        return
    elapsed = time.perf_counter() - started

    print(f"  confidence: {result.confidence:.4f}   ({elapsed:.2f}s)")
    print(f"  regions: {len(result.lines)}")
    print("  text:")
    for line in (result.text or "(empty)").splitlines():
        print(f"    | {line}")
    print()


def main() -> None:
    print(f"device = {settings.inference_device}\n")
    slugs = sys.argv[1:] or list(MODELS)
    for slug in slugs:
        run(slug)


if __name__ == "__main__":
    main()
