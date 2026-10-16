"""
Smoke test for the multi-model server backend.

Exercises every task the server exposes - OCR, classification and detection - so you can prove
the backend works even though the Streamlit UI only surfaces OCR. It builds a small invoice
image in memory (no network needed), posts it to the image endpoints, and prints a PASS/FAIL
summary.

Usage:
    uvicorn main:app --reload --port 8005     # terminal 1
    python smoke_test.py                      # terminal 2
    python smoke_test.py http://127.0.0.1:8005
"""

from __future__ import annotations

import io
import sys

import requests
from PIL import Image, ImageDraw

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8005").rstrip("/")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make_invoice_image() -> bytes:
    """A synthetic invoice so the test never depends on the internet."""
    image = Image.new("RGB", (900, 460), "white")
    draw = ImageDraw.Draw(image)
    lines = [
        "ABC CO., LTD",
        "INVOICE",
        "Date: 2026-10-14   No: INV-0001",
        "Item        Qty    Price     Amount",
        "T-Shirt       2    150000     300000",
        "Jeans         1    450000     450000",
        "Total: 750000 VND",
    ]
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black")
        y += 55
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def get_json(path: str):
    r = requests.get(BASE + path, timeout=30)
    return r.status_code, (r.json() if r.content else None)


def post_file(path: str, filename: str, content: bytes, data: dict):
    r = requests.post(BASE + path, files={"file": (filename, content, "image/png")}, data=data, timeout=600)
    return r.status_code, (r.json() if r.content else None)


def post_json(path: str, payload: dict):
    r = requests.post(BASE + path, json=payload, timeout=60)
    return r.status_code, (r.json() if r.content else None)


def main() -> int:
    print(f"Testing server at {BASE}\n")

    try:
        status, health = get_json("/health")
    except requests.exceptions.RequestException as exc:
        print(f"Could not reach the server: {exc}")
        print("Start it first:  uvicorn main:app --reload --port 8005")
        return 2

    check("GET /health", status == 200 and health.get("status") == "ok", str(health))

    status, devices = get_json("/devices")
    check("GET /devices", status == 200 and "cpu" in devices.get("devices", []), str(devices))

    status, ocr_models = get_json("/ocr/models")
    check("GET /ocr/models", status == 200 and bool(ocr_models), str([m["name"] for m in (ocr_models or [])]))

    status, models = get_json("/models")
    names = {m["name"] for m in (models or {}).get("models", [])}
    check("GET /models lists every task", {"rapidocr", "classifier", "detector"} <= names, str(sorted(names)))

    invoice = make_invoice_image()

    # --- OCR -----------------------------------------------------------------
    status, payload = post_file("/ocr/upload", "invoice.png", invoice, {"model": "rapidocr", "device": "cpu"})
    text = (payload or {}).get("text", "")
    check("POST /ocr/upload (rapidocr)", status == 200 and bool(text.strip()), f"{len(text)} chars")

    status, payload = post_file(
        "/ocr/upload", "invoice.png", invoice, {"model": "trocr-small-printed", "device": "cpu"}
    )
    check("POST /ocr/upload (trocr-small-printed)", status == 200, str(payload or {})[:120])

    # --- Classification ------------------------------------------------------
    status, info = get_json("/classification/info")
    check("GET /classification/info", status == 200 and bool((info or {}).get("classes")), str(info))

    status, payload = post_json(
        "/classification/predict",
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    )
    check("POST /classification/predict", status == 200 and "predicted_label" in (payload or {}),
          str((payload or {}).get("predicted_label")))

    # --- Detection -----------------------------------------------------------
    status, payload = post_file("/detection/upload", "invoice.png", invoice, {"threshold": "0.5", "device": "cpu"})
    check("POST /detection/upload", status == 200 and "count" in (payload or {}),
          f"count={(payload or {}).get('count')}")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
