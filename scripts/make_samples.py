"""Generate the bundled example images under ``frontend/samples/``.

These are purpose-built (license-free, deterministic) so each model has an input
it is actually good at — the frontend offers them as one-click examples.

    python scripts/make_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "frontend" / "samples"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
SERIF = FONT_DIR / "DejaVuSerif.ttf"
SERIF_BOLD = FONT_DIR / "DejaVuSerif-Bold.ttf"
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
MONO = FONT_DIR / "DejaVuSansMono.ttf"

INK = (26, 26, 30)
PAPER = (252, 251, 247)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def new_canvas(width: int, height: int, bg=PAPER) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), bg)
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, subdir: str, name: str) -> None:
    out_dir = SAMPLES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / name)
    print(f"  wrote {out_dir.relative_to(PROJECT_ROOT)}/{name}")


def line_image(text: str, size: int, font_path: Path, pad: int = 24) -> Image.Image:
    """Render a single line tightly cropped to the text (TrOCR-friendly)."""
    f = font(font_path, size)
    bbox = ImageDraw.Draw(Image.new("RGB", (10, 10))).textbbox((0, 0), text, font=f)
    width = bbox[2] - bbox[0] + 2 * pad
    height = bbox[3] - bbox[1] + 2 * pad
    image, draw = new_canvas(width, height)
    draw.text((pad - bbox[0], pad - bbox[1]), text, fill=INK, font=f)
    return image


def printed_line() -> None:
    # TrOCR is a single-line model and is most accurate on short lines.
    save(line_image("The quick brown fox", 48, SERIF), "printed", "printed_line.png")


def printed_headline() -> None:
    save(line_image("Total amount due", 48, SERIF), "printed", "printed_headline.png")


def receipt() -> None:
    image, draw = new_canvas(720, 960, (255, 255, 255))
    mono = font(MONO, 30)
    x = 50
    draw.text((x, 40), "CORNER MARKET", fill=INK, font=font(MONO, 40))
    draw.text((x, 92), "123 Main Street", fill=INK, font=mono)
    draw.text((x, 130), "Tel: 555-0142", fill=INK, font=mono)
    draw.line((x, 185, 670, 185), fill=(120, 120, 120), width=2)

    items = [
        ("Milk 2L", "3.50"),
        ("Bread", "2.25"),
        ("Eggs (12)", "4.10"),
        ("Coffee 500g", "9.99"),
        ("Bananas 1kg", "1.85"),
    ]
    y = 215
    for label, price in items:
        draw.text((x, y), label, fill=INK, font=mono)
        draw.text((520, y), price, fill=INK, font=mono)
        y += 48
    draw.line((x, y - 6, 670, y - 6), fill=(120, 120, 120), width=2)
    y += 20
    draw.text((x, y), "TOTAL", fill=INK, font=font(MONO, 36))
    draw.text((500, y), "21.69", fill=INK, font=font(MONO, 36))
    draw.text((x, y + 70), "THANK YOU!", fill=INK, font=font(MONO, 30))
    save(image, "multiline", "receipt.png")


def street_sign() -> None:
    image, draw = new_canvas(1200, 520, (18, 70, 140))
    draw.rectangle((30, 30, 1170, 490), outline=(255, 255, 255), width=6)
    draw.text((80, 90), "CAUTION", fill=(255, 255, 255), font=font(SANS_BOLD, 84))
    draw.text((80, 220), "Road work ahead", fill=(255, 255, 255), font=font(SANS, 60))
    draw.text((80, 320), "Reduce speed now", fill=(255, 255, 255), font=font(SANS, 60))
    save(image, "multiline", "street_sign.png")


def address_label() -> None:
    image, draw = new_canvas(960, 420, (255, 255, 255))
    lines = [
        "Nguyen Van A",
        "45 Le Loi Street, District 1",
        "Ho Chi Minh City",
        "Vietnam 700000",
    ]
    y = 50
    for line in lines:
        draw.text((60, y), line, fill=INK, font=font(SANS, 44))
        y += 80
    save(image, "multiline", "address_label.png")


def page_with_table() -> None:
    image, draw = new_canvas(1240, 1560, (255, 255, 255))
    draw.text((80, 60), "Quarterly Sales Report", fill=INK, font=font(SERIF_BOLD, 56))
    draw.text((80, 140), "Prepared by the analytics team — Q3 2026", fill=(90, 90, 96),
              font=font(SERIF, 32))

    paragraph = (
        "Revenue grew across all regions this quarter. The table below "
        "summarizes units sold and total revenue per region. All figures are "
        "in thousands of US dollars and exclude taxes."
    )
    y = 215
    for line in _wrap(paragraph, 56):
        draw.text((80, y), line, fill=INK, font=font(SERIF, 32))
        y += 44

    y += 30
    cols = [80, 420, 700, 1000]
    headers = ["Region", "Units", "Revenue"]
    rows = [
        ("North", "12,400", "1,240"),
        ("South", "9,850", "985"),
        ("East", "7,300", "730"),
        ("West", "5,120", "512"),
    ]
    draw.rectangle((70, y - 12, 1170, y + 60), fill=(238, 238, 240))
    for x, header in zip(cols, headers):
        draw.text((x, y), header, fill=INK, font=font(SANS_BOLD, 34))
    y += 80
    for row in rows:
        for x, cell in zip(cols, row):
            draw.text((x, y), cell, fill=INK, font=font(SANS, 32))
        y += 64
    draw.line((70, y - 6, 1170, y - 6), fill=(150, 150, 150), width=2)
    y += 30
    draw.text((cols[0], y), "Total", fill=INK, font=font(SANS_BOLD, 34))
    draw.text((cols[2], y), "3,467", fill=INK, font=font(SANS_BOLD, 34))
    save(image, "document", "page_with_table.png")


def invoice() -> None:
    image, draw = new_canvas(1000, 1300, (255, 255, 255))
    draw.text((60, 50), "INVOICE", fill=INK, font=font(SANS_BOLD, 60))
    draw.text((60, 140), "Invoice No: INV-2026-0042", fill=INK, font=font(SANS, 32))
    draw.text((60, 185), "Date: 03 October 2026", fill=INK, font=font(SANS, 32))
    draw.text((60, 230), "Bill To: Acme Corporation", fill=INK, font=font(SANS, 32))
    draw.text((60, 275), "Account: AC-9931", fill=INK, font=font(SANS, 32))

    y = 360
    draw.rectangle((50, y - 12, 950, y + 50), fill=(240, 240, 242))
    draw.text((80, y), "Description", fill=INK, font=font(SANS_BOLD, 32))
    draw.text((600, y), "Qty", fill=INK, font=font(SANS_BOLD, 32))
    draw.text((760, y), "Amount", fill=INK, font=font(SANS_BOLD, 32))
    y += 80
    for desc, qty, amount in [
        ("GPU inference server", "1", "4,200.00"),
        ("Model storage (2TB)", "2", "300.00"),
        ("Support plan", "12", "1,800.00"),
    ]:
        draw.text((80, y), desc, fill=INK, font=font(SANS, 30))
        draw.text((600, y), qty, fill=INK, font=font(SANS, 30))
        draw.text((760, y), amount, fill=INK, font=font(SANS, 30))
        y += 60
    draw.text((600, y + 30), "Total due", fill=INK, font=font(SANS_BOLD, 34))
    draw.text((760, y + 30), "6,300.00", fill=INK, font=font(SANS_BOLD, 34))
    save(image, "document", "invoice.png")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def main() -> None:
    print(f"Writing samples to {SAMPLES_DIR.relative_to(PROJECT_ROOT)}/")
    for builder in (
        printed_line,
        printed_headline,
        receipt,
        street_sign,
        address_label,
        page_with_table,
        invoice,
    ):
        builder()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
