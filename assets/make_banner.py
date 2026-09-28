"""Render the repository banner (used at the top of the README).

Usage: python assets/make_banner.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Palette: a forest canopy over an urban grid.
BACKGROUND = (16, 34, 28)
ACCENT = (140, 200, 90)
SECONDARY = (90, 170, 200)
TEXT = (233, 244, 238)


def font(size: int):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    width, height = 1600, 420
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    # A grid of cells with a few "vegetation" cells highlighted.
    step = 40
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill=(28, 52, 44), width=1)
    for y in range(0, height, step):
        draw.line([(0, y), (width, y)], fill=(28, 52, 44), width=1)
    import random

    rng = random.Random(7)
    for _ in range(60):
        cx = rng.randrange(0, width, step)
        cy = rng.randrange(0, height, step)
        colour = ACCENT if rng.random() < 0.6 else SECONDARY
        draw.rectangle([cx + 2, cy + 2, cx + step - 2, cy + step - 2], fill=colour)

    # Title panel.
    draw.rectangle([0, height - 190, width, height], fill=BACKGROUND)
    draw.text((70, height - 150), "VERDI", font=font(72), fill=TEXT)
    draw.text((72, height - 66),
              "Vegetation-Environment Resilience Diagnostic Intelligence",
              font=font(26), fill=ACCENT)
    draw.text((width - 480, height - 58),
              "label construction  |  data processing  |  model reference",
              font=font(18), fill=SECONDARY)

    out = Path(__file__).with_name("banner.png")
    image.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
