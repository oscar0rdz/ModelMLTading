#!/usr/bin/env python3
"""
Small helper to draw a high-level pipeline diagram without external dependencies.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


BOXES = [
    ("Datos\n(OHLCV + sample)", (30, 100, 200, 230)),
    ("Feature\nEngineering", (250, 100, 420, 230)),
    ("Entrenamiento\n+ Tuning", (470, 100, 640, 230)),
    ("Calibración\n(Isotónica)", (690, 100, 860, 230)),
    ("Selector EV\n(umbral)", (910, 100, 1080, 230)),
    ("Walk-Forward\nBacktest", (1130, 100, 1300, 230)),
]


def draw_diagram(output_path: Path) -> None:
    width, height = 1360, 360
    image = Image.new("RGB", (width, height), color="#0d1117")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for label, coords in BOXES:
        draw.rectangle(coords, outline="#58a6ff", width=3)
        bbox = draw.multiline_textbbox((0, 0), label, font=font, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x_center = (coords[0] + coords[2]) / 2 - text_w / 2
        y_center = (coords[1] + coords[3]) / 2 - text_h / 2
        draw.multiline_text((x_center, y_center), label, font=font, fill="#e6edf3", align="center")

    # Arrows between boxes
    for (_, (x1, y1, x2, y2)), (_, (nx1, ny1, nx2, ny2)) in zip(BOXES, BOXES[1:]):
        start = (x2, (y1 + y2) / 2)
        end = (nx1, (ny1 + ny2) / 2)
        draw.line([start, end], fill="#58a6ff", width=3)
        arrow_tip = (end[0], end[1])
        draw.polygon(
            [
                arrow_tip,
                (arrow_tip[0] - 15, arrow_tip[1] - 8),
                (arrow_tip[0] - 15, arrow_tip[1] + 8),
            ],
            fill="#58a6ff",
        )

    title = "BTC/USDT · 15m — Pipeline resumido"
    bbox = draw.textbbox((0, 0), title, font=font)
    title_w = bbox[2] - bbox[0]
    draw.text(((width - title_w) / 2, 30), title, fill="#e6edf3", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    print(f"[OK] Pipeline diagram saved to {output_path}")


if __name__ == "__main__":
    draw_diagram(Path("docs/figs/pipeline.png"))
