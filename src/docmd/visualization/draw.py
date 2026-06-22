from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from docmd.schema import LayoutRegion
from docmd.utils.images import read_image
from docmd.utils.io import ensure_dir


PALETTE = [
    (37, 99, 235),
    (5, 150, 105),
    (220, 38, 38),
    (217, 119, 6),
    (124, 58, 237),
    (8, 145, 178),
    (219, 39, 119),
    (82, 82, 91),
    (22, 163, 74),
    (202, 138, 4),
    (14, 116, 144),
]


def color_for(class_id: int) -> tuple[int, int, int]:
    r, g, b = PALETTE[class_id % len(PALETTE)]
    return b, g, r


def draw_layout(
    image_path: str | Path,
    regions: list[LayoutRegion],
    output_path: str | Path,
    show_order: bool = False,
    draw_masks: bool = True,
) -> Path:
    image = read_image(image_path)
    overlay = image.copy()
    for region in regions:
        color = color_for(region.class_id)
        if draw_masks and region.polygon:
            pts = np.array(region.polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(overlay, [pts], color)
        x1, y1, x2, y2 = [int(round(v)) for v in region.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = region.class_name
        if region.confidence is not None:
            label += f" {region.confidence:.2f}"
        if show_order and region.order is not None:
            label = f"{region.order}. {label}"
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if draw_masks:
        image = cv2.addWeighted(overlay, 0.22, image, 0.78, 0)
    target = Path(output_path)
    ensure_dir(target.parent)
    cv2.imwrite(str(target), image)
    return target

