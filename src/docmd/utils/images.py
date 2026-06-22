from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from docmd.schema import BBox, Polygon


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def image_size(path: str | Path) -> tuple[int, int]:
    image = read_image(path)
    h, w = image.shape[:2]
    return w, h


def clamp_bbox(bbox: BBox, width: int, height: int, pad: int = 0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(round(x1)) - pad),
        max(0, int(round(y1)) - pad),
        min(width, int(round(x2)) + pad),
        min(height, int(round(y2)) + pad),
    )


def crop_bbox(image: np.ndarray, bbox: BBox, pad: int = 2) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h, pad=pad)
    return image[y1:y2, x1:x2].copy()


def mask_crop(image: np.ndarray, bbox: BBox, polygon: Polygon | None, pad: int = 2) -> np.ndarray:
    crop = crop_bbox(image, bbox, pad=pad)
    if not polygon:
        return crop
    h, w = image.shape[:2]
    x1, y1, _, _ = clamp_bbox(bbox, w, h, pad=pad)
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] -= x1
    pts[:, 1] -= y1
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    white = np.full_like(crop, 255)
    return np.where(mask[..., None] > 0, crop, white)

