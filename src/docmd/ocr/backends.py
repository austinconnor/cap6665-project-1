from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import os

import numpy as np


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float | None = None
    backend: str = "unknown"
    lines: list[dict[str, object]] = field(default_factory=list)


class OCRBackend(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OCRResult:
        ...


class StubOCRBackend:
    name = "stub"

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult(text="", confidence=None, backend=self.name, lines=[])


class PaddleOCRBackend:
    name = "paddleocr"

    def __init__(
        self,
        device: str | None = None,
        text_detection_model_name: str = "PP-OCRv6_tiny_det",
        text_recognition_model_name: str = "PP-OCRv6_tiny_rec",
    ) -> None:
        cache_root = Path("outputs/model_cache").resolve()
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_root / "paddlex"))
        os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))

        from paddleocr import PaddleOCR

        kwargs: dict[str, Any] = {
            "text_detection_model_name": text_detection_model_name,
            "text_recognition_model_name": text_recognition_model_name,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": "transformers",
        }
        if device:
            kwargs["device"] = device
        self.ocr = PaddleOCR(**kwargs)

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        import cv2

        prepared = cv2.copyMakeBorder(
            image,
            top=8,
            bottom=8,
            left=8,
            right=8,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        height, width = prepared.shape[:2]
        if height < 96:
            scale = min(4.0, 96 / max(1, height))
            prepared = cv2.resize(
                prepared,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
        return prepared

    def recognize(self, image: np.ndarray) -> OCRResult:
        results = self.ocr.predict(self._prepare_image(image))
        return _paddle_results_to_ocr_result(results, self.name)


def create_ocr_backend(name: str = "paddleocr", **kwargs: object) -> OCRBackend:
    normalized = name.lower()
    if normalized in {"paddle", "paddleocr"}:
        return PaddleOCRBackend(**kwargs)
    if normalized == "stub":
        return StubOCRBackend()
    raise ValueError(f"Unsupported OCR backend: {name}")


def _paddle_results_to_ocr_result(results: object, backend_name: str) -> OCRResult:
    texts: list[str] = []
    confidences: list[float] = []
    lines: list[dict[str, object]] = []
    for result in results or []:
        payload = getattr(result, "json", result)
        if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
            payload = payload["res"]
        if not isinstance(payload, dict):
            continue
        rec_texts = payload.get("rec_texts") or []
        rec_scores = payload.get("rec_scores") or []
        rec_polys = payload.get("rec_polys") or payload.get("dt_polys") or []
        for index, raw_text in enumerate(rec_texts):
            text = str(raw_text).strip()
            if not text:
                continue
            score = _at(rec_scores, index)
            polygon = _at(rec_polys, index)
            confidence = _to_float(score)
            texts.append(text)
            if confidence is not None:
                confidences.append(confidence)
            lines.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": _quad_to_bbox(polygon),
                    "polygon": _flatten_quad(polygon),
                }
            )
    return OCRResult(
        text=" ".join(texts).strip(),
        confidence=(sum(confidences) / len(confidences) if confidences else None),
        backend=backend_name,
        lines=lines,
    )


def _at(values: object, index: int) -> object | None:
    try:
        return values[index]  # type: ignore[index]
    except Exception:
        return None


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _flatten_quad(box: object) -> list[float] | None:
    try:
        points = np.asarray(box, dtype=float).reshape(-1, 2)
    except Exception:
        return None
    return [float(v) for point in points for v in point]


def _quad_to_bbox(box: object) -> list[float] | None:
    try:
        points = np.asarray(box, dtype=float).reshape(-1, 2)
    except Exception:
        return None
    xs = points[:, 0]
    ys = points[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _self_check() -> None:
    result = _paddle_results_to_ocr_result(
        [
            {
                "res": {
                    "rec_texts": ["Hello", "world"],
                    "rec_scores": [0.9, 0.8],
                    "rec_polys": [
                        [[0, 0], [10, 0], [10, 5], [0, 5]],
                        [[0, 6], [10, 6], [10, 11], [0, 11]],
                    ],
                }
            }
        ],
        "paddleocr",
    )
    assert result.text == "Hello world"
    assert result.confidence is not None and abs(result.confidence - 0.85) < 1e-9
    assert result.lines[0]["bbox"] == [0.0, 0.0, 10.0, 5.0]


if __name__ == "__main__":
    _self_check()