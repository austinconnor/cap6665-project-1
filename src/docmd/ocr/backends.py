from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import os

import numpy as np


DEFAULT_REC_MODEL_DIR = Path("models/ocr/PP-OCRv6_medium_rec_safetensors")


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
    name = "paddleocr-rec"

    def __init__(
        self,
        device: str | None = None,
        model_dir: str | Path = DEFAULT_REC_MODEL_DIR,
    ) -> None:
        model_dir = Path(model_dir).resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"Packaged OCR recognition model not found: {model_dir}")

        cache_root = Path("outputs/model_cache").resolve()
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_root / "paddlex"))
        os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))

        from paddleocr import TextRecognition

        kwargs: dict[str, Any] = {
            "model_dir": str(model_dir),
            "engine": "transformers",
        }
        if device:
            kwargs["device"] = device
        self.recognizer = TextRecognition(**kwargs)

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
        prepared = self._prepare_image(image)
        result = self.recognizer.predict(prepared)
        return _recognition_results_to_ocr_result(result, self.name, prepared.shape[:2])


def create_ocr_backend(name: str = "paddleocr", **kwargs: object) -> OCRBackend:
    normalized = name.lower()
    if normalized in {"paddle", "paddleocr", "paddleocr-rec"}:
        return PaddleOCRBackend(**kwargs)
    if normalized == "stub":
        return StubOCRBackend()
    raise ValueError(f"Unsupported OCR backend: {name}")


def _recognition_results_to_ocr_result(
    results: object,
    backend_name: str,
    image_shape: tuple[int, int] | None = None,
) -> OCRResult:
    texts: list[str] = []
    confidences: list[float] = []
    lines: list[dict[str, object]] = []
    height, width = image_shape or (0, 0)
    bbox = [0.0, 0.0, float(width), float(height)] if width and height else None

    for result in results or []:
        payload = getattr(result, "json", result)
        if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
            payload = payload["res"]
        if not isinstance(payload, dict):
            continue
        text = str(payload.get("rec_text") or "").strip()
        if not text:
            continue
        confidence = _to_float(payload.get("rec_score"))
        texts.append(text)
        if confidence is not None:
            confidences.append(confidence)
        lines.append({"text": text, "confidence": confidence, "bbox": bbox, "polygon": None})

    return OCRResult(
        text=" ".join(texts).strip(),
        confidence=(sum(confidences) / len(confidences) if confidences else None),
        backend=backend_name,
        lines=lines,
    )


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _self_check() -> None:
    result = _recognition_results_to_ocr_result(
        [{"res": {"rec_text": "Hello world", "rec_score": 0.9}}],
        "paddleocr-rec",
        (32, 128),
    )
    assert result.text == "Hello world"
    assert result.confidence == 0.9
    assert result.lines[0]["bbox"] == [0.0, 0.0, 128.0, 32.0]


if __name__ == "__main__":
    _self_check()