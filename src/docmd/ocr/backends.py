from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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

    def _recognize_chunk(self, image: np.ndarray, bbox: list[float]) -> OCRResult:
        prepared = self._prepare_image(image)
        result = self.recognizer.predict(prepared)
        return _recognition_results_to_ocr_result(result, self.name, bbox)

    def recognize(self, image: np.ndarray) -> OCRResult:
        line_texts: list[str] = []
        confidences: list[float] = []
        lines: list[dict[str, object]] = []

        for line_image, line_bbox in _split_text_lines(image):
            chunk_texts: list[str] = []
            for chunk_image, chunk_bbox in _split_line_chunks(line_image, line_bbox):
                result = self._recognize_chunk(chunk_image, chunk_bbox)
                if not result.text:
                    continue
                chunk_texts.append(result.text)
                if result.confidence is not None:
                    confidences.append(result.confidence)
                lines.extend(result.lines)
            if chunk_texts:
                line_texts.append(" ".join(chunk_texts))

        return OCRResult(
            text="\n".join(line_texts).strip(),
            confidence=(sum(confidences) / len(confidences) if confidences else None),
            backend=self.name,
            lines=lines,
        )


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
    bbox: list[float] | None = None,
) -> OCRResult:
    texts: list[str] = []
    confidences: list[float] = []
    lines: list[dict[str, object]] = []

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


def _threshold_text(image: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _value, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = threshold.shape[:2]
    horizontal = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, width // 5), 1)),
    )
    vertical = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, height // 5))),
    )
    return cv2.subtract(threshold, cv2.bitwise_or(horizontal, vertical))


def _projection_bands(has_ink: np.ndarray, max_gap: int, min_len: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    last: int | None = None
    gap = 0
    for index, value in enumerate(has_ink):
        if bool(value):
            if start is None:
                start = index
            last = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                if last is not None and last - start + 1 >= min_len:
                    bands.append((start, last + 1))
                start = None
                last = None
                gap = 0
    if start is not None and last is not None and last - start + 1 >= min_len:
        bands.append((start, last + 1))
    return bands


# ponytail: whitespace chunker keeps PP-OCR rec usable without adding a text detector.
# Upgrade to a stronger local line segmenter if noisy scans need it.
def _split_text_lines(image: np.ndarray) -> list[tuple[np.ndarray, list[float]]]:
    height, width = image.shape[:2]
    threshold = _threshold_text(image)
    row_ink = (threshold > 0).sum(axis=1)
    min_row_ink = max(2, int(max(width * 0.003, row_ink.max(initial=0) * 0.10)))
    row_has_ink = row_ink > min_row_ink
    bands = _projection_bands(row_has_ink, max_gap=2, min_len=2)
    if not bands:
        return [(image, [0.0, 0.0, float(width), float(height)])]

    lines: list[tuple[np.ndarray, list[float]]] = []
    for top, bottom in bands:
        pad = 2
        y1 = max(0, top - pad)
        y2 = min(height, bottom + pad)
        lines.append((image[y1:y2, :], [0.0, float(y1), float(width), float(y2)]))
    return lines


def _split_line_chunks(
    image: np.ndarray,
    line_bbox: list[float],
    max_width_ratio: float = 10.0,
) -> list[tuple[np.ndarray, list[float]]]:
    height, width = image.shape[:2]
    if width <= max(32, int(max_width_ratio * max(1, height))):
        return [(image, line_bbox)]

    threshold = _threshold_text(image)
    col_has_ink = (threshold > 0).sum(axis=0) > 0
    word_bands = _projection_bands(col_has_ink, max_gap=2, min_len=1)
    if not word_bands:
        return [(image, line_bbox)]

    max_width = max(32, int(max_width_ratio * max(1, height)))
    chunks: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for start, end in word_bands:
        if current_start is None:
            current_start, current_end = start, end
        elif end - current_start <= max_width:
            current_end = end
        else:
            chunks.append((current_start, int(current_end)))
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        chunks.append((current_start, current_end))

    out: list[tuple[np.ndarray, list[float]]] = []
    x_offset, y1, _x2, y2 = line_bbox
    for start, end in chunks:
        pad = 4
        x1 = max(0, start - pad)
        x2 = min(width, end + pad)
        out.append((image[:, x1:x2], [x_offset + x1, y1, x_offset + x2, y2]))
    return out


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _self_check() -> None:
    result = _recognition_results_to_ocr_result(
        [{"res": {"rec_text": "Hello world", "rec_score": 0.9}}],
        "paddleocr-rec",
        [0.0, 0.0, 128.0, 32.0],
    )
    assert result.text == "Hello world"
    assert result.confidence == 0.9
    assert result.lines[0]["bbox"] == [0.0, 0.0, 128.0, 32.0]

    image = np.full((14, 487, 3), 255, dtype=np.uint8)
    image[4:10, 5:55] = 0
    image[4:10, 160:210] = 0
    image[4:10, 320:370] = 0
    chunks = _split_line_chunks(image, [0.0, 0.0, 487.0, 14.0])
    assert len(chunks) == 3

    two_line = np.full((40, 220, 3), 255, dtype=np.uint8)
    for x in range(5, 180, 30):
        two_line[5:12, x : x + 18] = 0
    for x in range(5, 120, 30):
        two_line[25:32, x : x + 18] = 0
    two_line[14:23, 100:104] = 0
    assert len(_split_text_lines(two_line)) == 2


if __name__ == "__main__":
    _self_check()
