from __future__ import annotations

from pathlib import Path
from typing import Any

from docmd.schema import LayoutRegion, PageResult
from docmd.detection.train import prepare_ultralytics_environment, resolve_device
from docmd.utils.images import image_size


def _names_from_model(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def infer_layout(
    image_path: str | Path,
    weights: str | Path,
    mode: str = "detection",
    conf: float = 0.25,
    imgsz: int = 1024,
    device: str = "cuda",
) -> PageResult:
    prepare_ultralytics_environment()
    device = resolve_device(device)
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    result = yolo.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    names = _names_from_model(result)
    width, height = image_size(image_path)
    regions: list[LayoutRegion] = []

    boxes = result.boxes
    polygons = None
    if getattr(result, "masks", None) is not None and result.masks is not None:
        polygons = result.masks.xy

    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for idx, bbox in enumerate(xyxy):
            cls = int(classes[idx])
            polygon = None
            if polygons is not None and idx < len(polygons):
                polygon = [float(v) for point in polygons[idx] for v in point]
            regions.append(
                LayoutRegion(
                    id=f"r{idx:04d}",
                    class_id=cls,
                    class_name=names.get(cls, str(cls)),
                    bbox=tuple(float(v) for v in bbox),
                    confidence=float(confs[idx]),
                    polygon=polygon,
                    source=mode,
                )
            )

    return PageResult(
        image_path=str(image_path),
        width=width,
        height=height,
        mode=mode,
        regions=regions,
        metadata={"weights": str(weights), "conf": conf, "imgsz": imgsz, "device": device},
    )
