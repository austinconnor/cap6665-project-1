from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BBox = tuple[float, float, float, float]
Polygon = list[float]


@dataclass(slots=True)
class LayoutRegion:
    id: str
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float | None = None
    polygon: Polygon | None = None
    mask: dict[str, Any] | None = None
    source: str = "yolo"
    order: int | None = None
    text: str = ""
    ocr_confidence: float | None = None
    ocr_lines: list[dict[str, Any]] = field(default_factory=list)
    asset_path: str | None = None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutRegion":
        bbox = tuple(float(v) for v in data["bbox"])
        return cls(**{**data, "bbox": bbox})


@dataclass(slots=True)
class PageResult:
    image_path: str
    width: int | None = None
    height: int | None = None
    mode: str = "detection"
    regions: list[LayoutRegion] = field(default_factory=list)
    markdown: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["regions"] = [region.to_dict() for region in self.regions]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageResult":
        regions = [LayoutRegion.from_dict(item) for item in data.get("regions", [])]
        return cls(**{**data, "regions": regions})


def stem_for_path(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_")
