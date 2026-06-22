from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from docmd.utils.io import ensure_dir, read_json, write_json


SPLITS = ("train", "val", "test")


@dataclass(slots=True)
class ConversionSummary:
    task: str
    output_dir: str
    splits: dict[str, dict[str, int]]
    names: list[str]
    skipped_annotations: int = 0


def coco_json_for_split(data_root: Path, split: str) -> Path:
    return data_root / split / f"{split}.json"


def source_image_for(data_root: Path, split: str, file_name: str) -> Path:
    direct = data_root / split / file_name
    if direct.exists():
        return direct
    png = data_root / split / "PNG" / file_name
    if png.exists():
        return png
    raise FileNotFoundError(f"Could not find {file_name} under {data_root / split}")


def category_mapping(categories: list[dict[str, Any]]) -> tuple[dict[int, int], list[str]]:
    ordered = sorted(categories, key=lambda item: int(item["id"]))
    id_to_index = {int(category["id"]): idx for idx, category in enumerate(ordered)}
    names = [str(category["name"]) for category in ordered]
    return id_to_index, names


def normalize_bbox(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    return cx, cy, w / width, h / height


def clean_polygon(segmentation: Any, width: float, height: float) -> list[float] | None:
    if not isinstance(segmentation, list):
        return None
    candidates = [poly for poly in segmentation if isinstance(poly, list) and len(poly) >= 6]
    if not candidates:
        return None
    polygon = max(candidates, key=len)
    if len(polygon) % 2 == 1:
        polygon = polygon[:-1]
    normalized: list[float] = []
    for idx, value in enumerate(polygon):
        denom = width if idx % 2 == 0 else height
        normalized.append(min(1.0, max(0.0, float(value) / denom)))
    return normalized if len(normalized) >= 6 else None


def grouped_annotations(annotations: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def convert_split(
    data_root: Path,
    output_root: Path,
    split: str,
    task: str,
    image_mode: str,
) -> tuple[dict[str, int], list[str], int]:
    coco = read_json(coco_json_for_split(data_root, split))
    id_to_index, names = category_mapping(coco["categories"])
    annotations_by_image = grouped_annotations(coco.get("annotations", []))
    image_dir = ensure_dir(output_root / "images" / split)
    label_dir = ensure_dir(output_root / "labels" / split)
    skipped = 0

    for image in coco.get("images", []):
        image_id = int(image["id"])
        width = float(image["width"])
        height = float(image["height"])
        file_name = str(image["file_name"])
        source = source_image_for(data_root, split, file_name)
        target_image = image_dir / source.name
        if image_mode == "copy":
            if not target_image.exists():
                shutil.copy2(source, target_image)
        elif image_mode == "hardlink":
            if not target_image.exists():
                try:
                    target_image.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, target_image)
        elif image_mode == "symlink":
            if not target_image.exists():
                try:
                    target_image.symlink_to(source)
                except OSError:
                    shutil.copy2(source, target_image)
        else:
            raise ValueError(f"Unsupported image_mode: {image_mode}")

        lines: list[str] = []
        for ann in annotations_by_image.get(image_id, []):
            class_idx = id_to_index[int(ann["category_id"])]
            if task == "detect":
                cx, cy, bw, bh = normalize_bbox(ann["bbox"], width, height)
                lines.append(f"{class_idx} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            elif task == "segment":
                polygon = clean_polygon(ann.get("segmentation"), width, height)
                if polygon is None:
                    skipped += 1
                    continue
                points = " ".join(f"{value:.8f}" for value in polygon)
                lines.append(f"{class_idx} {points}")
            else:
                raise ValueError(f"Unsupported task: {task}")
        (label_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "labels": len(list(label_dir.glob("*.txt"))),
    }
    return summary, names, skipped


def write_dataset_yaml(output_root: Path, names: list[str], task: str) -> Path:
    yaml_path = output_root / f"dataset_{'segmentation' if task == 'segment' else 'detection'}.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"task: {'segment' if task == 'segment' else 'detect'}\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def convert_coco_dataset(
    data_root: str | Path,
    output_root: str | Path,
    task: str,
    copy_images: bool = True,
    image_mode: str | None = None,
    splits: Iterable[str] = SPLITS,
) -> ConversionSummary:
    data_root = Path(data_root)
    output_root = ensure_dir(output_root)
    split_summaries: dict[str, dict[str, int]] = {}
    names: list[str] = []
    skipped = 0
    resolved_image_mode = image_mode or ("copy" if copy_images else "hardlink")
    for split in splits:
        summary, split_names, split_skipped = convert_split(
            data_root=data_root,
            output_root=output_root,
            split=split,
            task=task,
            image_mode=resolved_image_mode,
        )
        if not names:
            names = split_names
        split_summaries[split] = summary
        skipped += split_skipped

    yaml_path = write_dataset_yaml(output_root, names, task)
    summary = ConversionSummary(
        task=task,
        output_dir=str(output_root),
        splits=split_summaries,
        names=names,
        skipped_annotations=skipped,
    )
    write_json(output_root / "conversion_summary.json", {**asdict(summary), "dataset_yaml": str(yaml_path)})
    return summary
