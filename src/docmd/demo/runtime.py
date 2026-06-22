from __future__ import annotations

import base64
import re
import shutil
from datetime import datetime
from importlib import reload
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from docmd.schema import PageResult
from docmd.utils.images import crop_bbox, read_image
from docmd.utils.io import IMAGE_SUFFIXES, ensure_dir


def make_demo_run_dir(base_dir: str | Path = "outputs/demo_runs") -> Path:
    return ensure_dir(Path(base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"))


def save_uploaded_image(data: bytes, file_name: str, run_dir: Path) -> Path:
    target = ensure_dir(run_dir / "input") / Path(file_name).name
    target.write_bytes(data)
    return target


def save_pasted_image(pasted: str, run_dir: Path) -> Path:
    text = pasted.strip().strip("\"'")
    if not text:
        raise ValueError("Paste an image path, image URL, data URI, or base64 image.")

    target_dir = ensure_dir(run_dir / "input")
    parsed = urlparse(text)

    if parsed.scheme == "file":
        path_text = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:/", path_text):
            path_text = path_text[1:]
        return save_sample_for_run(Path(path_text), run_dir)

    if parsed.scheme in {"http", "https"}:
        response = requests.get(text, timeout=30)
        response.raise_for_status()
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            content_type = response.headers.get("content-type", "").lower()
            suffix = ".jpg" if "jpeg" in content_type else ".png"
        target = target_dir / f"pasted_url{suffix}"
        target.write_bytes(response.content)
        return target

    data_uri = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", text, re.DOTALL)
    if data_uri:
        ext = "jpg" if data_uri.group(1).lower() == "jpeg" else data_uri.group(1).lower()
        target = target_dir / f"pasted_image.{ext}"
        target.write_bytes(base64.b64decode(data_uri.group(2)))
        return target

    path = Path(text).expanduser()
    if path.exists():
        return save_sample_for_run(path, run_dir)

    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise ValueError(f"Could not resolve pasted image input: {text[:80]}") from exc
    target = target_dir / "pasted_image.png"
    target.write_bytes(data)
    return target


def save_sample_for_run(sample_path: str | Path, run_dir: Path) -> Path:
    source = Path(sample_path)
    if not source.exists():
        raise FileNotFoundError(f"Sample image not found: {source}")
    target = ensure_dir(run_dir / "input") / source.name
    shutil.copy2(source, target)
    return target


def save_region_crops(page: PageResult, run_dir: Path) -> list[Path]:
    image = read_image(page.image_path)
    crop_dir = ensure_dir(run_dir / "region_crops")
    paths: list[Path] = []
    for region in page.regions:
        crop = crop_bbox(image, region.bbox, pad=2)
        target = crop_dir / f"{region.order or 0:03d}_{region.id}_{region.class_name}.png"
        import cv2

        cv2.imwrite(str(target), crop)
        paths.append(target)
    return paths


def run_demo_pipeline(
    image_path: str | Path,
    weights: str | Path,
    run_dir: str | Path,
    mode: str,
    conf: float,
    imgsz: int,
    device: str,
    ocr_backend: str,
) -> tuple[PageResult, list[Path]]:
    import docmd.markdown.render as markdown_render
    import docmd.pipeline as pipeline

    reload(markdown_render)
    pipeline = reload(pipeline)

    run_dir = Path(run_dir)
    page = pipeline.run_page_pipeline(
        image_path=image_path,
        weights=weights,
        output_dir=run_dir,
        mode=mode,
        conf=conf,
        imgsz=imgsz,
        device=device,
        ocr_backend=ocr_backend,
        use_masks_for_ocr=(mode == "segmentation"),
    )
    return page, save_region_crops(page, run_dir)