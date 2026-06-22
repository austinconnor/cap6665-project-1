from __future__ import annotations

from datetime import datetime
from pathlib import Path

from docmd.detection.infer import infer_layout
from docmd.markdown.render import regions_to_markdown
from docmd.ocr.extract import run_region_ocr
from docmd.ordering.spatial import order_regions
from docmd.schema import LayoutRegion, PageResult, stem_for_path
from docmd.utils.images import crop_bbox, read_image
from docmd.utils.io import ensure_dir, image_paths, write_json, write_text
from docmd.visualization.draw import draw_layout


def save_markdown_assets(image_path: str | Path, regions: list[LayoutRegion], output_dir: Path, stem: str) -> None:
    image_regions = [
        region
        for region in regions
        if "picture" in region.class_name.lower() or "figure" in region.class_name.lower()
    ]
    if not image_regions:
        return
    image = read_image(image_path)
    asset_dir = ensure_dir(output_dir / "markdown" / "assets" / stem)
    legacy_dir = ensure_dir(output_dir / "markdown" / "figures")
    import cv2

    for region in image_regions:
        crop = crop_bbox(image, region.bbox, pad=8)
        target = asset_dir / f"{region.order or 0:03d}_{region.id}.png"
        legacy_target = legacy_dir / f"{region.id}.png"
        cv2.imwrite(str(target), crop)
        cv2.imwrite(str(legacy_target), crop)
        asset_path = f"assets/{stem}/{target.name}"
        if hasattr(region, "asset_path"):
            region.asset_path = asset_path
        mask = region.mask if isinstance(region.mask, dict) else {}
        mask["markdown_asset_path"] = asset_path
        region.mask = mask


def run_page_pipeline(
    image_path: str | Path,
    weights: str | Path,
    output_dir: str | Path,
    mode: str = "detection",
    conf: float = 0.25,
    imgsz: int = 1024,
    device: str = "cuda",
    ocr_backend: str = "paddleocr",
    use_masks_for_ocr: bool | None = None,
) -> PageResult:
    image_path = Path(image_path)
    output_dir = ensure_dir(output_dir)
    stem = stem_for_path(image_path)
    use_masks = mode == "segmentation" if use_masks_for_ocr is None else use_masks_for_ocr

    page = infer_layout(image_path, weights, mode=mode, conf=conf, imgsz=imgsz, device=device)
    write_json(output_dir / "layout_json" / f"{stem}.json", page.to_dict())
    draw_layout(image_path, page.regions, output_dir / "visualizations" / f"{stem}_layout.png")

    page = run_region_ocr(page, backend_name=ocr_backend, use_masks=use_masks)
    write_json(output_dir / "ocr_json" / f"{stem}.json", page.to_dict())

    page.regions = order_regions(page.regions, page.width)
    save_markdown_assets(image_path, page.regions, output_dir, stem)
    draw_layout(
        image_path,
        page.regions,
        output_dir / "visualizations" / f"{stem}_ordered.png",
        show_order=True,
    )

    page.markdown = regions_to_markdown(page.regions, title=image_path.name)
    write_text(output_dir / "markdown" / f"{stem}.md", page.markdown)
    write_json(output_dir / "page_results" / f"{stem}.json", page.to_dict())
    return page


def run_batch_pipeline(
    input_path: str | Path,
    weights: str | Path,
    output_dir: str | Path | None = None,
    **kwargs: object,
) -> list[PageResult]:
    if output_dir is None:
        output_dir = Path("outputs") / "pipeline_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = image_paths(input_path)
    return [run_page_pipeline(path, weights, output_dir, **kwargs) for path in paths]
