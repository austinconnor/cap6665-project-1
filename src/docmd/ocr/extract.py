from __future__ import annotations

from pathlib import Path

from docmd.ocr.backends import OCRBackend, create_ocr_backend
from docmd.schema import LayoutRegion, PageResult
from docmd.utils.images import mask_crop, read_image


TEXT_CLASS_HINTS = {
    "caption",
    "footnote",
    "formula",
    "list-item",
    "list",
    "page-footer",
    "page-header",
    "section-header",
    "text",
    "title",
    "table",
}


def is_text_region(region: LayoutRegion) -> bool:
    name = region.class_name.lower()
    return any(hint in name for hint in TEXT_CLASS_HINTS)


def run_region_ocr(
    page: PageResult,
    backend: OCRBackend | None = None,
    backend_name: str = "paddleocr",
    use_masks: bool = False,
    crop_pad: int = 4,
) -> PageResult:
    image = read_image(page.image_path)
    backend = backend or create_ocr_backend(backend_name)
    for region in page.regions:
        if not is_text_region(region):
            continue
        crop = mask_crop(image, region.bbox, region.polygon if use_masks else None, pad=crop_pad)
        result = backend.recognize(crop)
        region.text = result.text
        region.ocr_confidence = result.confidence
        region.ocr_lines = result.lines
    page.metadata["ocr_backend"] = getattr(backend, "name", backend_name)
    page.metadata["ocr_use_masks"] = use_masks
    return page


def ocr_from_layout_json(
    layout_json: str | Path,
    output_json: str | Path,
    backend_name: str = "paddleocr",
    use_masks: bool = False,
) -> PageResult:
    from docmd.schema import PageResult
    from docmd.utils.io import read_json, write_json

    page = PageResult.from_dict(read_json(layout_json))
    page = run_region_ocr(page, backend_name=backend_name, use_masks=use_masks)
    write_json(output_json, page.to_dict())
    return page
