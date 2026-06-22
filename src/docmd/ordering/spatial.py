from __future__ import annotations

from statistics import median

from docmd.schema import LayoutRegion


STRUCTURAL_CONTAINERS = ("table", "picture", "figure")
NESTED_TEXT = ("text", "list", "section-header", "caption", "formula")


def intersection_area(a: LayoutRegion, b: LayoutRegion) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area(region: LayoutRegion) -> float:
    return region.width * region.height


def class_priority(region: LayoutRegion) -> int:
    name = region.class_name.lower()
    if "title" == name:
        return 5
    if any(token in name for token in STRUCTURAL_CONTAINERS):
        return 4
    if "section-header" in name:
        return 3
    if "text" in name or "list" in name:
        return 2
    return 1


def remove_redundant_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    sorted_regions = sorted(
        regions,
        key=lambda region: (
            class_priority(region),
            region.confidence or 0.0,
            area(region),
        ),
        reverse=True,
    )
    kept: list[LayoutRegion] = []
    for region in sorted_regions:
        region_name = region.class_name.lower()
        region_area = max(1.0, area(region))
        redundant = False
        for other in kept:
            other_name = other.class_name.lower()
            overlap = intersection_area(region, other)
            smaller_overlap = overlap / max(1.0, min(region_area, area(other)))
            contained = overlap / region_area
            same_class = region_name == other_name
            nested_in_container = (
                any(token in other_name for token in STRUCTURAL_CONTAINERS)
                and any(token in region_name for token in NESTED_TEXT)
                and contained > 0.72
            )
            duplicate = same_class and smaller_overlap > 0.82
            if nested_in_container or duplicate:
                redundant = True
                break
        if not redundant:
            kept.append(region)
    return kept


def estimate_columns(regions: list[LayoutRegion], page_width: int | None) -> list[tuple[float, float]]:
    if not regions or page_width is None:
        return []
    centers = sorted(region.center_x for region in regions)
    if len(centers) < 4:
        return [(0, float(page_width))]
    widths = [region.width for region in regions if region.width > 0]
    typical_width = median(widths) if widths else page_width
    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    largest_gap, idx = max(gaps, default=(0, 0))
    if largest_gap < max(typical_width * 0.75, page_width * 0.12):
        return [(0, float(page_width))]
    split = (centers[idx] + centers[idx + 1]) / 2
    return [(0, split), (split, float(page_width))]


def assign_column(region: LayoutRegion, columns: list[tuple[float, float]]) -> int:
    if not columns:
        return 0
    for idx, (left, right) in enumerate(columns):
        if left <= region.center_x <= right:
            return idx
    return min(range(len(columns)), key=lambda idx: abs(region.center_x - sum(columns[idx]) / 2))


def order_regions(regions: list[LayoutRegion], page_width: int | None = None) -> list[LayoutRegion]:
    filtered = remove_redundant_regions(
        [region for region in regions if region.width > 1 and region.height > 1]
    )
    columns = estimate_columns(filtered, page_width)
    ordered = sorted(
        filtered,
        key=lambda region: (
            assign_column(region, columns),
            round(region.y1 / 12) * 12,
            region.x1,
            region.y1,
        ),
    )
    for idx, region in enumerate(ordered, start=1):
        region.order = idx
    return ordered
