from __future__ import annotations

import re
from statistics import median

from docmd.schema import LayoutRegion


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def list_lines(text: str) -> list[str]:
    pieces = [piece.strip(" -\t") for piece in re.split(r"(?:\n| {2,}|•)", text) if piece.strip(" -\t")]
    return [f"- {piece}" for piece in pieces] or ["- " + clean_text(text)]


def escape_table_cell(text: str) -> str:
    return clean_text(text).replace("|", "\\|")


def markdown_table(rows: list[list[str]]) -> str:
    rows = [[escape_table_cell(cell) for cell in row] for row in rows if any(clean_text(cell) for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if width == 1:
        rows = [[str(i + 1), row[0]] for i, row in enumerate(rows)]
        rows.insert(0, ["#", "Text"])
    header = rows[0]
    body = rows[1:] or [[""] * width]
    sep = ["---"] * len(header)
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
            *("| " + " | ".join(row) + " |" for row in body),
        ]
    )


def line_bbox(line: dict[str, object]) -> tuple[float, float, float, float] | None:
    bbox = line.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def table_from_line_geometry(lines: list[dict[str, object]]) -> str:
    items: list[dict[str, object]] = []
    for line in lines:
        text = clean_text(str(line.get("text", "")))
        bbox = line_bbox(line)
        if not text or bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        items.append(
            {
                "text": text,
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "height": max(1.0, y2 - y1),
            }
        )
    if len(items) < 2:
        return ""

    row_threshold = max(10.0, median(float(item["height"]) for item in items) * 0.85)
    rows: list[list[dict[str, object]]] = []
    for item in sorted(items, key=lambda item: (float(item["y"]), float(item["x"]))):
        if not rows:
            rows.append([item])
            continue
        row_y = sum(float(existing["y"]) for existing in rows[-1]) / len(rows[-1])
        if abs(float(item["y"]) - row_y) <= row_threshold:
            rows[-1].append(item)
        else:
            rows.append([item])

    first_row = sorted(rows[0], key=lambda item: float(item["x"]))
    first_text = " ".join(str(item["text"]).lower() for item in first_row)
    if 2 <= len(first_row) <= 6 and ("field" in first_text or "description" in first_text):
        centers = [float(item["x"]) for item in first_row]
    else:
        centers: list[float] = []
        for item in sorted(items, key=lambda item: float(item["x"])):
            x = float(item["x"])
            nearest = min(range(len(centers)), key=lambda i: abs(x - centers[i])) if centers else None
            if nearest is not None and abs(x - centers[nearest]) <= 120:
                centers[nearest] = (centers[nearest] + x) / 2
            else:
                centers.append(x)
        centers.sort()
    if len(centers) < 2:
        return ""
    if len(centers) > 6:
        return ""

    rendered_rows: list[list[str]] = []
    for row in rows:
        cells = [""] * len(centers)
        for item in sorted(row, key=lambda item: float(item["x"])):
            index = min(range(len(centers)), key=lambda i: abs(float(item["x"]) - centers[i]))
            cells[index] = clean_text(f"{cells[index]} {item['text']}")
        if any(cells):
            rendered_rows.append(cells)

    if len(rendered_rows) < 2:
        return ""
    return markdown_table(rendered_rows)


def table_from_definition_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    keys = [
        "Text 'description'",
        "From file",
        "Library",
        "Text",
        "To file",
        "*CURLIB",
    ]
    keys = sorted(keys, key=len, reverse=True)
    pattern = re.compile(rf"(?<!\w)({'|'.join(re.escape(key) for key in keys)})(?!\w)")
    matches = list(pattern.finditer(text))
    rows: list[list[str]] = [["Field", "Description"]]
    for index, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        description = text[start:end].strip(" :-")
        if description and len(description) > 8:
            rows.append([label, description])
    return markdown_table(rows) if len(rows) > 1 else ""


def table_placeholder(region: LayoutRegion) -> str:
    text = clean_text(region.text)
    if "|" in text:
        return text
    if "field name" in text.lower() and "description" in text.lower():
        definition_table = table_from_definition_text(text)
        if definition_table:
            return definition_table
    geometry_table = table_from_line_geometry(region.ocr_lines)
    if geometry_table:
        return geometry_table
    definition_table = table_from_definition_text(text)
    if definition_table:
        return definition_table
    if text:
        return f"```text\n{text}\n```"
    return "<!-- Table region detected; reliable cell reconstruction not available. -->"


def render_region(region: LayoutRegion) -> str:
    name = region.class_name.lower()
    text = clean_text(region.text)
    if "title" == name:
        return f"# {text}" if text else "# Untitled"
    if "section-header" in name or "header" in name and "page" not in name:
        return f"## {text}" if text else "## Section"
    if "list" in name:
        return "\n".join(list_lines(text)) if text else "- "
    if "table" in name:
        return table_placeholder(region)
    if "picture" in name or "figure" in name:
        alt = clean_text(region.text) or f"{region.class_name} {region.order or region.id}"
        mask_asset = region.mask.get("markdown_asset_path") if isinstance(region.mask, dict) else None
        asset_path = getattr(region, "asset_path", None) or mask_asset
        return f"![{alt}]({asset_path})" if asset_path else f"![{alt}](figures/{region.id}.png)"
    if "caption" in name:
        return f"*{text}*" if text else "*Caption*"
    if "formula" in name:
        return f"$$\n{text}\n$$" if text else "$$ $$"
    if "page-footer" in name or "page-header" in name:
        return f"<!-- {region.class_name}: {text} -->" if text else ""
    return text


def regions_to_markdown(regions: list[LayoutRegion], title: str | None = None) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"<!-- Source: {title} -->")
    for region in regions:
        block = render_region(region).strip()
        if block:
            parts.append(block)
    return "\n\n".join(parts).strip() + "\n"
