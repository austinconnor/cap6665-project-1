from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmd.markdown.render import regions_to_markdown
from docmd.ordering.spatial import order_regions
from docmd.schema import PageResult
from docmd.utils.io import read_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct Markdown from OCR-enriched layout JSON.")
    parser.add_argument("ocr_json")
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()
    page = PageResult.from_dict(read_json(args.ocr_json))
    ordered = order_regions(page.regions, page.width)
    markdown = regions_to_markdown(ordered, title=Path(page.image_path).name)
    output = args.output_md or Path("outputs/markdown") / f"{Path(page.image_path).stem}.md"
    write_text(output, markdown)
    print(f"Wrote Markdown to {output}")


if __name__ == "__main__":
    main()
