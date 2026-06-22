from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmd.ocr.extract import ocr_from_layout_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR over regions from a layout JSON file.")
    parser.add_argument("layout_json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--backend", default="paddleocr", choices=["paddleocr", "stub"])
    parser.add_argument("--use-masks", action="store_true")
    args = parser.parse_args()
    output = args.output_json or Path("outputs/ocr_json") / Path(args.layout_json).name
    ocr_from_layout_json(args.layout_json, output, args.backend, args.use_masks)
    print(f"Wrote OCR JSON to {output}")


if __name__ == "__main__":
    main()