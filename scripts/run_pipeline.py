from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmd.pipeline import run_batch_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO + OCR + reading-order + Markdown pipeline.")
    parser.add_argument("input", help="Page image or folder.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--mode", choices=["detection", "segmentation"], default="detection")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-backend", default="paddleocr", choices=["paddleocr", "stub"])
    parser.add_argument("--use-masks-for-ocr", action="store_true")
    args = parser.parse_args()
    results = run_batch_pipeline(
        input_path=args.input,
        weights=args.weights,
        output_dir=args.output_dir,
        mode=args.mode,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        ocr_backend=args.ocr_backend,
        use_masks_for_ocr=args.use_masks_for_ocr or None,
    )
    print(f"Processed {len(results)} page(s).")


if __name__ == "__main__":
    main()