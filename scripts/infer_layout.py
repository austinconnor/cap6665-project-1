from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmd.detection.infer import infer_layout
from docmd.utils.io import image_paths, write_json
from docmd.visualization.draw import draw_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO layout inference and save JSON/visual overlays.")
    parser.add_argument("input", help="Page image or folder of page images.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--mode", choices=["detection", "segmentation"], default="detection")
    parser.add_argument("--output-dir", default="outputs/layout_inference")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for image_path in image_paths(args.input):
        page = infer_layout(image_path, args.weights, args.mode, args.conf, args.imgsz, args.device)
        stem = Path(image_path).stem
        write_json(Path(args.output_dir) / "layout_json" / f"{stem}.json", page.to_dict())
        draw_layout(image_path, page.regions, Path(args.output_dir) / "visualizations" / f"{stem}.png")
        print(f"Wrote layout for {image_path}")


if __name__ == "__main__":
    main()
