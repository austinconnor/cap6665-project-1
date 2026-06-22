from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmd.dataset.coco_to_yolo import convert_coco_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert local COCO annotations to YOLO segmentation format.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="data/yolo_segmentation")
    parser.add_argument("--image-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = parser.parse_args()
    summary = convert_coco_dataset(
        data_root=args.data_root,
        output_root=args.output_root,
        task="segment",
        image_mode=args.image_mode,
    )
    print(summary)


if __name__ == "__main__":
    main()
