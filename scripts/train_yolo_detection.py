from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmd.detection.train import train_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26m document layout detector on CUDA.")
    parser.add_argument("--data", default="data/yolo_detection/dataset_detection.yaml")
    parser.add_argument("--model", default="models/yolo26m.pt")
    parser.add_argument("--project", default="outputs/training")
    parser.add_argument("--name", default="yolo26m_detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--erasing", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.02)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--no-val", action="store_true", help="Skip validation during training; run scripts/evaluate.py afterward.")
    parser.add_argument("--save-period", type=int, default=1, help="Save checkpoints every N epochs.")
    parser.add_argument("--resume", action="store_true", help="Resume from the model checkpoint passed via --model.")
    parser.add_argument("--cache", default=None, choices=["ram", "disk"], help="Cache images for faster training.")
    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data,
        model=args.model,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        task="detect",
        extra_args={
            "fraction": args.fraction,
            "patience": args.patience,
            "mosaic": args.mosaic,
            "fliplr": args.fliplr,
            "flipud": args.flipud,
            "erasing": args.erasing,
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "cos_lr": args.cos_lr,
            "val": not args.no_val,
            "save_period": args.save_period,
            "resume": args.resume,
            **({"cache": args.cache} if args.cache else {}),
        },
    )


if __name__ == "__main__":
    main()
