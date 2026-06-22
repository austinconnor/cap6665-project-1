from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docmd.evaluation.metrics import evaluate_markdown_file, write_report


def evaluate_yolo(weights: str, data: str, task: str, device: str, imgsz: int) -> dict:
    from docmd.detection.train import prepare_ultralytics_environment

    prepare_ultralytics_environment()
    from ultralytics import YOLO

    model = YOLO(weights, task=task)
    metrics = model.val(data=data, device=device, imgsz=imgsz, plots=True)
    return {
        "weights": weights,
        "data": data,
        "task": task,
        "device": device,
        "results_dict": getattr(metrics, "results_dict", {}),
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO metrics or reconstructed Markdown.")
    parser.add_argument("--prediction", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--output", default="outputs/evaluation/markdown_report.json")
    args = parser.parse_args()

    if args.yolo_weights:
        if not args.data:
            raise SystemExit("--data is required with --yolo-weights")
        report = evaluate_yolo(args.yolo_weights, args.data, args.task, args.device, args.imgsz)
    else:
        if not args.prediction or not args.reference:
            raise SystemExit("--prediction and --reference are required for Markdown evaluation")
        pred = Path(args.prediction)
        ref = Path(args.reference)
        if pred.is_file() and ref.is_file():
            report = evaluate_markdown_file(pred, ref)
        else:
            report = {
                "items": [
                    evaluate_markdown_file(p, ref / p.name)
                    for p in sorted(pred.glob("*.md"))
                    if (ref / p.name).exists()
                ]
            }
    write_report(report, args.output)
    print(f"Wrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
