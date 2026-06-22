from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def numeric(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def write_rows(csv_path: Path, writer: SummaryWriter, seen_epochs: set[int]) -> int:
    if not csv_path.exists():
        return 0
    wrote = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            epoch_value = numeric((row.get("epoch") or "").strip())
            if epoch_value is None:
                continue
            epoch = int(epoch_value)
            if epoch in seen_epochs:
                continue
            for key, value in row.items():
                clean_key = key.strip()
                if clean_key in {"epoch", "time"}:
                    continue
                scalar = numeric((value or "").strip())
                if scalar is not None:
                    writer.add_scalar(clean_key, scalar, epoch)
            elapsed = numeric((row.get("time") or "").strip())
            if elapsed is not None:
                writer.add_scalar("time/seconds", elapsed, epoch)
            seen_epochs.add(epoch)
            wrote += 1
    if wrote:
        writer.flush()
    return wrote


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror Ultralytics results.csv into TensorBoard events.")
    parser.add_argument("run_dir", help="Ultralytics run directory containing results.csv.")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    csv_path = run_dir / "results.csv"
    event_dir = run_dir / "tb_from_csv"
    event_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(event_dir))
    seen_epochs: set[int] = set()

    try:
        while True:
            wrote = write_rows(csv_path, writer, seen_epochs)
            if args.once:
                print(f"Wrote {wrote} epoch row(s) from {csv_path} to {event_dir}")
                return
            time.sleep(args.interval)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
