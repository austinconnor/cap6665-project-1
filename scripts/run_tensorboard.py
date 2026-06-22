from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    logdir = Path("outputs/training").resolve()
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        "127.0.0.1",
        "--port",
        "6006",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
