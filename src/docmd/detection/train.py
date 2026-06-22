from __future__ import annotations

from pathlib import Path
from typing import Any


def prepare_ultralytics_environment() -> None:
    import os

    config_dir = Path("outputs/ultralytics_config").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir / "matplotlib"))


def cuda_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def resolve_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested.startswith("cuda") or requested == "0":
        if not cuda_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment cannot see a CUDA GPU. "
                "Verify that the active interpreter is .venv\\Scripts\\python.exe and that "
                "torch is a CUDA build, for example torch.__version__ should include '+cu'."
            )
    return device


def assert_cuda_available(device: str) -> None:
    if not str(device).startswith("cuda") and str(device) != "0":
        return
    if not cuda_available():
        raise RuntimeError(
            "Training was requested with device='cuda', but torch.cuda.is_available() is False. "
            "Check the CUDA-enabled PyTorch install in the active venv."
        )


def train_yolo(
    data_yaml: str | Path,
    model: str,
    project: str | Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str = "cuda",
    task: str | None = None,
    workers: int = 8,
    extra_args: dict[str, Any] | None = None,
) -> Any:
    prepare_ultralytics_environment()
    device = resolve_device(device)
    assert_cuda_available(device)
    from ultralytics import YOLO

    yolo = YOLO(model, task=task) if task else YOLO(model)
    args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(project),
        "name": name,
        "workers": workers,
        "exist_ok": True,
    }
    if extra_args:
        args.update(extra_args)
    return yolo.train(**args)
