# Colab Upload and Training Setup

Archive created locally:

```text
docmd-reconstruction-colab.zip
```

Upload that file to Google Drive, for example:

```text
MyDrive/docmd-reconstruction-colab.zip
```

The archive contains the project code and the original local dataset under `data/train`, `data/val`, and `data/test`. It intentionally does not include `.venv`, `.uv-cache`, `outputs`, `runs`, or converted YOLO folders. Colab will regenerate YOLO-format datasets from the included COCO annotations.

## Colab Cells

Select an A100 runtime first:

```text
Runtime -> Change runtime type -> GPU
```

Then run:

```python
!nvidia-smi
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

Mount Drive and unzip the project to Colab local disk:

```python
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ZIP = "/content/drive/MyDrive/docmd-reconstruction-colab.zip"
LOCAL_PROJECT = "/content/docmd-reconstruction"

!rm -rf "$LOCAL_PROJECT"
!mkdir -p "$LOCAL_PROJECT"
!unzip -q "$DRIVE_ZIP" -d "$LOCAL_PROJECT"
%cd "$LOCAL_PROJECT"
```

Install dependencies with `uv`:

```python
!python -m pip install -q uv
!uv pip install --system -e .
!python -c "from ultralytics import YOLO; import torch; print(torch.__version__, torch.cuda.is_available())"
```

Regenerate YOLO datasets from the included COCO data:

```python
!python scripts/convert_coco_to_yolo_detection.py
!python scripts/convert_coco_to_yolo_segmentation.py
```

Start TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir outputs/training
```

Train detection on full data:

```python
import subprocess, sys
subprocess.Popen([
    sys.executable,
    "scripts/csv_to_tensorboard.py",
    "outputs/training/yolo26m_detection_full_a100",
    "--interval",
    "15",
])
```

```bash
python scripts/train_yolo_detection.py \
  --model models/yolo26m.pt \
  --epochs 80 \
  --batch 16 \
  --imgsz 1024 \
  --device cuda \
  --workers 8 \
  --fraction 1.0 \
  --patience 25 \
  --project outputs/training \
  --name yolo26m_detection_full_a100 \
  --cos-lr \
  --save-period 5
```

Validate detection:

```bash
python scripts/evaluate.py \
  --yolo-weights outputs/training/yolo26m_detection_full_a100/weights/best.pt \
  --data data/yolo_detection/dataset_detection.yaml \
  --task detect \
  --device cuda \
  --imgsz 1024 \
  --output outputs/evaluation/yolo26m_detection_full_a100_val.json
```

Train segmentation on full data:

```python
import subprocess, sys
subprocess.Popen([
    sys.executable,
    "scripts/csv_to_tensorboard.py",
    "outputs/training/yolo26m_segmentation_full_a100",
    "--interval",
    "15",
])
```

```bash
python scripts/train_yolo_segmentation.py \
  --model models/yolo26m-seg.pt \
  --epochs 60 \
  --batch 8 \
  --imgsz 1024 \
  --device cuda \
  --workers 8 \
  --fraction 1.0 \
  --patience 20 \
  --project outputs/training \
  --name yolo26m_segmentation_full_a100 \
  --cos-lr \
  --save-period 5
```

Validate segmentation:

```bash
python scripts/evaluate.py \
  --yolo-weights outputs/training/yolo26m_segmentation_full_a100/weights/best.pt \
  --data data/yolo_segmentation/dataset_segmentation.yaml \
  --task segment \
  --device cuda \
  --imgsz 1024 \
  --output outputs/evaluation/yolo26m_segmentation_full_a100_val.json
```

Sync results back to Drive:

```python
DRIVE_RESULTS = "/content/drive/MyDrive/docmd-training-results"
!mkdir -p "$DRIVE_RESULTS"
!rsync -a outputs/training "$DRIVE_RESULTS/"
!rsync -a outputs/evaluation "$DRIVE_RESULTS/"
```

Resume after interruption:

```bash
python scripts/train_yolo_detection.py \
  --model outputs/training/yolo26m_detection_full_a100/weights/last.pt \
  --resume \
  --device cuda
```

```bash
python scripts/train_yolo_segmentation.py \
  --model outputs/training/yolo26m_segmentation_full_a100/weights/last.pt \
  --resume \
  --device cuda
```
