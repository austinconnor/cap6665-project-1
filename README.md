# Document-to-Markdown Reconstruction

This project converts scanned document pages into Markdown with a small, reproducible pipeline:

1. YOLO layout detection or segmentation.
2. PaddleOCR PP-OCRv6 tiny OCR.
3. Geometry-based reading order.
4. Markdown rendering.

No VLM comparison path is included.

## Included Models

The repo packages the small YOLO weights needed for demo and training startup:

```text
models/yolo26m_detection_docaug_b8.pt   # trained detection checkpoint
models/yolo26m_segmentation_proto.pt    # trained segmentation checkpoint
models/yolo26m.pt                       # base detection model for training
models/yolo26m-seg.pt                   # base segmentation model for training
```

OCR uses PaddleOCR with `text_detection_model_name="PP-OCRv6_tiny_det"`, matching the safetensors model at https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_safetensors. Text recognition uses the matching `PP-OCRv6_tiny_rec` model because detection alone does not produce text.

## Setup

Use `uv` for Python commands:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv sync
```

For GPU runs, install a CUDA-compatible PyTorch build for your machine if the default wheel is not enough.

## Demo

Run the Streamlit demo:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/run_demo.py
```

The demo uses images in `demo_samples/`, packaged YOLO weights from `models/`, and PaddleOCR PP-OCRv6 tiny OCR. Outputs are written to `outputs/demo_runs/` and are intentionally gitignored.

## Batch Inference

Detection pipeline:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/run_pipeline.py demo_samples --weights models/yolo26m_detection_docaug_b8.pt --mode detection --device cpu
```

Segmentation pipeline:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/run_pipeline.py demo_samples --weights models/yolo26m_segmentation_proto.pt --mode segmentation --device cpu --use-masks-for-ocr
```

Use `--device cuda` when CUDA is available.

## Dataset and Training

The full DocLayNet-derived dataset is not packaged. Download it when needed:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python data/download.py
uv run python scripts/convert_coco_to_yolo_detection.py
uv run python scripts/convert_coco_to_yolo_segmentation.py
```

Train from the packaged base YOLO models:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python scripts/train_yolo_detection.py --model models/yolo26m.pt --epochs 20 --batch 8 --fraction 0.2 --device cuda --cos-lr --no-val --name yolo26m_detection_docaug_b8
uv run python scripts/train_yolo_segmentation.py --model models/yolo26m-seg.pt --epochs 20 --batch 4 --fraction 0.2 --device cuda --cos-lr --no-val --name yolo26m_segmentation_docaug_b4
```

Training outputs, converted YOLO data, cache files, and demo artifacts are ignored so clones stay small.

## Checks

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python src/docmd/ocr/backends.py
uv run python -m compileall src scripts demo data
```