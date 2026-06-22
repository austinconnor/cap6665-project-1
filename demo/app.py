from __future__ import annotations

import re
import sys
from importlib import reload
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from components import output_links, show_crops, show_region_table
import docmd.demo.runtime as demo_runtime
from docmd.detection.train import cuda_available
from docmd.utils.io import image_paths, load_yaml

demo_runtime = reload(demo_runtime)


def config() -> dict:
    return load_yaml(ROOT / "configs" / "demo.yaml")


def markdown_for_render(markdown: str) -> str:
    return re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL).strip()


def resolve_markdown_image(target: str, markdown_dir: Path) -> Path | str | None:
    if target.startswith(("http://", "https://")):
        return target

    path = Path(target)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([markdown_dir / target, markdown_dir.parent / target])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if path.parent.name == "figures":
        region_id = path.stem
        crop_dir = markdown_dir.parent / "region_crops"
        matches = sorted(
            crop
            for crop in crop_dir.glob(f"*_{region_id}_*.png")
            if "picture" in crop.name.lower() or "figure" in crop.name.lower()
        )
        if not matches:
            matches = sorted(crop_dir.glob(f"*_{region_id}_*.png"))
        if matches:
            return matches[0]

    return None


def render_markdown_document(markdown: str, markdown_dir: Path) -> None:
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    cursor = 0
    cleaned = markdown_for_render(markdown)
    for match in image_pattern.finditer(cleaned):
        chunk = cleaned[cursor : match.start()].strip()
        if chunk:
            st.markdown(chunk)
        alt = match.group(1).strip()
        target = match.group(2).strip()
        resolved = resolve_markdown_image(target, markdown_dir)
        if isinstance(resolved, str):
            st.image(resolved, caption=alt or None, width=560)
        elif resolved is not None:
            st.image(str(resolved), caption=alt or None, width=560)
        else:
            st.warning(f"Image not found: {target}")
        cursor = match.end()
    tail = cleaned[cursor:].strip()
    if tail:
        st.markdown(tail)


def preview_pasted_image(pasted: str) -> bool:
    text = pasted.strip().strip("\"'")
    if not text:
        return False
    if text.startswith(("http://", "https://", "data:image/")):
        st.image(text, caption="Original page", use_container_width=True)
        return True
    path = Path(text).expanduser()
    if path.exists():
        st.image(str(path), caption="Original page", use_container_width=True)
        return True
    st.info("Paste a valid local image path, file URL, image URL, data URI, or base64 image.")
    return False


def main() -> None:
    cfg = config()
    st.set_page_config(page_title="Document to Markdown Reconstruction", layout="wide")
    st.title("Document to Markdown Reconstruction")

    sample_dir = ROOT / cfg.get("sample_dir", "demo_samples")
    samples = image_paths(sample_dir) if sample_dir.exists() else []

    with st.sidebar:
        st.caption("Defaults use packaged YOLO weights and packaged PP-OCRv6 medium recognition.")
        st.header("Input")
        source = st.radio("Page source", ["Sample", "Upload", "Paste"], horizontal=True)
        selected_sample = None
        upload = None
        pasted = ""
        if source == "Sample":
            names = [path.name for path in samples]
            selected_name = st.selectbox("Sample page", names, index=0 if names else None)
            selected_sample = next((path for path in samples if path.name == selected_name), None)
        elif source == "Upload":
            upload = st.file_uploader("Upload a page image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        else:
            pasted = st.text_area(
                "Paste image input",
                placeholder="Local path, file:// URL, image URL, data:image/...;base64,..., or raw base64 image",
                height=120,
            )

        st.header("Pipeline")
        mode = st.radio("YOLO mode", ["detection", "segmentation"], horizontal=True)
        weights = st.text_input("Weights", value=str(ROOT / cfg["weights"][mode]))
        weight_path = Path(weights)
        if weight_path.exists():
            st.success(f"Checkpoint ready: {weight_path.name}")
        else:
            st.warning("Checkpoint not found yet.")
        conf = st.slider("Confidence", 0.05, 0.95, float(cfg.get("conf", 0.25)), 0.05)
        imgsz = st.number_input(
            "Image size",
            min_value=320,
            max_value=2048,
            value=int(cfg.get("imgsz", 1024)),
            step=64,
        )
        configured_device = str(cfg.get("device", "auto"))
        if configured_device.startswith("cuda") and not cuda_available():
            st.warning("CUDA is not visible to this Python environment; using CPU for demo inference.")
            configured_device = "cpu"
        device = st.text_input("Device", value=configured_device)
        ocr_backend = str(cfg.get("ocr_backend", "paddleocr"))
        start = st.button("Run live reconstruction", type="primary")

    if source == "Sample" and selected_sample:
        st.image(str(selected_sample), caption="Original page", use_container_width=True)
    elif upload is not None:
        st.image(upload, caption="Original page", use_container_width=True)
    elif source == "Paste" and preview_pasted_image(pasted):
        pass
    else:
        st.info("Choose a sample, upload a page image, or paste an image input.")
        return

    if not start:
        return

    weights_path = Path(weights)
    if not weights_path.exists():
        st.error(f"Weight file not found: {weights_path}")
        st.stop()

    run_dir = demo_runtime.make_demo_run_dir(ROOT / cfg.get("outputs_dir", "outputs/demo_runs"))
    if upload is not None:
        image_path = demo_runtime.save_uploaded_image(upload.getvalue(), upload.name, run_dir)
    elif source == "Paste":
        try:
            image_path = demo_runtime.save_pasted_image(pasted, run_dir)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    elif selected_sample is not None:
        image_path = demo_runtime.save_sample_for_run(selected_sample, run_dir)
    else:
        st.stop()

    with st.status("Running fresh YOLO inference, OCR, ordering, and Markdown reconstruction...", expanded=True) as status:
        st.write("1. Loading selected page")
        st.write("2. Running YOLO layout prediction on the selected image")
        st.write("3. Cropping detected regions and running OCR")
        st.write("4. Sorting blocks with spatial reading-order heuristics")
        st.write("5. Rendering Markdown and saving artifacts")
        page, crops = demo_runtime.run_demo_pipeline(
            image_path=image_path,
            weights=weights_path,
            run_dir=run_dir,
            mode=mode,
            conf=conf,
            imgsz=int(imgsz),
            device=device,
            ocr_backend=ocr_backend,
        )
        status.update(label="Reconstruction complete", state="complete")

    stem = Path(image_path).stem
    tabs = st.tabs(["Overlay", "Regions", "Reading Order", "Rendered Markdown", "Markdown Source"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.image(str(image_path), caption="Original", use_container_width=True)
        c2.image(str(run_dir / "visualizations" / f"{stem}_layout.png"), caption="YOLO layout overlay", use_container_width=True)
    with tabs[1]:
        show_crops(crops, page)
    with tabs[2]:
        st.image(str(run_dir / "visualizations" / f"{stem}_ordered.png"), caption="Ordered blocks", use_container_width=True)
        show_region_table(page)
    with tabs[3]:
        render_markdown_document(page.markdown, run_dir / "markdown")
    with tabs[4]:
        st.code(page.markdown, language="markdown")
    output_links(run_dir)


if __name__ == "__main__":
    main()