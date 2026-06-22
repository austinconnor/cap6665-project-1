from __future__ import annotations

from pathlib import Path

import streamlit as st

from docmd.schema import PageResult


def show_region_table(page: PageResult) -> None:
    rows = [
        {
            "order": region.order,
            "class": region.class_name,
            "det_conf": None if region.confidence is None else round(region.confidence, 3),
            "ocr_conf": None if region.ocr_confidence is None else round(region.ocr_confidence, 3),
            "text": (region.text[:160] + "...") if len(region.text) > 160 else region.text,
        }
        for region in page.regions
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_crops(crops: list[Path], page: PageResult) -> None:
    for crop, region in zip(crops, page.regions, strict=False):
        with st.expander(f"{region.order}. {region.class_name}"):
            st.image(str(crop), use_container_width=True)
            st.text_area("OCR text", region.text, height=90, key=f"ocr_{region.id}")


def output_links(run_dir: Path) -> None:
    st.caption(f"Outputs saved to `{run_dir}`")

