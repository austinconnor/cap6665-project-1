from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(prediction: str, reference: str) -> float:
    return levenshtein(prediction, reference) / max(1, len(reference))


def wer(prediction: str, reference: str) -> float:
    pred_words = prediction.split()
    ref_words = reference.split()
    return levenshtein(" ".join(pred_words), " ".join(ref_words)) / max(1, len(" ".join(ref_words)))


def block_signature(markdown: str) -> list[str]:
    signatures: list[str] = []
    for block in re.split(r"\n\s*\n", markdown.strip()):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            signatures.append("heading")
        elif stripped.startswith("- "):
            signatures.append("list")
        elif stripped.startswith("!["):
            signatures.append("figure")
        elif stripped.startswith("```") or "|" in stripped:
            signatures.append("table")
        elif stripped.startswith("*") and stripped.endswith("*"):
            signatures.append("caption")
        else:
            signatures.append("paragraph")
    return signatures


def structural_similarity(prediction: str, reference: str) -> dict[str, Any]:
    pred = block_signature(prediction)
    ref = block_signature(reference)
    distance = levenshtein(" ".join(pred), " ".join(ref))
    return {
        "pred_blocks": pred,
        "ref_blocks": ref,
        "block_sequence_error": distance / max(1, len(" ".join(ref))),
        "heading_order_match": [b for b in pred if b == "heading"] == [b for b in ref if b == "heading"],
    }


def evaluate_markdown_file(prediction_path: str | Path, reference_path: str | Path) -> dict[str, Any]:
    pred = Path(prediction_path).read_text(encoding="utf-8")
    ref = Path(reference_path).read_text(encoding="utf-8")
    return {
        "prediction": str(prediction_path),
        "reference": str(reference_path),
        "cer": cer(pred, ref),
        "wer_like": wer(pred, ref),
        "structure": structural_similarity(pred, ref),
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target

