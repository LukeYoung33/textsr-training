"""Build reproducible P17 OCR-box/ink-height statistics from the training set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _sha256_file(path: Path) -> str:
    """@luna-comment Hash one immutable input artifact for reproducible statistics.

    Input: readable file path. Output: lowercase SHA-256 hex digest. Key Steps:
    stream the file in bounded chunks. Key Params: none. State/Dependencies:
    filesystem read only; the digest is written into the derived statistics.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[float], q: float) -> float:
    """@luna-comment Return one finite empirical quantile for validated ratios.

    Input: non-empty finite scalar samples and quantile in [0, 1]. Output: a
    JSON-safe float. Key Steps: use NumPy's deterministic linear quantile.
    Key Params: q selects the requested distribution point. State/Dependencies:
    stateless numerical reduction.
    """
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def build_stats(manifest_path: Path) -> dict[str, Any]:
    """@luna-comment Derive training OCR-box geometry ratios from labels and text masks.

    Input: JSONL manifest whose files are relative to its dataset root. Output:
    versioned counts and q05/q50/q95 ratios for OCR box height, ink height, and
    top/bottom margins. Key Steps: load each supplied mask and OCR annotation,
    restrict mask support to the actual annotation bbox, reject empty/invalid
    cases, then aggregate ratios. Key Params: mask pixels above 127 represent
    supplied text support. State/Dependencies: read-only dataset access; no
    model preprocessing or image data is altered.
    """
    root = manifest_path.parent.parent
    samples = 0
    annotations = 0
    accepted = 0
    rejected = 0
    top_margin: list[float] = []
    bottom_margin: list[float] = []
    hbox_over_hink: list[float] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            samples += 1
            entry = json.loads(raw_line)
            files = entry.get("files", {})
            ocr_path = root / str(files.get("ocr", ""))
            mask_path = root / str(files.get("text_mask", ""))
            if not ocr_path.is_file() or not mask_path.is_file():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            try:
                ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for annotation in ocr.get("ocr", {}).get("annotations", []):
                annotations += 1
                bbox = annotation.get("bbox") if isinstance(annotation, dict) else None
                if not isinstance(bbox, list) or len(bbox) != 4:
                    rejected += 1
                    continue
                x0, y0, x1, y1 = map(float, bbox)
                left, right = max(0, int(np.floor(min(x0, x1)))), min(mask.shape[1], int(np.ceil(max(x0, x1))))
                top, bottom = max(0, int(np.floor(min(y0, y1)))), min(mask.shape[0], int(np.ceil(max(y0, y1))))
                if right - left < 2 or bottom - top < 2:
                    rejected += 1
                    continue
                support = mask[top:bottom, left:right] > 127
                rows = np.flatnonzero(support.any(axis=1))
                if rows.size < 2:
                    rejected += 1
                    continue
                ink_top, ink_bottom = int(rows[0]), int(rows[-1] + 1)
                ink_height = float(ink_bottom - ink_top)
                box_height = float(bottom - top)
                if ink_height <= 0.0 or box_height < ink_height:
                    rejected += 1
                    continue
                top_margin.append(float(ink_top) / ink_height)
                bottom_margin.append(float((bottom - top) - ink_bottom) / ink_height)
                hbox_over_hink.append(box_height / ink_height)
                accepted += 1
    if accepted < 10:
        raise RuntimeError(f"insufficient valid OCR/mask annotations: {accepted}")
    payload: dict[str, Any] = {
        "stats_version": "p17_mask_guided_geometry_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "dataset_root": str(root),
        "samples_scanned": samples,
        "annotations_seen": annotations,
        "annotations_accepted": accepted,
        "annotations_rejected": rejected,
        "quantiles": {
            "top_margin_over_hink_q05": _quantile(top_margin, .05),
            "top_margin_over_hink_q50": _quantile(top_margin, .50),
            "top_margin_over_hink_q95": _quantile(top_margin, .95),
            "bottom_margin_over_hink_q05": _quantile(bottom_margin, .05),
            "bottom_margin_over_hink_q50": _quantile(bottom_margin, .50),
            "bottom_margin_over_hink_q95": _quantile(bottom_margin, .95),
            "hbox_over_hink_q05": _quantile(hbox_over_hink, .05),
            "hbox_over_hink_q50": _quantile(hbox_over_hink, .50),
            "hbox_over_hink_q95": _quantile(hbox_over_hink, .95),
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["stats_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    """@luna-comment Parse the stats CLI and write one validated JSON artifact.

    Input: a training manifest path and output JSON path. Output: a formatted
    versioned statistics file and zero exit status on success. Key Steps: call
    ``build_stats`` and serialize its reproducibility fields. Key Params: none.
    State/Dependencies: read-only training inputs and caller-selected output.
    """
    parser = argparse.ArgumentParser(description="Build P17 mask-guided OCR geometry statistics")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_stats(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("stats_version", "annotations_accepted", "stats_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
