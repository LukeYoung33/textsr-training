"""Create labeled full-frame HX comparisons from frozen-OCR V4 and V5 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 260), 30), (0, 0, 0), -1)
    cv2.putText(result, text, (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return result


def _read(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lq-dir", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--v5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison_dir = args.output_dir / "comparisons"
    contact_dir = args.output_dir / "contact_sheets"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, np.ndarray]] = []
    for lq_path in sorted(args.lq_dir.glob("*")):
        if lq_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        stem = lq_path.stem
        v4_path = args.v4_dir / f"{stem}_basic_v2.png"
        v5_path = args.v5_dir / f"{stem}_basic_v2.png"
        lq, v4, v5 = _read(lq_path), _read(v4_path), _read(v5_path)
        if lq.shape != v4.shape or lq.shape != v5.shape:
            raise ValueError(f"full-frame shape mismatch for {stem}: {lq.shape}, {v4.shape}, {v5.shape}")
        comparison = cv2.hconcat((_label(lq, "LQ input"), _label(v4, "V4 r1"), _label(v5, "V5 r1")))
        cv2.imwrite(str(comparison_dir / f"{stem}__lq_v4_v5.png"), comparison)
        preview_h = 180
        preview_w = max(2, int(round(comparison.shape[1] * preview_h / comparison.shape[0])))
        panels.append((stem, cv2.resize(comparison, (preview_w, preview_h), interpolation=cv2.INTER_AREA)))
    per_sheet = 8
    for sheet_index in range(0, len(panels), per_sheet):
        group = panels[sheet_index:sheet_index + per_sheet]
        width = max(item.shape[1] for _, item in group)
        rows = []
        for stem, item in group:
            row = cv2.copyMakeBorder(item, 24, 0, 0, width - item.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(row, stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            rows.append(row)
        cv2.imwrite(str(contact_dir / f"hx_lq_v4_v5_{sheet_index // per_sheet + 1:02d}.jpg"), cv2.vconcat(rows))
    print(f"comparisons={len(panels)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
