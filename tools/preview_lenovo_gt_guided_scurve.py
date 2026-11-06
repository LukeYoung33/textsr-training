"""Render one aligned raw/baseline/GT-guided/GT line for A/B inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def render(root: Path) -> dict:
    """@luna-comment
    Input: Completed GT-guided variant root.
    Output: A four-panel PNG and small metrics JSON under reports/.
    Key Steps: Select an actively changed normal-size line, read paired images, and label equal crops.
    Key Params: Preview sample has width 280-1000 and height at least 48 pixels.
    State/Dependencies: Reads both A/B datasets and writes only reports in the variant root.
    """
    row = None
    with (root / "manifests" / "train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            width, height = candidate["cropped_size_wh"]
            if candidate["preprocess"]["fallback_reason"] is None and 280 <= width <= 1000 and height >= 48:
                row = candidate
                break
    if row is None:
        raise ValueError("no active normal-size line found for preview")
    baseline_lq = Path(row["source_variant_root"]) / "train" / "LQ_scurve" / f"{row['sample_id']}.png"
    paths = [Path(row["files"][name]) for name in ("lq_raw", "lq")]
    paths.insert(1, baseline_lq)
    paths.append(Path(row["files"]["gt"]))
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images) or len({image.shape for image in images}) != 1:
        raise ValueError("preview source images are unreadable or misaligned")
    titles = ("raw LQ", "LQ-derived v1", "GT-guided v1", "GT")
    panels = []
    for title, image in zip(titles, images, strict=True):
        header = np.full((30, image.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(header, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(np.vstack((header, image)))
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    preview = report_dir / "ab_preview.png"
    if not cv2.imwrite(str(preview), np.hstack(panels)):
        raise OSError(f"cannot write preview: {preview}")
    gt = images[-1].astype(np.int16)
    report = {"sample_id": row["sample_id"], "preview": str(preview),
              "mean_abs_error_to_gt": {name: float(np.mean(np.abs(image.astype(np.int16) - gt)))
                                       for name, image in zip(titles[:3], images[:3], strict=True)},
              "changed_pixels": row["preprocess"]["changed_pixels"]}
    (report_dir / "ab_preview.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """@luna-comment
    Input: GT-guided variant root on the command line.
    Output: Printed preview report and process exit code.
    Key Steps: Parse the root and render one deterministic representative line.
    Key Params: The first eligible active training line is selected.
    State/Dependencies: Writes only reports under the given root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
