"""Export auditable native LQ/GT training-pair samples from a V5 crop cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _stats(row: dict) -> dict | None:
    lq = cv2.imread(row["lq_crop"], cv2.IMREAD_COLOR)
    gt = cv2.imread(row["gt_crop"], cv2.IMREAD_COLOR)
    mask = cv2.imread(row["text_mask_crop"], cv2.IMREAD_GRAYSCALE)
    if lq is None or gt is None or mask is None or np.count_nonzero(mask) < 24:
        return None
    pick = mask > 127
    lq_lab = cv2.cvtColor(lq, cv2.COLOR_BGR2LAB)[pick]
    gt_lab = cv2.cvtColor(gt, cv2.COLOR_BGR2LAB)[pick]
    # Dark foreground makes the color behaviour easiest to audit; a/b measure chroma shift.
    return {
        "row": row,
        "gt_l": float(np.median(gt_lab[:, 0])),
        "lq_l": float(np.median(lq_lab[:, 0])),
        "lab_ab_delta": float(np.linalg.norm(np.median(gt_lab[:, 1:], axis=0) - np.median(lq_lab[:, 1:], axis=0))),
    }


def _panel(lq: np.ndarray, gt: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    h = max(lq.shape[0], gt.shape[0], mask.shape[0])
    def fit(image: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(image, 0, h - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    core = np.concatenate((fit(lq), fit(gt), fit(mask_bgr)), axis=1)
    header = np.full((34, core.shape[1], 3), 245, dtype=np.uint8)
    width = lq.shape[1]
    for pos, title in ((8, "LQ"), (width + 8, "GT"), (2 * width + 8, "TEXT MASK")):
        cv2.putText(header, title, (pos, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, label[:150], (max(2 * width + 125, 8), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate((header, core), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-dataset", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # First retain a deterministic small candidate pool from every batch.  This
    # keeps an audit export quick without scanning/decompressing every crop.
    candidates: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    with Path(args.manifest).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = hashlib.sha1(row["source_line_id"].encode("utf-8")).hexdigest()
            candidates[row["dataset"]].append((key, row))
    groups: dict[str, list[dict]] = defaultdict(list)
    for dataset, values in candidates.items():
        for _, row in sorted(values, key=lambda x: x[0])[:160]:
            found = _stats(row)
            if found:
                groups[dataset].append(found)

    selected: list[dict] = []
    for dataset, values in sorted(groups.items()):
        # Pick distinct highly visible/dark target text examples with the largest LQ-to-GT chroma shift.
        values.sort(key=lambda x: (x["gt_l"], -x["lab_ab_delta"], x["row"]["source_line_id"]))
        take = values[: max(1, args.per_dataset)]
        selected.extend(take)

    records = []
    contact_rows = []
    for index, item in enumerate(selected):
        row = item["row"]
        stem = f"{index:02d}_{row['dataset']}_{row['source_line_token']}"
        lq = cv2.imread(row["lq_crop"], cv2.IMREAD_COLOR)
        gt = cv2.imread(row["gt_crop"], cv2.IMREAD_COLOR)
        mask = cv2.imread(row["text_mask_crop"], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(output / f"{stem}_lq.png"), lq)
        cv2.imwrite(str(output / f"{stem}_gt.png"), gt)
        cv2.imwrite(str(output / f"{stem}_mask.png"), mask)
        label = f"{row['dataset']} | {row['source_line_id']} | GT_L={item['gt_l']:.1f} LQ_L={item['lq_l']:.1f} dAB={item['lab_ab_delta']:.1f}"
        panel = _panel(lq, gt, mask, label)
        cv2.imwrite(str(output / f"{stem}_panel.png"), panel)
        contact_rows.append(panel)
        records.append({**item, "sample_stem": stem})

    width = max(row.shape[1] for row in contact_rows)
    padded = [cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)) for row in contact_rows]
    cv2.imwrite(str(output / "contact_sheet.png"), np.concatenate(padded, axis=0))
    (output / "samples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(records), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
