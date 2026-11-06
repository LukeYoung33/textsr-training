"""Render original OCR polygons and selected inference crops for Lenovo 2x2 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SAMPLE_IDS = {
    "lenovo20260920": (
        "000-049_000076", "050-132_000516", "178-219_000264", "236-304_000755",
        "407-520_000486", "407-520_001368", "521-628_000948", "521-628_001497",
    ),
    "lenovo260926": (
        "batch_1_000-042_000051", "batch_2_000-016_000180", "batch_3_043-128_000315",
        "batch_3_043-128_000541", "batch_3_043-128_000586", "batch_3_043-128_000888",
        "batch_3_043-128_000975", "batch_3_043-128_001119",
    ),
}


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(path)


def _polygon(annotation: dict) -> np.ndarray:
    points = annotation.get("points") or annotation.get("polygon")
    if points is None:
        x0, y0, x1, y1 = annotation["bbox"]
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)


def _outline(image: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    cv2.polylines(image, [polygon], True, color, thickness, cv2.LINE_AA)


def _draw_envelope(image: np.ndarray, box: list[int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = map(int, box)
    for edge in (
        ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)),
    ):
        length = max(abs(edge[1][0] - edge[0][0]), abs(edge[1][1] - edge[0][1]))
        for start in range(0, length, 16):
            stop = min(start + 9, length)
            a = tuple(round(edge[0][k] + (edge[1][k] - edge[0][k]) * start / max(length, 1)) for k in (0, 1))
            b = tuple(round(edge[0][k] + (edge[1][k] - edge[0][k]) * stop / max(length, 1)) for k in (0, 1))
            cv2.line(image, a, b, color, 1, cv2.LINE_AA)


def _zoom_pair(image: np.ndarray, polygon: np.ndarray, envelope: list[int]) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    margin_y = max(18, round((y1 - y0) * .4))
    xa, ya = max(0, int(x0) - 24), max(0, int(y0) - margin_y)
    xb, yb = min(w, int(x1) + 24), min(h, int(y1) + margin_y)
    original = image[ya:yb, xa:xb].copy()
    marked = original.copy()
    _outline(marked, polygon - [xa, ya], (0, 255, 255), 1)
    local_envelope = [envelope[0] - xa, envelope[1] - ya, envelope[2] - xa, envelope[3] - ya]
    _draw_envelope(marked, local_envelope, (0, 255, 0))
    both = np.hstack((original, marked))
    # Keep source pixels unaltered in each panel; the browser can zoom further.
    return both


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    rows = json.loads(args.summary.read_text(encoding="utf-8"))["rows"]
    lookup = {(row["dataset"], row["sample_id"]): row for row in rows}
    index = []
    for dataset, sample_ids in SAMPLE_IDS.items():
        for sample_id in sample_ids:
            row = lookup[(dataset, sample_id)]
            image_path = Path(row["parent_lq"])
            ocr_path = image_path.parent.parent / "OCR" / (sample_id + ".json")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            annotations = ocr["ocr"]["annotations"]
            selected = {int(line["line_id"].rsplit("line", 1)[1]): line for line in row["lines"]}
            marked = image.copy()
            crops = image.copy()
            for ordinal, annotation in enumerate(annotations):
                if annotation.get("ignore") or annotation.get("visible") is False:
                    continue
                polygon = _polygon(annotation)
                _outline(marked, polygon, (0, 255, 255) if ordinal in selected else (255, 255, 0), 2)
                if ordinal in selected:
                    _outline(crops, polygon, (0, 255, 255), 2)
                    _draw_envelope(crops, selected[ordinal]["envelope_xyxy"], (0, 255, 0))
            sample_dir = args.output_root / dataset / sample_id
            _write(sample_dir / "01_lq_original.png", image)
            _write(sample_dir / "02_ocr_polygons.png", marked)
            _write(sample_dir / "03_ocr_and_model_crops.png", crops)
            ordinals = sorted(selected)
            chosen = sorted({ordinals[0], ordinals[len(ordinals) // 2], ordinals[-1]})
            zooms = []
            for ordinal in chosen:
                polygon = _polygon(annotations[ordinal])
                envelope = selected[ordinal]["envelope_xyxy"]
                name = f"line_{ordinal:03d}_lq_vs_box.png"
                _write(sample_dir / name, _zoom_pair(image, polygon, envelope))
                zooms.append(name)
            index.append({"dataset": dataset, "sample_id": sample_id, "ocr_annotations": len(annotations),
                          "lines_processed": len(selected), "selected_ordinals": ordinals,
                          "source_lq": str(image_path), "source_ocr": str(ocr_path), "zooms": zooms})
            print(f"{dataset}/{sample_id}: {len(selected)} selected lines, {len(annotations)} OCR annotations", flush=True)
    (args.output_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
