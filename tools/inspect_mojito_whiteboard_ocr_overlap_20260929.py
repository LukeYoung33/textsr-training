"""Inspect how often whiteboard OCR line rectangles overlap substantially."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/guang.yang/data/lenovo_mojito_textsr_20260928_whiteboard_v4")


def overlap(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / min(aa, ab)


def main() -> None:
    records = []
    for line in (ROOT / "manifests/train.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        sample_id = row["sample_id"]
        data = json.loads((ROOT / row["files"]["ocr"]).read_text(encoding="utf-8"))
        items = data["lines"]
        boxes = [item.get("bbox_xyxy") for item in items]
        counts = {}
        examples = []
        for threshold in (0.2, 0.3, 0.4, 0.5):
            pairs = []
            for i, first in enumerate(boxes):
                if first is None:
                    continue
                for j in range(i + 1, len(boxes)):
                    if boxes[j] is None:
                        continue
                    ratio = overlap(first, boxes[j])
                    if ratio >= threshold:
                        pairs.append((i, j, round(ratio, 3)))
            counts[str(threshold)] = len(pairs)
            if threshold == 0.3:
                examples = pairs[:8]
        records.append({"sample_id": sample_id, "lines": len(items), "pairs": counts, "examples_0.3": examples})
    print(json.dumps({"images": len(records), "totals": {key: sum(item["pairs"][key] for item in records) for key in ("0.2", "0.3", "0.4", "0.5")}, "records": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
