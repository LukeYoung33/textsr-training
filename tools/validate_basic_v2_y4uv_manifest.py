"""Validate Y4UV V3 segment geometry against the current source assets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_basic_v2_y4uv_v3_source_manifest import _image_hw


def validate_manifest(manifest: Path) -> dict[str, Any]:
    """@shanlan-comment Check every segment crop against all referenced image bounds."""
    shape_cache: dict[str, tuple[int, int] | None] = {}
    bad: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    total = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            crop = record.get("crop_xyxy_even")
            if not isinstance(crop, list) or len(crop) != 4:
                reason = "MISSING_CROP"
                counts[reason] += 1
                bad.append({"line": line_number, "reason": reason})
                continue
            x0, y0, x1, y1 = (int(value) for value in crop)
            if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0:
                reason = "INVALID_CROP"
                counts[reason] += 1
                bad.append({"line": line_number, "reason": reason, "crop": crop})
                continue
            for key in ("lq", "gt", "valid_mask", "text_mask"):
                path = str(record.get(key, ""))
                if path not in shape_cache:
                    shape_cache[path] = _image_hw(path) if path else None
                shape = shape_cache[path]
                if shape is None:
                    reason = f"MISSING_OR_UNREADABLE_{key.upper()}"
                    counts[reason] += 1
                    bad.append({"line": line_number, "reason": reason, "path": path})
                    continue
                height, width = shape
                if x1 > width or y1 > height:
                    reason = f"CROP_OUT_OF_BOUNDS_{key.upper()}"
                    counts[reason] += 1
                    bad.append(
                        {
                            "line": line_number,
                            "reason": reason,
                            "path": path,
                            "crop": crop,
                            "image_hw": [height, width],
                        }
                    )
    return {
        "manifest": str(manifest),
        "record_count": total,
        "unique_asset_count": len([path for path in shape_cache if path]),
        "bad_count": sum(counts.values()),
        "bad_reason_counts": dict(sorted(counts.items())),
        "bad_examples": bad[:20],
    }


def main() -> int:
    """@shanlan-comment Run strict geometry validation and return nonzero on bad records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_manifest(args.manifest)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["bad_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
