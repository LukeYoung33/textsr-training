"""Create an auditable, geometry-aligned source manifest for Y4UV V3 training.

The published 20260820 list intentionally keeps the original photometric LQ
path, while V3 OCR polygons and ``GT_V3`` live in the spatial-aligned image
coordinate system.  This tool derives a separate manifest that pairs the
spatial LQ and masks with the published V3 GT without modifying either source
list.  Legacy records are copied unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DERIVED_MANIFEST_VERSION = "y4uv_v3_spatial_pair_v1"


def _sha256(path: Path) -> str:
    """@shanlan-comment Hash one source or derived manifest for reproducibility."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _image_hw(path: str) -> tuple[int, int] | None:
    """@shanlan-comment Read one image header and return height/width when decodable."""
    # All published V3 assets are PNGs. Reading the IHDR avoids decoding every
    # multi-megapixel image during manifest preparation; retain a conservative
    # ``None`` result for unsupported/corrupt headers so the row is audited out.
    try:
        with Path(path).open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", header[16:24])
        if width <= 0 or height <= 0:
            return None
        return int(height), int(width)
    except (OSError, struct.error):
        return None


def _v3_spatial_pair(record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """@shanlan-comment Map one V3 row into a same-coordinate image pair or audit exclusion."""
    ocr_path = Path(str(record.get("ocr", "")))
    try:
        payload = json.loads(ocr_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {"reason": "OCR_READ_FAILED", "detail": str(exc)}
    lines = payload.get("lines") or []
    if not lines:
        # 0812 and other legacy rows already carry paired paths and must not be
        # rewritten as V3 spatial data.
        return dict(record), None
    image = payload.get("image") or {}
    spatial_gt_path = str(image.get("path") or "")
    spatial_valid_path = str(image.get("valid_mask_path") or "")
    stem = str(record.get("stem", ""))
    if not spatial_gt_path or not spatial_valid_path or not stem:
        return None, {"reason": "MISSING_V3_SPATIAL_METADATA"}
    spatial_lq_path = str(Path(spatial_gt_path).with_name(f"{stem}_lq.png")) if spatial_gt_path else ""
    # Newer V3 lists sometimes carry a directly aligned record.lq/valid_mask,
    # while older published rows require the spatial sibling next to GT_V3.
    # Try both layouts and accept only a complete, dimension-identical pair.
    pair_candidates: list[tuple[str, str, str]] = []
    for space, lq_path, mask_path in (
        ("record_paths", str(record.get("lq", "")), str(record.get("valid_mask", ""))),
        ("record_lq_spatial_mask", str(record.get("lq", "")), spatial_valid_path),
        ("spatial_gt", spatial_lq_path, spatial_valid_path),
        ("spatial_lq_record_mask", spatial_lq_path, str(record.get("valid_mask", ""))),
    ):
        item = (space, lq_path, mask_path)
        if item not in pair_candidates:
            pair_candidates.append(item)

    mismatch_details: list[dict[str, Any]] = []
    for space, lq_path, mask_path in pair_candidates:
        candidate = dict(record)
        candidate.update(
            {
                "lq": lq_path,
                "valid_mask": mask_path,
                "source_pair_space": space,
                "source_pair_manifest_version": DERIVED_MANIFEST_VERSION,
                "source_pair_lq": lq_path,
                "source_pair_gt": str(record.get("gt", "")),
            }
        )
        required = {
            "lq": lq_path,
            "gt": str(candidate.get("gt", "")),
            "valid_mask": mask_path,
            "text_mask": str(candidate.get("text_mask", "")),
            "ocr": str(candidate.get("ocr", "")),
        }
        missing = [key for key, path in required.items() if not path or not Path(path).is_file()]
        if missing:
            continue
        shapes = {key: _image_hw(path) for key, path in required.items() if key != "ocr"}
        if shapes.get("lq") is None or shapes.get("gt") is None:
            continue
        if shapes["lq"] != shapes["gt"]:
            mismatch_details.append(
                {"pair_space": space, "lq_hw": list(shapes["lq"]), "gt_hw": list(shapes["gt"])}
            )
            continue
        mask_mismatch = False
        for key in ("valid_mask", "text_mask"):
            if shapes.get(key) != shapes["lq"]:
                mask_mismatch = True
                mismatch_details.append(
                    {
                        "pair_space": space,
                        "mask_key": key,
                        "pair_hw": list(shapes["lq"]),
                        "mask_hw": None if shapes.get(key) is None else list(shapes[key]),
                    }
                )
                break
        if not mask_mismatch:
            return candidate, None

    if mismatch_details:
        return None, {"reason": "V3_LQ_GT_SHAPE_MISMATCH", "attempts": mismatch_details}
    return None, {"reason": "MISSING_V3_SPATIAL_PAIR"}


def build_source_manifest(
    input_jsonl: Path,
    output_jsonl: Path,
    excluded_jsonl: Path,
    summary_json: Path,
) -> dict[str, Any]:
    """@shanlan-comment Derive aligned rows and persist accepted/excluded audit artifacts."""
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    schema_counts: Counter[str] = Counter()
    with input_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            result, audit = _v3_spatial_pair(record)
            if audit is not None:
                audit = {"source_line": line_number, "stem": record.get("stem", ""), **audit}
                excluded.append(audit)
                continue
            assert result is not None
            schema_counts["v3_spatial" if result.get("source_pair_space") else "legacy"] += 1
            accepted.append(result)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    excluded_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in accepted:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with excluded_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in excluded:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "manifest_version": DERIVED_MANIFEST_VERSION,
        "input_manifest": str(input_jsonl),
        "input_sha256": _sha256(input_jsonl),
        "output_manifest": str(output_jsonl),
        "output_sha256": _sha256(output_jsonl),
        "excluded_manifest": str(excluded_jsonl),
        "input_count": len(accepted) + len(excluded),
        "output_count": len(accepted),
        "excluded_count": len(excluded),
        "schema_counts": dict(sorted(schema_counts.items())),
        "excluded_reason_counts": dict(
            sorted(Counter(item["reason"] for item in excluded).items())
        ),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    """@shanlan-comment Build one derived source manifest from CLI paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--excluded-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_source_manifest(args.input_jsonl, args.output_jsonl, args.excluded_jsonl, args.summary_json), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
