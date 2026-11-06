"""Build the HX fixed-canvas text-line manifest used by the 250K dump jobs.

The source HX ``pairs.jsonl`` stores paths relative to the dataset directory.
This adapter makes them absolute and delegates the actual crop-capacity and
size filtering to the production ``native_capacity_v1`` manifest builder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_basic_v2_segment_manifest_fast.py"


def _load_builder():
    """Load the existing production segment-manifest builder."""
    spec = importlib.util.spec_from_file_location("basic_v2_manifest_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load manifest builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _absolute(root: Path, value: str) -> str:
    """Resolve a dataset-relative path without changing already-absolute paths."""
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def build_hx_manifest(dataset_root: Path, pairs_jsonl: Path, output_dir: Path) -> dict:
    """Create absolute frame records and expand them into accepted text-line tiles."""
    builder = _load_builder()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_manifest = output_dir / "frames_absolute.jsonl"
    segment_manifest = output_dir / "segments.jsonl"
    summary_path = output_dir / "summary.json"

    frame_records = []
    for line_number, line in enumerate(pairs_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        pair = json.loads(line)
        required = ("stem", "lq", "gt", "valid_mask", "text_mask", "ocr")
        missing = [key for key in required if not pair.get(key)]
        if missing:
            raise ValueError(f"pairs.jsonl line {line_number} missing keys: {missing}")
        frame_records.append({
            "dataset": "Artguru_textHD/HX_testset_training",
            "stem": str(pair["stem"]),
            "lq": _absolute(dataset_root, pair["lq"]),
            "gt": _absolute(dataset_root, pair["gt"]),
            "valid_mask": _absolute(dataset_root, pair["valid_mask"]),
            "text_mask": _absolute(dataset_root, pair["text_mask"]),
            "ocr": _absolute(dataset_root, pair["ocr"]),
        })

    frame_manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in frame_records),
        encoding="utf-8",
    )
    summary = builder.build(
        frame_manifest,
        segment_manifest,
        summary_path,
        canvas_hw=(96, 320),
        overlap=64,
        min_short_side_ratio=0.5,
        min_long_side_ratio=0.5,
    )
    summary["pairs_jsonl"] = str(pairs_jsonl)
    summary["frame_manifest"] = str(frame_manifest)
    summary["segment_manifest"] = str(segment_manifest)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    pairs = args.pairs_jsonl or (args.dataset_root / "pairs.jsonl")
    result = build_hx_manifest(args.dataset_root, pairs, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
