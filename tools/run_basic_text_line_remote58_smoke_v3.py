"""Remote/CPU-compatible launcher contract for BASIC V3 segment smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_segment_manifest(path: Path) -> dict:
    """@shanlan-comment Inspect expanded segment records and report source-line/tile coverage."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = {}
    for record in records:
        line_id = str(record.get("source_line_id", record.get("line_id", "")))
        lines.setdefault(line_id, []).append(int(record.get("tile_index", 0)))
    return {"records": len(records), "source_lines": len(lines), "multi_tile_lines": sum(len(v) > 1 for v in lines.values()), "max_tiles": max((len(v) for v in lines.values()), default=0)}


def main(argv=None) -> int:
    """@shanlan-comment Validate a segment manifest before formal remote training."""
    parser = argparse.ArgumentParser(); parser.add_argument("manifest", type=Path); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv); result = inspect_segment_manifest(args.manifest); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
