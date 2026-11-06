"""SHAPE-S0 manifest statistics for BASIC V2 line candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_manifest(path, candidates):
    """@shanlan-comment Read line records and calculate candidate fit/tile/padding summaries."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    for height, width in candidates:
        shorts = np.array([float(r.get("short_side_px", r.get("aabb_h", 0))) for r in rows], dtype=np.float32)
        longs = np.array([float(r.get("long_side_px", r.get("aabb_w", width))) for r in rows], dtype=np.float32)
        valid = shorts > 0
        if not valid.any():
            result.append({"Hc": height, "Wc": width, "count": 0}); continue
        standard = valid & (shorts * 1.3 <= height)
        constrained = valid & ~standard & (shorts <= min(height - 16, 1.5 * height / 1.3))
        result.append({"Hc": height, "Wc": width, "count": int(valid.sum()),
                       "standard_fit_rate": float(standard.sum() / valid.sum()),
                       "constrained_rate": float(constrained.sum() / valid.sum()),
                       "oversize_rate": float(1 - (standard | constrained).sum() / valid.sum()),
                       "mean_tiles": float(np.maximum(1, np.ceil(longs / width)[valid]).mean())})
    return result


def main(argv=None):
    """@shanlan-comment Parse a manifest, emit JSON candidate summaries, and avoid model execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    candidates = [(h, w) for h in (64, 96, 128) for w in (256, 320, 384, 512)]
    Path(args.output).write_text(json.dumps(summarize_manifest(args.manifest, candidates), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
