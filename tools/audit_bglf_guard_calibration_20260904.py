#!/usr/bin/env python3
"""Audit post-ramp bgLF-guard contribution from a real training log.

The calibration probe runs with ``bg_lf_weight=1``.  This utility extracts the
logged weighted bgLF and total objective at every print interval after the
R1/recall ramp, then reports robust contribution statistics.  It deliberately
does not inspect image outputs: the result is a loss-scale calibration audit,
not a visual-quality claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


ITER_RE = re.compile(r"\biter:\s*([0-9,]+)")
BG_RE = re.compile(r"\bl_bg_lf_weighted:\s*([-+0-9.eE]+)")
TOTAL_RE = re.compile(r"\bl_total:\s*([-+0-9.eE]+)")
RAW_RE = re.compile(r"\bl_bg_lf_raw:\s*([-+0-9.eE]+)")
COVERAGE_RE = re.compile(r"\bbg_lf_far_coverage:\s*([-+0-9.eE]+)")


def _float(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def audit(log_path: Path, ramp_end: int, probe_weight: float) -> dict[str, object]:
    samples: list[dict[str, float | int]] = []
    malformed_after_ramp = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        iter_match = ITER_RE.search(line)
        if iter_match is None:
            continue
        iteration = int(iter_match.group(1).replace(",", ""))
        if iteration < ramp_end:
            continue
        weighted = _float(BG_RE.search(line))
        total = _float(TOTAL_RE.search(line))
        raw = _float(RAW_RE.search(line))
        coverage = _float(COVERAGE_RE.search(line))
        if weighted is None or total is None or total <= 0:
            malformed_after_ramp += 1
            continue
        sample: dict[str, float | int] = {
            "iter": iteration,
            "bg_lf_weighted": weighted,
            "l_total": total,
            "share": weighted / total,
        }
        if raw is not None:
            sample["bg_lf_raw"] = raw
        if coverage is not None:
            sample["far_coverage"] = coverage
        samples.append(sample)

    if not samples:
        raise SystemExit(f"No valid post-ramp samples found in {log_path} (ramp_end={ramp_end})")

    shares = [float(sample["share"]) for sample in samples]
    weighted_values = [float(sample["bg_lf_weighted"]) for sample in samples]
    total_values = [float(sample["l_total"]) for sample in samples]
    raw_values = [float(sample["bg_lf_raw"]) for sample in samples if "bg_lf_raw" in sample]
    coverage_values = [float(sample["far_coverage"]) for sample in samples if "far_coverage" in sample]

    # For a probe weight of one, use the observed share to estimate the weight
    # that would land at a requested target while retaining the logged
    # non-bgLF objective: w = target/(1-target) * (1-s)/s.
    median_share = statistics.median(shares)
    mean_share = statistics.fmean(shares)

    def projected_weight(target: float, share: float) -> float:
        return target / (1.0 - target) * (1.0 - share) / share

    return {
        "log": str(log_path),
        "probe_weight": probe_weight,
        "ramp_end_exclusive": ramp_end,
        "sample_count": len(samples),
        "malformed_after_ramp": malformed_after_ramp,
        "iter_first": int(samples[0]["iter"]),
        "iter_last": int(samples[-1]["iter"]),
        "share_mean": mean_share,
        "share_median": median_share,
        "share_p10": _percentile(shares, 0.10),
        "share_p90": _percentile(shares, 0.90),
        "share_min": min(shares),
        "share_max": max(shares),
        "weighted_bg_lf_mean": statistics.fmean(weighted_values),
        "total_mean": statistics.fmean(total_values),
        "raw_bg_lf_mean": statistics.fmean(raw_values) if raw_values else None,
        "far_coverage_mean": statistics.fmean(coverage_values) if coverage_values else None,
        "projected_weight_for_5pct_mean_share": projected_weight(0.05, mean_share),
        "projected_weight_for_7_5pct_mean_share": projected_weight(0.075, mean_share),
        "projected_weight_for_10pct_mean_share": projected_weight(0.10, mean_share),
        "projected_weight_for_5pct_median_share": projected_weight(0.05, median_share),
        "projected_weight_for_7_5pct_median_share": projected_weight(0.075, median_share),
        "projected_weight_for_10pct_median_share": projected_weight(0.10, median_share),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--ramp-end", type=int, default=7000)
    parser.add_argument("--probe-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.log, args.ramp_end, args.probe_weight)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
