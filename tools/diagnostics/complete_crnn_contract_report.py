#!/usr/bin/env python3
"""Complete stratified and mode-diagnostic summaries from a finished CRNN run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_csv(path: Path) -> list[dict[str, str]]:
    """@guang.yang-comment
    Input: one UTF-8 CSV result artifact.
    Output: ordered row dictionaries with original string values.
    Key Steps: open with universal newline handling and parse the header once.
    Key Params: DictReader preserves the diagnostic column names for downstream grouping.
    State/Dependencies: reads only the requested results file.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, Any], key: str) -> float | None:
    """@guang.yang-comment
    Input: one result row and a numeric field name.
    Output: finite float or None for empty/non-finite cells.
    Key Steps: parse the CSV scalar and reject NaN/Infinity.
    Key Params: missing metrics remain missing rather than becoming zero.
    State/Dependencies: pure conversion.
    """
    value = row.get(key, "")
    if value in (None, "", "NA"):
        return None
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return 1.0 if value.lower() == "true" else 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def aspect_bucket(row: dict[str, Any]) -> str:
    """@guang.yang-comment
    Input: one per-sample metric row containing the source crop aspect ratio.
    Output: a stable source aspect-ratio bucket.
    Key Steps: classify ratio into <4, 4-8, 8-16, or >=16.
    Key Params: boundaries expose the text-line geometry ranges relevant to fixed 32x256/32x128 resizing.
    State/Dependencies: pure numeric bucketing.
    """
    ratio = number(row, "aspect_ratio")
    if ratio is None:
        return "unknown"
    if ratio < 4:
        return "<4"
    if ratio < 8:
        return "4-8"
    if ratio < 16:
        return "8-16"
    return ">=16"


def deformation_bucket(row: dict[str, Any]) -> str:
    """@guang.yang-comment
    Input: one per-sample metric row with condition geometry and source aspect ratio.
    Output: bucket for absolute horizontal deformation |log2(s_x)|.
    Key Steps: derive target aspect 8 or 4 for fixed widths, then classify zero/mild/strong deformation.
    Key Params: boundaries 0.25 and 0.75 expose approximately quarter- and three-quarter-octave changes.
    State/Dependencies: pure arithmetic based on recorded condition metadata.
    """
    ratio = number(row, "aspect_ratio")
    if ratio is None:
        return "unknown"
    if row.get("geometry") != "fixed":
        value = 0.0
    else:
        target = 4.0 if row.get("condition") == "C06-TATSR" else 8.0
        value = abs(math.log(max(target, 1e-6) / max(ratio, 1e-6), 2))
    if value < 0.25:
        return "<0.25"
    if value < 0.75:
        return "0.25-0.75"
    return ">=0.75"


def group_value(row: dict[str, Any], dimension: str) -> str:
    """@guang.yang-comment
    Input: one metric or CP row and a requested stratification dimension.
    Output: the dimension bucket used by the report tables.
    Key Steps: map language/length/structure/aspect/deformation fields and provide an all bucket.
    Key Params: all is an explicit aggregate, not an omitted grouping.
    State/Dependencies: reads row metadata only.
    """
    if dimension == "all":
        return "all"
    if dimension == "aspect_ratio":
        return aspect_bucket(row)
    if dimension == "deformation":
        return deformation_bucket(row)
    return str(row.get(dimension, "unknown") or "unknown")


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """@guang.yang-comment
    Input: result rows and one numeric key.
    Output: finite arithmetic mean or None when no values are present.
    Key Steps: parse with number and average only valid values.
    Key Params: empty groups remain explicit as null in CSV/JSON artifacts.
    State/Dependencies: pure aggregation.
    """
    values = [value for row in rows if (value := number(row, key)) is not None]
    return float(np.mean(values)) if values else None


def bootstrap(values: list[float], seed: int, iterations: int = 2000) -> dict[str, float | int | None]:
    """@guang.yang-comment
    Input: paired differences, seed, and bootstrap iteration count.
    Output: n, mean, and percentile 95% confidence interval.
    Key Steps: resample paired differences with replacement and compute means.
    Key Params: 2,000 iterations and 2.5/97.5 percentiles match the experiment plan.
    State/Dependencies: local NumPy generator only.
    """
    if not values:
        return {"n": 0, "mean": None, "low": None, "high": None}
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "low": value, "high": value}
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(iterations, len(array)))
    samples = array[indices].mean(axis=1)
    return {"n": int(len(array)), "mean": float(array.mean()), "low": float(np.percentile(samples, 2.5)), "high": float(np.percentile(samples, 97.5))}


def stratified_metrics(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    """@guang.yang-comment
    Input: all per-sample condition metrics.
    Output: OCR/blank summaries by condition, view, and language/length/structure/geometry buckets.
    Key Steps: attach derived geometry buckets, group rows, and aggregate finite metrics.
    Key Params: LQ rows are retained for target-distribution reporting; text metrics exclude no-text rows.
    State/Dependencies: reads in-memory CSV rows and does not alter them.
    """
    rows = []
    dimensions = ("language", "length_bucket", "structure_bucket", "aspect_ratio", "deformation")
    for condition in sorted({row["condition"] for row in metrics}):
        for view in ("lq", "gt"):
            source = [row for row in metrics if row["condition"] == condition and row["view"] == view]
            for dimension in ("all", *dimensions):
                buckets = {group_value(row, dimension) for row in source}
                for bucket in sorted(buckets):
                    grouped = [row for row in source if group_value(row, dimension) == bucket]
                    valid = [row for row in grouped if row.get("text_valid") == "True"]
                    rows.append({
                        "condition": condition,
                        "view": view,
                        "dimension": dimension,
                        "bucket": bucket,
                        "n": len(grouped),
                        "valid_text_n": len(valid),
                        "acc_mean": mean(valid, "acc"),
                        "cer_mean": mean(valid, "cer"),
                        "ned_mean": mean(valid, "ned"),
                        "blank_rate": mean(grouped, "blank_output"),
                        "pred_length_mean": mean(grouped, "pred_length"),
                    })
    return rows


def stratified_cp(cp_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """@guang.yang-comment
    Input: per-pair CP ranking rows.
    Output: positive-vs-negative ordering and margin summaries by condition and semantic bucket.
    Key Steps: group by language, length, structure, and all; average pos_better/margin.
    Key Params: the same 400-pair sample pool is retained for every condition.
    State/Dependencies: pure aggregation over CP CSV rows.
    """
    rows = []
    for condition in sorted({row["condition"] for row in cp_rows}):
        source = [row for row in cp_rows if row["condition"] == condition]
        for dimension in ("all", "language", "length_bucket", "structure_bucket"):
            for bucket in sorted({group_value(row, dimension) for row in source}):
                grouped = [row for row in source if group_value(row, dimension) == bucket]
                rows.append({
                    "condition": condition,
                    "dimension": dimension,
                    "bucket": bucket,
                    "n": len(grouped),
                    "pos_better_probability": mean(grouped, "pos_better"),
                    "margin_mean": mean(grouped, "margin"),
                    "d_pos_mean": mean(grouped, "d_pos"),
                    "d_neg_mean": mean(grouped, "d_neg"),
                })
    return rows


def mode_diagnostics(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    """@guang.yang-comment
    Input: per-sample metrics containing eval and train diagnostic conditions.
    Output: paired train-minus-eval NED/CER deltas and prediction disagreement rates.
    Key Steps: compare C00/C10 and C05/C11 by sample_id and LQ view.
    Key Params: matching geometry/range pairs isolate mode effects without checkpoint contamination.
    State/Dependencies: reads metric rows only; no model execution occurs.
    """
    pairs = (("C00-SRC", "C10-SRC-TRAIN"), ("C05-REALCE", "C11-REALCE-TRAIN"))
    output = []
    for eval_id, train_id in pairs:
        eval_rows = {row["sample_id"]: row for row in metrics if row["condition"] == eval_id and row["view"] == "lq"}
        train_rows = {row["sample_id"]: row for row in metrics if row["condition"] == train_id and row["view"] == "lq"}
        matched = [sample for sample in sorted(eval_rows) if sample in train_rows]
        for group_key, group_name in ((None, "all"), ("language", "zh"), ("language", "en"), ("length_bucket", ">32"), ("structure_bucket", "high")):
            selected = [sample for sample in matched if group_key is None or eval_rows[sample].get(group_key) == group_name]
            eval_selected = [eval_rows[sample] for sample in selected]
            train_selected = [train_rows[sample] for sample in selected]
            ned_delta = [float(train["ned"]) - float(evaluate["ned"]) for evaluate, train in zip(eval_selected, train_selected) if number(evaluate, "ned") is not None and number(train, "ned") is not None]
            cer_delta = [float(train["cer"]) - float(evaluate["cer"]) for evaluate, train in zip(eval_selected, train_selected) if number(evaluate, "cer") is not None and number(train, "cer") is not None]
            disagreement = [float(evaluate.get("prediction", "") != train.get("prediction", "")) for evaluate, train in zip(eval_selected, train_selected)]
            output.append({
                "eval_condition": eval_id,
                "train_condition": train_id,
                "group": group_name,
                "n": len(selected),
                "ned_train_minus_eval": mean([{"value": value} for value in ned_delta], "value"),
                "cer_train_minus_eval": mean([{"value": value} for value in cer_delta], "value"),
                "prediction_disagreement_rate": float(np.mean(disagreement)) if disagreement else None,
            })
    return output


def fixed_intervals(metrics: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    """@guang.yang-comment
    Input: per-sample metrics and bootstrap seed.
    Output: paired fixed-minus-aspect CER/NED intervals by language, length, structure, aspect, and deformation groups.
    Key Steps: index paired LQ samples and calculate bootstrap intervals for C00/C06 against C03.
    Key Params: 2,000 bootstrap iterations are delegated to bootstrap; all available groups are retained.
    State/Dependencies: read-only aggregation of finished metrics.
    """
    output = []
    for fixed_id in ("C00-SRC", "C06-TATSR"):
        for dimension in ("all", "language", "length_bucket", "structure_bucket", "aspect_ratio", "deformation"):
            fixed_rows = {row["sample_id"]: row for row in metrics if row["condition"] == fixed_id and row["view"] == "lq" and row.get("text_valid") == "True"}
            aspect_rows = {row["sample_id"]: row for row in metrics if row["condition"] == "C03-DYN-NORM" and row["view"] == "lq" and row.get("text_valid") == "True"}
            for bucket in sorted({group_value(row, dimension) for row in fixed_rows.values()}):
                samples = [sample for sample in sorted(set(fixed_rows) & set(aspect_rows)) if group_value(fixed_rows[sample], dimension) == bucket]
                for metric in ("cer", "ned"):
                    values = [float(fixed_rows[sample][metric]) - float(aspect_rows[sample][metric]) for sample in samples if number(fixed_rows[sample], metric) is not None and number(aspect_rows[sample], metric) is not None]
                    interval = bootstrap(values, seed + len(output))
                    output.append({"fixed": fixed_id, "aspect": "C03-DYN-NORM", "dimension": dimension, "bucket": bucket, "metric": metric, **interval})
    return output


def numeric_range_intervals(metrics: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    """@guang.yang-comment
    Input: per-sample LQ metrics for paired normalized-range conditions.
    Output: bootstrap intervals for alternative-minus-baseline CER and NED deltas.
    Key Steps: pair C00/C01 and C03/C04 by sample_id, then resample each metric delta.
    Key Params: 2,000 bootstrap iterations and the experiment seed preserve the run identity.
    State/Dependencies: read-only aggregation of finished metrics.
    """
    output = []
    for baseline, alternative in (("C00-SRC", "C01-SRC-01"), ("C03-DYN-NORM", "C04-DYN-01")):
        baseline_rows = {
            row["sample_id"]: row
            for row in metrics
            if row["condition"] == baseline and row["view"] == "lq" and row.get("text_valid") == "True"
        }
        alternative_rows = {
            row["sample_id"]: row
            for row in metrics
            if row["condition"] == alternative and row["view"] == "lq" and row.get("text_valid") == "True"
        }
        for metric in ("cer", "ned"):
            values = [
                float(alternative_rows[sample][metric]) - float(baseline_rows[sample][metric])
                for sample in sorted(set(baseline_rows) & set(alternative_rows))
                if number(baseline_rows[sample], metric) is not None and number(alternative_rows[sample], metric) is not None
            ]
            interval = bootstrap(values, seed + len(output))
            output.append({
                "baseline": baseline,
                "alternative": alternative,
                "metric": metric,
                "delta_definition": "alternative_minus_baseline",
                **interval,
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """@guang.yang-comment
    Input: output path and row mappings.
    Output: stable UTF-8 CSV artifact.
    Key Steps: infer ordered fields from first appearance and write empty values for missing cells.
    Key Params: newline='' preserves portable CSV formatting.
    State/Dependencies: writes only the requested path.
    """
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    """@guang.yang-comment
    Input: one summary scalar.
    Output: compact report text with NA for missing values.
    Key Steps: handle null/non-finite values and format finite floats.
    Key Params: six significant digits balance readability and evidence precision.
    State/Dependencies: pure formatting.
    """
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.6g}"
    return str(value)


def append_report(
    report_path: Path,
    stratified: list[dict[str, Any]],
    cp_summary: list[dict[str, Any]],
    mode_summary: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    range_intervals: list[dict[str, Any]],
    source_root: str,
) -> None:
    """@guang.yang-comment
    Input: report path and completed stratified/mode/interval summaries.
    Output: the report extended with measured subgroup evidence and corrected data-source wording.
    Key Steps: replace stale source wording, remove any prior generated appendix, then append compact tables for all required strata.
    Key Params: language, long-text, high-density, aspect, and deformation groups are explicitly surfaced.
    State/Dependencies: reads and rewrites only the finished Markdown report.
    """
    text = report_path.read_text(encoding="utf-8")
    text = text.replace("paired with `52mm`", f"paired with `{source_root}`")
    generated_marker = "\n## 7. Stratified measured results"
    if generated_marker in text:
        text = text.split(generated_marker, 1)[0].rstrip()
    text = text.replace("Best OCR condition by measured CER/NED:", "Best overall OCR row (includes non-candidate train diagnostics):")
    text = text.replace(
        "Best CP ordering condition by measured P(D_pos < D_neg): `C04-DYN-01`.",
        "Best CP ordering among normalized eval candidates by measured P(D_pos < D_neg): `C04-DYN-01` (C05 ties numerically but is rejected by OCR/activation/gradient evidence).",
    )
    lines = ["", "## 7. Stratified measured results", "", "The following tables are from LQ rows in the same frozen manifest.", "", "| Condition | Dimension | Bucket | N | CER | NED | Blank rate |", "|---|---|---|---:|---:|---:|---:|"]
    selected = [row for row in stratified if row["view"] == "lq" and (row["dimension"], row["bucket"]) in {("language", "zh"), ("language", "en"), ("length_bucket", ">32"), ("structure_bucket", "high"), ("aspect_ratio", ">=16"), ("deformation", ">=0.75")} and row["condition"] in {"C00-SRC", "C03-DYN-NORM", "C06-TATSR"}]
    for row in selected:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (row["condition"], row["dimension"], row["bucket"], row["valid_text_n"], format_value(row["cer_mean"]), format_value(row["ned_mean"]), format_value(row["blank_rate"])))
    lines.extend(["", "## 8. Train-mode diagnostics", "", "Train-mode rows are diagnostic only. Positive CER or negative NED means train mode is worse than the paired eval mode.", "", "| Eval | Train | Group | N | CER train-eval | NED train-eval | Prediction disagreement |", "|---|---|---|---:|---:|---:|---:|"])
    for row in mode_summary:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (row["eval_condition"], row["train_condition"], row["group"], row["n"], format_value(row["cer_train_minus_eval"]), format_value(row["ned_train_minus_eval"]), format_value(row["prediction_disagreement_rate"])))
    lines.extend(["", "## 9. Extended fixed-width intervals", "", "| Fixed | Group dimension | Bucket | Metric | N | Mean fixed-aspect | 95% CI |", "|---|---|---|---|---:|---:|---|"])
    selected_intervals = [row for row in intervals if row["dimension"] in {"language", "length_bucket", "structure_bucket", "aspect_ratio", "deformation"} and (row["bucket"] in {"zh", "en", ">32", "high", ">=16", ">=0.75"})]
    for row in selected_intervals:
        lines.append("| %s | %s | %s | %s | %s | %s | [%s, %s] |" % (row["fixed"], row["dimension"], row["bucket"], row["metric"], row["n"], format_value(row["mean"]), format_value(row["low"]), format_value(row["high"])))
    lines.extend(["", "## 10. CP subgroup evidence", "", "| Condition | Dimension | Bucket | N | P(pos<neg) | Margin |", "|---|---|---|---:|---:|---:|"])
    selected_cp = [row for row in cp_summary if (row["dimension"], row["bucket"]) in {("language", "zh"), ("language", "en"), ("length_bucket", ">32"), ("structure_bucket", "high")} and row["condition"] in {"C00-SRC", "C03-DYN-NORM", "C04-DYN-01", "C06-TATSR"}]
    for row in selected_cp:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (row["condition"], row["dimension"], row["bucket"], row["n"], format_value(row["pos_better_probability"]), format_value(row["margin_mean"])))
    candidate_rows = [row for row in stratified if row["view"] == "lq" and row["dimension"] == "all" and row["condition"] in {"C00-SRC", "C01-SRC-01", "C02-SRC-255", "C03-DYN-NORM", "C04-DYN-01", "C05-REALCE", "C06-TATSR"}]
    lines.extend(["", "## 11. Eval-only candidate selection", "", "C10/C11 are excluded from this table because train mode is a diagnostic, not a frozen-loss candidate.", "", "| Candidate | N | CER | NED |", "|---|---:|---:|---:|"])
    for row in candidate_rows:
        lines.append("| %s | %s | %s | %s |" % (row["condition"], row["valid_text_n"], format_value(row["cer_mean"]), format_value(row["ned_mean"])))
    lines.extend(["", "## 12. Normalized-range paired intervals", "", "Deltas are alternative minus baseline on the same LQ text rows; intervals including zero do not establish a statistically decisive range preference.", "", "| Baseline | Alternative | Metric | N | Mean delta | Bootstrap 95% CI |", "|---|---|---|---:|---:|---|"])
    for row in range_intervals:
        lines.append("| %s | %s | %s | %s | %s | [%s, %s] |" % (row["baseline"], row["alternative"], row["metric"], row["n"], format_value(row["mean"]), format_value(row["low"]), format_value(row["high"])))
    report_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """@guang.yang-comment
    Input: command-line result directory.
    Output: validated argument namespace.
    Key Steps: register the finished run path.
    Key Params: no inference options are accepted; this tool is post-processing only.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--real-root", default="26mm")
    return parser.parse_args()


def main() -> None:
    """@guang.yang-comment
    Input: a finished CRNN result directory and the real-camera root label.
    Output: stratified_summary.csv, cp_stratified_summary.csv, mode_diagnostics.csv, fixed_geometry_intervals.csv, numeric_range_intervals.csv, and an extended report.
    Key Steps: load finished CSVs, derive all required strata, compute paired intervals, and append evidence without model execution.
    Key Params: seed 20260814 and 2,000 bootstrap iterations preserve the experiment identity.
    State/Dependencies: reads/writes only the supplied result directory.
    """
    args = parse_args()
    metrics = load_csv(args.results_dir / "per_sample_metrics.csv")
    cp_rows = load_csv(args.results_dir / "cp_pair_ranking.csv")
    stratified = stratified_metrics(metrics)
    cp_summary = stratified_cp(cp_rows)
    mode_summary = mode_diagnostics(metrics)
    intervals = fixed_intervals(metrics, 20260814)
    range_intervals = numeric_range_intervals(metrics, 20260814)
    write_csv(args.results_dir / "stratified_summary.csv", stratified)
    write_csv(args.results_dir / "cp_stratified_summary.csv", cp_summary)
    write_csv(args.results_dir / "mode_diagnostics.csv", mode_summary)
    write_csv(args.results_dir / "fixed_geometry_intervals.csv", intervals)
    write_csv(args.results_dir / "numeric_range_intervals.csv", range_intervals)
    append_report(args.results_dir / "report.md", stratified, cp_summary, mode_summary, intervals, range_intervals, args.real_root)
    summary = {"stratified_rows": len(stratified), "cp_stratified_rows": len(cp_summary), "mode_rows": len(mode_summary), "interval_rows": len(intervals), "numeric_range_interval_rows": len(range_intervals)}
    (args.results_dir / "logs" / "postprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
