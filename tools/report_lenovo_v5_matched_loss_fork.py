"""Read and render an append-only Lenovo V5 matched-loss-fork result folder.

This is deliberately independent of the fork runner: it consumes only
``run_report.json``, ``steps.jsonl``, and optional saved snapshot arrays.  A
missing field or snapshot is reported as unavailable rather than inferred.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROIS = ("stroke_a", "stroke_b")
CROP = (145, 0, 230, 96)


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def cell(record: dict[str, Any] | None) -> str:
    if not record:
        return "unavailable"
    energy = record.get("band_energy")
    phase = record.get("phase_to_original_r1_250k")
    mean = record.get("mean_luminance")
    values = ("NA" if energy is None else f"{float(energy):.6g}",
              "NA" if phase is None else f"{float(phase):+.4f}",
              "NA" if mean is None else f"{float(mean):.6f}")
    return "energy=%s; phase=%s; meanY=%s" % values


def arm_metrics(snapshot: dict[str, Any], arm: str, roi: str) -> dict[str, Any] | None:
    return snapshot.get("arms", {}).get(arm, {}).get("metrics", {}).get("rois", {}).get(roi)


def equal_data(event: dict[str, Any]) -> bool | None:
    hashes = event.get("arm_data_hashes")
    expected = event.get("data_hashes")
    if not isinstance(hashes, dict) or not isinstance(expected, dict):
        return None
    return all(value == expected for value in hashes.values())


def cp_result(event: dict[str, Any]) -> bool | None:
    losses = event.get("losses")
    if not isinstance(losses, dict):
        return None
    for loss in losses.values():
        scale = loss.get("cp_scale")
        if scale is None or not np.isclose(float(scale), 0.0005, rtol=1e-6, atol=1e-10):
            return False
        eligible = float(loss.get("cp_eligible_lines", 0.0))
        if eligible > 0 and abs(float(loss.get("l_cp_raw", 0.0))) == 0.0:
            return False
    return True


def lr_result(event: dict[str, Any]) -> tuple[bool | None, str]:
    values = event.get("lr")
    if not isinstance(values, dict) or not values:
        return None, "unavailable"
    normalized = {name: tuple(float(v) for v in value) if isinstance(value, list) else (float(value),)
                  for name, value in values.items()}
    answer = len(set(normalized.values())) == 1
    return answer, ", ".join(f"{name}={value}" for name, value in normalized.items())


def group_l2(groups: Any) -> float | None:
    """Return the all-group L2 from runner group-difference records, if present."""
    if not isinstance(groups, dict) or not groups:
        return None
    values = [entry.get("l2") for entry in groups.values() if isinstance(entry, dict)]
    if not values or any(value is None for value in values):
        return None
    return float(np.sqrt(sum(float(value) ** 2 for value in values)))


def parameter_noise_vs_effect(event: dict[str, Any]) -> dict[str, Any]:
    """Compare observed OFF A/A parameter divergence with ON-minus-OFF effect."""
    diffs = event.get("parameter_group_diffs")
    if not isinstance(diffs, dict):
        return {"status": "not_measured"}
    noise = group_l2(diffs.get("off_a_vs_off_b"))
    effect = group_l2(diffs.get("on_vs_off_a"))
    if noise is None or effect is None:
        return {"status": "not_measured"}
    return {"status": "measured", "aa_parameter_l2": noise, "on_minus_off_parameter_l2": effect,
            "on_over_aa_l2": None if noise == 0.0 else effect / noise}


def raw_snapshot_noise_vs_effect(run_folder: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compare saved raw historical RGB A/A noise and ON-minus-OFF effect when all files exist."""
    arrays: dict[str, np.ndarray] = {}
    for arm in ("off_a", "off_b", "on"):
        path = run_folder / arm / f"step_{int(snapshot.get('step', -1)):06d}" / "historical_output.npz"
        if not path.is_file():
            return {"status": "not_measured", "reason": f"missing {path.relative_to(run_folder)}"}
        with np.load(path) as payload:
            if "raw_rgb" not in payload:
                return {"status": "not_measured", "reason": f"raw_rgb absent in {path.relative_to(run_folder)}"}
            arrays[arm] = np.asarray(payload["raw_rgb"], dtype=np.float64)
    noise = arrays["off_a"] - arrays["off_b"]
    effect = arrays["on"] - arrays["off_a"]
    noise_rms = float(np.sqrt(np.mean(noise * noise)))
    effect_rms = float(np.sqrt(np.mean(effect * effect)))
    return {"status": "measured", "aa_raw_rms": noise_rms, "on_minus_off_raw_rms": effect_rms,
            "on_over_aa_raw_rms": None if noise_rms == 0.0 else effect_rms / noise_rms,
            "aa_raw_max_abs": float(np.max(np.abs(noise))), "on_minus_off_raw_max_abs": float(np.max(np.abs(effect)))}


def inspect(report: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots = sorted(report.get("snapshots", []), key=lambda item: int(item.get("step", -1)))
    aa_snapshots = []
    for snapshot in snapshots:
        arms = snapshot.get("arms", {})
        aa_snapshots.append({"step": snapshot.get("step"), "equal": (
            arms.get("off_a", {}).get("historical_raw_rgb_digest") ==
            arms.get("off_b", {}).get("historical_raw_rgb_digest")) if {"off_a", "off_b"} <= set(arms) else None})
    rows = []
    for event in events:
        lr_equal, lr_values = lr_result(event)
        rows.append({"step": event.get("relative_step"), "absolute_iter": event.get("absolute_iter"),
                     "off_aa_equal": event.get("off_aa_equal"), "data_hashes_same": equal_data(event),
                     "cp_present": cp_result(event), "lr_equal": lr_equal, "lr": lr_values})
        rows[-1]["parameter_noise_vs_effect"] = parameter_noise_vs_effect(event)
    exact = all(item["off_aa_equal"] is True for item in rows) if rows else None
    return {"run_status": report.get("status", "unavailable"), "contract": report.get("contract"),
            "loss_contract": report.get("loss_contract"), "snapshots": snapshots,
            "aa_snapshot_checks": aa_snapshots, "step_checks": rows,
            "all_step_aa_equal": exact,
            "aa_interpretation": "exact_reproducibility_observed" if exact is True else (
                "measured_nondeterministic_noise_pending_root_acceptance" if exact is False else "not_measured"),
            "all_data_hashes_same": all(item["data_hashes_same"] is True for item in rows) if rows else None,
            "all_cp_present": all(item["cp_present"] is True for item in rows) if rows else None,
            "all_lr_equal": all(item["lr_equal"] is True for item in rows) if rows else None}


def render_crop(run_folder: Path, snapshots: list[dict[str, Any]], output: Path) -> list[str]:
    selected = []
    # The start is common by construction; show it once, then compare final OFF/ON.
    if snapshots:
        initial, final = snapshots[0], snapshots[-1]
        for arm, snapshot in (("off_a", initial), ("off_a", final), ("on", final)):
            path = run_folder / arm / f"step_{int(snapshot.get('step', -1)):06d}" / "historical_output.npz"
            if path.is_file():
                selected.append((arm, int(snapshot["step"]), path))
    if not selected:
        return []
    panel_w, panel_h, label_h = (CROP[2] - CROP[0]) * 3, (CROP[3] - CROP[1]) * 3, 24
    sheet = Image.new("RGB", (len(selected) * panel_w, panel_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    names = []
    for column, (arm, step, path) in enumerate(selected):
        with np.load(path) as contents:
            if "raw_rgb" not in contents:
                continue
            raw = contents["raw_rgb"]
        array = raw[0] if raw.ndim == 4 else raw
        if array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
        image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).crop(CROP).resize(
            (panel_w, panel_h), Image.Resampling.NEAREST)
        x = column * panel_w
        sheet.paste(image, (x, label_h))
        draw.text((x + 2, 3), f"{arm} step {step}", fill="black")
        names.append(str(path.relative_to(run_folder)))
    if names:
        sheet.save(output)
    return names


def markdown(result: dict[str, Any], crop_files: list[str]) -> str:
    lines = ["# Independent matched-loss-fork reader", "",
             f"- Run status: `{result['run_status']}`", f"- Contract: {result['contract'] or 'unavailable'}", "",
             "## Snapshot display-domain primary prediction metric", "",
             "Each cell is clipped-float prediction luma: 0.25--0.40 cycles/pixel band energy, phase to supplied original R1 250k (if present), and mean luminance. It is not an error metric.", "",
             "| Step | OFF A | OFF B | ON A | ON B |", "|---:|---|---|---|---|"]
    for snapshot in result["snapshots"]:
        values = []
        for arm in ("off_a", "on"):
            for roi in ROIS:
                values.append(cell(arm_metrics(snapshot, arm, roi)))
        lines.append("| %s | %s | %s | %s | %s |" % (snapshot.get("step"), *values))
    lines += ["", "## Integrity checks", "",
              "| Scope | A/A exact equal | shared data hashes | CP present at 0.0005 | equal LR | parameter noise vs ON effect |",
              "|---|---|---|---|---|---|"]
    for row in result["step_checks"]:
        comparison = row["parameter_noise_vs_effect"]
        noise_text = "not_measured" if comparison["status"] != "measured" else (
            "AA=%.6g; ON-OFF=%.6g; ratio=%s" % (comparison["aa_parameter_l2"], comparison["on_minus_off_parameter_l2"],
            "NA" if comparison["on_over_aa_l2"] is None else f"{comparison['on_over_aa_l2']:.6g}"))
        lines.append("| step %s | %s | %s | %s | %s (%s) | %s |" % (
            row["step"], row["off_aa_equal"], row["data_hashes_same"], row["cp_present"], row["lr_equal"], row["lr"], noise_text))
    lines += ["", "Snapshot A/A raw-output checks: " + ", ".join(
        f"step {item['step']}={item['equal']}" for item in result["aa_snapshot_checks"]) + ".",
        "",
        "| Aggregate | Value |", "|---|---|",
        f"| all logged A/A rows exactly equal | {result['all_step_aa_equal']} |",
        f"| A/A interpretation | {result['aa_interpretation']} |",
        f"| all logged arm data hashes equal | {result['all_data_hashes_same']} |",
        f"| all logged CP checks valid | {result['all_cp_present']} |",
        f"| all logged arm LRs equal | {result['all_lr_equal']} |",
        "",
        "An exact A/A false result is recorded as measured nondeterministic noise; it is not automatically classified as invalid. Root acceptance requires a measured ON-minus-OFF effect relative to that noise.",
        "", "## Saved raw-output noise versus ON effect", "",
        "| Snapshot step | Status | A/A raw RMS | ON-minus-OFF raw RMS | Ratio |",
        "|---:|---|---:|---:|---:|"]
    for value in result.get("raw_snapshot_noise_vs_effect", []):
        if value.get("status") == "measured":
            ratio = value.get("on_over_aa_raw_rms")
            lines.append("| %s | measured | %.6g | %.6g | %s |" % (
                value.get("step"), value["aa_raw_rms"], value["on_minus_off_raw_rms"],
                "NA" if ratio is None else f"{ratio:.6g}"))
        else:
            lines.append(f"| {value.get('step')} | not_measured ({value.get('reason', 'fields unavailable')}) | NA | NA | NA |")
    lines += [
        "", "## Optional snapshot display", "",
        "Saved raw-RGB snapshots rendered (fixed clip, no normalization, nearest 3x): " +
        (", ".join(f"`{value}`" for value in crop_files) if crop_files else "unavailable") + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_folder.resolve()
    report_path = root / "run_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    output = (args.output_dir or root / "independent_reader").resolve()
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = inspect(report, load_json_lines(root / "steps.jsonl"))
    result["raw_snapshot_noise_vs_effect"] = [
        {"step": snapshot.get("step"), **raw_snapshot_noise_vs_effect(root, snapshot)} for snapshot in result["snapshots"]]
    output.mkdir(parents=True)
    crop_files = render_crop(root, result["snapshots"], output / "historical_glyph_crop_sheet.png")
    (output / "reader_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "reader_report.md").write_text(markdown(result, crop_files), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(output), "snapshots": len(result["snapshots"]), "events": len(result["step_checks"])}))


if __name__ == "__main__":
    main()
