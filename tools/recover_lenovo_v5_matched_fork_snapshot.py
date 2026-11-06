"""Independently recover one saved matched-fork snapshot without editing its run report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROIS = {"stroke_a": (slice(14, 39), slice(171, 177)), "stroke_b": (slice(55, 80), slice(197, 201))}
RAW_RMS_LIMIT = 1e-4


def raw_rgb(path: Path) -> np.ndarray:
    with np.load(path) as content:
        value = np.asarray(content["raw_rgb"], dtype=np.float64)
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"expected BCHW raw_rgb in {path}, got {value.shape}")
    return value


def profile_fft(rgb: np.ndarray, roi: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(rgb, 0.0, 1.0)
    y = .299 * clipped[:, 0] + .587 * clipped[:, 1] + .114 * clipped[:, 2]
    profile = y[0, roi[0], roi[1]].mean(axis=1)
    index = np.arange(profile.size, dtype=np.float64)
    design = np.stack((np.ones_like(index), index, index * index), axis=1)
    detrended = profile - design @ np.linalg.lstsq(design, profile, rcond=None)[0]
    return np.fft.rfft(detrended * np.hanning(profile.size)), np.fft.rfftfreq(profile.size)


def display_metrics(rgb: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, roi in ROIS.items():
        spectrum, frequency = profile_fft(rgb, roi)
        band = (frequency >= .25) & (frequency <= .4)
        result[name] = {"band_frequencies": frequency[band].tolist(),
                        "band_energy": float(np.sum(np.abs(spectrum[band]) ** 2)),
                        "band_coefficients_real": spectrum[band].real.tolist(),
                        "band_coefficients_imag": spectrum[band].imag.tolist()}
    return result


def difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    delta = first - second
    return {"raw_rms": float(np.sqrt(np.mean(delta * delta))), "raw_max_abs": float(np.max(np.abs(delta)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_folder.resolve()
    arrays = {arm: raw_rgb(root / arm / f"step_{args.step:06d}" / "historical_output.npz")
              for arm in ("off_a", "off_b", "on")}
    metrics = {arm: display_metrics(value) for arm, value in arrays.items()}
    aa_raw, on_raw = difference(arrays["off_a"], arrays["off_b"]), difference(arrays["on"], arrays["off_a"])
    roi_gate: dict[str, Any] = {}
    for roi in ROIS:
        off_energy = metrics["off_a"][roi]["band_energy"]
        aa_delta = metrics["off_a"][roi]["band_energy"] - metrics["off_b"][roi]["band_energy"]
        on_delta = metrics["on"][roi]["band_energy"] - off_energy
        threshold = max(1e-5, .01 * off_energy)
        roi_gate[roi] = {"off_a_band_energy": off_energy, "off_b_band_energy": metrics["off_b"][roi]["band_energy"],
                         "on_band_energy": metrics["on"][roi]["band_energy"], "aa_signed_energy_delta": aa_delta,
                         "aa_abs_energy_delta": abs(aa_delta), "on_minus_off_signed_energy_delta": on_delta,
                         "on_minus_off_abs_energy_delta": abs(on_delta), "aa_threshold": threshold,
                         "aa_gate_pass": abs(aa_delta) <= threshold,
                         "on_over_aa_abs_energy_ratio": None if abs(aa_delta) == 0 else abs(on_delta) / abs(aa_delta)}
    output = {"status": "RECOVERED_FROM_NPZ_ONLY", "run_report_not_modified": True, "step": args.step,
              "domain": {"raw_difference": "unclipped float RGB", "band_metric": "clipped float RGB luma; quadratic detrend; Hann; 0.25--0.40 cycles/pixel"},
              "raw_comparison": {"aa_off_a_minus_off_b": aa_raw, "on_minus_off_a": on_raw,
                                 "on_over_aa_raw_rms_ratio": None if aa_raw["raw_rms"] == 0 else on_raw["raw_rms"] / aa_raw["raw_rms"]},
              "display_metrics": metrics, "aa_band_gate": {"raw_rms_limit": RAW_RMS_LIMIT,
                  "raw_rms_pass": aa_raw["raw_rms"] < RAW_RMS_LIMIT, "rois": roi_gate,
                  "pass": aa_raw["raw_rms"] < RAW_RMS_LIMIT and all(item["aa_gate_pass"] for item in roi_gate.values())},
              "effect_note": "ON-minus-OFF ratios are descriptive for this aborted step; they are not a final causal result."}
    destination = args.output or root / f"recovered_snapshot{args.step}.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(destination), "aa_gate_pass": output["aa_band_gate"]["pass"]}))


if __name__ == "__main__":
    main()
