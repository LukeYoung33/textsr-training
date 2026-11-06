"""Supplement matched-fork primary clipped metrics with raw-luma and clamp diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROIS = {"stroke_a": (slice(14, 39), slice(171, 177)), "stroke_b": (slice(55, 80), slice(197, 201))}


def band_spectrum(y: np.ndarray, roi: tuple[slice, slice]) -> np.ndarray:
    profile = y[roi].mean(axis=1)
    rows = np.arange(profile.size, dtype=np.float64)
    design = np.stack((np.ones_like(rows), rows, rows * rows), axis=1)
    detrended = profile - design @ np.linalg.lstsq(design, profile, rcond=None)[0]
    spectrum = np.fft.rfft(detrended * np.hanning(profile.size))
    frequencies = np.fft.rfftfreq(profile.size)
    return spectrum[(frequencies >= .25) & (frequencies <= .4)]


def band_energy(y: np.ndarray, roi: tuple[slice, slice]) -> float:
    return float(np.sum(np.abs(band_spectrum(y, roi)) ** 2))


def raw_luma(path: Path) -> np.ndarray:
    with np.load(path) as source:
        raw = np.asarray(source["raw_rgb"], dtype=np.float64)
    rgb = raw[0] if raw.ndim == 4 else raw
    if rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)
    return .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]


def metrics(path: Path) -> dict[str, dict[str, float]]:
    with np.load(path) as source:
        raw = np.asarray(source["raw_rgb"], dtype=np.float64)
    rgb = raw[0] if raw.ndim == 4 else raw
    if rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)
    raw_y = .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]
    clipped_rgb = np.clip(rgb, 0, 1)
    clipped_y = .299 * clipped_rgb[..., 0] + .587 * clipped_rgb[..., 1] + .114 * clipped_rgb[..., 2]
    output: dict[str, dict[str, float]] = {}
    for name, roi in ROIS.items():
        raw_region, clipped_region = raw_y[roi], clipped_y[roi]
        rgb_region = rgb[roi]
        output[name] = {"raw_y_band_energy": band_energy(raw_y, roi),
                        "display_clipped_y_band_energy": band_energy(clipped_y, roi),
                        "raw_y_mean": float(raw_region.mean()), "display_clipped_y_mean": float(clipped_region.mean()),
                        "raw_y_row_profile": raw_region.mean(axis=1).tolist(),
                        "display_clipped_y_row_profile": clipped_region.mean(axis=1).tolist(),
                        "rgb_component_clamp_fraction": float(np.mean((rgb_region < 0) | (rgb_region > 1))),
                        "rgb_component_below_zero_fraction": float(np.mean(rgb_region < 0)),
                        "rgb_component_above_one_fraction": float(np.mean(rgb_region > 1))}
    return output


def render_final_profiles(result: dict[str, object], destination: Path) -> None:
    """Render final OFF/ON raw/clipped profiles on one shared y-axis."""
    import matplotlib.pyplot as plt
    final = result["steps"].get("500", {})
    if not all(isinstance(final.get(arm), dict) and "stroke_a" in final[arm] for arm in ("off_a", "on")):
        return
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
    for axis, roi in zip(axes, ROIS):
        for arm, color in (("off_a", "tab:blue"), ("on", "tab:orange")):
            record = final[arm][roi]
            axis.plot(record["raw_y_row_profile"], color=color, linestyle="-", label=f"{arm} raw Y")
            axis.plot(record["display_clipped_y_row_profile"], color=color, linestyle="--", label=f"{arm} clipped Y")
        axis.set_title(f"{roi}: 25-row interior profile")
        axis.set_ylabel("row mean Y")
        axis.grid(alpha=.25)
    axes[-1].set_xlabel("row within ROI")
    axes[0].legend(ncol=2, fontsize=8)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 100, 250, 500])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_folder.resolve()
    result: dict[str, object] = {"domain_note": "Primary fork evidence remains display-clipped luma. This supplement separately measures unclipped raw RGB->Y and RGB component clipping.",
                                 "band": "quadratic detrend, Hann, 0.25--0.40 cycles/pixel", "steps": {}}
    for step in args.steps:
        row: dict[str, object] = {}
        for arm in ("off_a", "off_b", "on"):
            path = root / arm / f"step_{step:06d}" / "historical_output.npz"
            row[arm] = metrics(path) if path.is_file() else {"status": "unavailable", "path": str(path)}
        result["steps"][str(step)] = row
    final_off = root / "off_a" / "step_000500" / "historical_output.npz"
    final_on = root / "on" / "step_000500" / "historical_output.npz"
    if final_off.is_file() and final_on.is_file():
        off_y, on_y = raw_luma(final_off), raw_luma(final_on)
        phase = {}
        for name, roi in ROIS.items():
            off_band, on_band = band_spectrum(off_y, roi), band_spectrum(on_y, roi)
            denominator = np.linalg.norm(off_band) * np.linalg.norm(on_band)
            phase[name] = None if denominator == 0 else float(np.real(np.vdot(off_band, on_band)) / denominator)
        result["final_raw_on_vs_off_band_real_cosine"] = phase
    destination = args.output or root / "raw_clamp_supplement_0_100_250_500_profiles.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot = destination.with_name(destination.stem + "_final_profiles.png")
    render_final_profiles(result, plot)
    print(json.dumps({"status": "COMPLETE", "output": str(destination)}))


if __name__ == "__main__":
    main()
