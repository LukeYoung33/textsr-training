"""Visualize P17 preprocessing support and luma transfer on affected background regions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.lenovo_p17_preprocess import preprocess_lq_rgb  # noqa: E402
from realesrgan.lenovo_p17_torch_gpu import (  # noqa: E402
    _contour_weight,
    _estimate_sigma,
    _gaussian_blur,
    _otsu_threshold,
    _segment_text,
    _wiener_deblur,
)
from tools.infer_basic_text_line_full_image import load_lq_preprocess_config  # noqa: E402


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(rgb * 255.0 + .5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(path)


def _starts(length: int, patch: int, stride: int) -> list[int]:
    return sorted(set(list(range(0, max(length - patch, 0) + 1, stride)) + [max(length - patch, 0)]))


def _centers(stats: torch.Tensor) -> tuple[float, float, float, float, bool]:
    threshold, _ = _otsu_threshold(stats[None])
    encoded = stats.mul(255.0).round().clamp(0, 255).long()
    high = encoded > threshold.mul(255.0).round().long()[:, None, None]
    low_values, high_values = stats[~high[0]], stats[high[0]]
    if not low_values.numel() or not high_values.numel():
        return 0., 0., float((~high[0]).float().mean()), float(high[0].float().mean()), False
    def trimmed_center(values: torch.Tensor) -> torch.Tensor:
        bounds = torch.quantile(values, torch.tensor((.35, .65), device=values.device, dtype=values.dtype))
        middle = values[(values >= bounds[0]) & (values <= bounds[1])]
        return torch.quantile(middle if middle.numel() else values, .5)
    low, high_value = trimmed_center(low_values), trimmed_center(high_values)
    low_fraction, high_fraction = float((~high[0]).float().mean()), float(high[0].float().mean())
    dynamic_range = float(stats.max() - stats.min())
    valid = low_fraction >= .12 and high_fraction >= .12 and abs(float(high_value - low)) >= max(12. / 255., .08 * dynamic_range)
    return float(min(low, high_value)), float(max(low, high_value)), low_fraction, high_fraction, valid


def _select_regions(score: np.ndarray, count: int, min_distance: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    remaining = score.copy()
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
        if remaining[y, x] <= 0:
            break
        selected.append((int(x), int(y)))
        cv2.circle(remaining, (int(x), int(y)), min_distance, 0.0, thickness=-1)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", type=int, default=3)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="restrict peak selection to an image-space rectangle")
    parser.add_argument("--model-output", type=Path,
                        help="optional full-frame output from the matching preprocess checkpoint")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    model_rgb: np.ndarray | None = None
    if args.model_output is not None:
        model_bgr = cv2.imread(str(args.model_output), cv2.IMREAD_COLOR)
        if model_bgr is None or model_bgr.shape != bgr.shape:
            raise ValueError("model output must exist and match input image shape")
        model_rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)[None].to(device)
    params = load_lq_preprocess_config(args.config)
    if not params["enabled"]:
        raise ValueError("diagnostic requires enabled lq_preprocess")
    with torch.no_grad():
        y = (0.299 * rgb[:, 0:1] + .587 * rgb[:, 1:2] + .114 * rgb[:, 2:3]).clamp(0, 1)
        mask, _ = _segment_text(y, params)
        sigma = _estimate_sigma(y, mask, params)
        deblurred = _wiener_deblur(y, mask, sigma, float(params["lambda"]))
        preprocessed = preprocess_lq_rgb(rgb, params)
    pre_np = preprocessed[0].permute(1, 2, 0).cpu().numpy()
    y_np = y[0, 0].cpu().numpy()
    pre_y = (.299 * pre_np[..., 0] + .587 * pre_np[..., 1] + .114 * pre_np[..., 2])
    contour = _contour_weight(mask, int(params["p16_write_full_radius"]), int(params["p16_write_zero_radius"]))[0, 0].cpu().numpy()
    text = mask[0, 0].cpu().numpy() > .5
    h, w = y_np.shape
    # Background-only ranking avoids merely selecting the intended glyph cores.
    score = np.abs(pre_y - y_np) * contour * (~text)
    if args.roi is not None:
        x0, y0, x1, y1 = args.roi
        allowed = np.zeros_like(score, dtype=bool)
        allowed[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
        score[~allowed] = 0.0
    regions = _select_regions(score, args.regions, min_distance=96)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    _write_rgb(out / "lq.png", rgb_np)
    _write_rgb(out / "preprocessed.png", pre_np)
    cv2.imwrite(str(out / "text_mask.png"), (text.astype(np.uint8) * 255))
    cv2.imwrite(str(out / "write_weight.png"), np.clip(contour * 255., 0, 255).astype(np.uint8))
    cv2.imwrite(str(out / "abs_luma_delta_x16.png"), np.clip(np.abs(pre_y - y_np) * 16. * 255., 0, 255).astype(np.uint8))

    patch_size, stride = int(params["p17_patch_size"]), int(params["p17_patch_stride"])
    ys, xs = _starts(h, patch_size, stride), _starts(w, patch_size, stride)
    overlay = bgr.copy()
    report: list[dict[str, float | int | list[int] | bool]] = []
    figure, axes = plt.subplots(1, len(regions), figsize=(5.2 * max(1, len(regions)), 4.2), squeeze=False)
    for number, (x, y_coord) in enumerate(regions, 1):
        left = min(xs, key=lambda value: abs((value + patch_size / 2.) - x))
        top = min(ys, key=lambda value: abs((value + patch_size / 2.) - y_coord))
        right, bottom = min(w, left + patch_size), min(h, top + patch_size)
        stats = _gaussian_blur(deblurred[:, :, top:bottom, left:right], float(params["p17_stats_gaussian_sigma"]))[0, 0]
        low, high, low_fraction, high_fraction, valid = _centers(stats)
        crop_radius = 56
        cx0, cy0, cx1, cy1 = max(0, x - crop_radius), max(0, y_coord - crop_radius), min(w, x + crop_radius), min(h, y_coord + crop_radius)
        panel = np.hstack((
            cv2.cvtColor((rgb_np[cy0:cy1, cx0:cx1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.cvtColor((pre_np[cy0:cy1, cx0:cx1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            cv2.applyColorMap(np.clip(np.abs(pre_y[cy0:cy1, cx0:cx1] - y_np[cy0:cy1, cx0:cx1]) * 16. * 255., 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO),
            cv2.applyColorMap(np.clip(contour[cy0:cy1, cx0:cx1] * 255., 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS),
        ))
        cv2.imwrite(str(out / f"region_{number:02d}_lq_pre_delta_weight.png"), panel)
        # Save the same diagnostics independently: viewers can then inspect
        # the support map without losing scale to a four-way mosaic.
        _write_rgb(out / f"region_{number:02d}_lq.png", rgb_np[cy0:cy1, cx0:cx1])
        _write_rgb(out / f"region_{number:02d}_preprocessed.png", pre_np[cy0:cy1, cx0:cx1])
        if model_rgb is not None:
            _write_rgb(out / f"region_{number:02d}_model_output.png", model_rgb[cy0:cy1, cx0:cx1])
            model_y = .299 * model_rgb[..., 0] + .587 * model_rgb[..., 1] + .114 * model_rgb[..., 2]
            cv2.imwrite(
                str(out / f"region_{number:02d}_model_minus_preprocessed_y_x16.png"),
                np.clip(np.abs(model_y[cy0:cy1, cx0:cx1] - pre_y[cy0:cy1, cx0:cx1]) * 16. * 255., 0, 255).astype(np.uint8),
            )
        cv2.imwrite(
            str(out / f"region_{number:02d}_abs_luma_delta_x16.png"),
            np.clip(np.abs(pre_y[cy0:cy1, cx0:cx1] - y_np[cy0:cy1, cx0:cx1]) * 16. * 255., 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(out / f"region_{number:02d}_write_weight.png"),
            np.clip(contour[cy0:cy1, cx0:cx1] * 255., 0, 255).astype(np.uint8),
        )
        cv2.imwrite(str(out / f"region_{number:02d}_text_mask.png"), (text[cy0:cy1, cx0:cx1].astype(np.uint8) * 255))
        cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (0, 255, 255), 2)
        cv2.circle(overlay, (x, y_coord), 5, (0, 0, 255), -1)
        cv2.putText(overlay, f"R{number}", (left + 3, top + 20), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2, cv2.LINE_AA)
        axis = axes[0, number - 1]
        values = np.linspace(0., 1., 512)
        # This is the transfer implemented today: a two-sided clamp.  It is
        # shown explicitly so it cannot be mistaken for a platform-preserving S-curve.
        mapped = np.clip(values, low, high)
        axis.plot(values, mapped, color="#d62728", linewidth=2, label="implemented transfer")
        axis.plot(values, values, "--", color="0.45", label="identity")
        axis.axvline(low, color="#1f77b4", linestyle=":", label=f"low={low * 255:.1f}")
        axis.axvline(high, color="#2ca02c", linestyle=":", label=f"high={high * 255:.1f}")
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="deblurred Y", ylabel="mapped Y", title=f"R{number}: patch ({left},{top})")
        axis.grid(alpha=.25); axis.legend(fontsize=7, loc="lower right")
        report.append({"region": number, "peak_xy": [x, y_coord], "patch_xyxy": [left, top, right, bottom], "low_center_255": low * 255., "high_center_255": high * 255., "low_fraction": low_fraction, "high_fraction": high_fraction, "valid": valid, "peak_abs_luma_delta_255": float(abs(pre_y[y_coord, x] - y_np[y_coord, x]) * 255.), "peak_write_weight": float(contour[y_coord, x])})
    figure.suptitle("Actual P17 regional Otsu transfer: clamp, not an S-curve", fontsize=12)
    figure.tight_layout(); figure.savefig(out / "regional_otsu_transfer_curves.png", dpi=180); plt.close(figure)
    cv2.imwrite(str(out / "overview_regions_and_processing_patches.png"), overlay)
    (out / "report.json").write_text(json.dumps({"image": str(args.image), "config": str(args.config), "regions": report}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
