"""Deterministic functional smoke for the new stroke geometry losses.

This is a plumbing test, not an image-quality experiment.  It uses a small
synthetic dark-text canvas with an invalid right-padding strip and checks:

* identical prediction is near the minimum;
* centerline deletion raises GT-coverage loss;
* an external spur lowers prediction-skeleton precision;
* a controlled wider stroke raises soft-distance width loss;
* prediction gradients are finite and exactly zero in invalid padding.

No network, checkpoint, dataset, or formal training process is started.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realesrgan.losses.stroke_geometry_losses import SoftCLDiceLoss, StrokeWidthLoss


def _make_canvas() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """@luna-comment Create a fixed dark-stroke canvas and valid-padding mask."""
    height, width = 48, 96
    valid = torch.ones(1, 1, height, width)
    valid[:, :, :, 90:] = 0.0
    text = torch.zeros_like(valid)
    text[:, :, 20:24, 8:72] = 1.0
    text[:, :, 12:34, 36:40] = 1.0
    rgb_text = text.expand(-1, 3, -1, -1)
    gt = rgb_text * 0.0 + (1.0 - rgb_text)
    return gt, text, valid


def _check_loss(name: str, loss_fn: torch.nn.Module, gt: torch.Tensor, text: torch.Tensor, valid: torch.Tensor) -> dict:
    """@luna-comment Run identity and controlled perturbation checks for one loss."""
    base, base_details = loss_fn(gt, gt, text, valid, return_details=True)

    broken = gt.clone()
    broken[:, :, 21:23, 40:45] = 0.5
    broken_loss, broken_details = loss_fn(broken, gt, text, valid, return_details=True)

    spur = gt.clone()
    spur[:, :, 30:34, 60:68] = 0.0
    spur_loss, spur_details = loss_fn(spur, gt, text, valid, return_details=True)

    widened = gt.clone()
    widened[:, :, 18:26, 48:68] = 0.0
    widened_loss, widened_details = loss_fn(widened, gt, text, valid, return_details=True)

    probe = gt.clone().requires_grad_(True)
    probe_loss, probe_details = loss_fn(probe, gt, text, valid, return_details=True)
    probe_loss.backward()
    padding_grad = probe.grad[:, :, :, 90:].abs().max().item()
    grad_norm = probe.grad.abs().sum().item()

    result = {
        "base": float(base.detach()),
        "broken": float(broken_loss.detach()),
        "spur": float(spur_loss.detach()),
        "widened": float(widened_loss.detach()),
        "finite": all(torch.isfinite(v.detach()).item() for v in (base, broken_loss, spur_loss, widened_loss, probe_loss)),
        "gradient_l1": float(grad_norm),
        "padding_gradient_max": float(padding_grad),
        "active": float(base_details["active"].mean()),
    }
    if name == "cldice":
        result.update({
            "broken_gt_coverage": float(broken_details["gt_coverage"].mean()),
            "base_gt_coverage": float(base_details["gt_coverage"].mean()),
            "spur_pred_precision": float(spur_details["pred_precision"].mean()),
            "base_pred_precision": float(base_details["pred_precision"].mean()),
        })
    else:
        result.update({
            "base_width_error": float(base_details["width_abs_error"].mean()),
            "widened_width_error": float(widened_details["width_abs_error"].mean()),
            "widened_predicted_width": float(widened_details["predicted_width_at_gt_skeleton"].mean()),
            "gt_width": float(widened_details["gt_width_at_gt_skeleton"].mean()),
        })
    return result


def main() -> int:
    """@luna-comment Execute both loss contracts and serialize pass/fail evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="", help="optional JSON output path")
    args = parser.parse_args()
    torch.manual_seed(20260907)
    gt, text, valid = _make_canvas()
    results = {
        "cldice": _check_loss("cldice", SoftCLDiceLoss(), gt, text, valid),
        "stroke_width": _check_loss("stroke_width", StrokeWidthLoss(), gt, text, valid),
    }
    # A weak (1.5% luma-contrast) GT stroke must still be an active example.
    # The same *relative* prediction error should not have a weaker gradient
    # than a high-contrast stroke merely because the page is faint.
    def weak_gradient(contrast: float) -> tuple[float, float, float, float]:
        gt, text, valid = _make_canvas()
        gt = (1.0 - text).expand(-1, 3, -1, -1) * contrast + (1.0 - contrast)
        probe = gt.clone()
        probe[:, :, 21:23, 40:45] += contrast * 0.25
        probe.requires_grad_(True)
        cl_loss, cl_details = SoftCLDiceLoss()(probe, gt, text, valid, return_details=True)
        width_loss, width_details = StrokeWidthLoss()(probe, gt, text, valid, return_details=True)
        (cl_loss + width_loss).backward()
        region_grad = probe.grad[:, :, 21:23, 40:45].abs().mean().item()
        return (
            float(cl_details["active"].mean()),
            float(width_details["active"].mean()),
            float(region_grad),
            float((cl_loss + width_loss).detach()),
        )

    weak = weak_gradient(0.015)
    strong = weak_gradient(0.30)
    checks = {
        "cldice_identity_small": results["cldice"]["base"] < 1.0e-2,
        "cldice_break_detected": results["cldice"]["broken"] > results["cldice"]["base"] + 1.0e-3,
        "cldice_spur_detected": results["cldice"]["spur"] > results["cldice"]["base"] + 1.0e-3,
        "width_identity_small": results["stroke_width"]["base"] < 1.0e-2,
        "width_widening_detected": results["stroke_width"]["widened"] > results["stroke_width"]["base"] + 1.0e-2,
        "finite": all(item["finite"] for item in results.values()),
        "nonzero_gradients": all(item["gradient_l1"] > 1.0e-8 for item in results.values()),
        "invalid_padding_zero_grad": all(item["padding_gradient_max"] == 0.0 for item in results.values()),
        "active_gt": all(item["active"] > 0.0 for item in results.values()),
        "weak_contrast_active": weak[0] == 1.0 and weak[1] == 1.0,
        "weak_contrast_gradient_not_smaller": weak[2] >= strong[2],
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "script": "smoke_stroke_geometry_losses.py",
        "seed": 20260907,
        "canvas": {"height": 48, "width": 96, "invalid_padding_columns": [90, 95]},
        "results": results,
        "contrast_normalization": {
            "weak_contrast": 0.015,
            "strong_contrast": 0.30,
            "weak": {"cldice_active": weak[0], "width_active": weak[1], "region_grad": weak[2], "combined_loss": weak[3]},
            "strong": {"cldice_active": strong[0], "width_active": strong[1], "region_grad": strong[2], "combined_loss": strong[3]},
        },
        "checks": checks,
        "interpretation": "functional loss plumbing only; no model quality claim",
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
