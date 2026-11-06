"""Deterministic functional smoke for the new stroke-core losses.

The checks are intentionally tensor-only and do not construct a trainer:

* identical prediction is zero for both losses;
* a centerline brightness error activates ``StrokeCoreLoss``;
* local bright/dark envelope violations activate ``OvershootLoss``;
* an invalid right-padding anomaly contributes no value or prediction
  gradient, while a valid anomaly still has a gradient;
* an explicit custom mask can suppress an otherwise active overshoot.
"""

from __future__ import annotations

import json

import torch

from realesrgan.losses.stroke_core_overshoot_loss import OvershootLoss, StrokeCoreLoss


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = 32, 64
    text = torch.zeros(1, 1, height, width)
    # A sufficiently wide glyph-like block has a stable interior skeleton.
    text[:, :, 9:23, 16:44] = 1.0
    valid = torch.ones_like(text)
    valid[:, :, :, 56:] = 0.0
    # Dark text on a bright page in normalized RGB.
    gt = (1.0 - text).repeat(1, 3, 1, 1)
    # Padding is deliberately arbitrary.  It must never become a target or
    # an envelope reference for a valid pixel.
    gt[:, :, :, 56:] = 0.37
    return gt, text, valid


def _assert_close_zero(name: str, value: torch.Tensor, tol: float = 1.0e-7) -> None:
    if not torch.isfinite(value) or float(value.detach().abs().item()) > tol:
        raise AssertionError(f"{name} should be zero, got {value}")


def main() -> None:
    torch.manual_seed(17)
    gt, text, valid = _case()
    core = StrokeCoreLoss()
    overshoot = OvershootLoss(kernel_size=5, tol=2.0 / 255.0, mask_radius=2, mask_mode="text_dilate")

    core_identity = core(gt, gt, text, valid)
    over_identity = overshoot(gt, gt, text, valid)
    _assert_close_zero("core identity", core_identity)
    _assert_close_zero("overshoot identity", over_identity)

    # The center of the block is on/near the detached GT skeleton.  Darken it
    # further to produce a controlled core error without changing the masks.
    core_bad = gt.clone()
    core_bad[:, :, 14:18, 28:32] = 0.35
    core_bad_value = core(core_bad, gt, text, valid)
    if not float(core_bad_value) > 1.0e-5:
        raise AssertionError(f"core controlled error did not activate: {core_bad_value}")

    over_bad = gt.clone()
    # Interior bright overshoot and a dark overshoot near the text edge.
    over_bad[:, :, 14:18, 28:32] = 1.0
    over_bad[:, :, 11:13, 16:20] = -0.25
    over_bad_value, over_details = overshoot(over_bad, gt, text, valid, return_details=True)
    if not float(over_bad_value) > 1.0e-5:
        raise AssertionError(f"overshoot controlled error did not activate: {over_bad_value}")
    if float(over_details["upper_active_fraction"].max()) <= 0.0:
        raise AssertionError("upper overshoot branch did not activate")

    # Both losses must ignore right padding even when its prediction is badly
    # wrong.  Add a valid anomaly too, to prove the gradient is not globally
    # disconnected.
    core_grad_pred = gt.clone().requires_grad_(True)
    with torch.no_grad():
        core_grad_pred[:, :, 14:18, 28:32] = 0.35
        core_grad_pred[:, :, :, 56:] = -3.0
    core_grad_value = core(core_grad_pred, gt, text, valid)
    core_grad_value.backward()
    core_padding_grad = float(core_grad_pred.grad[:, :, :, 56:].abs().max())
    core_active_grad = float(core_grad_pred.grad[:, :, :, :56].abs().max())
    if core_padding_grad > 1.0e-8 or core_active_grad <= 0.0:
        raise AssertionError(
            f"core padding gradient contract failed: padding={core_padding_grad}, active={core_active_grad}"
        )

    over_grad_pred = gt.clone().requires_grad_(True)
    with torch.no_grad():
        over_grad_pred[:, :, 14:18, 28:32] = 1.0
        over_grad_pred[:, :, :, 56:] = -3.0
    over_grad_value = overshoot(over_grad_pred, gt, text, valid)
    over_grad_value.backward()
    over_padding_grad = float(over_grad_pred.grad[:, :, :, 56:].abs().max())
    over_active_grad = float(over_grad_pred.grad[:, :, :, :56].abs().max())
    if over_padding_grad > 1.0e-8 or over_active_grad <= 0.0:
        raise AssertionError(
            f"overshoot padding gradient contract failed: padding={over_padding_grad}, active={over_active_grad}"
        )

    # Explicit custom mask gating: hide the bright anomaly and verify it no
    # longer contributes.  The gate is intersected with valid/text-dilate.
    custom_gate = torch.ones_like(text)
    custom_gate[:, :, 14:18, 28:32] = 0.0
    masked_over = overshoot(over_bad, gt, text, valid, mask=custom_gate)
    if float(masked_over) >= float(over_bad_value):
        raise AssertionError(
            f"custom overshoot mask did not suppress the selected anomaly: full={over_bad_value}, masked={masked_over}"
        )

    result = {
        "core_identity": float(core_identity),
        "overshoot_identity": float(over_identity),
        "core_controlled": float(core_bad_value),
        "overshoot_controlled": float(over_bad_value),
        "overshoot_masked": float(masked_over),
        "core_padding_grad_max": core_padding_grad,
        "overshoot_padding_grad_max": over_padding_grad,
        "core_active_grad_max": core_active_grad,
        "overshoot_active_grad_max": over_active_grad,
        "overshoot_upper_fraction": float(over_details["upper_active_fraction"].max()),
        "overshoot_lower_fraction": float(over_details["lower_active_fraction"].max()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

