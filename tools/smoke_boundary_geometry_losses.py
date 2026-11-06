"""Functional smoke for the training-only SDF and normal-profile losses.

The smoke tests loss ordering for controlled geometric mistakes and verifies
that invalid right padding has no direct prediction gradient.  It is deliberately
independent of the V4 trainer; the trainer smoke is run only after integration.
"""

from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from realesrgan.losses.gt_sdf_boundary_loss import GTSDFBoundaryLoss
from realesrgan.losses.normal_profile_loss import BoundaryNormalProfileLoss


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.zeros(1, 1, 32, 64)
    mask[:, :, 9:23, 19:45] = 1.0
    valid = torch.ones_like(mask)
    valid[:, :, :, 56:] = 0.0
    # Dark ink on a bright page.  The inactive right area is intentionally
    # arbitrary, so a correct loss cannot learn from it.
    gt = (1.0 - mask).repeat(1, 3, 1, 1)
    gt[:, :, :, 56:] = 0.37
    return gt, mask, valid


def _variants(gt: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    correct = gt.clone()
    eroded = gt.clone()
    eroded[:, :, 11:21, 21:43] = 1.0
    dilated_mask = F.max_pool2d(mask, 5, 1, 2)
    dilated = (1.0 - dilated_mask).repeat(1, 3, 1, 1)
    broken = gt.clone()
    broken[:, :, 15:18, 19:45] = 1.0
    spur = gt.clone()
    spur[:, :, 12:20, 45:50] = 0.0
    return {"correct": correct, "eroded": eroded, "dilated": dilated, "broken": broken, "spur": spur}


def _losses(loss: torch.nn.Module, gt: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    values = {}
    for name, pred in _variants(gt, mask).items():
        values[name] = float(loss(pred, gt, mask, valid).detach().cpu())
    return values


def _assert_geometric_order(name: str, values: dict[str, float]) -> None:
    correct = values["correct"]
    failed = {key: value for key, value in values.items() if key != "correct" and value <= correct + 1.0e-5}
    if failed:
        raise AssertionError(f"{name} does not penalize geometric failures over correct: {values}")


def _assert_padding_gradient(loss: torch.nn.Module, gt: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> float:
    pred = gt.clone().requires_grad_(True)
    # Make invalid padding maximally wrong.  It must not receive direct loss gradient.
    with torch.no_grad():
        pred[:, :, :, 56:] = 0.0
    value = loss(pred, gt, mask, valid)
    value.backward()
    padded = pred.grad[:, :, :, 56:].abs().max().item()
    active = pred.grad[:, :, :, :56].abs().max().item()
    if padded > 1.0e-8 or active <= 0.0:
        raise AssertionError(f"padding-gradient contract failed: padded={padded}, active={active}")
    return padded


def main() -> None:
    torch.manual_seed(7)
    gt, mask, valid = _case()
    sdf = GTSDFBoundaryLoss(radius=3, sdf_temperature=1.0, ink_temperature=0.05, min_contrast=0.03)
    profile = BoundaryNormalProfileLoss(radius=3, tolerance_pixels=0.75, tolerance_samples=3)
    sdf_values = _losses(sdf, gt, mask, valid)
    profile_values = _losses(profile, gt, mask, valid)
    _assert_geometric_order("sdf", sdf_values)
    _assert_geometric_order("profile", profile_values)
    result = {
        "sdf": sdf_values,
        "profile": profile_values,
        "sdf_padding_grad_max": _assert_padding_gradient(sdf, gt, mask, valid),
        "profile_padding_grad_max": _assert_padding_gradient(profile, gt, mask, valid),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
