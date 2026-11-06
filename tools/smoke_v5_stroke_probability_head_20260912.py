"""CPU smoke for the V5 training-only stroke probability branch."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch

# Make direct ``python tools/...py`` execution resolve the repository package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch import (
    BasicTextLineR4StudentY4UVV5,
)


def run_smoke(canvas_hw: tuple[int, int] = (32, 64)) -> dict[str, Any]:
    """@luna-comment Check V4-compatible probability shapes, gradients, and deployment removal."""
    torch.manual_seed(20260912)
    model = BasicTextLineR4StudentY4UVV5(
        canvas_hw=canvas_hw, stroke_prob_enabled=True
    ).train()
    sample = torch.randn(1, 6, *model.expected_input_hw, requires_grad=True)
    plain = model(sample)
    output, logits = model(sample, return_stroke_logits=True)
    if not isinstance(plain, torch.Tensor):
        raise AssertionError("enabled plain forward must remain a Tensor")
    if tuple(logits.shape) != (1, 1, *canvas_hw):
        raise AssertionError(f"full-resolution logits shape mismatch: {tuple(logits.shape)}")
    if not torch.equal(plain, output) or not torch.equal(output, sample):
        raise AssertionError("probability head changed the initial identity output")

    loss = output.square().mean() + 0.1 * logits.square().mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("probability smoke loss is non-finite")
    head = model.stroke_prob_head
    if head is None or head.weight.grad is None or not torch.isfinite(head.weight.grad).all():
        raise AssertionError("probability head did not receive finite gradients")

    model.eval()
    model.switch_to_deploy()
    deployed = model(sample.detach())
    audit = model.deployment_audit()
    if model.stroke_prob_head is not None or audit["training_only_modules"]:
        raise AssertionError(f"training-only head survived deployment: {audit}")
    if not torch.equal(deployed, sample.detach()):
        raise AssertionError("deployment changed the initial identity output")

    disabled = BasicTextLineR4StudentY4UVV5(
        canvas_hw=canvas_hw, stroke_prob_enabled=False
    ).eval()
    disabled_output = disabled(sample.detach())
    if not isinstance(disabled_output, torch.Tensor) or disabled.stroke_prob_head is not None:
        raise AssertionError("disabled V5 does not expose the plain Tensor contract")

    return {
        "status": "PASS",
        "canvas_hw": list(canvas_hw),
        "logits_shape": list(logits.shape),
        "head_grad_abs": float(head.weight.grad.detach().abs().sum()),
        "deployment_audit": audit,
    }


def main() -> None:
    """@luna-comment Run the V5 probability-head smoke and print JSON evidence."""
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
