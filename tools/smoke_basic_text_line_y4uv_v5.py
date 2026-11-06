"""Minimal V5 contract smoke: shape, phase, fusion, gradients, carry, and export audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch import (  # noqa: E402
    BasicTextLineR4StudentY4UVV5,
    FixedPhasePack,
    V5RepECB,
)


def _check(condition: bool, message: str) -> None:
    """@luna-comment Raise an assertion with a stable acceptance message."""
    if not condition:
        raise AssertionError(message)


def run_smoke(canvas_hw=(32, 64), export_path: str | None = None) -> Dict[str, Any]:
    """@luna-comment Execute all low-cost V5 structural and numerical acceptance checks."""
    torch.manual_seed(20260907)
    model = BasicTextLineR4StudentY4UVV5(canvas_hw=canvas_hw)
    input_hw = model.expected_input_hw
    x = torch.randn(2, 6, *input_hw, requires_grad=True)
    context_terminal_initial_abs = sum(
        float(parameter.detach().abs().sum()) for parameter in model.context_terminal.parameters()
    )
    _check(context_terminal_initial_abs > 0.0, "context terminal is unexpectedly zero-initialized")

    volumes = model.encoder_stage_volumes
    _check(volumes[1] >= volumes[0] and volumes[2] >= volumes[1], f"volume decrease: {volumes}")

    phase = FixedPhasePack(1)
    impulse = torch.arange(1.0, 17.0).view(1, 1, 4, 4).requires_grad_(True)
    packed = phase(impulse)
    expected = torch.stack(
        (impulse[:, :, 0::2, 0::2], impulse[:, :, 0::2, 1::2], impulse[:, :, 1::2, 0::2], impulse[:, :, 1::2, 1::2]),
        dim=1,
    ).flatten(1, 2)
    _check(torch.equal(packed, expected), "four phase selector order mismatch")
    _check(not phase.weight.requires_grad, "phase selector unexpectedly trainable")
    before = phase.weight.detach().clone()
    optimizer = torch.optim.SGD(phase.parameters(), lr=1.0)
    packed.sum().backward()
    optimizer.step()
    _check(torch.equal(before, phase.weight), "phase selector changed after optimizer step")

    block = V5RepECB(4, 5, act_type="prelu")
    block.eval()
    block_input = torch.randn(2, 4, 7, 9)
    unfused = block(block_input)
    block.switch_to_deploy()
    fused = block(block_input)
    fusion_error = float((unfused - fused).abs().max().detach())
    _check(fusion_error < 2e-5, f"ECB fusion parity failed: {fusion_error}")

    model.train()
    output = model(x)
    _check(output.shape == x.shape, f"I/O shape mismatch: {tuple(output.shape)}")
    _check(torch.isfinite(output).all().item(), "non-finite forward output")
    _check(torch.equal(output, x), "initial context warm-up changed input identity")
    output.square().mean().backward()
    _check(x.grad is not None and torch.isfinite(x.grad).all().item(), "non-finite input gradient")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _check(torch.isfinite(parameter.grad).all().item(), f"non-finite gradient: {name}")
    context_scale_grad = model.context_scale.grad
    _check(context_scale_grad is not None, "context_scale has no first-backward gradient")
    context_scale_grad_abs = float(context_scale_grad.detach().abs().sum())
    _check(context_scale_grad_abs > 0.0, "context_scale first-backward gradient is zero")
    context_terminal_grad_abs = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.context_terminal.parameters()
        if parameter.grad is not None
    )
    _check(context_terminal_grad_abs > 0.0, "context terminal first-backward gradient is zero")

    model.eval()
    branches = model(x.detach(), return_branches=True)
    _check(torch.equal(branches["full"], branches["input"]), "zero-init graph is not input identity")
    _check(float(branches["local_residual"][:, 4:].abs().max().detach()) == 0.0, "local branch leaked into UV")
    _check(model.context_scale_value == 0.0, "context scale is not zero initialized")

    debug_before = model(x.detach(), return_debug=True)
    with torch.no_grad():
        for parameter in model.dec1.parameters():
            parameter.zero_()
        for parameter in model.dec2.parameters():
            parameter.zero_()
    debug_carry = model(x.detach(), return_debug=True)
    _check(torch.allclose(debug_carry["decoded1"], debug_carry["phase_carry1"], atol=1e-6), "P1 carry changed after zeroing decoder residual")
    _check(torch.allclose(debug_carry["decoded0"], debug_carry["stem_carry"], atol=1e-6), "S0 carry changed after zeroing decoder residual")

    # Restore a nonzero bounded context amplitude for a non-trivial deploy parity check.
    with torch.no_grad():
        model.context_scale.fill_(0.7)
    parity_reference = model(x.detach())
    model.switch_to_deploy()
    parity_deploy = model(x.detach())
    deploy_error = float((parity_reference - parity_deploy).abs().max().detach())
    _check(deploy_error < 2e-5, f"full-model fusion parity failed: {deploy_error}")
    module_audit = model.deployment_audit()
    _check(module_audit["passed"], f"deployment module audit failed: {module_audit}")

    export_report = None
    if export_path:
        from tools.export_basic_text_line_y4uv_v5_deploy_onnx import export_v5_deploy

        export_report = export_v5_deploy(export_path, canvas_hw=canvas_hw)
        _check(export_report["graph_audit"]["passed"], "ONNX graph audit failed")

    return {
        "canvas_hw": list(canvas_hw),
        "input_hw": list(input_hw),
        "stage_shapes": [list(shape) for shape in model.stage_shapes],
        "stage_volumes": list(volumes),
        "fusion_error": fusion_error,
        "full_model_deploy_error": deploy_error,
        "context_scale_grad_abs": context_scale_grad_abs,
        "context_terminal_grad_abs": context_terminal_grad_abs,
        "context_terminal_initial_abs": context_terminal_initial_abs,
        "parameter_count": model.parameter_count(),
        "macs": model.estimate_macs(),
        "analytic_receptive_field": model.analytic_receptive_field,
        "module_audit": module_audit,
        "export": export_report,
        "passed": True,
    }


def main() -> None:
    """@luna-comment Parse smoke arguments and print the acceptance report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas-h", type=int, default=32)
    parser.add_argument("--canvas-w", type=int, default=64)
    parser.add_argument("--export", default=None, help="optional ONNX output path")
    args = parser.parse_args()
    report = run_smoke((args.canvas_h, args.canvas_w), args.export)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
