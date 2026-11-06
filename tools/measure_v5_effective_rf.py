"""Measure V5 effective receptive-field mass from an input gradient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch import (  # noqa: E402
    BasicTextLineR4StudentY4UVV5,
)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Optional[str]) -> None:
    """@luna-comment Load a saved generator state using common BasicSR checkpoint keys."""
    if checkpoint is None:
        return
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint container: {type(state)!r}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def _mass_box(gradient: torch.Tensor, fraction: float) -> Dict[str, Any]:
    """@luna-comment Return the top-gradient support box reaching a cumulative mass fraction."""
    flat = gradient.reshape(-1)
    values, indices = torch.sort(flat, descending=True)
    total = float(values.sum())
    if total <= 0.0:
        return {"fraction": fraction, "height": 0, "width": 0, "area": 0, "mass": 0.0}
    cumulative = torch.cumsum(values, dim=0)
    count = int(torch.searchsorted(cumulative, torch.tensor(total * fraction, device=gradient.device)).item()) + 1
    selected = indices[:count]
    rows = torch.div(selected, gradient.shape[1], rounding_mode="floor")
    cols = selected.remainder(gradient.shape[1])
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    return {
        "fraction": fraction,
        "height": bottom - top + 1,
        "width": right - left + 1,
        "area": (bottom - top + 1) * (right - left + 1),
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "mass": float(values[:count].sum()),
        "total_mass": total,
    }


def measure_effective_rf(
    checkpoint: Optional[str] = None,
    canvas_hw=(96, 480),
    output_channel: int = 0,
) -> Dict[str, Any]:
    """@luna-comment Measure 50/90/95 percent input-gradient mass for one output pixel."""
    model = BasicTextLineR4StudentY4UVV5(canvas_hw=tuple(canvas_hw))
    _load_checkpoint(model, checkpoint)
    model.eval()
    height, width = model.expected_input_hw
    input_tensor = torch.zeros(1, 6, height, width, requires_grad=True)
    output = model(input_tensor)
    row, col = height // 2, width // 2
    if output_channel < 0 or output_channel >= output.shape[1]:
        raise ValueError(f"output_channel must be in [0,{output.shape[1]})")
    output[0, output_channel, row, col].backward()
    gradient = input_tensor.grad.detach().abs().sum(dim=1)[0]
    report = {
        "checkpoint": checkpoint,
        "canvas_hw": list(canvas_hw),
        "input_hw": [height, width],
        "probe": {"output_channel": output_channel, "row": row, "col": col},
        "analytic_receptive_field": model.analytic_receptive_field,
        "gradient_mass_boxes": [_mass_box(gradient, fraction) for fraction in (0.5, 0.9, 0.95)],
    }
    return report


def main() -> None:
    """@luna-comment Parse effective-RF arguments and emit a JSON measurement report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--canvas-h", type=int, default=96)
    parser.add_argument("--canvas-w", type=int, default=480)
    parser.add_argument("--output-channel", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            measure_effective_rf(
                checkpoint=args.checkpoint,
                canvas_hw=(args.canvas_h, args.canvas_w),
                output_channel=args.output_channel,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
