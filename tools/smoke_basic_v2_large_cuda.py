"""Run one CUDA forward/backward/checkpoint smoke for a measured BASIC V2 Large preset."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn import functional as F

from profile_basic_v2_large_macs import CANVAS_HW, LARGE_PRESETS, PACKED_INPUT_SHAPE, build_large_preset


def smoke_cuda(preset: str, batch_size: int) -> dict[str, int | float | str | list[int]]:
    """@shanlan-comment
    Input: measured preset name and positive CUDA batch size.
    Output: finite one-update/checkpoint result with peak allocated CUDA MiB.
    Key Steps: create generator/input/target, run Adam-backed forward/backward/step,
    save and strict-load a temporary checkpoint, then report peak allocator memory.
    Key Params: batch_size scales the exact [B,1,144,320] packed-NV21 input.
    State/Dependencies: requires CUDA; creates and removes one temporary checkpoint.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device("cuda")
    model = build_large_preset(preset).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    packed = torch.randn((batch_size, *PACKED_INPUT_SHAPE[1:]), device=device)
    target = torch.randn((batch_size, 3, *CANVAS_HW), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    output = model(packed)
    loss = F.l1_loss(output, target)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise RuntimeError("non-finite output or loss")
    with tempfile.TemporaryDirectory(prefix="basic-v2-large-smoke-") as directory:
        checkpoint = Path(directory) / "net_g.pth"
        torch.save(model.state_dict(), checkpoint)
        restored = build_large_preset(preset).to(device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    torch.cuda.synchronize(device)
    return {
        "preset": preset,
        "batch_size": batch_size,
        "input_shape": [batch_size, *PACKED_INPUT_SHAPE[1:]],
        "output_shape": list(output.shape),
        "loss": float(loss.detach().cpu()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "params": sum(parameter.numel() for parameter in model.parameters()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """@shanlan-comment
    Input: CLI preset and batch-size arguments.
    Output: one JSON CUDA-smoke record and zero exit status after a valid update.
    Key Steps: parse arguments, run smoke_cuda, and serialize result for ST-04.
    Key Params: --preset fixes measured architecture; --batch-size verifies feasibility.
    State/Dependencies: CUDA availability and the temporary checkpoint lifecycle.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(LARGE_PRESETS), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(smoke_cuda(args.preset, args.batch_size), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
