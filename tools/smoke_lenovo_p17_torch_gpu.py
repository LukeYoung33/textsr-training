"""Run a bounded Remote58 CUDA contract and timing smoke for opt-in Torch P17."""

from __future__ import annotations

import json
import time

import torch

from realesrgan.lenovo_p17_preprocess import preprocess_lq_rgb


def main() -> None:
    """@luna-comment Time one fixed CUDA P17-B candidate invocation and export only factual tensor metrics."""
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires CUDA")
    value = torch.full((1, 3, 96, 480), 0.25, device="cuda")
    value[:, :, 24:72, 160:320] = 0.80
    config = {"enabled": True, "implementation": "torch_gpu", "lambda": 0.03, "alpha": 1.0}
    preprocess_lq_rgb(value, config)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    output = preprocess_lq_rgb(value, config)
    end.record()
    torch.cuda.synchronize()
    cpu_value = value.cpu()
    cpu_start = time.perf_counter()
    cpu_output = preprocess_lq_rgb(cpu_value, {**config, "implementation": "cpu_exact"})
    cpu_elapsed_ms = (time.perf_counter() - cpu_start) * 1000.0
    absolute_error = (output.cpu() - cpu_output).abs()
    report = {
        "device": str(output.device),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(torch.isfinite(output).all().item()),
        "minimum": float(output.amin().item()),
        "maximum": float(output.amax().item()),
        "cuda_elapsed_ms": float(start.elapsed_time(end)),
        "cpu_reference_elapsed_ms": cpu_elapsed_ms,
        "cpu_reference_mae": float(absolute_error.mean().item()),
        "cpu_reference_p99_abs_error": float(torch.quantile(absolute_error.reshape(-1), 0.99).item()),
        "mean_abs_delta_from_input": float((output - value).abs().mean().item()),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
