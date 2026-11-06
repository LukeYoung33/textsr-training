"""Export the CP BASIC V2 Student core to static-batch ONNX.

The training/inference entry point accepts packed NV21 ``[B, 1, 144, 320]``
and returns YUV444 ``[B, 3, 96, 320]``.  The deployment graph used here follows
the ECBSR export contract in ``E:/tools/pytorch2tflite/archs/ecbsr_Unet_arch.py``:

    y4uv: [B, 12, H/2, W/2]
    skip: [B, 3, H, W]
    output: [B, 3, H, W]

ECB/ECB2 branches are fused into ordinary Conv2d layers before ONNX export.
The fixed Haar and inverse-Haar transforms are also represented as frozen
1x1 Conv2d(12, 12) layers.  The output graph therefore contains no
training-time multi-branch ECB blocks and no Haar Slice/Concat path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realesrgan.archs.basic_text_line_r4_student_v2_arch import (  # noqa: E402
    BasicTextLineR4StudentV2,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(model: nn.Module, checkpoint: Path) -> str:
    try:
        raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        raw = torch.load(str(checkpoint), map_location="cpu")
    state = raw
    if isinstance(raw, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a parameter mapping: {checkpoint}")
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return sha256_file(checkpoint)


class FusedECB(nn.Module):
    """Static equivalent of an eval-mode ECB block."""

    def __init__(self, source: nn.Module):
        super().__init__()
        kernel, bias = source.rep_params()
        stride = 2 if getattr(source, "phase", "Conv") == "Encoder" else 1
        self.conv = nn.Conv2d(
            int(kernel.shape[1]),
            int(kernel.shape[0]),
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
            self.conv.bias.copy_(bias)
        act_type = getattr(source, "act_type", "linear")
        self.act = source.act if act_type != "linear" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class FusedECB2(nn.Module):
    """Static equivalent of an eval-mode decoder ECB2 block."""

    def __init__(self, source: nn.Module):
        super().__init__()
        kernel, bias = source.rep_params()
        self.upsample = source.upsample
        self.with_proj = bool(getattr(source, "with_proj", False))
        self.proj_conv = source.proj_conv if self.with_proj else nn.Identity()
        self.conv = nn.Conv2d(
            int(kernel.shape[1]),
            int(kernel.shape[0]),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
            self.conv.bias.copy_(bias)
        act_type = getattr(source, "act_type", "linear")
        self.act = source.act if act_type != "linear" else nn.Identity()

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, skip = inputs
        skip = self.proj_conv(skip)
        y = self.conv(self.upsample(x)) + skip
        return self.act(y)


def _haar_matrix(inverse: bool) -> torch.Tensor:
    """Return the 4x4 Haar matrix used by the trained ECBSR graph."""
    matrix = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return matrix if inverse else matrix / 4.0


class FixedHaar12Conv(nn.Module):
    """Frozen 12-channel 1x1 transform: Haar/identity or inverse/identity."""

    def __init__(self, inverse: bool):
        super().__init__()
        self.conv = nn.Conv2d(12, 12, kernel_size=1, bias=False)
        matrix = torch.eye(12, dtype=torch.float32)
        matrix[:4, :4] = _haar_matrix(inverse=inverse)
        with torch.no_grad():
            self.conv.weight.copy_(matrix.view(12, 12, 1, 1))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def reparameterize_ecb_modules(module: nn.Module) -> int:
    """Replace every ECB/ECB2 child with its static Conv2d equivalent."""
    replaced = 0
    for name, child in list(module.named_children()):
        class_name = child.__class__.__name__
        if class_name == "ECB2":
            setattr(module, name, FusedECB2(child))
            replaced += 1
        elif class_name == "ECB":
            setattr(module, name, FusedECB(child))
            replaced += 1
        else:
            replaced += reparameterize_ecb_modules(child)
    return replaced


class BasicTextLineInternalExport(nn.Module):
    """Reference-file-compatible two-input ECBSR export core."""

    def __init__(self, model: BasicTextLineR4StudentV2):
        super().__init__()
        self.model = model

    def forward(self, y4uv: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Reference contract: y4uv is [Y1,Y2,Y3,Y4,U1..U4,V1..V4] at half res.
        y4 = y4uv[:, :4, ...]
        u4v4 = y4uv[:, 4:, ...]
        y_haar = self.model.y4_2_haar(y4)
        x = torch.cat([y_haar, u4v4], dim=1)

        enc0 = self.model.conv0(x)
        enc1 = self.model.enc1(enc0)
        enc2 = self.model.enc2(enc1)
        enc3 = self.model.enc3(enc2)

        dec0 = self.model.encdec(enc3)
        dec1 = self.model.dec1((dec0, enc2))
        dec2 = self.model.dec2((dec1, enc1))
        dec3 = self.model.dec3((dec2, enc0))

        y = self.model.out(dec3)
        y = torch.cat([self.model.haar_2_y4(y[:, :4, ...]), y[:, 4:, ...]], dim=1)
        return self.model.out_layer(y) + skip


class BasicTextLineFixed12Export(nn.Module):
    """Same core with the Haar/split path replaced by two fixed 1x1 Conv2d."""

    def __init__(self, model: BasicTextLineR4StudentV2):
        super().__init__()
        self.model = model
        self.input_transform = FixedHaar12Conv(inverse=False)
        self.output_transform = FixedHaar12Conv(inverse=True)

    def forward(self, y4uv: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # The first transform produces [LL,LH,HL,HH,U1..U4,V1..V4].
        x = self.input_transform(y4uv)

        enc0 = self.model.conv0(x)
        enc1 = self.model.enc1(enc0)
        enc2 = self.model.enc2(enc1)
        enc3 = self.model.enc3(enc2)

        dec0 = self.model.encdec(enc3)
        dec1 = self.model.dec1((dec0, enc2))
        dec2 = self.model.dec2((dec1, enc1))
        dec3 = self.model.dec3((dec2, enc0))

        # The network emits [LL,LH,HL,HH,U1..U4,V1..V4].
        y = self.output_transform(self.model.out(dec3))
        return self.model.out_layer(y) + skip


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a - b)).item())


def export_one(
    reference: nn.Module,
    fused: nn.Module,
    output_path: Path,
    batch_size: int,
    input_hw: tuple[int, int],
    opset: int,
) -> dict[str, Any]:
    input_h, input_w = input_hw
    y4uv = torch.randn(batch_size, 12, input_h, input_w, dtype=torch.float32)
    skip = torch.randn(batch_size, 3, input_h * 2, input_w * 2, dtype=torch.float32)
    with torch.inference_mode():
        reference_output = reference(y4uv, skip)
        fused_output = fused(y4uv, skip)
    reparam_max_abs = _max_abs(reference_output, fused_output)
    reparam_mean_abs = _mean_abs(reference_output, fused_output)
    if reparam_max_abs > 2e-4:
        raise RuntimeError(
            f"reparameterization mismatch for batch={batch_size}: "
            f"max_abs={reparam_max_abs:.6g}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        model=fused,
        args=(y4uv, skip),
        f=str(output_path),
        input_names=["y4uv", "skip"],
        output_names=["yuv444"],
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    try:
        torch.onnx.export(**export_kwargs, dynamo=False)
    except TypeError:  # compatibility with older torch exporters
        torch.onnx.export(**export_kwargs)

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    onnx_inputs = {
        value.name: [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
        for value in onnx_model.graph.input
    }
    onnx_outputs = {
        value.name: [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
        for value in onnx_model.graph.output
    }

    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_output = torch.from_numpy(
        session.run(["yuv444"], {"y4uv": y4uv.numpy(), "skip": skip.numpy()})[0]
    )
    onnx_max_abs = _max_abs(fused_output, ort_output)
    onnx_mean_abs = _mean_abs(fused_output, ort_output)
    if onnx_max_abs > 2e-4:
        raise RuntimeError(
            f"ONNX mismatch for batch={batch_size}: max_abs={onnx_max_abs:.6g}"
        )

    return {
        "batch_size": batch_size,
        "output": str(output_path),
        "input_shapes": onnx_inputs,
        "output_shapes": onnx_outputs,
        "reparameterization_max_abs": reparam_max_abs,
        "reparameterization_mean_abs": reparam_mean_abs,
        "onnxruntime_max_abs": onnx_max_abs,
        "onnxruntime_mean_abs": onnx_mean_abs,
        "onnx_nodes": len(onnx_model.graph.node),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CP BASIC V2 core to reparameterized ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-h", type=int, default=48)
    parser.add_argument("--input-w", type=int, default=240)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.input_h <= 0 or args.input_w <= 0:
        raise ValueError("input spatial dimensions must be positive")
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")

    model = BasicTextLineR4StudentV2(
        canvas_hw=(96, 320),
        in_channel=12,
        out_channel=12,
        layer_num=3,
        expand_channel=(0.5, 1.0, 2.0, 4.0),
        depth_factor=1,
        scale_factor=2,
        with_idt=False,
        act_type="prelu",
        scale=2,
        with_proj=True,
    ).eval()
    checkpoint_sha256 = load_checkpoint(model, args.checkpoint)
    reference_model = copy.deepcopy(model).eval()
    fused_model = model.eval()
    replaced = reparameterize_ecb_modules(fused_model)
    fused_model.eval()
    reference = BasicTextLineInternalExport(reference_model).eval()
    fused = BasicTextLineFixed12Export(fused_model).eval()

    reports = []
    for batch_size in args.batch_sizes:
        output_path = args.output_dir / (
            f"basic_text_line_cp_direct_reparam_b{batch_size}_"
            f"12x{args.input_h}x{args.input_w}_to_3x{args.input_h * 2}x{args.input_w * 2}.onnx"
        )
        reports.append(
            export_one(
                reference,
                fused,
                output_path,
                batch_size,
                (args.input_h, args.input_w),
                args.opset,
            )
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": "BasicTextLineR4StudentV2 / ECBSRUnet_StdV6",
        "reference_contract": "E:/tools/pytorch2tflite/archs/ecbsr_Unet_arch.py::ECBSRUnet_StdV6.forward(x, skip)",
        "reparameterized_ecb_modules": replaced,
        "fixed_channel_linear_transforms": {
            "input": "Conv2d(12,12,kernel_size=1,bias=False): Haar/identity",
            "output": "Conv2d(12,12,kernel_size=1,bias=False): inverse-Haar/identity",
            "trainable": False,
        },
        "main_input_contract": ["B", 12, args.input_h, args.input_w],
        "skip_input_contract": ["B", 3, args.input_h * 2, args.input_w * 2],
        "output_contract": ["B", 3, args.input_h * 2, args.input_w * 2],
        "opset": args.opset,
        "exports": reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
