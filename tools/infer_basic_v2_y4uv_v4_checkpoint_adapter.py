"""Run the V4 Y4UV checkpoint through the existing full-image geometry pipeline.

The production V3 entry point remains untouched.  This isolated adapter reuses
its detector, ROI geometry, orientation normalization, aspect resize, tiling,
and compositing code, while converting each legacy tile to the native V4
six-channel contract immediately before the checkpoint is called.  The V4
model output is decoded back to a temporary YUV444 ROI only at the compositor
boundary, because the existing full-frame compositor operates on planar YUV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLS_ROOT = Path(__file__).resolve().parent
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from infer_basic_v2_deployment_pipeline import (  # noqa: E402
    DetectorONNXRunner,
    OUT_SHAPE,
    RestoreGeometry,
    RestoreONNXRunner,
    SKIP_SHAPE,
    Y4UV_SHAPE,
    YUV420Frame,
    run_bgr_adapter_frame,
    yuv420_to_bgr,
)
from realesrgan.archs.basic_text_line_r4_student_y4uv_v4_arch import (  # noqa: E402
    BasicTextLineR4StudentY4UVV4,
    V4_CANVAS_HW,
)
from realesrgan.data.basic_text_line_y4uv_v3 import numpy_rgb_to_y4uv, y4uv_to_yuv444  # noqa: E402
from realesrgan.utils import YUV2RGB  # noqa: E402
from tools.basic_v2_eval_network import build_eval_network  # noqa: E402


V4_INPUT_SHAPE = (1, 6, V4_CANVAS_HW[0] // 2, V4_CANVAS_HW[1] // 2)
V4_OUTPUT_SHAPE = V4_INPUT_SHAPE
V4_TILE_OVERLAP = 64
V4_RESIZE_INTERPOLATION = cv2.INTER_AREA


def _sha256(path: Path) -> str:
    """@shanlan-comment Compute a stable checkpoint identity for the run trace."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_params(checkpoint: Any) -> dict[str, torch.Tensor]:
    """@shanlan-comment Extract a generator state dict from common BasicSR checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    for key in ("params_ema", "params", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint has no generator parameter dictionary")


def load_v4_checkpoint(
    path: Path,
    *,
    haar_switch: bool,
    device: torch.device,
    stage_channels: tuple[int, int, int, int] = (6, 24, 96, 384),
    refine_channels: int = 16,
    stage_repeat: int = 1,
    bottleneck_repeat: int = 2,
) -> torch.nn.Module:
    """@shanlan-comment Build the V4 graph, load one checkpoint, and validate an exact state match."""
    model = BasicTextLineR4StudentY4UVV4(
        canvas_hw=V4_CANVAS_HW,
        haar_switch=haar_switch,
        stage_channels=tuple(int(value) for value in stage_channels),
        refine_channels=int(refine_channels),
        stage_repeat=int(stage_repeat),
        bottleneck_repeat=int(bottleneck_repeat),
    ).to(device).eval()
    checkpoint = torch.load(path, map_location="cpu")
    params = _checkpoint_params(checkpoint)
    normalized: dict[str, torch.Tensor] = {}
    for key, value in params.items():
        name = str(key)
        for prefix in ("module.", "net_g.", "generator."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        normalized[name] = value
    result = model.load_state_dict(normalized, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint state mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model


def _legacy_to_v4(legacy_y4uv: np.ndarray) -> np.ndarray:
    """Bridge legacy limited-range I420 tiles to V4 full-range RGB/Y4UV."""
    value = np.asarray(legacy_y4uv, dtype=np.float32)
    if tuple(value.shape) != Y4UV_SHAPE:
        raise ValueError(f"legacy y4uv shape must be {Y4UV_SHAPE}, got {value.shape}")
    y = value[0, :4].transpose(1, 2, 0).reshape(48, 240, 2, 2).transpose(0, 2, 1, 3).reshape(96, 480)
    bgr = yuv420_to_bgr(YUV420Frame(
        np.clip(y * 255.0, 0, 255).astype(np.uint8),
        np.clip(value[0, 4] * 255.0, 0, 255).astype(np.uint8),
        np.clip(value[0, 8] * 255.0, 0, 255).astype(np.uint8),
    ))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return numpy_rgb_to_y4uv(rgb).unsqueeze(0).numpy().astype(np.float32)


def _v4_to_yuv444(native_output: np.ndarray) -> np.ndarray:
    """Bridge V4 full-range Y4UV output to legacy limited-range compositor YUV."""
    value = np.asarray(native_output, dtype=np.float32)
    if tuple(value.shape) != V4_OUTPUT_SHAPE:
        raise ValueError(f"V4 output shape must be {V4_OUTPUT_SHAPE}, got {value.shape}")
    yuv444 = y4uv_to_yuv444(torch.from_numpy(value))
    rgb = YUV2RGB(yuv444, format="RGB", is_train=False, uv_normalized=True)[0]
    bgr = cv2.cvtColor(np.transpose(rgb.numpy(), (1, 2, 0)).astype(np.float32), cv2.COLOR_RGB2BGR)
    packed = cv2.cvtColor(np.clip(bgr * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YUV_I420)
    h, w = V4_CANVAS_HW
    y = packed[:h].astype(np.float32) / 255.0
    u = packed[h:h + h // 4].reshape(h // 2, w // 2).astype(np.float32) / 255.0
    v = packed[h + h // 4:].reshape(h // 2, w // 2).astype(np.float32) / 255.0
    uv = cv2.resize(np.stack((u, v), axis=-1), (w, h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    return np.concatenate((y[None], uv), axis=0)[None].astype(np.float32)


class V4CheckpointRunner:
    """@shanlan-comment Adapt old pipeline tile inputs to one native V4 checkpoint call."""

    def __init__(self, model: torch.nn.Module, device: torch.device, *, preserve_lq_uv: bool = False) -> None:
        self.model = model
        self.device = device
        self.preserve_lq_uv = bool(preserve_lq_uv)
        self.calls = 0
        self.native_input_shapes: set[tuple[int, ...]] = set()
        self.native_output_shapes: set[tuple[int, ...]] = set()

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """@shanlan-comment Run one tile and return the temporary YUV444 shape expected by the legacy merger."""
        if set(("y4uv", "skip")) - set(inputs):
            raise ValueError("adapter requires legacy y4uv and skip keys")
        if tuple(inputs["y4uv"].shape) != Y4UV_SHAPE or tuple(inputs["skip"].shape) != SKIP_SHAPE:
            raise ValueError("adapter received an invalid legacy tile shape")
        native = _legacy_to_v4(inputs["y4uv"])
        self.native_input_shapes.add(tuple(native.shape))
        tensor = torch.from_numpy(native).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor).detach().cpu().numpy().astype(np.float32)
        if self.preserve_lq_uv:
            # Native Y4UV contract: channels 0..3 are the four luma samples;
            # 4..5 are chroma.  This diagnostic isolates luma restoration.
            output[:, 4:6] = native[:, 4:6]
        self.native_output_shapes.add(tuple(output.shape))
        self.calls += 1
        return _v4_to_yuv444(output)


class V4IdentityRunner(V4CheckpointRunner):
    """Diagnostic-only runner that proves the two color-space bridges are lossless enough."""

    def __init__(self) -> None:
        self.model = None
        self.device = torch.device("cpu")
        self.calls = 0
        self.native_input_shapes: set[tuple[int, ...]] = set()
        self.native_output_shapes: set[tuple[int, ...]] = set()

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        if tuple(inputs["y4uv"].shape) != Y4UV_SHAPE:
            raise ValueError("identity adapter received an invalid legacy tile shape")
        native = _legacy_to_v4(inputs["y4uv"])
        self.native_input_shapes.add(tuple(native.shape))
        self.native_output_shapes.add(tuple(native.shape))
        self.calls += 1
        return _v4_to_yuv444(native)


def _rewrite_trace_for_v4(
    trace: dict[str, Any],
    runner: V4CheckpointRunner,
    checkpoint: Path,
    checkpoint_sha256: str,
    haar_switch: bool,
    stage_channels: tuple[int, int, int, int],
    refine_channels: int,
    stage_repeat: int,
    bottleneck_repeat: int,
) -> None:
    """@shanlan-comment
    Input: completed shared-pipeline trace, native runner evidence, checkpoint identity, and Haar mode.
    Output: trace enriched with verified V4 tensor and 20260812 training-geometry metadata.
    Key Steps: reject any pixel-processing contract mismatch before labeling the run as V4-compatible.
    Key Params: height 96, overlap 64, area interpolation, even resized width, and Global-S disabled.
    State/Dependencies: mutates only the caller-owned trace dictionary.
    """
    config = trace.setdefault("config", {})
    expected = {
        "restore_content_height": V4_CANVAS_HW[0],
        "restore_overlap": V4_TILE_OVERLAP,
        "restore_resize_interpolation": V4_RESIZE_INTERPOLATION,
        "restore_force_even_width": True,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"V4 preprocessing contract mismatch: actual={actual}, expected={expected}")
    config.update({
        "pipeline_variant": "y4uv_v4_checkpoint_adapter",
        "y4uv_format": "native_y4uv420_six_channel",
        "all_text_rois_resized": True,
        "resize_content_height": V4_CANVAS_HW[0],
        "resize_interpolation": "area",
        "tile_overlap": V4_TILE_OVERLAP,
        "vertical_padding": [0, 0],
        "horizontal_padding": "right_edge_replicate",
        "enable_global_s": False,
    })
    trace["restorer"] = {
        "architecture": "BasicTextLineR4StudentY4UVV4",
        "weights": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "haar_switch": bool(haar_switch),
        "stage_channels": [int(value) for value in stage_channels],
        "refine_channels": int(refine_channels),
        "stage_repeat": int(stage_repeat),
        "bottleneck_repeat": int(bottleneck_repeat),
        "native_input_shape": list(V4_INPUT_SHAPE),
        "native_output_shape": list(V4_OUTPUT_SHAPE),
        "calls": int(runner.calls),
        "observed_native_input_shapes": [list(shape) for shape in sorted(runner.native_input_shapes)],
        "observed_native_output_shapes": [list(shape) for shape in sorted(runner.native_output_shapes)],
        "output_adapter": "V4 Y4UV output decoded to temporary YUV444 only at compositor boundary",
    }
    for roi in trace.get("rois", []):
        for tile in roi.get("tiles", []):
            tile["native_input_y4uv_shape"] = list(V4_INPUT_SHAPE)
            tile["native_output_y4uv_shape"] = list(V4_OUTPUT_SHAPE)
            tile["legacy_compositor_output_shape"] = list(OUT_SHAPE)


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    """@shanlan-comment
    Input: one image, detector, V4 checkpoint, device, output paths, and fixed training content height.
    Output: restored image files and a trace proving the native tensor and preprocessing contracts.
    Key Steps: strict-load V4, run shared full-image logic with explicit V4 geometry, validate trace, and save.
    Key Params: content height 96, overlap 64, area interpolation, even width, and Haar switch.
    State/Dependencies: OpenCV, PyTorch, detector ONNX Runtime, and local filesystem outputs.
    """
    input_path = Path(args.input)
    checkpoint = Path(args.checkpoint)
    output_path = Path(args.output)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    if args.threads is not None and device.type == "cpu":
        torch.set_num_threads(max(1, int(args.threads)))
    checkpoint_sha256 = _sha256(checkpoint)
    if args.identity:
        model = None
    elif args.network_config:
        model, checkpoint_sha256 = build_eval_network(checkpoint, device, Path(args.network_config))
        if tuple(getattr(model, "expected_input_hw", V4_INPUT_SHAPE[-2:])) != V4_INPUT_SHAPE[-2:]:
            raise ValueError("configured native Y4UV model does not use the required 48x240 input geometry")
    else:
        model = load_v4_checkpoint(
            checkpoint,
            haar_switch=bool(args.haar_switch),
            device=device,
            stage_channels=tuple(int(value) for value in args.stage_channels),
            refine_channels=int(args.refine_channels),
            stage_repeat=int(args.stage_repeat),
            bottleneck_repeat=int(args.bottleneck_repeat),
        )
    detector = DetectorONNXRunner(args.detector)
    runner = V4IdentityRunner() if args.identity else V4CheckpointRunner(
        model, device, preserve_lq_uv=bool(args.preserve_lq_uv)
    )
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(input_path)
    geometry = RestoreGeometry(
        content_h=int(args.content_height),
        overlap=V4_TILE_OVERLAP,
        interpolation=V4_RESIZE_INTERPOLATION,
        force_even_width=True,
    )
    out_frame, output, trace = run_bgr_adapter_frame(
        image,
        detector,
        runner,
        debug_dir=None,
        restore_geometry=geometry,
    )
    if args.network_config:
        trace["restorer"] = {
            "architecture": type(model).__name__, "weights": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256, "network_config": str(args.network_config),
            "native_input_shape": list(V4_INPUT_SHAPE), "native_output_shape": list(V4_OUTPUT_SHAPE),
            "calls": int(runner.calls),
            "output_adapter": "native Y4UV output decoded to temporary YUV444 only at compositor boundary",
        }
    else:
        _rewrite_trace_for_v4(
            trace, runner, checkpoint, checkpoint_sha256, bool(args.haar_switch),
            tuple(int(value) for value in args.stage_channels), int(args.refine_channels),
            int(args.stage_repeat), int(args.bottleneck_repeat),
        )
    trace["input"] = str(input_path)
    trace["output"] = str(output_path)
    trace["detector"] = str(args.detector)
    trace["device"] = str(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise IOError(f"failed to write {output_path}")
    preview_path = output_path.with_name(output_path.stem + "_preview.jpg")
    if not cv2.imwrite(str(preview_path), output):
        raise IOError(f"failed to write {preview_path}")
    if args.output_i420:
        from infer_basic_v2_deployment_pipeline import write_yuv420

        write_yuv420(args.output_i420, out_frame)
    if args.trace:
        trace_path = Path(args.trace)
    else:
        trace_path = output_path.with_suffix(output_path.suffix + ".json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace


def main(argv: list[str] | None = None) -> int:
    """@shanlan-comment Parse one V4 checkpoint test command and emit a compact JSON result."""
    parser = argparse.ArgumentParser(description="BASIC V2 native Y4UV V4 full-image checkpoint test")
    parser.add_argument("--input", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--network-config", default=None,
                        help="Optional native-Y4UV training YAML; enables a registry-built architecture such as V5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--output-i420")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--content-height",
        type=int,
        choices=(V4_CANVAS_HW[0],),
        default=V4_CANVAS_HW[0],
        help="Fixed V4 training content height; only 96 is accepted",
    )
    parser.add_argument("--haar-switch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stage-channels", type=int, nargs=4, metavar=("C0", "C1", "C2", "C3"),
        default=(6, 24, 96, 384),
        help="V4 encoder channels; each 2x downsample must expand sufficiently",
    )
    parser.add_argument("--refine-channels", type=int, default=16)
    parser.add_argument("--stage-repeat", type=int, default=1)
    parser.add_argument("--bottleneck-repeat", type=int, default=2)
    parser.add_argument("--identity", action="store_true", help="diagnostic: bypass the checkpoint and round-trip color only")
    parser.add_argument("--preserve-lq-uv", action="store_true", help="diagnostic: use model Y and input LQ U/V")
    args = parser.parse_args(argv)
    trace = run_one(args)
    print(json.dumps({
        "status": "passed",
        "accepted_count": trace.get("accepted_count", 0),
        "fallbacks": trace.get("fallbacks", []),
        "restorer": trace.get("restorer", {}),
        "output": trace.get("output"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
