"""Run an append-only, matched-state Lenovo V5 R1 loss-fork diagnostic.

This deliberately does not replay the historical data-loader/RNG state (those
states were not saved).  Instead, it restores the real 150k generator, Adam,
and scheduler state into three identical arms and drives a new, deterministic
stream.  OFF-A and OFF-B are duplicate controls; ON differs only by the two
production recall-aware weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
# CUDA reads this at its first initialization, so it must precede torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from torch.utils.data._utils.collate import default_collate

from basicsr.data import build_dataset
from basicsr.models import build_model
from realesrgan.archs.basic_text_line_r4_student_v2_arch import load_strict_checkpoint
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.utils import YUV2RGB
# Match realesrgan/train.py registration before resolving config type strings.
import realesrgan.archs as _realesrgan_archs
import realesrgan.data as _realesrgan_data
import realesrgan.losses as _realesrgan_losses
import realesrgan.models as _realesrgan_models


SNAPSHOT_STEPS = (0, 1, 10, 50, 100, 250, 500)
ROIS = {
    "stroke_a": (slice(14, 39), slice(171, 177)),
    "stroke_b": (slice(55, 80), slice(197, 201)),
}


def _args() -> argparse.Namespace:
    """@luna-comment
    Input: explicit immutable source artifacts and one fresh output root.
    Output: parsed matched-fork run contract.
    Key Steps: require the real checkpoint/state, fixed historical pair, and
    append-only destination; expose an optional original 250k R1 reference.
    Key Params: steps defaults to 500 and seed drives the new controlled stream.
    State/Dependencies: no path is inferred from an experiment root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--start-checkpoint", required=True, type=Path)
    parser.add_argument("--start-state", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--historical-lq", required=True, type=Path)
    parser.add_argument("--historical-gt", required=True, type=Path)
    parser.add_argument("--historical-r1-full", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic-resize-backward", "--cp-bicubic-cuda-forward-cpu-backward", "--cp-bicubic-cpu", "--cpu-bicubic-cp",
                        dest="cp_bicubic_cpu", action="store_true",
                        help="diagnostic: preserve CUDA bicubic/bilinear forward and use deterministic CPU resize transpose in backward")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    """@luna-comment
    Input: readable immutable file path.
    Output: streamed SHA-256 digest.
    Key Steps: hash in fixed-size chunks without loading a checkpoint twice.
    Key Params: chunk size is one MiB.
    State/Dependencies: used only for provenance; never mutates the source.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(value: Any) -> str:
    """@luna-comment
    Input: tensor/scalar/container state with deterministic iteration order.
    Output: SHA-256 digest over type, shape, dtype, and tensor bytes.
    Key Steps: recurse mappings by sorted key and preserve list order.
    Key Params: tensors are detached and copied to CPU before byte hashing.
    State/Dependencies: validates equality without serializing or changing state.
    """
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            cpu = item.detach().contiguous().cpu()
            digest.update(f"tensor:{tuple(cpu.shape)}:{cpu.dtype}".encode())
            digest.update(cpu.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"seq:{len(item)}".encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _clone_tree(value: Any) -> Any:
    """@luna-comment
    Input: device-resident feed_data fields and nested metadata.
    Output: an exact-value, independent tensor tree for one arm.
    Key Steps: detach/clone tensors and recursively copy containers.
    Key Params: non-tensor scalar values remain immutable references.
    State/Dependencies: prevents one arm from aliasing another arm's batch.
    """
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(child) for child in value)
    return copy.deepcopy(value)


def _seed_all(seed: int) -> None:
    """@luna-comment
    Input: one integer seed for an arm/absolute iteration.
    Output: aligned Python, NumPy, CPU Torch, and CUDA random streams.
    Key Steps: set every stochastic source before each arm optimization.
    Key Params: the caller derives a unique seed from absolute iteration.
    State/Dependencies: deterministic algorithms and TF32 policy are set once.
    """
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_determinism() -> None:
    """@luna-comment
    Input: current Torch runtime.
    Output: disabled TF32 and best-effort deterministic kernel selection.
    Key Steps: turn off matmul/cuDNN TF32 and enable deterministic algorithms.
    Key Params: warn_only avoids silently switching to an unsafe fallback.
    State/Dependencies: may reject a nondeterministic operation at execution.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        # The remote production V5 backward uses bicubic interpolation, for
        # which this Torch build has no deterministic CUDA kernel.  Keep the
        # deterministic policy for supported operations and let the required
        # OFF A/A byte checks reject any observed divergence.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError as error:
        raise RuntimeError("matched fork requires deterministic algorithms") from error


class _CudaForwardCpuResizeBackward(torch.autograd.Function):
    """Use the exact CUDA linear-resize forward and CPU's deterministic transpose."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, original: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        ctx.input_shape, ctx.input_dtype, ctx.input_device = tuple(input.shape), input.dtype, input.device
        ctx.original, ctx.args, ctx.kwargs = original, args, kwargs
        return original(input, *args, **kwargs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        # Bilinear and bicubic interpolation are linear in their input.  Their Jacobians therefore
        # depends only on shape/options, not on input values or CRNN activations.
        with torch.enable_grad():
            zero = torch.zeros(ctx.input_shape, dtype=ctx.input_dtype, device="cpu", requires_grad=True)
            cpu_output = ctx.original(zero, *ctx.args, **ctx.kwargs)
            cpu_gradient = torch.autograd.grad(cpu_output, zero, grad_output.detach().to("cpu"),
                                               retain_graph=False, create_graph=False)[0]
        return cpu_gradient.to(ctx.input_device), None, None, None


@contextmanager
def _cpu_bicubic_context(enabled: bool):
    """@luna-comment Preserve CUDA bicubic/bilinear forwards while routing only their backward transposes through CPU."""
    if not enabled:
        yield
        return
    original = functional.interpolate
    def routed(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
        if mode in ("bicubic", "bilinear") and input.is_cuda:
            return _CudaForwardCpuResizeBackward.apply(input, original, args, kwargs)
        return original(input, *args, **kwargs)
    functional.interpolate = routed
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        functional.interpolate = original
        torch.use_deterministic_algorithms(True, warn_only=True)


def _cp_value_and_prediction_gradient(model: Any, prediction_source: torch.Tensor, use_cpu_bicubic: bool) -> dict[str, Any]:
    """@luna-comment
    Input: one fed production arm and a temporary CP interpolation route choice.
    Output: CP scalar and dCP/dRGB for a detached, identical float prediction.
    Key Steps: decode once without graph, require grad on an independent RGB leaf,
    then run only the frozen CP criterion under the selected interpolation mode.
    Key Params: raw CP (not its .0005 training weight) is compared before step 1.
    State/Dependencies: does not change model parameters, optimizer, or batch fields.
    """
    if model.cri_cp is None or model.gt_rgb is None:
        raise RuntimeError("CP equivalence requires cri_cp and gt_rgb")
    prediction = prediction_source.float().detach().clone().requires_grad_(True)
    if bool(getattr(model.cri_cp, "expects_segment_metadata", False)):
        cp_args: tuple[Any, ...] = (prediction, model.gt_rgb.detach(), model.segment_metadata)
    else:
        cp_args = (prediction, model.gt_rgb.detach(), model._build_cp_annotations(model.segment_metadata, model.lq.shape[0]))
    was_training = bool(model.cri_cp.training)
    model.cri_cp.eval()
    try:
        with _cpu_bicubic_context(use_cpu_bicubic):
            value = model.cri_cp(*cp_args)
            gradient = torch.autograd.grad(value, prediction, retain_graph=False, create_graph=False)[0]
    finally:
        model.cri_cp.train(was_training)
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("non-finite CP equivalence value or gradient")
    return {"value": float(value.detach().cpu()), "gradient": gradient.detach()}


def _cp_cpu_equivalence(model: Any) -> dict[str, Any]:
    """@luna-comment Compare CP CUDA and CPU-bicubic values/gradients before the first optimizer step."""
    # One detached forward output is deliberately reused, so the comparison
    # cannot be contaminated by a second train-mode/STE generator forward.
    with torch.no_grad():
        prediction = _decode_rgb(_historical_forward(model, model.lq)).detach()
    prediction_digest = _hash_tree(prediction)
    cuda = _cp_value_and_prediction_gradient(model, prediction, use_cpu_bicubic=False)
    cpu = _cp_value_and_prediction_gradient(model, prediction, use_cpu_bicubic=True)
    value_denom = max(abs(cuda["value"]), 1e-12)
    grad_denom = max(float(torch.linalg.vector_norm(cuda["gradient"]).cpu()), 1e-12)
    relative_value = abs(cpu["value"] - cuda["value"]) / value_denom
    relative_gradient = float(torch.linalg.vector_norm(cpu["gradient"] - cuda["gradient"]).cpu()) / grad_denom
    passed = relative_value <= 1e-5 and relative_gradient <= 1e-4
    return {"shared_detached_prediction_digest": prediction_digest,
            "cuda_warn_only": {"value": cuda["value"], "gradient_l2": float(torch.linalg.vector_norm(cuda["gradient"]).cpu())},
            "cpu_strict": {"value": cpu["value"], "gradient_l2": float(torch.linalg.vector_norm(cpu["gradient"]).cpu())},
            "relative_value_error": relative_value, "relative_gradient_l2_error": relative_gradient,
            "limits": {"relative_value": 1e-5, "relative_gradient_l2": 1e-4}, "all_finite": True, "pass": passed}


def _named_gradient_snapshot(model: Any) -> dict[str, torch.Tensor | None]:
    """Return detached named generator gradients without changing parameter/state values."""
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in _model_parameters(model)}


def _gradient_relative_l2(left: dict[str, torch.Tensor | None], right: dict[str, torch.Tensor | None]) -> tuple[float, bool]:
    """Compare two complete named gradient trees and reject non-finite/missing mismatches."""
    if left.keys() != right.keys():
        return float("inf"), False
    numerator = denominator = 0.0
    finite = True
    for name in left:
        first, second = left[name], right[name]
        if first is None or second is None:
            if first is not second:
                return float("inf"), False
            continue
        finite = finite and bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all())
        delta = (first - second).float()
        numerator += float((delta * delta).sum().cpu())
        baseline = first.float()
        denominator += float((baseline * baseline).sum().cpu())
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-12), finite


def _full_optimize_equivalence(model: Any, current_iter: int, seed: int) -> dict[str, Any]:
    """@luna-comment
    Input: a fed ON arm and the intended optimizer iteration/seed.
    Output: no-step CUDA-vs-custom-resize total-loss/full-gradient comparison.
    Key Steps: replace optimizer.step temporarily, run production optimize twice
    from identical state/RNG, collect all named gradients, and prove state did not change.
    Key Params: strict custom resize uses CUDA forward plus CPU linear transpose.
    State/Dependencies: gradients/logs are transient; parameter/Adam digests must match before/after.
    """
    parameter_before = _parameter_digest(model)[1]
    optimizer_before = _optimizer_summary(model)["digest"]
    original_step = model.optimizer_g.step

    def run(custom_resize: bool) -> tuple[float, dict[str, torch.Tensor | None]]:
        model.optimizer_g.step = lambda *args, **kwargs: None
        _seed_all(seed)
        try:
            with _cpu_bicubic_context(custom_resize):
                model.optimize_parameters(current_iter)
            total = float(model.log_dict["l_total"])
            return total, _named_gradient_snapshot(model)
        finally:
            model.optimizer_g.step = original_step

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        cuda_total, cuda_gradients = run(False)
        custom_total, custom_gradients = run(True)
        relative_grad, finite = _gradient_relative_l2(cuda_gradients, custom_gradients)
        relative_value = abs(cuda_total - custom_total) / max(abs(cuda_total), 1e-12)
        parameter_after = _parameter_digest(model)[1]
        optimizer_after = _optimizer_summary(model)["digest"]
        unchanged = parameter_before == parameter_after and optimizer_before == optimizer_after
        passed = finite and unchanged and relative_value <= 1e-5 and relative_grad <= 1e-4
        return {"cuda_warn_only": {"l_total": cuda_total}, "cuda_forward_cpu_resize_backward_strict": {"l_total": custom_total},
                "relative_total_value_error": relative_value, "relative_full_gradient_l2_error": relative_grad,
                "all_gradients_finite": finite, "parameter_digest_before": parameter_before,
                "parameter_digest_after": parameter_after, "adam_digest_before": optimizer_before,
                "adam_digest_after": optimizer_after, "state_unchanged": unchanged,
                "limits": {"relative_total_value": 1e-5, "relative_full_gradient_l2": 1e-4}, "pass": passed}
    finally:
        model.optimizer_g.step = original_step
        model.optimizer_g.zero_grad(set_to_none=True)
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    """@luna-comment
    Input: RGB PNG or HWC/CHW RGB NumPy array path.
    Output: float BCHW RGB tensor without display clipping beyond PNG decoding.
    Key Steps: normalize uint8 PNG, validate exactly three channels, move device.
    Key Params: arrays are accepted only when their channels identify RGB.
    State/Dependencies: historical input is independent of the training stream.
    """
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.ndim == 3 and array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"historical RGB must be HWC/CHW 3-channel: {path} {array.shape}")
        array = array.astype(np.float32, copy=False)
        if float(np.max(array)) > 1.0:
            array /= 255.0
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0).to(device)


def _decode_rgb(y4uv: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Input: packed six-channel Y4UV network output.
    Output: raw decoded RGB tensor, intentionally not clipped.
    Key Steps: expand chroma to YUV444 then use the production RGB decoder.
    Key Params: is_train remains true to retain the production tensor contract.
    State/Dependencies: raw output is retained alongside clipped-float metrics.
    """
    return YUV2RGB(y4uv_to_yuv444(y4uv), format="RGB", is_train=True, uv_normalized=True)


def _historical_forward(model: Any, packed_lq: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Input: restored production arm and packed historical LQ under P17-off config.
    Output: generator Y4UV inference through the model's available forward route.
    Key Steps: use the newer wrapper when present, otherwise its bare V5 net.
    Key Params: P17 is recorded as disabled/absent and therefore needs no skip source.
    State/Dependencies: only evaluation uses this compatibility adapter; training uses optimize_parameters.
    """
    if hasattr(model, "_forward_v5"):
        return model._forward_v5(packed_lq)
    return model.get_bare_model(model.net_g)(packed_lq)


def _luma_clipped(rgb: torch.Tensor) -> np.ndarray:
    """@luna-comment
    Input: raw RGB BCHW tensor.
    Output: display-clipped continuous-float luminance as NumPy BCHW.
    Key Steps: clip RGB only, then apply the fixed RGB-to-Y coefficients.
    Key Params: this never quantizes to 8-bit.
    State/Dependencies: mirrors the corrected display-domain RCA metric.
    """
    value = rgb.detach().float().clamp(0.0, 1.0)
    return (.299 * value[:, :1] + .587 * value[:, 1:2] + .114 * value[:, 2:3]).cpu().numpy()


def _profile_fft(y: np.ndarray, roi: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    """@luna-comment
    Input: clipped-float luminance BCHW and a fixed stroke ROI.
    Output: quadratic-detrended, Hann-windowed row-profile FFT and frequencies.
    Key Steps: mean x columns, least-squares quadratic removal, rFFT.
    Key Params: the profile frequency band is measured later at 0.25--0.40.
    State/Dependencies: row coordinate follows the original 96x480 image.
    """
    profile = np.asarray(y)[0, 0, roi[0], roi[1]].mean(axis=1)
    rows = np.arange(profile.size, dtype=np.float64)
    design = np.stack((np.ones_like(rows), rows, rows * rows), axis=1)
    trend = design @ np.linalg.lstsq(design, profile, rcond=None)[0]
    return np.fft.rfft((profile - trend) * np.hanning(profile.size)), np.fft.rfftfreq(profile.size)


def _three_px_metrics(pred: torch.Tensor, gt: torch.Tensor, reference: torch.Tensor | None) -> dict[str, Any]:
    """@luna-comment
    Input: raw prediction/GT and optional original-250k R1 RGB tensors.
    Output: clipped-float luminance, contrast, band-energy, and signed phase metrics.
    Key Steps: process both fixed ROIs with quadratic detrending and Hann FFT.
    Key Params: 0.25--0.40 cycles/pixel is the observed three-pixel candidate band.
    State/Dependencies: reference phase is absent rather than fabricated when omitted.
    """
    pred_y, gt_y = _luma_clipped(pred), _luma_clipped(gt)
    ref_y = None if reference is None else _luma_clipped(reference)
    output: dict[str, Any] = {"domain": "display_clipped_float", "rois": {}}
    for name, roi in ROIS.items():
        pred_fft, freq = _profile_fft(pred_y, roi)
        gt_fft, _ = _profile_fft(gt_y, roi)
        error_fft = pred_fft - gt_fft
        band = (freq >= 0.25) & (freq <= 0.40)
        record: dict[str, Any] = {
            "mean_luminance": float(pred_y[0, 0, roi[0], roi[1]].mean()),
            "contrast_std": float(pred_y[0, 0, roi[0], roi[1]].std()),
            "band_energy": float(np.abs(pred_fft[band]).dot(np.abs(pred_fft[band]))),
            "error_band_energy": float(np.abs(error_fft[band]).dot(np.abs(error_fft[band]))),
            "band_frequencies": freq[band].tolist(),
        }
        if ref_y is not None:
            ref_fft, _ = _profile_fft(ref_y, roi)
            reference_error = ref_fft - gt_fft
            cross = np.vdot(ref_fft[band], pred_fft[band])
            denominator = np.linalg.norm(ref_fft[band]) * np.linalg.norm(pred_fft[band])
            record["phase_to_original_r1_250k"] = None if denominator == 0 else float(np.angle(cross))
            record["signed_phase_correlation_to_original_r1_250k"] = None if denominator == 0 else float(np.real(cross) / denominator)
            error_cross = np.vdot(reference_error[band], error_fft[band])
            error_denominator = np.linalg.norm(reference_error[band]) * np.linalg.norm(error_fft[band])
            record["error_phase_to_original_r1_250k"] = None if error_denominator == 0 else float(np.angle(error_cross))
        output["rois"][name] = record
    return output


def _model_parameters(model: Any) -> Iterable[tuple[str, torch.Tensor]]:
    """@luna-comment
    Input: BasicSR model wrapper.
    Output: stable ordered bare-generator parameter pairs.
    Key Steps: unwrap BasicSR's generator once and retain named_parameters order.
    Key Params: no parameter filtering is permitted in the full-network fork.
    State/Dependencies: ordering is part of the arm identity check.
    """
    return model.get_bare_model(model.net_g).named_parameters()


def _parameter_digest(model: Any) -> tuple[list[str], str]:
    """@luna-comment
    Input: one restored arm.
    Output: ordered parameter names and a content digest.
    Key Steps: concatenate name boundaries and exact tensor byte hashes.
    Key Params: all generator parameters participate, including untouched groups.
    State/Dependencies: used before and after each production optimizer step.
    """
    named = list(_model_parameters(model))
    return [name for name, _ in named], _hash_tree({name: tensor for name, tensor in named})


def _optimizer_summary(model: Any) -> dict[str, Any]:
    """@luna-comment
    Input: arm with production Adam optimizer.
    Output: digest plus exp_avg/exp_avg_sq/step aggregate diagnostics.
    Key Steps: enumerate state by generator parameter name in stable order.
    Key Params: all Adam groups are checked, not only localized B3/B13 tensors.
    State/Dependencies: proves duplicate OFF arms retain identical optimizer state.
    """
    names = {id(parameter): name for name, parameter in _model_parameters(model)}
    state: dict[str, Any] = {}
    total_avg = total_sq = 0.0
    steps: list[float] = []
    for parameter, item in model.optimizer_g.state.items():
        name = names.get(id(parameter), f"unmapped:{id(parameter)}")
        entry: dict[str, Any] = {}
        for key, value in item.items():
            entry[key] = value.detach().cpu() if torch.is_tensor(value) else value
            if key == "exp_avg":
                total_avg += float(torch.linalg.vector_norm(value.detach()).cpu())
            elif key == "exp_avg_sq":
                total_sq += float(torch.linalg.vector_norm(value.detach()).cpu())
            elif key == "step":
                steps.append(float(value.detach().cpu()) if torch.is_tensor(value) else float(value))
        state[name] = entry
    return {"digest": _hash_tree(model.optimizer_g.state_dict()), "parameter_state_digest": _hash_tree(state),
            "exp_avg_l2_sum": total_avg, "exp_avg_sq_l2_sum": total_sq,
            "step_min": min(steps, default=0.0), "step_max": max(steps, default=0.0)}


def _scheduler_digest(model: Any) -> str:
    """@luna-comment
    Input: arm with restored production learning-rate schedulers.
    Output: stable digest of every scheduler state dictionary.
    Key Steps: collect scheduler state in configured list order.
    Key Params: preserves the historical 150k milestone transition.
    State/Dependencies: checked alongside Adam before any optimizer update.
    """
    return _hash_tree([scheduler.state_dict() for scheduler in model.schedulers])


def _group_diffs(left: Any, right: Any) -> dict[str, dict[str, float]]:
    """@luna-comment
    Input: two full-network arms with matching parameter ordering.
    Output: L2/max differences grouped by first module path component.
    Key Steps: subtract every matching parameter and accumulate group norms.
    Key Params: group labels are descriptive only; no branch is frozen.
    State/Dependencies: detects both duplicate-control divergence and ON effects.
    """
    left_items, right_items = list(_model_parameters(left)), list(_model_parameters(right))
    if [name for name, _ in left_items] != [name for name, _ in right_items]:
        raise RuntimeError("parameter ordering differs between matched arms")
    groups: dict[str, dict[str, float]] = {}
    for (name, first), (_, second) in zip(left_items, right_items):
        group = name.split(".", 1)[0]
        delta = (first.detach() - second.detach()).float()
        record = groups.setdefault(group, {"l2_sq": 0.0, "max_abs": 0.0, "parameter_count": 0.0})
        record["l2_sq"] += float((delta * delta).sum().cpu())
        record["max_abs"] = max(record["max_abs"], float(delta.abs().max().cpu()))
        record["parameter_count"] += float(delta.numel())
    for record in groups.values():
        record["l2"] = float(np.sqrt(record.pop("l2_sq")))
    return groups


def _collect_batch(dataset: Any, indices: list[int]) -> dict[str, Any]:
    """@luna-comment
    Input: real Lenovo dataset and deterministic row indices.
    Output: native collated batch preserving masks and segment metadata fields.
    Key Steps: call the dataset's collate function when it defines one.
    Key Params: indices always have the configured historical batch size.
    State/Dependencies: avoids DataLoader workers and their unsaved RNG state.
    """
    rows = [dataset[index] for index in indices]
    if hasattr(dataset, "collate_fn"):
        return dataset.collate_fn(rows)
    return default_collate(rows)


def _batch_indices(length: int, batch_size: int, steps: int, seed: int) -> Iterable[list[int]]:
    """@luna-comment
    Input: dataset cardinality, fixed batch size, step count, and stream seed.
    Output: deterministic batches over repeated seeded permutations.
    Key Steps: consume a CPU Generator only and wrap at epoch boundaries.
    Key Params: batch size must remain the production config's value.
    State/Dependencies: creates a documented new stream, not historical replay.
    """
    if length < batch_size:
        raise ValueError(f"dataset has {length} rows but production batch size is {batch_size}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(length, generator=generator).tolist()
    cursor = 0
    for _ in range(steps):
        if cursor + batch_size > length:
            order = torch.randperm(length, generator=generator).tolist()
            cursor = 0
        yield order[cursor:cursor + batch_size]
        cursor += batch_size


def _copy_generated_fields(source: Any, target: Any) -> None:
    """@luna-comment
    Input: source arm after production feed_data and an un-fed target arm.
    Output: target fields containing exact LQ/GT/masks/metadata values.
    Key Steps: clone every device field consumed by optimize_parameters.
    Key Params: segment_metadata includes Lenovo degradation metadata and CP data.
    State/Dependencies: target does not call degradation or advance its RNG.
    """
    fields = ("lq_rgb", "gt_rgb", "lq", "gt", "valid_mask", "text_mask", "valid_mask_rgb", "text_mask_rgb",
              "structure_available", "segment_metadata", "original_lq")
    for field in fields:
        if hasattr(source, field):
            setattr(target, field, _clone_tree(getattr(source, field)))


def _restore_arm(opt: dict[str, Any], checkpoint: Path, state: dict[str, Any], arm: str) -> Any:
    """@luna-comment
    Input: one complete option mapping plus real generator/state artifacts.
    Output: a train-mode full Lenovo model with restored Adam and scheduler state.
    Key Steps: build production model, strict-load generator, then resume state.
    Key Params: arm changes only recall-aware missing/false weights.
    State/Dependencies: build_model deep-copies options; source files remain read-only.
    """
    arm_opt = copy.deepcopy(opt)
    train = arm_opt["train"]
    train["loss_mode"] = "native"
    train["train_reference_loss"] = False
    train["cp_weight"] = 0.0005
    if arm == "on":
        train["recall_aware_weight_missing"] = 1.0
        train["recall_aware_weight_false"] = 0.25
    else:
        train["recall_aware_weight_missing"] = 0.0
        train["recall_aware_weight_false"] = 0.0
    model = build_model(arm_opt)
    load_strict_checkpoint(model.get_bare_model(model.net_g), checkpoint, map_location="cpu")
    model.resume_training(copy.deepcopy(state))
    model.net_g.train()
    return model


def _save_snapshot(root: Path, step: int, absolute_iter: int, arms: dict[str, Any], historical_lq: torch.Tensor, historical_gt: torch.Tensor,
                   original_r1: torch.Tensor | None) -> dict[str, Any]:
    """@luna-comment
    Input: append-only root, current arms, and fixed historical evaluation tensors.
    Output: per-arm raw/packed arrays, PNG previews, model state, and metrics.
    Key Steps: eval each arm, restore train mode, then serialize a self-contained snapshot.
    Key Params: snapshots are requested at 0/1/10/50/100/250/500 and final.
    State/Dependencies: eval is never used for subsequent training STE gradients.
    """
    result: dict[str, Any] = {"step": step, "absolute_iter": absolute_iter, "arms": {}}
    raw_outputs: dict[str, torch.Tensor] = {}
    packed_lq = rgb_to_y4uv(historical_lq)
    for name, model in arms.items():
        arm_root = root / name / f"step_{step:06d}"
        arm_root.mkdir(parents=True, exist_ok=False)
        was_training = bool(model.net_g.training)
        model.net_g.eval()
        with torch.no_grad():
            packed = _historical_forward(model, packed_lq)
            raw_rgb = _decode_rgb(packed)
        model.net_g.train(was_training)
        raw_outputs[name] = raw_rgb
        raw_cpu, packed_cpu = raw_rgb.detach().cpu().numpy(), packed.detach().cpu().numpy()
        np.savez_compressed(arm_root / "historical_output.npz", raw_rgb=raw_cpu, packed_y4uv=packed_cpu,
                            historical_lq_rgb=historical_lq.detach().cpu().numpy(), historical_gt_rgb=historical_gt.detach().cpu().numpy())
        preview = np.moveaxis(np.clip(raw_cpu[0], 0.0, 1.0), 0, -1)
        Image.fromarray(np.rint(preview * 255.0).astype(np.uint8), mode="RGB").save(arm_root / "historical_output_display.png")
        torch.save({"params": model.get_bare_model(model.net_g).state_dict(),
                    "optimizer": model.optimizer_g.state_dict(), "schedulers": [item.state_dict() for item in model.schedulers],
                    "absolute_iter": absolute_iter}, arm_root / "fork_state.pth")
        result["arms"][name] = {"train_mode_before_eval": was_training, "train_mode_after_eval": bool(model.net_g.training),
                                "metrics": _three_px_metrics(raw_rgb, historical_gt, original_r1),
                                "historical_raw_rgb_digest": _hash_tree(raw_rgb), "historical_packed_y4uv_digest": _hash_tree(packed),
                                "parameter_digest": _parameter_digest(model)[1], "optimizer": _optimizer_summary(model)}
    result["off_aa_comparison"] = _arm_comparison(raw_outputs["off_a"], raw_outputs["off_b"], historical_gt, original_r1, gate=True)
    result["on_vs_off_a_comparison"] = _arm_comparison(raw_outputs["on"], raw_outputs["off_a"], historical_gt, original_r1, gate=False)
    return result


def _historical_output(model: Any, historical_lq: torch.Tensor) -> torch.Tensor:
    """@luna-comment
    Input: one train-mode arm and fixed historical RGB LQ tensor.
    Output: raw historical RGB after temporary eval inference.
    Key Steps: enter eval, run the production forward/decode path, restore train.
    Key Params: raw output is hashed before display clipping.
    State/Dependencies: explicitly preserves training-mode STE for the next step.
    """
    was_training = bool(model.net_g.training)
    model.net_g.eval()
    with torch.no_grad():
        raw = _decode_rgb(_historical_forward(model, rgb_to_y4uv(historical_lq)))
    model.net_g.train(was_training)
    return raw


def _arm_comparison(left: torch.Tensor, right: torch.Tensor, gt: torch.Tensor, reference: torch.Tensor | None, *, gate: bool) -> dict[str, Any]:
    """@luna-comment Compare two historical outputs against the calibrated A/A noise bounds."""
    delta = left - right
    first, second = _three_px_metrics(left, gt, reference), _three_px_metrics(right, gt, reference)
    signed = {roi: first["rois"][roi]["band_energy"] - second["rois"][roi]["band_energy"] for roi in ROIS}
    thresholds = {roi: max(1e-5, .01 * max(first["rois"][roi]["band_energy"], second["rois"][roi]["band_energy"])) for roi in ROIS}
    result = {"raw_max_abs": float(delta.abs().max()), "raw_rms": float(torch.sqrt((delta * delta).mean())),
              "roi_band_energy_signed_delta": signed, "roi_band_energy_threshold": thresholds}
    if gate:
        result["gate"] = {"raw_rms_limit": 1e-4, "raw_rms_pass": result["raw_rms"] <= 1e-4,
                          "roi_pass": {roi: abs(signed[roi]) <= thresholds[roi] for roi in ROIS}}
        result["gate"]["pass"] = result["gate"]["raw_rms_pass"] and all(result["gate"]["roi_pass"].values())
    return result


def main() -> None:
    """@luna-comment
    Input: CLI artifacts for a real R1 150k state and new controlled data stream.
    Output: append-only matched ON/OFF/OFF run evidence, or a hard failure.
    Key Steps: restore identical arms, generate LQ once per batch, run production
    train STE optimizer calls, verify duplicate controls, and save checkpoints.
    Key Params: ON=(missing 1,false .25); OFF=(0,0); CP always remains .0005.
    State/Dependencies: historical RNG is unavailable, so this is not a bit-exact replay.
    """
    args = _args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    smoke_noise_calibration = args.steps <= 3
    required = (args.start_checkpoint, args.start_state, args.config, args.historical_lq, args.historical_gt)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.historical_r1_full is not None and not args.historical_r1_full.is_file():
        raise FileNotFoundError(args.historical_r1_full)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output root already exists: {root}")
    _configure_determinism()
    _seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("config must parse to a YAML mapping")
    # Use the repository configuration, not the known malformed archived copy.
    document = copy.deepcopy(document)
    document.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1, "num_gpu": 1 if device.type == "cuda" else 0})
    document.setdefault("path", {})
    document["path"].update({"pretrain_network_g": None, "resume_state": None})
    state = torch.load(args.start_state, map_location="cpu")
    if not isinstance(state, dict) or "iter" not in state or "optimizers" not in state or "schedulers" not in state:
        raise ValueError("start state must contain iter, optimizers, and schedulers")
    start_iter = int(state["iter"])
    arms = {name: _restore_arm(document, args.start_checkpoint, state, name) for name in ("off_a", "off_b", "on")}
    ordering, initial_digest = _parameter_digest(arms["off_a"])
    initial_optim = _optimizer_summary(arms["off_a"])
    initial_scheduler = _scheduler_digest(arms["off_a"])
    for name, arm in arms.items():
        names, digest = _parameter_digest(arm)
        if names != ordering or digest != initial_digest or _optimizer_summary(arm)["digest"] != initial_optim["digest"] or _scheduler_digest(arm) != initial_scheduler:
            raise RuntimeError(f"initial state mismatch in {name}")
    cp_state_digests = {name: _hash_tree(model.cri_cp.state_dict()) for name, model in arms.items()}
    if len(set(cp_state_digests.values())) != 1:
        raise RuntimeError("initial cri_cp state differs between matched arms")
    dataset_option = copy.deepcopy(document["datasets"]["train"])
    dataset_option["num_worker_per_gpu"] = 0
    dataset = build_dataset(dataset_option)
    batch_size = int(dataset_option["batch_size_per_gpu"])
    historical_lq, historical_gt = _load_rgb(args.historical_lq, device), _load_rgb(args.historical_gt, device)
    original_r1 = None if args.historical_r1_full is None else _load_rgb(args.historical_r1_full, device)
    if tuple(historical_lq.shape[-2:]) != (96, 480) or tuple(historical_gt.shape[-2:]) != (96, 480):
        raise ValueError("historical evaluation pair must be 96x480 RGB")
    root.mkdir(parents=True)
    source_files = (Path(__file__), Path("realesrgan/models/lenovo_text_line_degradation_model.py"),
                    Path("realesrgan/models/basic_text_line_y4uv_training_model_v3.py"), Path("realesrgan/lenovo_p17_preprocess.py"))
    report: dict[str, Any] = {
        "status": "RUNNING", "contract": "new matched deterministic stream; not a bit-exact historical RNG/dataloader replay",
        "start_iter": start_iter, "steps": args.steps, "seed": args.seed, "device": str(device), "tf32_disabled": True,
        "deterministic_algorithms": "warn_only unless deterministic resize-backward mode temporarily enables strict mode inside validation/optimize",
        "resize_backward_mode": "cuda_forward_cpu_deterministic_bicubic_and_bilinear_backward" if args.cp_bicubic_cpu else "disabled",
        "smoke_noise_calibration": smoke_noise_calibration,
        "config": {"path": str(args.config), "sha256": _sha256_file(args.config)}, "source_checkpoint": {"path": str(args.start_checkpoint), "sha256": _sha256_file(args.start_checkpoint)},
        "source_state": {"path": str(args.start_state), "sha256": _sha256_file(args.start_state)},
        "historical": {"lq": str(args.historical_lq), "gt": str(args.historical_gt), "original_r1_full": None if args.historical_r1_full is None else str(args.historical_r1_full)},
        "initial": {"parameter_order": ordering, "parameter_digest": initial_digest, "optimizer": initial_optim,
                    "scheduler_digest": initial_scheduler, "cri_cp_state_digests": cp_state_digests},
        "loss_contract": {"off": {"native": True, "cp_weight": .0005, "missing": 0.0, "false": 0.0},
                          "on": {"native": True, "cp_weight": .0005, "missing": 1.0, "false": .25}},
        "p17_config": document.get("lq_preprocess", {"enabled": False, "reason": "absent_from_R1_config"}),
        "architecture_sha256": _sha256_file(Path("realesrgan/archs/basic_text_line_r4_student_y4uv_v5_arch.py")),
        "source_sha256": {str(path): _sha256_file(path) for path in source_files if path.is_file()}, "snapshots": [],
    }
    (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    initial_snapshot = _save_snapshot(root, 0, start_iter, arms, historical_lq, historical_gt, original_r1)
    if not initial_snapshot["off_aa_comparison"]["gate"]["pass"]:
        raise RuntimeError("initial OFF A/A output exceeds numerical noise gate")
    report["snapshots"].append(initial_snapshot)
    (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (root / "steps.jsonl").open("x", encoding="utf-8") as log:
        for relative_step, indices in enumerate(_batch_indices(len(dataset), batch_size, args.steps, args.seed), start=1):
            step_started = time.perf_counter()
            absolute_iter = start_iter + relative_step
            _seed_all(args.seed + relative_step)
            batch = _collect_batch(dataset, indices)
            # Exactly one production degradation call creates the shared LQ/masks.
            arms["off_a"]._lenovo_seed = args.seed
            _seed_all(args.seed + absolute_iter)
            arms["off_a"].feed_data(batch)
            for target in (arms["off_b"], arms["on"]):
                _copy_generated_fields(arms["off_a"], target)
            data_hashes = {field: _hash_tree(getattr(arms["off_a"], field)) for field in ("lq_rgb", "gt_rgb", "lq", "gt", "valid_mask", "text_mask", "valid_mask_rgb", "text_mask_rgb", "segment_metadata")}
            arm_data_hashes = {name: {field: _hash_tree(getattr(model, field)) for field in data_hashes} for name, model in arms.items()}
            if any(hashes != data_hashes for hashes in arm_data_hashes.values()):
                raise RuntimeError(f"generated batch copy mismatch at step {relative_step}")
            if relative_step == 1 and args.cp_bicubic_cpu:
                cp_equivalence = _cp_cpu_equivalence(arms["off_a"])
                report["cp_cpu_equivalence_before_step1"] = cp_equivalence
                (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                if not cp_equivalence["pass"]:
                    raise RuntimeError("CPU bicubic CP equivalence did not meet value/gradient limits")
                full_equivalence = _full_optimize_equivalence(arms["on"], absolute_iter, args.seed + absolute_iter * 17)
                report["full_optimize_equivalence_before_step1"] = full_equivalence
                (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                if not full_equivalence["pass"]:
                    raise RuntimeError("deterministic resize backward full optimize equivalence did not meet limits")
            losses: dict[str, Any] = {}
            for name, model in arms.items():
                model.net_g.train()
                _seed_all(args.seed + absolute_iter * 17)
                model.update_learning_rate(absolute_iter, warmup_iter=document["train"].get("warmup_iter", -1))
                with _cpu_bicubic_context(args.cp_bicubic_cpu):
                    model.optimize_parameters(absolute_iter)
                losses[name] = dict(model.log_dict)
                actual_cp_scale = float(model.log_dict.get("cp_scale", -1.0))
                if not math.isclose(actual_cp_scale, 0.0005, rel_tol=1e-6, abs_tol=1e-10):
                    raise RuntimeError(f"{name} changed production CP scale: expected=.0005 actual={actual_cp_scale:.12g}")
                eligible = sum(bool(value) for value in model.segment_metadata.get("cp_eligible", []))
                if eligible and abs(float(model.log_dict.get("l_cp_raw", 0.0))) == 0.0:
                    raise RuntimeError(f"{name} has {eligible} CP-eligible rows but no CP loss")
            a_digest, b_digest = _parameter_digest(arms["off_a"])[1], _parameter_digest(arms["off_b"])[1]
            a_optim, b_optim = _optimizer_summary(arms["off_a"]), _optimizer_summary(arms["off_b"])
            duplicate_required = relative_step <= min(5, args.steps)
            duplicate_equal = a_digest == b_digest and a_optim["digest"] == b_optim["digest"]
            # Unsupported bicubic CUDA backward may differ at low bits; the
            # quantitative historical-output gate below is authoritative.
            historical_outputs = {name: _historical_output(model, historical_lq) for name, model in arms.items()} if duplicate_required else {}
            output_digests = {name: _hash_tree(value) for name, value in historical_outputs.items()}
            # Byte digests are retained as evidence, but unsupported bicubic
            # CUDA backward is accepted/rejected only by the numeric A/A gate.
            output_comparison: dict[str, Any] = {}
            on_off_comparison: dict[str, Any] = {}
            if duplicate_required:
                output_comparison = _arm_comparison(historical_outputs["off_a"], historical_outputs["off_b"], historical_gt, original_r1, gate=True)
                on_off_comparison = _arm_comparison(historical_outputs["on"], historical_outputs["off_a"], historical_gt, original_r1, gate=False)
                if not smoke_noise_calibration and not output_comparison["gate"]["pass"]:
                    raise RuntimeError(f"OFF A/A output exceeds numerical noise gate at step {relative_step}")
            event = {"relative_step": relative_step, "absolute_iter": absolute_iter, "indices": indices, "data_hashes": data_hashes,
                     "arm_data_hashes": arm_data_hashes,
                     "losses": losses, "lr": {name: model.get_current_learning_rate() for name, model in arms.items()},
                     "train_mode": {name: bool(model.net_g.training) for name, model in arms.items()}, "off_aa_equal": duplicate_equal,
                     "optimizer": {name: _optimizer_summary(model) for name, model in arms.items()}, "scheduler_digests": {name: _scheduler_digest(model) for name, model in arms.items()},
                     "parameter_digests": {name: _parameter_digest(model)[1] for name, model in arms.items()}, "historical_output_digests": output_digests, "historical_output_comparison": output_comparison, "on_vs_off_a_comparison": on_off_comparison,
                     "step_elapsed_seconds": time.perf_counter() - step_started,
                     "parameter_group_diffs": {"off_a_vs_off_b": _group_diffs(arms["off_a"], arms["off_b"]), "on_vs_off_a": _group_diffs(arms["on"], arms["off_a"])}}
            log.write(json.dumps(event) + "\n")
            log.flush()
            if relative_step in SNAPSHOT_STEPS or relative_step == args.steps:
                snapshot = _save_snapshot(root, relative_step, absolute_iter, arms, historical_lq, historical_gt, original_r1)
                if not smoke_noise_calibration and not snapshot["off_aa_comparison"]["gate"]["pass"]:
                    raise RuntimeError(f"OFF A/A snapshot exceeds numerical noise gate at step {relative_step}")
                report["snapshots"].append(snapshot)
                (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["status"] = "SMOKE_NOISE_CALIBRATION" if smoke_noise_calibration else "COMPLETE"
    (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_root": str(root)}))


if __name__ == "__main__":
    main()
