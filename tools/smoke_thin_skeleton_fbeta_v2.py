"""Run the SkelFbeta contract against a real V2 training batch.

The smoke is data-level and training-only.  It loads the configured
``BasicTextLineGTDegradeDataset`` records, checks a V4 network forward when
available, and applies controlled RGB perturbations to the same GT batch.  It
does not change source images, checkpoints, generator inputs, or optimizer
state.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.data import build_dataset
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.utils import YUV2RGB

# Load this new module by path so a preparation smoke does not execute the
# legacy ``realesrgan.losses`` package initializer.  The source-home checkout
# used for Remote58 preparation can contain an unrelated partial overlay;
# direct loading keeps this diagnostic isolated while still registering the
# class in BasicSR's LOSS_REGISTRY.
_LOSS_PATH = Path(__file__).resolve().parents[1] / "realesrgan" / "losses" / "thin_skeleton_fbeta_loss.py"
_LOSS_SPEC = importlib.util.spec_from_file_location("skelfbeta_loss_preparation", _LOSS_PATH)
if _LOSS_SPEC is None or _LOSS_SPEC.loader is None:
    raise ImportError(f"cannot load standalone SkelFbeta loss from {_LOSS_PATH}")
_LOSS_MODULE = importlib.util.module_from_spec(_LOSS_SPEC)
_LOSS_SPEC.loader.exec_module(_LOSS_MODULE)
ThinSkeletonFbetaLoss = _LOSS_MODULE.ThinSkeletonFbetaLoss

# These imports populate the repository registries used by build_dataset and
# build_network.  The new loss remains directly imported above until the
# parent trainer receives the integration greenlight.
import realesrgan.archs  # noqa: F401,E402
import realesrgan.data  # noqa: F401,E402


def _parser() -> argparse.ArgumentParser:
    """@luna-comment Build CLI options for a real-V2 SkelFbeta smoke run.

    Input: command-line arguments naming a training YAML, output directory,
    device, and bounded sample count.
    Output: an argparse parser; parsing itself has no filesystem side effect.
    Key Steps: declare the source contract and optional root override.
    Key Params: ``--max-samples`` bounds data work; ``--device`` selects CPU or
    CUDA without changing the experiment configuration.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", "--opt", required=True, help="R1+bgLF V2-only training YAML")
    parser.add_argument("--output-dir", required=True, help="directory for JSON/report/panel outputs")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument(
        "--dataset-root",
        default="",
        help="optional V2 root override; otherwise use the YAML source root",
    )
    parser.add_argument("--skip-network", action="store_true", help="skip optional V4 forward probe")
    return parser


def _device(value: str) -> torch.device:
    """@luna-comment Resolve the requested smoke device with CUDA validation.

    Input: ``auto``, ``cpu``, or a torch device string.
    Output: a torch.device suitable for dataset tensors and the network probe.
    Key Steps: use CUDA when auto is requested and available.
    Key Params: ``value`` controls only this diagnostic process.
    State/Dependencies: torch.cuda availability.
    """
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    """@luna-comment Convert a collated value to float32 on the smoke device.

    Input: a tensor, NumPy array, or array-like value from a dataset batch.
    Output: float32 tensor on ``device``.
    Key Steps: use ``torch.as_tensor`` only for non-tensor values, then move.
    Key Params: device is explicit to avoid accidental CPU/GPU mixing.
    State/Dependencies: torch tensor conversion.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _finite(value: torch.Tensor | float) -> bool:
    """@luna-comment Return whether every value in a smoke result is finite.

    Input: scalar or tensor loss/metric result.
    Output: Python boolean for JSON checks.
    Key Steps: evaluate torch.isfinite without changing the value.
    Key Params: none.
    State/Dependencies: torch.
    """
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all())
    return bool(np.isfinite(float(value)))


def _float(value: Any) -> float:
    """@luna-comment Convert a scalar tensor/number into a JSON-safe float.

    Input: scalar tensor or numeric value.
    Output: Python float.
    Key Steps: detach and move tensors to CPU before conversion.
    Key Params: none.
    State/Dependencies: torch for tensor inputs.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """@luna-comment Dilate a BCHW binary mask using the loss-compatible kernel.

    Input: BCHW mask and non-negative pixel radius.
    Output: BCHW max-pooled dilation with the same shape.
    Key Steps: apply zero-outside max pooling for positive radius.
    Key Params: radius defines the excluded near-text neighborhood.
    State/Dependencies: torch.nn.functional.max_pool2d.
    """
    radius = int(radius)
    if radius <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)


def _patches(candidate: torch.Tensor, *, half_size: int = 2) -> torch.Tensor:
    """@luna-comment Select one bounded candidate patch per batch sample.

    Input: detached BCHW candidate mask where nonzero pixels are eligible.
    Output: a same-shape binary patch intersected with each candidate.
    Key Steps: choose the median candidate coordinate and clip a square to the
    canvas; empty candidates produce an empty patch.
    Key Params: half_size controls perturbation extent, not loss behavior.
    State/Dependencies: torch.nonzero; output is detached and diagnostic-only.
    """
    result = torch.zeros_like(candidate)
    height, width = candidate.shape[-2:]
    half_size = max(0, int(half_size))
    for index in range(int(candidate.shape[0])):
        coords = torch.nonzero(candidate[index, 0] > 0.0, as_tuple=False)
        if coords.numel() == 0:
            continue
        center = coords[len(coords) // 2]
        cy, cx = int(center[0]), int(center[1])
        y0, y1 = max(0, cy - half_size), min(height, cy + half_size + 1)
        x0, x1 = max(0, cx - half_size), min(width, cx + half_size + 1)
        result[index, 0, y0:y1, x0:x1] = 1.0
    return result * (candidate > 0.0).to(result.dtype)


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """@luna-comment Compute RGB luminance for diagnostic perturbations.

    Input: BCHW RGB tensor in the training range.
    Output: BCHW one-channel luminance tensor.
    Key Steps: apply the same Rec.601 coefficients as the loss.
    Key Params: fixed coefficients preserve polarity semantics.
    State/Dependencies: torch autograd remains active when input requires it.
    """
    return 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _shift_to_ink(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    polarity: torch.Tensor,
    amount: float,
) -> torch.Tensor:
    """@luna-comment Move selected pixels toward GT-derived ink polarity.

    Input: BCHW RGB, BCHW patch mask, per-sample polarity (+dark/-light), and
    positive luminance amount.
    Output: clipped RGB tensor with a differentiable-free controlled edit.
    Key Steps: add ``-polarity * amount`` equally to RGB channels on the patch.
    Key Params: amount is bounded by [0,1] and is used only for sensitivity.
    State/Dependencies: no source or model state is modified.
    """
    delta = -polarity.reshape(-1, 1, 1, 1).to(device=rgb.device, dtype=rgb.dtype)
    return (rgb + delta * float(amount) * mask).clamp(0.0, 1.0)


def _erase_to_midpoint(
    rgb: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    midpoint: torch.Tensor,
) -> torch.Tensor:
    """@luna-comment Replace selected GT ink with its local midpoint value.

    Input: RGB GT/prediction, BCHW selected centerline patch, and per-sample
    GT-derived midpoint.
    Output: clipped RGB tensor in which the selected centerline is weakened.
    Key Steps: shift all RGB channels by midpoint minus GT luminance.
    Key Params: mask controls deletion/break size; midpoint comes from GT only.
    State/Dependencies: diagnostic tensor construction only.
    """
    delta = midpoint.reshape(-1, 1, 1, 1).to(rgb.dtype) - _luma(target)
    return (rgb + delta * mask).clamp(0.0, 1.0)


def _load_batch(option_path: Path, dataset_root: str, batch_size: int, max_samples: int) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """@luna-comment Load verified-structure samples from the real V2 dataset.

    Input: R1 YAML path, optional root override, batch-size hint, and sample cap.
    Output: collated V2 batch, dataset object, and provenance metadata.
    Key Steps: build the exact configured train dataset, retain rows with a
    verified structure mask, and call its native collate function.
    Key Params: no fallback/synthetic data is allowed; max_samples bounds work.
    State/Dependencies: BasicSR and Real-ESRGAN dataset registries; source files
    are read only.
    """
    opt = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    if not isinstance(opt, dict) or not isinstance(opt.get("datasets", {}).get("train"), dict):
        raise ValueError("training YAML must contain datasets.train")
    train_opt = copy.deepcopy(opt["datasets"]["train"])
    train_opt.update({"phase": "train", "batch_size_per_gpu": int(batch_size), "num_worker_per_gpu": 0})
    if dataset_root:
        for source in train_opt.get("sources", []):
            source["root"] = str(dataset_root)
            source.pop("manifest", None)
            source.pop("manifest_jsonl", None)
    dataset = build_dataset(train_opt)
    requested = max(1, int(max_samples))
    samples = []
    examined = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        examined += 1
        if bool(sample.get("structure_available", True)):
            samples.append(sample)
        if len(samples) >= requested:
            break
    if len(samples) < requested:
        raise RuntimeError(
            f"found only {len(samples)} verified-structure V2 samples after examining {examined}; "
            "SkelFbeta must not silently train on mask-less rows"
        )
    batch = dataset.collate_fn(samples)
    provenance = {
        "dataset_type": train_opt.get("type"),
        "dataset_length": len(dataset),
        "sample_count": len(samples),
        "examined_indices": examined,
        "dataset_sources": train_opt.get("sources"),
        "canvas_hw": train_opt.get("canvas_hw"),
        "resize_mode": train_opt.get("resize_mode"),
        "structure_available": [bool(sample.get("structure_available", True)) for sample in samples],
    }
    return batch, dataset, {"option": opt, "train_opt": train_opt, "provenance": provenance}


def _network_probe(option: dict[str, Any], batch: dict[str, Any], device: torch.device, skip: bool) -> dict[str, Any]:
    """@luna-comment Run an optional V4 forward on the same actual V2 batch.

    Input: parsed R1 YAML, collated V2 batch, target device, and skip switch.
    Output: JSON-safe shape/finiteness report; exceptions are recorded.
    Key Steps: build configured network, pack ``lq_rgb`` as Y4UV, decode output.
    Key Params: skip is diagnostic-only and never changes the loss contract.
    State/Dependencies: network registry and Y4UV conversion helpers; no
    checkpoint or optimizer state is touched.
    """
    if skip:
        return {"attempted": False, "finite": None, "skipped": True}
    try:
        network = build_network(copy.deepcopy(option["network_g"])).to(device).eval()
        lq_rgb = _tensor(batch["lq_rgb"], device)
        lq_y4uv = rgb_to_y4uv(lq_rgb)
        with torch.no_grad():
            output_y4uv = network(lq_y4uv)
            output_rgb = YUV2RGB(
                y4uv_to_yuv444(output_y4uv), format="RGB", is_train=True, uv_normalized=True
            )
        report = {
            "attempted": True,
            "finite": bool(torch.isfinite(output_y4uv).all() and torch.isfinite(output_rgb).all()),
            "input_shape": list(lq_y4uv.shape),
            "output_shape": list(output_y4uv.shape),
            "decoded_rgb_shape": list(output_rgb.shape),
        }
        del network, output_y4uv, output_rgb
        return report
    except Exception as error:  # smoke report must preserve the concrete failure
        return {"attempted": True, "finite": False, "error": repr(error)}


def _run_checks(batch: dict[str, Any], loss: ThinSkeletonFbetaLoss, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """@luna-comment Evaluate all SkelFbeta perturbation categories on one batch.

    Input: collated actual V2 batch, standalone loss module, and device.
    Output: JSON-safe check record plus first-sample visualization tensors.
    Key Steps: compute base maps, apply centerline deletion/break, external
    spur/overgrowth, forced-padding, and polarity-inversion probes, then check
    finite values and gradient support.
    Key Params: perturbation amplitudes are fixed diagnostic values; loss
    hyperparameters come solely from the module.
    State/Dependencies: autograd is used only for a local prediction tensor;
    no generator/optimizer state is mutated.
    """
    gt = _tensor(batch["gt_rgb"], device)
    valid = _tensor(batch["valid_mask_rgb"], device)
    text = _tensor(batch["text_mask_rgb"], device)
    if text.ndim == 3:
        text = text.unsqueeze(1)
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    base_loss, base_details = loss(
        gt,
        gt,
        text_mask_rgb=text,
        valid_mask_rgb=valid,
        return_details=True,
        return_maps=True,
    )
    skeleton = base_details["skeleton"]
    tube = base_details["tube"]
    text_map = base_details["text"]
    valid_map = base_details["valid"]
    polarity = base_details["polarity"]
    midpoint = base_details["midpoint"]
    far_candidate = valid_map * (1.0 - _dilate(text_map, loss.polarity_radius + 2)).clamp_min(0.0)
    external_candidate = valid_map * (1.0 - text_map).clamp_min(0.0)
    spur_patch = _patches(far_candidate if bool(torch.any(far_candidate)) else external_candidate, half_size=3)
    overgrowth_patch = _patches(external_candidate, half_size=3)
    break_patch = _patches(skeleton, half_size=0)
    deletion_patch = _patches(skeleton, half_size=2)

    break_pred = _erase_to_midpoint(gt, gt, break_patch, midpoint)
    deletion_pred = _erase_to_midpoint(gt, gt, deletion_patch, midpoint)
    spur_pred = _shift_to_ink(gt, spur_patch, polarity, amount=0.35)
    overgrowth_pred = _shift_to_ink(gt, overgrowth_patch, polarity, amount=0.35)

    break_loss, break_details = loss(break_pred, gt, text, valid, return_details=True)
    deletion_loss, deletion_details = loss(deletion_pred, gt, text, valid, return_details=True)
    spur_loss, spur_details = loss(spur_pred, gt, text, valid, return_details=True)
    overgrowth_loss, overgrowth_details = loss(overgrowth_pred, gt, text, valid, return_details=True)

    # Force an external patch to act as padding and compare perturbed/base
    # losses under the identical forced valid mask.  This works even when the
    # configured stretch path has no natural padding columns.
    padding_patch = _patches(external_candidate, half_size=3)
    forced_valid = (valid_map * (1.0 - padding_patch)).clamp_min(0.0)
    forced_base_loss = loss(gt, gt, text_map, forced_valid)
    forced_padding_loss = loss(
        _shift_to_ink(gt, padding_patch, polarity, amount=0.35), gt, text_map, forced_valid
    )

    inverse_loss = loss(1.0 - gt, 1.0 - gt, text_map, valid_map)

    grad_pred = (gt + 0.01 * torch.ones_like(gt)).clamp(0.0, 1.0).detach().requires_grad_(True)
    grad_loss = loss(grad_pred, gt, text, valid)
    grad_loss.backward()
    gradient_norm = grad_pred.grad.detach().abs().sum()

    def mean_metric(details: dict[str, torch.Tensor], key: str) -> float:
        active = details["active"] > 0.0
        value = details[key]
        if bool(active.any()):
            return _float(value[active].mean())
        return float("nan")

    base_recall = mean_metric(base_details, "skeleton_recall")
    break_recall = mean_metric(break_details, "skeleton_recall")
    deletion_recall = mean_metric(deletion_details, "skeleton_recall")
    base_precision = mean_metric(base_details, "occupancy_precision")
    spur_precision = mean_metric(spur_details, "occupancy_precision")
    overgrowth_precision = mean_metric(overgrowth_details, "occupancy_precision")
    base_loss_value = _float(base_loss)
    checks = {
        "finite_base_loss": _finite(base_loss),
        "finite_perturbation_losses": all(
            _finite(value) for value in (break_loss, deletion_loss, spur_loss, overgrowth_loss, inverse_loss)
        ),
        "nonzero_prediction_gradient": _float(gradient_norm) > 1e-8,
        "verified_active_structure": bool((base_details["active"] > 0.0).any()),
        "break_sensitivity": _float(break_loss) > base_loss_value + 1e-6 and break_recall < base_recall - 1e-4,
        "deletion_sensitivity": _float(deletion_loss) > base_loss_value + 1e-6 and deletion_recall < base_recall - 1e-4,
        "false_spur_penalty": _float(spur_loss) > base_loss_value + 1e-6 and spur_precision < base_precision - 1e-4,
        "overgrowth_penalty": _float(overgrowth_loss) > base_loss_value + 1e-6 and overgrowth_precision < base_precision - 1e-4,
        "padding_exclusion": abs(_float(forced_padding_loss) - _float(forced_base_loss)) < 1e-5,
        "polarity_robustness": abs(_float(inverse_loss) - base_loss_value) < 2e-2,
    }
    record = {
        "loss_base": base_loss_value,
        "loss_break": _float(break_loss),
        "loss_deletion": _float(deletion_loss),
        "loss_false_spur": _float(spur_loss),
        "loss_overgrowth": _float(overgrowth_loss),
        "loss_forced_padding_base": _float(forced_base_loss),
        "loss_forced_padding_perturbed": _float(forced_padding_loss),
        "loss_polarity_inverted": _float(inverse_loss),
        "gradient_norm_l1": _float(gradient_norm),
        "base_skeleton_recall": base_recall,
        "break_skeleton_recall": break_recall,
        "deletion_skeleton_recall": deletion_recall,
        "base_occupancy_precision": base_precision,
        "spur_occupancy_precision": spur_precision,
        "overgrowth_occupancy_precision": overgrowth_precision,
        "active_samples": int((base_details["active"] > 0.0).sum().item()),
        "batch_size": int(gt.shape[0]),
        "resolution": [int(gt.shape[-2]), int(gt.shape[-1])],
        "valid_pixels": [_float(value) for value in base_details["valid_pixels"]],
        "text_pixels": [_float(value) for value in base_details["text_pixels"]],
        "skeleton_pixels": [_float(value) for value in base_details["skeleton_pixels"]],
        "tube_pixels": [_float(value) for value in base_details["tube_pixels"]],
        "polarity": [_float(value) for value in polarity],
        "midpoint": [_float(value) for value in midpoint],
        "contrast": [_float(value) for value in base_details["contrast"]],
        "checks": checks,
    }
    tensors = {
        "gt": gt[0].detach(),
        "text": text_map[0].detach(),
        "skeleton": skeleton[0].detach(),
        "tube": tube[0].detach(),
        "ink_probability": base_details["ink_probability"][0].detach(),
        "break_patch": break_patch[0].detach(),
        "deletion_patch": deletion_patch[0].detach(),
        "spur_patch": spur_patch[0].detach(),
        "overgrowth_patch": overgrowth_patch[0].detach(),
        "padding_patch": padding_patch[0].detach(),
        "valid": valid_map[0].detach(),
    }
    return record, tensors


def _to_gray(value: torch.Tensor) -> np.ndarray:
    """@luna-comment Convert one diagnostic BCHW/CHW tensor into uint8 gray.

    Input: one- or three-channel tensor with spatial dimensions.
    Output: HxW uint8 grayscale visualization.
    Key Steps: detach, average RGB when needed, clamp, and scale.
    Key Params: fixed [0,1] display range.
    State/Dependencies: torch/numpy only.
    """
    array = value.detach().float().cpu()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            array = 0.299 * array[0] + 0.587 * array[1] + 0.114 * array[2]
    return np.rint(array.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def _write_panel(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """@luna-comment Write a loss-support and perturbation mask panel.

    Input: output path and first-sample visualization tensors.
    Output: lossless PNG with labeled column tiles; raises on write failure.
    Key Steps: convert GT/masks/probes to grayscale and concatenate columns.
    Key Params: nearest-neighbor display preserves one-pixel skeleton geometry.
    State/Dependencies: OpenCV writes only the requested diagnostic artifact.
    """
    labels = (
        "gt", "text", "skeleton", "tube", "ink_probability", "break_patch",
        "deletion_patch", "spur_patch", "overgrowth_patch", "padding_patch", "valid",
    )
    tiles = []
    for label in labels:
        image = _to_gray(tensors[label])
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE) if label == "gt" else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(image, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    panel = np.concatenate(tiles, axis=1)
    if not cv2.imwrite(str(path), panel):
        raise IOError(f"failed to write panel: {path}")


def main() -> int:
    """@luna-comment Run the real-V2 SkelFbeta smoke and serialize its verdict.

    Input: parsed CLI options naming a config, output directory, and device.
    Output: exit 0 only when all required contract checks pass; JSON, Markdown,
    and PNG evidence are written under the output directory.
    Key Steps: load batch, run optional network probe, evaluate loss categories,
    apply checks, and write durable diagnostics.
    Key Params: SkelFbeta options come from ``train.skel_fbeta_opt`` when given.
    State/Dependencies: no source data/checkpoint/trainer mutation; output
    artifacts are the only writes.
    """
    args = _parser().parse_args()
    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    batch, _, metadata = _load_batch(option_path, args.dataset_root, args.batch_size, args.max_samples)
    device = _device(args.device)
    loss_opt = copy.deepcopy(metadata["option"].get("train", {}).get("skel_fbeta_opt") or {})
    loss_opt.pop("type", None)
    loss = ThinSkeletonFbetaLoss(**loss_opt).to(device)
    record, tensors = _run_checks(batch, loss, device)
    network = _network_probe(metadata["option"], batch, device, bool(args.skip_network))
    checks = dict(record["checks"])
    checks["network_forward_finite"] = network.get("finite") is not False
    status = "pass" if all(bool(value) for value in checks.values()) else "fail"
    payload = {
        "status": status,
        "script": "smoke_thin_skeleton_fbeta_v2.py",
        "config": str(option_path),
        "experiment_name": metadata["option"].get("name"),
        "device": str(device),
        "skel_fbeta_opt": loss_opt,
        "provenance": metadata["provenance"],
        "tests": record,
        "network_forward": network,
        "checks": checks,
        "interpretation": {
            "skeleton": "detached GT-only one-pixel morphological skeleton; prediction is never skeletonized",
            "recall": "inverse-radius weighted soft ink probability sampled on the GT skeleton",
            "precision": "GT occupancy true positives divided by true positives plus external false ink and light overgrowth penalty",
            "padding": "forced invalid patch is perturbed under the same valid mask and must leave loss unchanged",
            "polarity": "inverting both GT and prediction should preserve the local polarity/midpoint contract",
            "tier": "plumbing smoke evidence, not an official image-quality result",
        },
    }
    json_path = output_dir / "skelfbeta_smoke.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    panel_path = output_dir / "skelfbeta_smoke_panel.png"
    _write_panel(panel_path, tensors)
    report_lines = [
        f"# SkelFbeta V2 smoke report: {payload['experiment_name']}",
        "",
        f"Status: **{status}** (plumbing smoke; not an image-quality result)",
        f"Config: `{option_path}`",
        f"Dataset: `{metadata['provenance']['dataset_type']}`, {metadata['provenance']['sample_count']} verified samples from {metadata['provenance']['dataset_length']}",
        f"Device: `{device}`; V4 network forward finite: `{network.get('finite')}`",
        "",
        "## Category checks",
        "",
    ]
    for name, passed in checks.items():
        report_lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    report_lines.extend([
        "",
        f"- Base loss: `{record['loss_base']:.8g}`; gradient L1: `{record['gradient_norm_l1']:.8g}`",
        f"- Skeleton recall base/break/delete: `{record['base_skeleton_recall']:.6f}` / `{record['break_skeleton_recall']:.6f}` / `{record['deletion_skeleton_recall']:.6f}`",
        f"- Occupancy precision base/spur/overgrowth: `{record['base_occupancy_precision']:.6f}` / `{record['spur_occupancy_precision']:.6f}` / `{record['overgrowth_occupancy_precision']:.6f}`",
        f"- Loss base/break/delete/spur/overgrowth: `{record['loss_base']:.8g}` / `{record['loss_break']:.8g}` / `{record['loss_deletion']:.8g}` / `{record['loss_false_spur']:.8g}` / `{record['loss_overgrowth']:.8g}`",
        f"- Forced padding base/perturbed: `{record['loss_forced_padding_base']:.8g}` / `{record['loss_forced_padding_perturbed']:.8g}`",
        f"- Polarity inversion loss: `{record['loss_polarity_inverted']:.8g}`",
        "",
        f"Mask panel: `{panel_path}` (GT, text, skeleton, tube, soft ink, break/delete/spur/overgrowth/padding patches, valid).",
    ])
    report_path = output_dir / "skelfbeta_smoke_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "report": str(report_path), "panel": str(panel_path)}, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
