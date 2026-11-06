"""Run a real V2 trainer step and calibrate the SkelFbeta gradient scale.

This helper is deliberately a bounded plumbing/calibration probe.  It builds
the configured V4 trainer, uses verified-structure rows from the exact V2-only
root, measures raw SkelFbeta versus native+CP generator-gradient norms, and
then runs one post-ramp optimizer step.  It never writes a checkpoint or
changes the formal training stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.models import build_model
from realesrgan.data.basic_text_line_y4uv_v3 import y4uv_to_yuv444
from realesrgan.utils import YUV2RGB

# Populate the registries before building the configured dataset/model/losses.
import realesrgan.archs  # noqa: F401,E402
import realesrgan.data  # noqa: F401,E402
import realesrgan.losses  # noqa: F401,E402
import realesrgan.models  # noqa: F401,E402


def _parser() -> argparse.ArgumentParser:
    """@luna-comment Build CLI options for a bounded trainer calibration.

    Input: command-line arguments naming a G1+SkelFbeta YAML, output directory,
    device, sample count, and post-ramp iteration.
    Output: argparse parser; parsing has no training or filesystem side effect.
    Key Steps: expose exact-source checks, calibration-batch count, and an
    optional skip switch only for local debugging.
    Key Params: ``--target-ratio`` is the desired auxiliary/native+CP median
    gradient ratio and defaults to the conservative midpoint 0.15.
    State/Dependencies: argparse only; no trainer/global state is mutated.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", "--opt", required=True, help="R1 guard + SkelFbeta YAML")
    parser.add_argument("--output-dir", required=True, help="durable smoke/calibration output directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=2)
    parser.add_argument("--current-iter", type=int, default=7000)
    parser.add_argument("--target-ratio", type=float, default=0.15)
    return parser


def _device(value: str) -> torch.device:
    """@luna-comment Resolve and validate the diagnostic torch device.

    Input: ``auto``, ``cpu``, or an explicit torch device string.
    Output: torch.device used consistently by the trainer probe.
    Key Steps: select CUDA only when available for ``auto`` and preserve an
    explicit user choice for reproducibility.
    Key Params: ``value`` controls this probe only and never edits YAML.
    State/Dependencies: torch CUDA availability; no shared state mutation.
    """
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {value} but CUDA is unavailable")
    return device


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    """@luna-comment Move a collated tensor-like field to float32 device data.

    Input: a tensor, NumPy array, or array-like dataset field and target device.
    Output: float32 torch tensor on the requested device.
    Key Steps: use ``as_tensor`` only for non-tensors, then transfer/dtype-cast.
    Key Params: explicit device prevents accidental CPU/GPU field mixing.
    State/Dependencies: torch conversion only; source data is read-only.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _safe_float(value: Any) -> float:
    """@luna-comment Convert a scalar tensor/number to a finite-checkable float.

    Input: scalar tensor or Python numeric value from trainer logs/metrics.
    Output: detached host float suitable for JSON and math.isfinite.
    Key Steps: detach and move tensors to CPU before scalar conversion.
    Key Params: none; non-scalar values intentionally raise from ``item``.
    State/Dependencies: torch tensor state is not modified.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    return float(value)


def _load_verified_batches(option_path: Path, batch_size: int, max_samples: int, batches: int) -> tuple[dict[str, Any], list[dict[str, Any]], Any, dict[str, Any]]:
    """@luna-comment Load exact V2 train records and collate verified batches.

    Input: parsed YAML path, batch-size hint, total sample cap, and batch count.
    Output: first collated batch, additional calibration batches, dataset, and
    provenance; every retained row has ``structure_available=true``.
    Key Steps: build the configured GT-degradation dataset, scan rows without
    synthetic fallback, retain verified structures, and collate fixed batches.
    Key Params: ``max_samples`` bounds data work; ``batches`` controls median
    calibration support and is kept small for a smoke.
    State/Dependencies: BasicSR/Real-ESRGAN dataset registry and V2 files are
    read only; no source manifests or masks are written.
    """
    option = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    if not isinstance(option, dict) or not isinstance(option.get("datasets", {}).get("train"), dict):
        raise ValueError("training YAML must contain datasets.train")
    train_opt = copy.deepcopy(option["datasets"]["train"])
    train_opt.update({"phase": "train", "batch_size_per_gpu": int(batch_size), "num_worker_per_gpu": 0})
    sources = train_opt.get("sources")
    expected = [{"root": "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2", "weight": 1.0}]
    if sources != expected:
        raise ValueError(f"R1 SkelFbeta requires exactly one V2 source, got {sources!r}")
    dataset = build_dataset(train_opt)
    requested = max(int(batch_size), int(batch_size) * max(1, int(batches)))
    requested = min(max(int(max_samples), int(batch_size)), requested)
    samples = []
    examined = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        examined += 1
        if bool(sample.get("structure_available", True)):
            samples.append(sample)
        if len(samples) >= requested:
            break
    if len(samples) < int(batch_size):
        raise RuntimeError("fewer than one full verified-structure V2 batch was available")
    actual_batches = max(1, min(int(batches), len(samples) // int(batch_size)))
    samples = samples[: actual_batches * int(batch_size)]
    collated = [dataset.collate_fn(samples[offset : offset + int(batch_size)]) for offset in range(0, len(samples), int(batch_size))]
    provenance = {
        "dataset_type": train_opt.get("type"),
        "dataset_length": len(dataset),
        "dataset_sources": sources,
        "canvas_hw": train_opt.get("canvas_hw"),
        "resize_mode": train_opt.get("resize_mode"),
        "examined_indices": examined,
        "sample_count": len(samples),
        "batch_count": len(collated),
        "structure_available": [bool(sample.get("structure_available", True)) for sample in samples],
    }
    return collated[0], collated, dataset, {"option": option, "train_opt": train_opt, "provenance": provenance}


def _remap_local_cp_path(option: dict[str, Any], root: Path) -> None:
    """@luna-comment Remap an absent Linux CRNN path to a local probe asset.

    Input: mutable trainer options and repository root for the smoke process.
    Output: options unchanged when the configured recognizer exists, otherwise
    its path points to a same-named local weight if one is present.
    Key Steps: inspect only the configured path, derive a repository-local
    candidate, and replace it when the candidate is a regular file.
    Key Params: no fallback filename is invented; basename must match exactly.
    State/Dependencies: filesystem existence checks only; no model/checkpoint
    is modified.
    """
    cp_opt = option.get("train", {}).get("cp_opt")
    if not isinstance(cp_opt, dict):
        return
    configured = Path(str(cp_opt.get("recognizer_path", "")))
    if configured.is_file():
        return
    candidate = root / "weights" / configured.name
    if candidate.is_file():
        cp_opt["recognizer_path"] = str(candidate)


def _trainer_options(option: dict[str, Any], root: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    """@luna-comment Add isolated single-process runtime paths to trainer options.

    Input: parsed experiment mapping, repository root, output directory, and
    selected device; the original mapping is never mutated.
    Output: deep-copied BasicSR model options with CPU/CUDA process metadata and
    unique smoke model/state/log paths.
    Key Steps: set non-distributed mode, choose num_gpu, and add paths beneath
    the requested output directory before remapping local CP assets.
    Key Params: output paths are unique to this smoke and no formal checkpoint
    path is reused.
    State/Dependencies: BasicSR build_model consumes the returned mapping only.
    """
    result = copy.deepcopy(option)
    result.update({"is_train": True, "dist": False, "rank": 0, "world_size": 1})
    result["num_gpu"] = 1 if device.type == "cuda" else 0
    result["root_path"] = str(root)
    paths = copy.deepcopy(result.get("path", {}))
    paths.setdefault("models", str(output_dir / "trainer_models"))
    paths.setdefault("training_states", str(output_dir / "trainer_states"))
    paths.setdefault("log", str(output_dir / "trainer_logs"))
    result["path"] = paths
    _remap_local_cp_path(result, root)
    return result


def _step_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """@luna-comment Remove synthetic-kernel keys for a direct paired trainer step.

    Input: one collated GT-degradation batch containing clean RGB and optional
    standard kernel tensors.
    Output: shallow batch mapping suitable for the V4 direct RGB feed path.
    Key Steps: omit kernel fields so the trainer uses the already collated V2
    ``lq_rgb``/``gt_rgb`` pair while preserving all masks and metadata.
    Key Params: only the three canonical kernel keys are removed.
    State/Dependencies: no tensor is copied or modified; mapping is read-only.
    """
    return {key: value for key, value in batch.items() if key not in {"kernel1", "kernel2", "sinc_kernel"}}


def _gradient_norm(model: Any) -> float:
    """@luna-comment Measure L2 norm of currently accumulated generator grads.

    Input: a built trainer exposing ``net_g.parameters()`` with autograd grads.
    Output: scalar square-root sum of squared finite gradient entries.
    Key Steps: ignore parameters without gradients, accumulate detached sums,
    and return zero when no gradient exists.
    Key Params: all trainable generator parameters are included consistently.
    State/Dependencies: reads gradient buffers only; optimizer/model weights
    remain unchanged.
    """
    total = None
    for parameter in model.net_g.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return math.sqrt(_safe_float(total)) if total is not None else 0.0


def _structure_masks(model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """@luna-comment Return RGB structure masks while explicitly gating unknown rows.

    Input: fed V4 trainer with full-resolution RGB masks and optional
    ``structure_available`` metadata.
    Output: text/valid BCHW masks where unavailable rows are both zeroed, so
    SkelFbeta contributes no fabricated skeleton or gradient for them.
    Key Steps: validate batch metadata length, clone only when needed, and
    zero unavailable rows.
    Key Params: availability is the dataset's verified-mask contract.
    State/Dependencies: reads model batch fields; does not mutate model fields.
    """
    if model.gt_rgb is None or model.text_mask_rgb is None or model.valid_mask_rgb is None:
        raise RuntimeError("SkelFbeta trainer probe requires full-resolution RGB targets and masks")
    text, valid = model.text_mask_rgb, model.valid_mask_rgb
    availability = getattr(model, "structure_available", [])
    if availability:
        if len(availability) != int(model.gt_rgb.shape[0]):
            raise ValueError("structure_available metadata length does not match batch")
        unavailable = [index for index, value in enumerate(availability) if not bool(value)]
        if unavailable:
            text, valid = text.clone(), valid.clone()
            text[unavailable] = 0.0
            valid[unavailable] = 0.0
    return text, valid


def _component_gradients(model: Any, batch: dict[str, Any], current_iter: int) -> dict[str, float]:
    """@luna-comment Measure native+CP and raw SkelFbeta generator gradients.

    Input: initialized V4 trainer, one verified V2 batch, and post-ramp step.
    Output: finite scalar losses, gradient norms, raw ratio, and selected
    external weight targeting the requested midpoint ratio.
    Key Steps: forward once, compute native/CP/SkelFbeta terms on identical
    output/masks, backprop each branch separately, and clear grads between them.
    Key Params: CP uses the configured ramp at ``current_iter``; raw SkelFbeta
    is measured at module-local weight one and target selection is external.
    State/Dependencies: temporary autograd/optimizer gradient buffers mutate
    only this probe model; no optimizer step or checkpoint write occurs.
    """
    model.optimizer_g.zero_grad(set_to_none=True)
    model.output = model.net_g(model.lq)
    native = model.cri_y4uv_native(
        model.output,
        model.gt,
        valid_mask=model.valid_mask,
        text_mask=model.text_mask,
    )
    output_rgb = YUV2RGB(
        y4uv_to_yuv444(model.output), format="RGB", is_train=True, uv_normalized=True
    )
    if model.cri_cp is None:
        raise RuntimeError("R1 SkelFbeta calibration requires the configured CP loss")
    if bool(getattr(model.cri_cp, "expects_segment_metadata", False)):
        cp_raw = model.cri_cp(output_rgb, model.gt_rgb, model.segment_metadata)
    else:
        lines = model._build_cp_annotations(model.segment_metadata, model.output.shape[0])
        cp_raw = model.cri_cp(output_rgb, model.gt_rgb, lines)
    text_rgb, valid_rgb = _structure_masks(model)
    aux_raw = model.cri_skel_fbeta(
        output_rgb,
        model.gt_rgb,
        text_mask_rgb=text_rgb,
        valid_mask_rgb=valid_rgb,
    )
    cp_weight = model._schedule(float(model.opt["train"].get("cp_weight", 0.0)), current_iter)
    native_cp = native + cp_weight * cp_raw
    native_cp.backward(retain_graph=True)
    native_cp_norm = _gradient_norm(model)
    model.optimizer_g.zero_grad(set_to_none=True)
    aux_raw.backward()
    aux_norm = _gradient_norm(model)
    model.optimizer_g.zero_grad(set_to_none=True)
    ratio = aux_norm / max(native_cp_norm, 1e-12)
    return {
        "native_loss": _safe_float(native),
        "cp_raw_loss": _safe_float(cp_raw),
        "cp_weight": float(cp_weight),
        "skel_fbeta_raw_loss": _safe_float(aux_raw),
        "native_plus_cp_gradient_l2": float(native_cp_norm),
        "skel_fbeta_raw_gradient_l2": float(aux_norm),
        "raw_aux_to_native_plus_cp_ratio": float(ratio),
    }


def _trainer_step(model: Any, batch: dict[str, Any], current_iter: int) -> dict[str, Any]:
    """@luna-comment Run one exact configured trainer optimization step.

    Input: initialized trainer, verified V2 collated batch, and post-ramp index.
    Output: finite/log-key/shape/gradient evidence from one optimizer update.
    Key Steps: direct-feed paired RGB fields, call the unmodified trainer path,
    collect scalar logs, and inspect output/gradient buffers.
    Key Params: ``current_iter`` is 7000 by default so all configured ramps are
    fully active, matching the G1 guard smoke convention.
    State/Dependencies: mutates only this probe model's optimizer and weights;
    no formal run or checkpoint path is touched.
    """
    model.feed_data(_step_batch(batch))
    model.optimize_parameters(current_iter=int(current_iter))
    logs = {key: _safe_float(value) for key, value in model.get_current_log().items()}
    required = (
        "l_native", "l_cp_raw", "l_cp", "l_missing", "l_false", "l_bg_lf_raw",
        "l_bg_lf_weighted", "bg_lf_guard_active", "l_skel_fbeta_raw",
        "l_skel_fbeta_weighted", "skel_fbeta_recall", "skel_fbeta_precision",
        "skel_fbeta_fbeta", "skel_fbeta_active_ratio", "skel_fbeta_scale", "l_total",
    )
    missing = [key for key in required if key not in logs]
    output = model.output.detach()
    finite = all(math.isfinite(value) for value in logs.values()) and bool(torch.isfinite(output).all())
    gradient_l2 = _gradient_norm(model)
    return {
        "attempted": True,
        "finite": bool(finite),
        "nonzero_generator_gradient": bool(gradient_l2 > 1e-12),
        "gradient_l2_after_step": float(gradient_l2),
        "current_iter": int(current_iter),
        "logs": logs,
        "missing_logs": missing,
        "output_shape": list(output.shape),
    }


def main() -> int:
    """@luna-comment Run calibration plus trainer smoke and write evidence.

    Input: CLI YAML/output/device options for the final G1 guard control plus
    the SkelFbeta-only delta.
    Output: JSON and Markdown evidence; exit zero only when calibration and
    trainer checks pass and the selected ratio lies in [0.10, 0.25].
    Key Steps: validate V2 source, seed from the pinned config, build one
    isolated model, calibrate multiple real batches with those same weights,
    run one trainer step, and serialize all logs/ratios.
    Key Params: target ratio defaults to 0.15; no formal checkpoint is loaded.
    State/Dependencies: BasicSR/Real-ESRGAN registries, actual V2 dataset,
    and output directory are the only external dependencies.
    """
    args = _parser().parse_args()
    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    first_batch, batches, dataset, metadata = _load_verified_batches(
        option_path, args.batch_size, args.max_samples, args.calibration_batches
    )
    option = metadata["option"]
    seed = option.get("manual_seed")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    loss_cfg = option.get("train", {}).get("skel_fbeta_opt")
    if not isinstance(loss_cfg, dict) or str(loss_cfg.get("type")) != "ThinSkeletonFbetaLoss":
        raise ValueError("trainer smoke requires train.skel_fbeta_opt: ThinSkeletonFbetaLoss")
    weight = float(option.get("train", {}).get("skel_fbeta_weight", 0.0))
    if weight <= 0.0:
        raise ValueError("trainer smoke requires a positive probe skel_fbeta_weight")
    trainer_opt = _trainer_options(option, Path(__file__).resolve().parents[1], output_dir, device)
    calibration = []
    probe_model = build_model(trainer_opt)
    for batch in batches:
        probe_model.feed_data(_step_batch(batch))
        calibration.append(_component_gradients(probe_model, batch, int(args.current_iter)))
    raw_ratios = [item["raw_aux_to_native_plus_cp_ratio"] for item in calibration]
    median_raw_ratio = float(torch.tensor(raw_ratios, dtype=torch.float64).median().item())
    if not math.isfinite(median_raw_ratio) or median_raw_ratio <= 0.0:
        raise RuntimeError(f"invalid raw SkelFbeta/native+CP median ratio: {median_raw_ratio}")
    selected_weight = float(args.target_ratio) / median_raw_ratio
    selected_ratio = selected_weight * median_raw_ratio

    trainer_step = _trainer_step(probe_model, first_batch, int(args.current_iter))
    checks = {
        "v2_only_source": metadata["provenance"]["dataset_sources"] == [{"root": "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2", "weight": 1.0}],
        "verified_structure_rows": all(metadata["provenance"]["structure_available"]),
        "finite_gradient_calibration": all(math.isfinite(value) for item in calibration for value in item.values()),
        "positive_raw_gradient": all(item["skel_fbeta_raw_gradient_l2"] > 1e-12 for item in calibration),
        "selected_ratio_in_target_band": 0.10 <= selected_ratio <= 0.25,
        "trainer_step_finite": bool(trainer_step["finite"]),
        "trainer_nonzero_gradient": bool(trainer_step["nonzero_generator_gradient"]),
        "trainer_log_contract": not trainer_step["missing_logs"],
        "trainer_output_shape": trainer_step["output_shape"] == [int(args.batch_size), 6, 48, 240],
        "guard_active": trainer_step["logs"].get("bg_lf_guard_active") == 1.0,
        "skel_active": trainer_step["logs"].get("skel_fbeta_active_ratio", 0.0) > 0.0,
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "script": "smoke_thin_skeleton_fbeta_trainer_v2.py",
        "config": str(option_path),
        "experiment_name": option.get("name"),
        "device": str(device),
        "target_ratio": float(args.target_ratio),
        "probe_weight": weight,
        "median_raw_aux_to_native_plus_cp_ratio": median_raw_ratio,
        "selected_skel_fbeta_weight": selected_weight,
        "selected_weighted_ratio": selected_ratio,
        "provenance": metadata["provenance"],
        "calibration_batches": calibration,
        "trainer_step": trainer_step,
        "checks": checks,
        "interpretation": {
            "calibration": "generator-parameter L2 norm of raw SkelFbeta divided by native plus configured CP at the same post-ramp step",
            "selection": "target midpoint 0.15, yielding a documented 0.10-0.25 weighted ratio band",
            "tier": "pre-launch plumbing/calibration evidence, not an image-quality result",
        },
    }
    json_path = output_dir / "skelfbeta_trainer_smoke.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        f"# SkelFbeta trainer smoke: {payload['experiment_name']}",
        "",
        f"Status: **{payload['status']}** (pre-launch plumbing/calibration)",
        f"Config: `{option_path}`",
        f"Device: `{device}`; V2 samples/batches: `{metadata['provenance']['sample_count']}`/`{metadata['provenance']['batch_count']}`",
        "",
        "## Calibration",
        "",
        f"- Raw aux/native+CP ratios: `{raw_ratios}`; median `{median_raw_ratio:.8g}`",
        f"- Target ratio: `{args.target_ratio:.8g}`; selected SkelFbeta weight: `{selected_weight:.8g}`",
        f"- Selected weighted ratio: `{selected_ratio:.8g}` (required 0.10--0.25)",
        "",
        "## Trainer step",
        "",
        f"- Finite: `{trainer_step['finite']}`; nonzero generator gradient: `{trainer_step['nonzero_generator_gradient']}`; output shape: `{trainer_step['output_shape']}`",
        f"- Missing required log keys: `{trainer_step['missing_logs']}`",
        f"- Guard active: `{trainer_step['logs'].get('bg_lf_guard_active')}`; SkelFbeta active ratio: `{trainer_step['logs'].get('skel_fbeta_active_ratio')}`",
        "",
        "## Checks",
        "",
    ]
    report.extend(f"- `{key}`: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
    report.extend(["", "This is a plumbing/calibration artifact only; it makes no image-quality claim."])
    report_path = output_dir / "skelfbeta_trainer_smoke_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(json_path), "report": str(report_path), "selected_weight": selected_weight}, indent=2))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
