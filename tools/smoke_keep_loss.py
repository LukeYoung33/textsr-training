"""Run the evidence-gated Keep loss contract on an actual V2 batch.

This is a loss/data smoke only. It reads the configured
BasicTextLineGTDegradeDataset records and does not update a generator,
checkpoint, or optimizer. The trainer one-step probe is intentionally a
separate post-integration gate because G2 owns the shared trainer edits.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from basicsr.data import build_dataset


def _load_keep_loss_class() -> type[torch.nn.Module]:
    """Load KeepLoss from the package, with an isolated-stage fallback.

    The fallback lets this data/loss-only probe run against a frozen G1 stage
    before the unique loss file is merged into that stage's loss package. It
    is selected only when ``KEEP_LOSS_PATH`` is explicitly supplied and the
    normal package import reports that the module is absent.
    """

    try:
        from realesrgan.losses.keep_loss import KeepLoss as loss_class

        return loss_class
    except ModuleNotFoundError as error:
        if error.name != "realesrgan.losses.keep_loss":
            raise
        source = os.environ.get("KEEP_LOSS_PATH")
        if not source:
            raise RuntimeError(
                "realesrgan.losses.keep_loss is unavailable; set KEEP_LOSS_PATH "
                "only for an isolated pre-integration smoke"
            ) from error
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"KEEP_LOSS_PATH does not exist: {path}") from error
        spec = importlib.util.spec_from_file_location("standalone_keep_loss", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load KeepLoss module from {path}") from error
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.KeepLoss


KeepLoss = _load_keep_loss_class()

# Populate dataset/network registries without constructing a model.
import realesrgan.archs  # noqa: F401,E402
import realesrgan.data  # noqa: F401,E402


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for an actual V2 Keep smoke."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", "--opt", required=True, help="G1 R1+bgLF_guard YAML")
    parser.add_argument("--output-dir", required=True, help="JSON report directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="optional exact dataset indices; useful for covering both polarities",
    )
    return parser


def _device(value: str) -> torch.device:
    """Resolve an explicit smoke device without changing the training config."""

    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert a collated tensor/array to float32 on the smoke device."""

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _float(value: Any) -> float:
    """Convert a scalar tensor/number into a JSON-safe float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def _finite(value: Any) -> bool:
    """Return whether a scalar/tensor contains only finite values."""

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all())
    return math.isfinite(float(value))


def _load_batch(
    option_path: Path,
    batch_size: int,
    max_samples: int,
    indices: list[int] | None = None,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Load verified text-mask rows from the exact configured V2 train root.

    Input: Keep YAML and bounded batch/sample sizes. Output: one collated
    dataset batch, dataset object, and provenance. Rows without a verified
    structure asset are skipped because no inferred mask may activate Keep.
    """

    option = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    train_cfg = copy.deepcopy(option.get("datasets", {}).get("train"))
    if not isinstance(train_cfg, dict):
        raise ValueError("training YAML must contain datasets.train")
    expected = "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2"
    sources = train_cfg.get("sources", [])
    if sources != [{"root": expected, "weight": 1.0}]:
        raise ValueError(f"Keep smoke requires exactly one V2 source, got {sources!r}")
    train_cfg.update(
        {
            "phase": "train",
            "batch_size_per_gpu": int(batch_size),
            "num_worker_per_gpu": 0,
        }
    )
    dataset = build_dataset(train_cfg)
    wanted = max(1, int(max_samples))
    samples = []
    examined = 0
    selected_indices: list[int] = []
    if indices:
        candidates = [int(index) for index in indices]
    else:
        candidates = list(range(len(dataset)))
    for index in candidates:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"dataset index out of range: {index}")
        sample = dataset[index]
        examined += 1
        if bool(sample.get("structure_available", False)):
            samples.append(sample)
            selected_indices.append(index)
        if len(samples) >= wanted and not indices:
            break
    if len(samples) < wanted:
        raise RuntimeError(
            f"found {len(samples)} verified-structure rows after {examined} samples; "
            "Keep must not infer masks for unavailable structure"
        )
    batch = dataset.collate_fn(samples)
    provenance = {
        "dataset_type": train_cfg.get("type"),
        "dataset_length": len(dataset),
        "sample_count": len(samples),
        "examined_indices": examined,
        "selected_indices": selected_indices,
        "sources": sources,
        "canvas_hw": train_cfg.get("canvas_hw"),
        "resize_mode": train_cfg.get("resize_mode"),
        "structure_available": [bool(item.get("structure_available", False)) for item in samples],
    }
    return batch, dataset, {"option": option, "provenance": provenance}


def _loss_config(option: dict[str, Any]) -> dict[str, Any]:
    """Extract the Keep module options and reject a missing/incorrect type."""

    train = option.get("train", {})
    keep = copy.deepcopy(train.get("keep_opt") or train.get("keep_loss_opt") or {})
    if keep.pop("type", None) != "KeepLoss":
        raise ValueError("smoke config must select KeepLoss in train.keep_opt")
    return keep


def _run(
    loss: KeepLoss,
    prediction: torch.Tensor,
    gt: torch.Tensor,
    lq: torch.Tensor,
    text: torch.Tensor,
    valid: torch.Tensor,
    availability: Any,
    *,
    return_maps: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate Keep with the production RGB/mask argument contract."""

    return loss(
        prediction,
        gt,
        lq,
        text_mask_rgb=text,
        valid_mask_rgb=valid,
        structure_available=availability,
        return_details=True,
        return_maps=return_maps,
    )


def _erase(prediction: torch.Tensor, mask: torch.Tensor, amount: float) -> torch.Tensor:
    """Move an RGB patch toward mid-gray, weakening local stroke contrast."""

    patch = mask.to(device=prediction.device, dtype=prediction.dtype)
    return (prediction * (1.0 - amount * patch) + 0.5 * amount * patch).clamp(0.0, 1.0)


def _ink_shift(prediction: torch.Tensor, mask: torch.Tensor, polarity: torch.Tensor, amount: float) -> torch.Tensor:
    """Add a controlled ink-polarity background spur for diagnostics only."""

    delta = -polarity.view(-1, 1, 1, 1).to(device=prediction.device, dtype=prediction.dtype)
    patch = mask.to(device=prediction.device, dtype=prediction.dtype)
    return (prediction + delta * float(amount) * patch).clamp(0.0, 1.0)


def _record_details(details: dict[str, torch.Tensor]) -> dict[str, float]:
    """Select scalar diagnostics from the loss detail mapping for JSON."""

    result = {}
    for name, value in details.items():
        if value.ndim == 0:
            result[name] = _float(value)
    return result


def _contract(
    loss: KeepLoss,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run finite/gradient, evidence, perturbation, padding, and polarity checks."""

    gt = _tensor(batch["gt_rgb"], device)
    lq = _tensor(batch["lq_rgb"], device)
    text = _tensor(batch["text_mask_rgb"], device)
    valid = _tensor(batch["valid_mask_rgb"], device)
    availability = batch.get("structure_available", [])
    base_prediction = gt.detach()

    base, base_details = _run(
        loss, base_prediction, gt, lq, text, valid, availability, return_maps=True
    )
    visible = base_details["keep_visible_skeleton"] > 0.5
    selected = base_details["keep_selected"] > 0.5
    valid_binary = base_details["keep_valid"] > 0.5
    if not bool((selected & valid_binary).any()):
        raise RuntimeError("Keep smoke selected no credible LQ-visible support")

    # A contrast measurement uses a local ring, so erase the full ring
    # neighborhood around visible skeleton points for an unambiguous deletion
    # diagnostic. This does not change the production support construction.
    erase_radius = max(1, int(loss.ring_radius))
    visible_neighborhood = (
        torch.nn.functional.max_pool2d(
            visible.float(), kernel_size=2 * erase_radius + 1, stride=1, padding=erase_radius
        )
        > 0.5
    )
    weakened = _erase(base_prediction, visible_neighborhood, 0.50)
    deleted = _erase(base_prediction, visible_neighborhood, 1.00)
    weak_loss, weak_details = _run(loss, weakened, gt, lq, text, valid, availability)
    deleted_loss, deleted_details = _run(loss, deleted, gt, lq, text, valid, availability)

    low_lq = torch.full_like(lq, 0.5)
    low_loss, low_details = _run(loss, base_prediction, gt, low_lq, text, valid, availability)

    # Force a right-edge invalid strip even on stretch canvases, then perturb
    # only that strip. The masked ring implementation must not see it.
    forced_valid = valid.clone()
    invalid_strip = torch.zeros_like(valid)
    cut = max(1, int(valid.shape[-1]) - 8)
    invalid_strip[..., cut:] = 1.0
    forced_valid[..., cut:] = 0.0
    forced_base, _ = _run(loss, base_prediction, gt, lq, text, forced_valid, availability)
    forced_padding, _ = _run(
        loss,
        _erase(base_prediction, invalid_strip, 1.0),
        gt,
        lq,
        text,
        forced_valid,
        availability,
    )

    inverse_loss, inverse_details = _run(
        loss,
        1.0 - base_prediction,
        1.0 - gt,
        1.0 - lq,
        text,
        valid,
        availability,
    )

    grad_prediction = base_prediction.clone().requires_grad_(True)
    grad_loss, _ = _run(loss, grad_prediction, gt, lq, text, forced_valid, availability)
    grad_loss.backward()
    gradient = grad_prediction.grad.detach().abs()
    padding_gradient = gradient[..., cut:]

    polarity = base_details["keep_polarity"]
    background = (valid_binary & ~(text > 0.5)).to(base_prediction.dtype)
    spur_prediction = _ink_shift(base_prediction, background, polarity, 0.20)
    spur_loss, spur_details = _run(loss, spur_prediction, gt, lq, text, valid, availability)

    scalar_base = _record_details(base_details)
    scalar_weak = _record_details(weak_details)
    scalar_deleted = _record_details(deleted_details)
    scalar_low = _record_details(low_details)
    scalar_inverse = _record_details(inverse_details)
    scalar_spur = _record_details(spur_details)
    checks = {
        "finite_base_and_gradient": _finite(base) and _finite(grad_loss) and bool(torch.isfinite(gradient).all()),
        "gradient_nonzero": _float(gradient.sum()) > 1e-8,
        "support_nonempty": _float(base_details["keep_support_count"]) > 0.0,
        "support_inside_valid": bool(
            (base_details["keep_selected"].bool() & ~valid_binary).sum() == 0
        ),
        "loss_increases_when_weakened": _float(weak_loss) > _float(base) + 1e-7,
        "loss_increases_when_deleted": _float(deleted_loss) > _float(weak_loss) + 1e-7,
        "deleted_visible_to_lost": _float(deleted_details["keep_visible_to_lost_rate"]) >= 0.99,
        "low_contrast_lq_is_gated": _float(low_details["keep_support_count"]) == 0.0 and abs(_float(low_loss)) < 1e-10,
        "padding_perturbation_excluded": abs(_float(forced_padding) - _float(forced_base)) < 1e-7,
        "padding_gradient_zero": _float(padding_gradient.sum()) == 0.0,
        "polarity_inversion_stable": abs(_float(inverse_loss) - _float(base)) < 2e-2,
        "spur_metric_finite": _finite(spur_loss) and _finite(spur_details["keep_background_activation_rate"]),
    }
    record = {
        "base_loss": _float(base),
        "weakened_loss": _float(weak_loss),
        "deleted_loss": _float(deleted_loss),
        "low_contrast_lq_loss": _float(low_loss),
        "forced_padding_base_loss": _float(forced_base),
        "forced_padding_perturbed_loss": _float(forced_padding),
        "polarity_inverted_loss": _float(inverse_loss),
        "spur_loss": _float(spur_loss),
        "gradient_l1": _float(gradient.sum()),
        "padding_gradient_l1": _float(padding_gradient.sum()),
        "base": scalar_base,
        "weakened": scalar_weak,
        "deleted": scalar_deleted,
        "low_contrast_lq": scalar_low,
        "polarity_inverted": scalar_inverse,
        "spur": scalar_spur,
        "checks": checks,
    }
    tensors = {
        "gt": gt[0].detach(),
        "lq": lq[0].detach(),
        "valid": valid_binary[0].detach(),
        "text": (text > 0.5).float()[0].detach(),
        "selected": selected.float()[0].detach(),
        "visible": visible.float()[0].detach(),
    }
    return record, tensors


def main() -> int:
    """Load the exact V2 batch, run the contract, and write a JSON report."""

    args = _parser().parse_args()
    option_path = Path(args.opt).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    option = yaml.safe_load(option_path.read_text(encoding="utf-8"))
    keep_cfg = _loss_config(option)
    device = _device(args.device)
    batch, dataset, metadata = _load_batch(option_path, args.batch_size, args.max_samples, args.indices)
    loss = KeepLoss(**keep_cfg).to(device)
    record, _ = _contract(loss, batch, device)
    payload = {
        "status": "pass" if all(record["checks"].values()) else "fail",
        "script": "smoke_keep_loss.py",
        "config": str(option_path),
        "experiment_name": option.get("name"),
        "device": str(device),
        "keep_opt": keep_cfg,
        "dataset": metadata["provenance"],
        "dataset_runtime_length": len(dataset),
        "tests": record,
        "contract": {
            "support": "GT text skeleton/tube AND positive GT/LQ signed local-ring contrast",
            "target": "detached LQ signed contrast; one-sided temperature*softplus((c_lq-c_sr)/temperature)",
            "padding": "valid RGB mask is applied to local sums/counts and prediction support",
            "model_interface": "loss-side RGB call only; no generator input or inferred mask branch",
        },
    }
    json_path = output_dir / "keep_smoke.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(json_path)}, indent=2))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
