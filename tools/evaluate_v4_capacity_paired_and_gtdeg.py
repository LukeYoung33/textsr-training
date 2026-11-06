"""Evaluate V4 capacity checkpoints on real paired and canonical GT-degradation validation.

The synthetic arm deliberately calls ``RealESRNetModel.feed_data`` through a
minimal runner.  Thus its two-stage blur/resize/noise/JPEG/sinc process is the
same implementation and the same option distribution used during training.
For each frozen paired-val GT sample, several deterministic random draws are
made and then bucketed by observed detail retention; a bucket is a stratum of
the training distribution, not a second hand-authored degradation recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from basicsr.data import build_dataset
from basicsr.utils import DiffJPEG

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.models.realesrnet_model import RealESRNetModel
from realesrgan.utils import YUV2RGB
from tools.basic_v2_eval_network import build_eval_network, resolve_canvas_hw
from tools.dump_basic_v2_validation_rgb_pairs import _label, _safe_component, _write_rgb


def _parse_model(value: str) -> tuple[str, Path, Path]:
    try:
        name, config, checkpoint = value.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model must be NAME=CONFIG=CHECKPOINT") from exc
    return name, Path(config), Path(checkpoint)


@contextmanager
def _seeded(seed: int):
    """Make one degradation draw repeatable without leaking RNG state."""
    py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


class _CanonicalDegrader:
    """Small adapter exposing exactly the RealESRNet GPU degradation path."""

    def __init__(self, opt: dict[str, Any], device: torch.device):
        self.opt = opt
        self.device = device
        self.is_train = True
        self.jpeger = DiffJPEG(differentiable=False).to(device)
        self.finetune_deg = -1.0
        self.queue_size = 0
        self.color = "rgb"

    def _dequeue_and_enqueue(self) -> None:
        return None

    feed_data = RealESRNetModel.feed_data


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    return np.clip(value.permute(1, 2, 0).numpy(), 0.0, 1.0)


def _prediction_rgb(network: torch.nn.Module, lq_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    packed = rgb_to_y4uv(torch.from_numpy(np.ascontiguousarray(lq_rgb)).permute(2, 0, 1).unsqueeze(0)).to(device)
    yuv = y4uv_to_yuv444(network(packed))
    return _rgb(YUV2RGB(yuv, format="RGB", is_train=False, uv_normalized=True))


def _detail(value: np.ndarray) -> float:
    luma = cv2.cvtColor(value.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return float(np.mean(np.abs(cv2.Laplacian(luma, cv2.CV_32F))))


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
    mse = float(np.mean(np.square(diff)))
    return {"mae_rgb": float(np.mean(np.abs(diff))), "psnr_rgb": float(99.0 if mse <= 1e-12 else -10.0 * np.log10(mse))}


def _load_paired_dataset(config: Path):
    option = yaml.safe_load(config.read_text(encoding="utf-8"))
    return build_dataset(dict(option["datasets"]["val"])), option


def _load_gtdeg_dataset(config: Path, seed: int):
    option = yaml.safe_load(config.read_text(encoding="utf-8"))
    val = dict(option["datasets"]["val"])
    val.update({"type": "BasicTextLineGTDegradeDataset", "kernel_seed": int(seed)})
    return build_dataset(val), option


def _write_record(root: Path, record: dict[str, Any], lq: np.ndarray, gt: np.ndarray, outputs: dict[str, np.ndarray]) -> None:
    filename = record["filename"]
    _write_rgb(root / "lq" / filename, lq)
    _write_rgb(root / "gt" / filename, gt)
    panels = [_label(lq, "LQ"), _label(gt, "GT")]
    for name, output in outputs.items():
        _write_rgb(root / "outputs" / _safe_component(name, "model") / filename, output)
        panels.append(_label(output, name))
    _write_rgb(root / "montages" / filename, np.concatenate(panels, axis=1))


def _summarize(records: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(records), "models": {}}
    buckets = sorted(set(str(item["bucket"]) for item in records))
    for name in names:
        all_rows = [item["metrics"][name] for item in records]
        model: dict[str, Any] = {
            "all": {key: float(np.mean([row[key] for row in all_rows])) for key in ("mae_rgb", "psnr_rgb")}
        }
        for bucket in buckets:
            rows = [item["metrics"][name] for item in records if item["bucket"] == bucket]
            model[bucket] = {"n": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in ("mae_rgb", "psnr_rgb")}}
        result["models"][name] = model
    return result


def _uniform_subset_indices(dataset, count: int, seed: int) -> tuple[list[int], dict[str, int]]:
    """Uniformly sample manifest ``subset`` scenes, preserving a frozen seed."""
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(dataset._entries):
        subset = str(entry["record"].get("subset", "unknown"))
        groups.setdefault(subset, []).append(index)
    if not groups:
        raise RuntimeError("validation dataset has no entries")
    if count > len(dataset):
        raise ValueError(f"requested {count} samples from only {len(dataset)} validation rows")
    names = sorted(groups)
    target = {name: count // len(names) for name in names}
    for name in names[:count % len(names)]:
        target[name] += 1
    rng = random.Random(seed)
    chosen: list[int] = []
    remaining = 0
    for name in names:
        take = min(target[name], len(groups[name]))
        chosen.extend(rng.sample(groups[name], take))
        remaining += target[name] - take
    if remaining:
        unused = [index for name in names for index in groups[name] if index not in chosen]
        chosen.extend(rng.sample(unused, remaining))
    rng.shuffle(chosen)
    counts = {name: sum(index in set(chosen) for index in values) for name, values in groups.items()}
    return chosen, counts


def _run_paired(dataset, networks, root: Path, indices: list[int], device: torch.device) -> list[dict[str, Any]]:
    records = []
    for order, index in enumerate(indices):
        item = dataset[index]
        # The paired dataset preserves the canonical RGB tensors beside the
        # legacy NV21/YUV tensors.  Use them directly: treating NV21 as Y4UV
        # would silently make a colour-invalid paired comparison.
        lq, gt = _rgb(item["lq_rgb"]), _rgb(item["gt_rgb"])
        outputs = {name: _prediction_rgb(network, lq, device) for name, network in networks.items()}
        stem = _safe_component(Path(str(item["line_id"])).stem, f"sample_{index:04d}")
        subset = str(dataset._entries[index]["record"].get("subset", "unknown"))
        record = {"index": index, "order": order, "subset": subset, "source": str(item["source_name"]), "line_id": str(item["line_id"]), "bucket": "paired_real", "filename": f"{order:04d}_{stem}.png", "metrics": {name: _metrics(output, gt) for name, output in outputs.items()}}
        _write_record(root, record, lq, gt, outputs)
        records.append(record)
        if (order + 1) % 25 == 0 or order + 1 == len(indices):
            print(f"[paired] {order + 1}/{len(indices)}", flush=True)
    return records


def _run_synthetic(dataset, option: dict[str, Any], networks, root: Path, indices: list[int], draws: int, seed: int, device: torch.device) -> list[dict[str, Any]]:
    degrader = _CanonicalDegrader(option, device)
    records = []
    for order, index in enumerate(indices):
        for draw in range(draws):
            draw_seed = int(seed + index * 1009 + draw * 97)
            with _seeded(draw_seed):
                item = dataset[index]
                batch = {key: (value.unsqueeze(0) if torch.is_tensor(value) and value.ndim in (2, 3) else value) for key, value in item.items()}
                RealESRNetModel.feed_data(degrader, batch)
                lq, gt = _rgb(degrader.lq), _rgb(degrader.gt)
            outputs = {name: _prediction_rgb(network, lq, device) for name, network in networks.items()}
            stem = _safe_component(Path(str(item["line_id"])).stem, f"sample_{index:04d}")
            subset = str(dataset._entries[index]["record"].get("subset", "unknown"))
            records.append({"index": index, "order": order, "draw": draw, "seed": draw_seed, "subset": subset, "source": str(item["source_name"]), "line_id": str(item["line_id"]), "bucket": "unassigned", "detail_retention": float(_detail(lq) / max(_detail(gt), 1e-8)), "filename": f"{order:04d}_d{draw}_{stem}.png", "_lq": lq, "_gt": gt, "_outputs": outputs, "metrics": {name: _metrics(output, gt) for name, output in outputs.items()}})
        if (order + 1) % 25 == 0 or order + 1 == len(indices):
            print(f"[synthetic-generate] {order + 1}/{len(indices)}", flush=True)
    values = np.asarray([record["detail_retention"] for record in records], dtype=np.float32)
    low, high = np.quantile(values, (1 / 3, 2 / 3))
    for record in records:
        value = record["detail_retention"]
        record["bucket"] = "heavy" if value <= low else "medium" if value <= high else "light"
        _write_record(root, record, record.pop("_lq"), record.pop("_gt"), record.pop("_outputs"))
    return records


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", type=Path, required=True, help="V4-base GT-degradation YAML; supplies data and degradation contract")
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--synthetic-draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    device = torch.device(args.device)
    paired, option = _load_paired_dataset(args.reference_config)
    synthetic, synthetic_option = _load_gtdeg_dataset(args.reference_config, args.seed)
    count = min(int(args.samples), len(paired), len(synthetic))
    if count <= 0:
        raise RuntimeError("validation set is empty")
    canvas = resolve_canvas_hw(args.reference_config)
    networks: dict[str, torch.nn.Module] = {}
    models = []
    for name, config, checkpoint in args.model:
        if resolve_canvas_hw(config) != canvas:
            raise ValueError(f"{name}: canvas differs from reference config")
        network, digest = build_eval_network(checkpoint, device, config)
        networks[name] = network
        models.append({"name": name, "config": str(config), "config_sha256": _digest(config), "checkpoint": str(checkpoint), "checkpoint_sha256": digest})
    args.output_root.mkdir(parents=True, exist_ok=True)
    indices, subset_counts = _uniform_subset_indices(paired, count, int(args.seed))
    with torch.inference_mode():
        paired_records = _run_paired(paired, networks, args.output_root / "paired_real", indices, device)
        synthetic_records = _run_synthetic(synthetic, synthetic_option, networks, args.output_root / "synthetic_train_distribution", indices, int(args.synthetic_draws), int(args.seed), device)
    report = {"status": "COMPLETE", "reference_config": str(args.reference_config), "reference_config_sha256": _digest(args.reference_config), "models": models, "samples_per_arm": count, "sampling": {"method": "uniform random by manifest subset", "seed": int(args.seed), "selected_dataset_indices": indices, "subset_counts": subset_counts}, "synthetic_draws_per_gt": int(args.synthetic_draws), "synthetic_contract": "exact configured RealESRNet two-stage training degradation; light/medium/heavy are post-hoc tertiles of observed luma detail retention", "paired_real": {"summary": _summarize(paired_records, list(networks)), "records": paired_records}, "synthetic_train_distribution": {"summary": _summarize(synthetic_records, list(networks)), "records": synthetic_records}}
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "paired": report["paired_real"]["summary"], "synthetic": report["synthetic_train_distribution"]["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
