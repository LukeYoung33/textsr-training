"""Create isolated full-frame ablations from the frozen September 28 options."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "options/train_bucket_3/lenovo_borderrail_8arm_20260928/train_v5_borderrail_h96_250k_20260928.yml"
ARMS = {
    "A": {"text_weight": 0.0},
    "B": {"text_weight": 1.0},
    "C1": {"text_weight": 1.0, "missing_weight": 1.0, "false_weight": .25},
    "C2": {"text_weight": 1.0, "sdf_weight": .1},
    "D": {"text_weight": 1.0, "cp_weight": .0005},
}


def digest(path):
    """@luna-comment
    Input: existing file path. Output: SHA256 hex provenance string.
    Key Steps: stream file bytes into SHA256. Key Params: fixed 1 MiB chunks.
    State/Dependencies: reads only the requested file; no model deserialization.
    """
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_manifest(path, root):
    """@luna-comment
    Input: full-frame JSON/JSONL manifest and its explicit image root.
    Output: records and scene/capture identities after path/duplicate validation.
    Key Steps: parse, require aligned-pair paths, reject duplicate records, and
        derive leakage identities using capture_group, source_id, or LQ path.
    Key Params: root resolves relative asset paths; mask fields are optional.
    State/Dependencies: read-only filesystem inspection; no dataset materialization.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(row) for row in text.splitlines() if row.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest must contain a non-empty list: {path}")
    identities, pairs = set(), set()
    for record in records:
        if not isinstance(record, dict) or not {"lq", "gt"} <= record.keys():
            raise ValueError("Each full-frame record needs lq and gt")
        resolved = {}
        for key in ("lq", "gt", "text_mask", "valid_mask"):
            if not record.get(key):
                continue
            target = Path(record[key])
            target = target if target.is_absolute() else Path(root) / target
            if not target.is_file():
                raise FileNotFoundError(f"{key} asset does not exist: {target}")
            resolved[key] = str(target.resolve())
        pair = (resolved["lq"], resolved["gt"])
        if pair in pairs:
            raise ValueError(f"Duplicate source pair in {path}: {pair}")
        pairs.add(pair)
        # Source metadata never substitutes for physical asset identity.
        identities.update("asset:" + str(Path(resolved[key])).casefold() for key in ("lq", "gt"))
        for key in ("capture_group", "source_id"):
            if record.get(key) not in (None, ""):
                identities.add(key + ":" + str(record[key]))
    return records, identities


def make_configs(*, train_manifest, val_manifest, root, output, experiments_root,
                 baseline=DEFAULT_BASELINE, checkpoint=None, canvas_hw=(256, 256),
                 halo=32, num_gpu=1, total_iter=None, arms=("A", "B", "C1", "C2"),
                 cp_opt=None):
    """@luna-comment
    Input: immutable baseline YAML, full-frame train/val assets, and new run paths.
    Output: new arm YAML paths plus a provenance JSON; legacy files stay unchanged.
    Key Steps: validate source split and geometry, copy backbone/optimizer/scheduler,
        replace only new pipeline settings, and record all input hashes.
    Key Params: arms isolate supervision changes; D requires explicit local CP options.
    State/Dependencies: writes only the requested new output directory; no training.
    """
    baseline, root, output = Path(baseline).resolve(), Path(root).resolve(), Path(output).resolve()
    train_manifest, val_manifest = Path(train_manifest).resolve(), Path(val_manifest).resolve()
    if total_iter is not None and (isinstance(total_iter, bool) or int(total_iter) != total_iter or total_iter <= 0):
        raise ValueError("total_iter must be a positive integer")
    if num_gpu not in (0, 1):
        raise ValueError("num_gpu must be zero for CPU or one GPU per process")
    if any(int(size) <= 0 or int(size) % 8 for size in canvas_hw):
        raise ValueError("canvas dimensions must be positive multiples of eight")
    if halo < 0 or halo % 8 or any(2 * halo >= size for size in canvas_hw):
        raise ValueError("halo must be a non-negative multiple of eight leaving a positive center")
    train_records, train_ids = read_manifest(train_manifest, root)
    val_records, val_ids = read_manifest(val_manifest, root)
    if train_ids & val_ids:
        raise ValueError(f"Train/validation source leakage: {sorted(train_ids & val_ids)}")
    source = yaml.safe_load(baseline.read_text(encoding="utf-8-sig"))
    if source.get("network_g", {}).get("type") != "BasicTextLineR4StudentY4UVV5":
        raise ValueError("Baseline must use the September 28 V5 backbone")
    if any(arm not in ARMS for arm in arms):
        raise ValueError(f"Unknown arm; choose from {list(ARMS)}")
    if "D" in arms and not cp_opt:
        raise ValueError("Arm D requires a local cp_opt configuration")
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if output == baseline.parent or baseline in [output / f"train_{arm}.yml" for arm in arms]:
        raise ValueError("New configurations must use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    if len(set(arms)) != len(arms) or any((output / f"train_{arm}.yml").exists() for arm in arms):
        raise FileExistsError("Duplicate arm or existing output; choose a fresh configuration directory")
    outputs = []
    for arm in arms:
        configuration = {
            "name": f"fullframe_text_priority_{arm}_20261014",
            "model_type": "FullFrameTextPriorityModel", "scale": 1,
            "num_gpu": int(num_gpu), "manual_seed": source.get("manual_seed", 20260928),
            "network_g": copy.deepcopy(source["network_g"]),
            "path": {"pretrain_network_g": str(Path(checkpoint).resolve()) if checkpoint else None,
                     "strict_load_g": True, "resume_state": None,
                     "experiments_root": str(Path(experiments_root).resolve())},
            "train": {
                "ema_decay": 0, "optim_g": copy.deepcopy(source["train"]["optim_g"]),
                "scheduler": copy.deepcopy(source["train"]["scheduler"]),
                "total_iter": int(total_iter or source["train"]["total_iter"]),
                "text_start_iter": 0, "text_ramp_iter": 0,
                "fullframe_loss_opt": {
                    "image_channel_weight": [1., 1., 1., 1., .25, .25],
                    "text_weight": 0., "pixel_weight": .1,
                    "missing_weight": 0., "false_weight": 0., "sdf_weight": 0.,
                    "cp_weight": 0., "bg_weight": 0., "uv_weight": 0.,
                    "area_floor": 4., "area_mode": "soft", "ring_radius": 3,
                    "sdf_radius": 3, "roi_weight_min": .25, "roi_weight_max": 4.,
                    **ARMS[arm],
                },
            },
            "datasets": {},
            "logger": {"print_freq": 100, "save_checkpoint_freq": 25000, "use_tb_logger": False},
            "val": {"val_freq": 25000, "save_img": False, "metrics": {}},
            "dist_params": {"backend": "nccl", "port": 29500},
            "inference": {"canvas_hw": list(canvas_hw), "halo": halo, "batch_size": 1},
            "pipeline_provenance": {"baseline": str(baseline), "baseline_sha256": digest(baseline),
                                    "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
                                    "initialization": "fixed_checkpoint" if checkpoint else "random_explicit"},
        }
        configuration["network_g"]["canvas_hw"] = list(canvas_hw)
        if arm == "D":
            configuration["train"]["fullframe_loss_opt"]["cp_opt"] = copy.deepcopy(cp_opt)
        for split, manifest in (("train", train_manifest), ("val", val_manifest)):
            configuration["datasets"][split] = {
                "name": f"fullframe_{split}", "type": "FullFrameTextPriorityDataset",
                "root": str(root), "manifest": str(manifest), "split": split,
                "canvas_hw": list(canvas_hw), "halo": halo,
                "seed": configuration["manual_seed"], "roi_context": 8,
                "batch_size_per_gpu": 4 if split == "train" else 1,
                "num_worker_per_gpu": 4 if split == "train" else 0,
                "dataset_enlarge_ratio": 1,
            }
            if split == "train":
                configuration["datasets"][split]["sampling"] = {
                    "ratios": {"text": .6, "hard": .2, "other": .2}, "source_balance": True}
        target = output / f"train_{arm}.yml"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing run configuration: {target}")
        target.write_text(yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True), encoding="utf-8")
        outputs.append(str(target))
    report = {"baseline": str(baseline), "baseline_sha256": digest(baseline),
              "train_manifest_sha256": digest(train_manifest), "val_manifest_sha256": digest(val_manifest),
              "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
              "train_records": len(train_records), "val_records": len(val_records), "configs": outputs,
              "note": "Candidate coefficients and budgets require calibration; no quality claim."}
    (output / "provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    """@luna-comment
    Input: CLI paths, geometry and selected ablation arms. Output: configs/provenance.
    Key Steps: parse explicit source paths and delegate validation/generation.
    Key Params: checkpoint optional for explicit random pilots; CP config needed for D.
    State/Dependencies: filesystem only, no remote access or model training.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--canvas-hw", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--num-gpu", type=int, default=1)
    parser.add_argument("--total-iter", type=int)
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C1", "C2"])
    parser.add_argument("--cp-config", help="YAML mapping containing an explicit local CP loss specification")
    args = parser.parse_args(argv)
    settings = vars(args)
    cp_path = settings.pop("cp_config")
    settings["cp_opt"] = yaml.safe_load(Path(cp_path).read_text(encoding="utf-8-sig")) if cp_path else None
    print(json.dumps(make_configs(**settings), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
