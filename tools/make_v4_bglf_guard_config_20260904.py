"""Create the isolated R1 V2-only strict BG-LF-guard training config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "options" / "train_bucket_3" / "train_v4_r1_recall_bglf_v2only_250k_20260903_lambda3.yml"
DEFAULT_OUTPUT = ROOT / "options" / "train_bucket_3" / "train_v4_r1_recall_bglf_guard_v2only_250k_20260904.yml"
DEFAULT_NAME = "v4_r1_recall_bglf_guard_v2only_250k_20260904"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping YAML: {path}")
    return value


def make_config(
    weight: float = 1.0,
    *,
    name: str = DEFAULT_NAME,
    port: int = 29704,
    total_iter: int = 250000,
    print_freq: int | None = None,
    save_checkpoint_freq: int | None = None,
    val_freq: int | None = None,
) -> dict:
    """Clone the completed R1 contract and replace only the BG-LF formulation."""
    config = copy.deepcopy(_load(BASE))
    train_cfg = config["train"]
    train = config["datasets"]["train"]
    val = config["datasets"]["val"]
    roots = train.get("sources")
    if roots != [{"root": "/home/guang.yang/workspace/MARCONetPlusPlus-main/datasets/real_finetune_v2", "weight": 1.0}]:
        raise ValueError("R1 guard must use exactly the V2-only train source")
    if val.get("sources") != roots:
        raise ValueError("R1 guard train/validation sources must match exactly")

    config["name"] = name
    train["name"] = f"{name}_train"
    val["name"] = f"{name}_val"
    train_cfg["bg_lf_weight"] = float(weight)
    train_cfg["bg_lf_opt"] = {
        "type": "BackgroundLowFrequencyGuardLoss",
        "radius": 7,
        "sigma": 4.0,
        "channel": "luma",
        "threshold": 0.5,
        "eps": 1.0e-6,
    }
    train_cfg["total_iter"] = int(total_iter)
    if print_freq is not None:
        config["logger"]["print_freq"] = int(print_freq)
    if save_checkpoint_freq is not None:
        config["logger"]["save_checkpoint_freq"] = int(save_checkpoint_freq)
    if val_freq is not None:
        config["val"]["val_freq"] = int(val_freq)
    config["dist_params"]["port"] = int(port)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--port", type=int, default=29704)
    parser.add_argument("--total-iter", type=int, default=250000)
    parser.add_argument("--print-freq", type=int)
    parser.add_argument("--save-checkpoint-freq", type=int)
    parser.add_argument("--val-freq", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    config = make_config(
        args.weight,
        name=args.name,
        port=args.port,
        total_iter=args.total_iter,
        print_freq=args.print_freq,
        save_checkpoint_freq=args.save_checkpoint_freq,
        val_freq=args.val_freq,
    )
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
