"""One-step production-path preflight for BASIC text-line V3 configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
import realesrgan.losses  # noqa: F401
import realesrgan.models  # noqa: F401
from basicsr.data import build_dataset
from basicsr.models import build_model
from basicsr.utils.options import parse_options


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--iter", type=int, default=1)
    args = parser.parse_args(argv)
    root = ROOT
    sys.argv = [sys.argv[0], "-opt", str(args.opt), "--launcher", "none"]
    opt, _ = parse_options(str(root), is_train=True)
    opt["root_path"] = str(root)
    dataset = build_dataset(opt["datasets"]["train"])
    sample = dataset.collate_fn([dataset[0]]) if hasattr(dataset, "collate_fn") else [dataset[0]]
    model = build_model(opt)
    model.feed_data(sample)
    model.optimize_parameters(args.iter)
    logs = model.get_current_log()
    result = {
        "config": str(args.opt),
        "name": opt["name"],
        "iter": int(args.iter),
        "dataset_length": len(dataset),
        "device": str(model.device),
        "input_shape": list(model.lq.shape),
        "output_shape": list(model.output.shape),
        "logs": {key: float(value) for key, value in logs.items()},
        "finite": all(torch.isfinite(value).item() for value in model.output.detach().flatten()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["finite"] or any(not torch.isfinite(torch.tensor(value)) for value in result["logs"].values()):
        raise FloatingPointError("non-finite output or loss in BASIC V3 preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
