"""Exercise Torch P17 over uniformly sampled OCR-line inputs."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from realesrgan.lenovo_p17_preprocess import preprocess_lq_rgb

import realesrgan.data  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    option = yaml.safe_load(Path(args.opt).read_text(encoding="utf-8"))
    dataset_opt = copy.deepcopy(option["datasets"]["train"])
    dataset_opt.update({"phase": "train", "batch_size_per_gpu": 1, "num_worker_per_gpu": 0})
    dataset = build_dataset(dataset_opt)
    indices = np.linspace(0, len(dataset) - 1, min(args.count, len(dataset)), dtype=int).tolist()
    for index in indices:
        sample = dataset[index]
        output = preprocess_lq_rgb(sample["lq_rgb"].unsqueeze(0).cuda(), option["lq_preprocess"])
        if not torch.isfinite(output).all():
            raise RuntimeError(f"P17 produced non-finite output at OCR-line index {index}")
    result = {"status": "pass", "dataset_length": len(dataset), "checked_indices": indices}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
