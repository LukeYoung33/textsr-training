"""Export paired P17 training inputs for visual data-contract inspection."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from realesrgan.data.basic_text_line_remote58_dataset import BasicTextLineRemote58Dataset
from realesrgan.lenovo_p17_preprocess import preprocess_lq_rgb


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {path}")


def _save_mask(path: Path, tensor: torch.Tensor) -> None:
    image = (tensor.detach().cpu().squeeze(0).numpy() > 0.5).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), image):
        raise IOError(f"failed to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="manifests/train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = BasicTextLineRemote58Dataset({
        "root": args.root,
        "manifest": args.manifest,
        "split": "train",
        "canvas_hw": [96, 480],
        "resize_mode": "height_crop_pad",
        "ocr_line_crop": True,
        "text_line_crop": False,
        "text_line_gap": 6,
        "text_line_vertical_context": 0.12,
        "text_line_horizontal_context": 0.03,
        "text_line_boundary_jitter": 0.05,
        "train_homography_prob": 0.35,
        "train_homography_jitter": 0.015,
    })
    p17_config = {"enabled": True, "implementation": "torch_gpu", "lambda": 0.03, "alpha": 1.0,
                  "p17_patch_size": 96, "p17_patch_stride": 32, "p17_stats_gaussian_sigma": 0.6,
                  "p17_min_class_fraction": 0.12, "p16_write_full_radius": 16, "p16_write_zero_radius": 24}
    rows = ["# P17 model-input inspection", "", "| index | sample_id | structure_available | files |", "|---:|---|---|---|"]
    for index in np.linspace(0, len(dataset) - 1, num=min(args.count, len(dataset)), dtype=int):
        sample = dataset[int(index)]
        raw = sample["lq_rgb"]
        processed = preprocess_lq_rgb(raw.unsqueeze(0).cuda(), p17_config).squeeze(0)
        prefix = f"{index:05d}"
        _save_rgb(output / f"{prefix}_lq_before_p17.png", raw)
        _save_rgb(output / f"{prefix}_lq_after_p17.png", processed)
        _save_rgb(output / f"{prefix}_gt.png", sample["gt_rgb"])
        _save_mask(output / f"{prefix}_text_mask.png", sample["text_mask_rgb"])
        _save_mask(output / f"{prefix}_valid_mask.png", sample["valid_mask_rgb"])
        rows.append(f"| {index} | {sample['line_id']} | {sample['structure_available']} | `*_lq_before_p17.png`, `*_lq_after_p17.png`, `*_gt.png`, `*_text_mask.png`, `*_valid_mask.png` |")
    (output / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
