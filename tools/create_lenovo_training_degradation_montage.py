"""Create a deterministic per-category montage from one clean RGB image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realesrgan.lenovo_degradation import DEFAULT_LENOVO_DEGRADATION_CONFIG, LenovoDegradationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/generated_images/lenovo_training_montage.png"))
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    with Image.open(args.input) as source:
        image = source.convert("RGB")
    tensor = torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    names = ("c1", "c2", "c3", "c4", "c5", "c7", "c8")
    panels = [image]
    for index, name in enumerate(names):
        output, _ = LenovoDegradationPipeline().torch_atomic(tensor, "defocus" if name == "c1" else name, torch.Generator().manual_seed(args.seed + index))
        panels.append(Image.fromarray(output[0].permute(1, 2, 0).mul(255).round().byte().numpy()))
    width, height = image.size
    montage = Image.new("RGB", (width * 4, (height + 20) * 2), "white")
    labels = ("clean",) + names
    draw = ImageDraw.Draw(montage)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (index % 4) * width, (index // 4) * (height + 20)
        montage.paste(panel, (x, y + 20)); draw.text((x + 2, y + 2), label, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
