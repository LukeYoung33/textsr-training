"""Generate deterministic visual previews for the eight Lenovo degradation categories.

The tool is for visual calibration only.  It keeps every category isolated so
that a reviewer can decide whether its appearance resembles the target test
set before combinations are added to the training pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realesrgan.lenovo_atomic import *
from realesrgan.lenovo_atomic import _rgb_to_bgr

def _source_images(input_root: Path) -> list[Path]:
    return [path for path in sorted(input_root.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def _contact_sheet(source: np.ndarray, variants: list[tuple[str, np.ndarray]], output: Path) -> None:
    thumb_w, thumb_h = 420, 300
    columns = 3
    rows = int(np.ceil((len(variants) + 1) / columns))
    sheet = Image.new("RGB", (thumb_w * columns, (thumb_h + 34) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    entries = [("BL original", source)] + variants
    for index, (label, bgr) in enumerate(entries):
        image = Image.fromarray(_rgb_to_bgr(bgr))
        image.thumbnail((thumb_w, thumb_h))
        x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + 34)
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 30 + (thumb_h - image.height) // 2))
        draw.text((x + 8, y + 8), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def run(input_root: Path, output_root: Path, seed: int) -> Path:
    input_root, output_root = input_root.resolve(), output_root.resolve()
    records = []
    for source_path in _source_images(input_root):
        image = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {source_path}")
        variants = []
        item = {"source": source_path.name, "size_wh": [image.shape[1], image.shape[0]], "categories": {}}
        for category_id, (label, function) in CATEGORIES.items():
            item_seed = int.from_bytes(hashlib.sha256(f"{seed}|{source_path.name}|{category_id}".encode()).digest()[:8], "little")
            result, params = function(image, np.random.default_rng(item_seed))
            destination = output_root / category_id / f"{source_path.stem}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError(f"Could not save {destination}")
            encoded.tofile(str(destination))
            item["categories"][category_id] = {"label": label, "seed": item_seed, "output": str(destination), "parameters": params}
            variants.append((category_id, result))
        _contact_sheet(image, variants, output_root / "contact_sheets" / f"{source_path.stem}.jpg")
        records.append(item)
    manifest = {"seed": seed, "input": str(input_root), "category_order": list(CATEGORIES), "records": records}
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/generated_images"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/generated_images/lenovo_eight_category_preview_20260910"))
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    print(run(args.input, args.output_root, args.seed))


if __name__ == "__main__":
    main()
