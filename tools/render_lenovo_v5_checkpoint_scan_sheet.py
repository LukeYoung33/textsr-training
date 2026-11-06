"""Render a fixed-scale contact sheet for already saved checkpoint scan RGB arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROIS = {"A": (171, 14, 177, 39), "B": (197, 55, 201, 80)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    scale, label_h = 2, 22
    arrays = sorted((path for path in args.directory.glob("*_net_g_*.npy")
                     if path.name.startswith(("c0_", "r1_"))),
                    key=lambda p: (p.name[0], int(p.stem.split("_")[-1])))
    if len(arrays) != 20:
        raise ValueError(f"expected 20 clipped RGB arrays, found {len(arrays)}")
    first = np.load(arrays[0]); height, width = first.shape[:2]
    sheet = Image.new("RGB", (10 * width * scale, 2 * (height * scale + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    manifest = {"domain": "clipped float RGB arrays only; no raw packed Y4UV was saved by the scan",
                "display": "nearest-neighbor, no per-panel normalization",
                "metric_field_note": "Existing report.json mean_darkness is a deprecated alias for mean_luminance; both have the same numeric value.",
                "arrays": []}
    for index, path in enumerate(arrays):
        array = np.load(path)
        arm, iteration = path.stem.split("_")[0], int(path.stem.split("_")[-1])
        row, col = (0 if arm == "c0" else 1), index % 10
        image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).resize(
            (width * scale, height * scale), Image.Resampling.NEAREST)
        x, y = col * width * scale, row * (height * scale + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 2, y + 2), f"{arm.upper()} {iteration // 1000}k", fill="black")
        for name, (x0, y0, x1, y1) in ROIS.items():
            draw.rectangle((x + x0 * scale, y + label_h + y0 * scale,
                            x + x1 * scale, y + label_h + y1 * scale), outline="red", width=1)
            if col == 0:
                draw.text((x + x1 * scale + 2, y + label_h + y0 * scale), name, fill="red")
        manifest["arrays"].append({"arm": arm, "iteration": iteration, "path": path.name,
                                   "sha256": digest(path), "shape": list(array.shape), "dtype": str(array.dtype)})
    output = args.directory / "c0_r1_checkpoint_threepx_contact_sheet.png"
    sheet.save(output)
    # A compact original-image crop keeps the approximately three-pixel pattern
    # legible in ordinary report viewers.  The values remain fixed clipped RGB.
    crop = (145, 0, 230, 96)
    panel_w, panel_h = (crop[2] - crop[0]) * 3, (crop[3] - crop[1]) * 3
    detail = Image.new("RGB", (5 * panel_w, 4 * (panel_h + label_h)), "white")
    detail_draw = ImageDraw.Draw(detail)
    for index, path in enumerate(arrays):
        array = np.load(path)
        arm, iteration = path.stem.split("_")[0], int(path.stem.split("_")[-1])
        arm_index = index % 10
        row = (0 if arm == "c0" else 2) + arm_index // 5
        col = arm_index % 5
        image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).crop(crop).resize(
            (panel_w, panel_h), Image.Resampling.NEAREST)
        x, y = col * panel_w, row * (panel_h + label_h)
        detail.paste(image, (x, y + label_h))
        detail_draw.text((x + 2, y + 2), f"{arm.upper()} {iteration // 1000}k", fill="black")
    detail_output = args.directory / "c0_r1_checkpoint_threepx_glyph_crop_sheet.png"
    detail.save(detail_output)
    manifest["contact_sheet"] = output.name
    manifest["glyph_crop_contact_sheet"] = {"path": detail_output.name, "crop_xyxy": crop,
                                              "scale": 3, "roi_overlays": False}
    (args.directory / "scan_artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "sheet": str(output), "arrays": len(arrays)}))


if __name__ == "__main__":
    main()
