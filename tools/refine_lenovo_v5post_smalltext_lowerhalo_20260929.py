"""Keep V5 background cleanup while softening its effect on small text strokes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def refine(normal: np.ndarray, base: np.ndarray, lines: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if normal.shape != base.shape:
        raise ValueError("normal and V5 output differ in shape")
    height, width = normal.shape[:2]
    normal_y = cv2.cvtColor(normal, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    attenuation = np.zeros((height, width), np.float32)
    for line in lines:
        x0, y0, x1, y1 = map(int, line["crop_xyxy"])
        crop_height = y1 - y0
        strength = float(np.clip(0.5 + 0.5 * (crop_height - 130.0) / 60.0, 0.5, 1.0))
        if strength >= 1.0:
            continue
        luminance = normal_y[y0:y1, x0:x1]
        # The high luminance quantile estimates the local paper/screen background.
        # Restrict attenuation to ink; keep full V5 correction on pale lower halos.
        background = float(np.percentile(luminance, 80))
        ink = smoothstep((background - 17.0 - luminance) / 38.0)
        ink = cv2.GaussianBlur(ink, (0, 0), 1.1)
        yy, xx = np.ogrid[:crop_height, :x1 - x0]
        edge = np.minimum(np.minimum(xx, x1 - x0 - 1 - xx),
                          np.minimum(yy, crop_height - 1 - yy)).astype(np.float32)
        feather = smoothstep((edge - 4.0) / 20.0)
        region = attenuation[y0:y1, x0:x1]
        np.maximum(region, (1.0 - strength) * ink * feather, out=region)
    result = np.clip(np.rint(base.astype(np.float32) + attenuation[:, :, None] *
                              (normal.astype(np.float32) - base.astype(np.float32))),
                     0, 255).astype(np.uint8)
    result = suppress_lower_band(result, lines)
    return result, attenuation


def suppress_lower_band(image: np.ndarray, lines: list[dict]) -> np.ndarray:
    """Suppress the bright row-wide halo just below small glyphs."""
    height, width = image.shape[:2]
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luma = ycrcb[:, :, 0].astype(np.float32)
    adjustment = np.zeros((height, width), np.float32)
    for line in lines:
        x0, y0, x1, y1 = map(int, line["crop_xyxy"])
        line_height = y1 - y0
        if line_height >= 190 or line_height < 80:
            continue
        start = max(0, y0 - 12)
        stop = min(height, y1 + 30)
        patch = luma[start:stop, x0:x1]
        if float(np.percentile(patch, 80)) < 105:
            continue
        row_background = np.percentile(patch, 80, axis=1).astype(np.float32)
        baseline = float(np.percentile(row_background, 30))
        yy = np.arange(start, stop, dtype=np.float32)
        phase = (yy - y0) / line_height
        lower = smoothstep((phase - 0.55) / 0.15) * (1.0 - smoothstep((phase - 0.94) / 0.15))
        excess = np.clip(row_background - baseline - 1.0, 0.0, 9.0) * lower
        background_weight = smoothstep((patch - (row_background[:, None] - 34.0)) / 23.0)
        xx = np.arange(x1 - x0, dtype=np.float32)
        horizontal = smoothstep(np.minimum(xx, x1 - x0 - 1 - xx) / 25.0)
        candidate = -excess[:, None] * background_weight * horizontal[None, :]
        np.minimum(adjustment[start:stop, x0:x1], candidate,
                   out=adjustment[start:stop, x0:x1])
    ycrcb[:, :, 0] = np.clip(np.rint(luma + adjustment), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--export-jpegs", action="store_true")
    args = parser.parse_args()
    if args.export_jpegs:
        if args.output_root is None:
            parser.error("--export-jpegs requires --output-root")
        destination = args.output_root / "preview_jpg"
        destination.mkdir(parents=True, exist_ok=True)
        pictures = sorted((args.output_root / "gtfit_v6_borderrail_h96").glob("*/output_full.jpg"))
        for picture in pictures:
            shutil.copy2(picture, destination / f"{picture.parent.name}.jpg")
        print(json.dumps({"exported": len(pictures), "directory": str(destination)}))
        return
    if args.batch:
        if not all((args.source_root, args.base_root, args.output_root)):
            parser.error("--batch requires source, base, and output roots")
        arm = "gtfit_v6_borderrail_h96"
        reports = sorted((args.base_root / arm).glob("*/result.json"))
        if not reports:
            raise ValueError("no source reports found")
        for index, report in enumerate(reports, 1):
            sample = report.parent.name
            normal = args.source_root / arm / "normal" / sample / "output.png"
            base = report.parent / "output.png"
            output = args.output_root / arm / sample / "output.png"
            process(normal, base, report, output)
            print(json.dumps({"done": index, "total": len(reports), "sample_id": sample},
                             ensure_ascii=False), flush=True)
        return
    if not all((args.normal, args.base, args.report, args.output)):
        parser.error("normal, base, report, and output paths are required")
    process(args.normal, args.base, args.report, args.output)


def process(normal_path: Path, base_path: Path, report_path: Path, output_path: Path) -> None:
    normal = cv2.imread(str(normal_path), cv2.IMREAD_COLOR)
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if normal is None or base is None:
        raise ValueError("missing normal or V5 output")
    lines = json.loads(report_path.read_text(encoding="utf-8"))["line_reports"]
    result, attenuation = refine(normal, base, lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"cannot write {output_path}")
    cv2.imwrite(str(output_path.with_name("output_full.jpg")), result,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(json.dumps({"output": str(output_path),
                      "attenuated_pixels": int(np.count_nonzero(attenuation > 0.01)),
                      "max_attenuation": float(attenuation.max())}), flush=True)


if __name__ == "__main__":
    main()
