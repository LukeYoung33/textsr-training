"""Deterministic previews for two Lenovo-oriented image degradation branches.

The tool intentionally keeps capture noise before resizing in the digital-zoom
branch: zoom changes the size and correlation of captured noise rather than
simply increasing independent output-pixel noise.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG coding while retaining an RGB ndarray API."""
    encoded = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(
        encoded, format="JPEG", quality=int(quality), subsampling=2
    )
    encoded.seek(0)
    with Image.open(encoded) as image:
        return np.asarray(image.convert("RGB")).copy()


def _blur(rgb: np.ndarray, sigma_x: float, sigma_y: float, angle_degrees: float) -> np.ndarray:
    if abs(sigma_x - sigma_y) < 1e-6:
        return cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    radius = max(3, int(np.ceil(max(sigma_x, sigma_y) * 3)) * 2 + 1)
    impulse = np.zeros((radius, radius), dtype=np.float32)
    impulse[radius // 2, radius // 2] = 1.0
    kernel = cv2.GaussianBlur(impulse, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    rotation = cv2.getRotationMatrix2D((radius // 2, radius // 2), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (radius, radius), flags=cv2.INTER_CUBIC)
    kernel /= max(float(kernel.sum()), 1e-8)
    return cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def _add_midfreq_chroma_noise(rgb: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Add zero-mean band-pass chroma variation while keeping luminance intact."""
    if strength <= 0:
        return rgb
    height, width = rgb.shape[:2]
    field = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    middle = cv2.GaussianBlur(field, (0, 0), 1.1) - cv2.GaussianBlur(field, (0, 0), 6.0)
    middle /= max(float(middle.std()), 1e-6)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycrcb[:, :, 1] += middle * strength
    ycrcb[:, :, 2] += cv2.GaussianBlur(middle, (0, 0), 0.65) * strength * float(rng.choice((-1.0, 1.0)))
    return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def _capture(
    rgb: np.ndarray, rng: np.random.Generator, midfreq_chroma_strength: float = 0.0
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Approximate optical blur -> sensor noise -> mild ISP chroma blur -> JPEG."""
    mode = "motion" if rng.random() < 0.25 else "gaussian"
    if mode == "motion":
        length = int(rng.integers(3, 10))
        angle = float(rng.uniform(0, 180))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        kernel = cv2.warpAffine(
            kernel, cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1.0), (length, length)
        )
        kernel /= max(float(kernel.sum()), 1e-8)
        blurred = cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REFLECT101)
        blur_params: dict[str, Any] = {"type": mode, "length": length, "angle_degrees": round(angle, 3)}
    else:
        sigma_x = float(rng.uniform(0.35, 1.45))
        sigma_y = sigma_x if rng.random() < 0.55 else float(rng.uniform(0.35, 1.45))
        angle = float(rng.uniform(0, 180))
        blurred = _blur(rgb, sigma_x, sigma_y, angle)
        blur_params = {"type": mode, "sigma_x": round(sigma_x, 4), "sigma_y": round(sigma_y, 4), "angle_degrees": round(angle, 3)}

    signal = blurred.astype(np.float32) / 255.0
    # D1/D2 should primarily show blur and sampling loss, not train denoising.
    shot = float(rng.uniform(0.00003, 0.00018))
    read = float(rng.uniform(0.0004, 0.0015))
    noisy = signal + rng.normal(0.0, np.sqrt(np.maximum(signal, 0.0) * shot + read * read), signal.shape)
    output = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    chroma_sigma = float(rng.uniform(0.0, 0.75))
    if chroma_sigma > 0.08:
        ycrcb = cv2.cvtColor(output, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 1:] = cv2.GaussianBlur(ycrcb[:, :, 1:], (0, 0), chroma_sigma)
        output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    output = _add_midfreq_chroma_noise(output, rng, midfreq_chroma_strength)
    quality = int(rng.integers(80, 96))
    return _jpeg_roundtrip(output, quality), {
        "blur": blur_params, "shot_noise": round(shot, 6), "read_noise": round(read, 6),
        "chroma_blur_sigma": round(chroma_sigma, 4),
        "midfreq_chroma_strength": round(float(midfreq_chroma_strength), 4),
        "jpeg_quality": quality,
    }


def normal_camera(
    rgb: np.ndarray, rng: np.random.Generator, midfreq_chroma_strength: float = 0.0
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Return normal camera degradation at the original image dimensions."""
    if midfreq_chroma_strength < 0:
        raise ValueError("midfreq_chroma_strength must be non-negative")
    return _capture(rgb, rng, midfreq_chroma_strength)


def digital_zoom(
    rgb: np.ndarray, rng: np.random.Generator, zoom: float = 2.0,
    interpolation: str = "bicubic", crop_center: Optional[Tuple[float, float]] = None,
    midfreq_chroma_strength: float = 0.0,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Capture first, then crop/resize to model digital zoom at fixed output size."""
    if zoom <= 1.0:
        raise ValueError("zoom must be greater than 1.0")
    if midfreq_chroma_strength < 0:
        raise ValueError("midfreq_chroma_strength must be non-negative")
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation: {interpolation}")
    captured, capture_params = _capture(rgb, rng, midfreq_chroma_strength)
    height, width = captured.shape[:2]
    crop_width, crop_height = max(2, round(width / zoom)), max(2, round(height / zoom))
    cx, cy = crop_center or (0.5, 0.5)
    cx, cy = float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1))
    left = int(round(cx * (width - crop_width)))
    top = int(round(cy * (height - crop_height)))
    crop = captured[top:top + crop_height, left:left + crop_width]
    output = cv2.resize(crop, (width, height), interpolation=INTERPOLATIONS[interpolation])
    sharpen_amount = float(rng.uniform(0.0, 0.23))
    if sharpen_amount > 0.01:
        softened = cv2.GaussianBlur(output, (0, 0), 0.7)
        output = cv2.addWeighted(output, 1.0 + sharpen_amount, softened, -sharpen_amount, 0)
    quality = int(rng.integers(82, 97))
    output = _jpeg_roundtrip(output, quality)
    return output, {
        "capture": capture_params, "zoom": float(zoom), "crop_box_xywh": [left, top, crop_width, crop_height],
        "crop_center_normalized": [round(cx, 4), round(cy, 4)], "interpolation": interpolation,
        "post_resize_sharpen_amount": round(sharpen_amount, 4), "jpeg_quality": quality,
    }


def _images(input_path: Path, excluded_root: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_SUFFIXES:
            yield input_path
        return
    for path in sorted(input_path.rglob("*")):
        relative_parts = path.relative_to(input_path).parts
        is_preview = any(part.startswith("lenovo_degradation_preview_") for part in relative_parts)
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and excluded_root not in path.parents
            and not is_preview
        ):
            yield path


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    output_root = Path(args.output_root).resolve()
    if input_path.is_file() and output_root == input_path.parent:
        raise ValueError("output_root must not be the input image directory")
    normal_root, zoom_root = output_root / "normal_camera", output_root / "digital_zoom"
    records = []
    rng = np.random.default_rng(args.seed)
    for source in _images(input_path, output_root):
        relative = source.name if input_path.is_file() else source.relative_to(input_path)
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        normal, normal_params = normal_camera(rgb, rng, args.midfreq_chroma_strength)
        zoomed, zoom_params = digital_zoom(
            rgb, rng, args.zoom, args.interpolation, args.crop_center, args.midfreq_chroma_strength
        )
        for root, output in ((normal_root, normal), (zoom_root, zoomed)):
            destination = (root / relative).with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output).save(destination)
        records.append({"source": str(relative), "source_size_wh": [rgb.shape[1], rgb.shape[0]], "normal_camera": normal_params, "digital_zoom": zoom_params})
    manifest = {"seed": args.seed, "input": str(input_path), "count": len(records), "records": records}
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _parse_center(value: str) -> Tuple[float, float]:
    try:
        x, y = (float(piece.strip()) for piece in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop center must be X,Y") from error
    return x, y


def main() -> None:
    default_input = Path("datasets/generated_images")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(default_input), help="Image file or directory (default: datasets/generated_images)")
    parser.add_argument("--output-root", default="datasets/generated_images/lenovo_degradation_preview_a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--interpolation", choices=sorted(INTERPOLATIONS), default="bicubic")
    parser.add_argument("--crop-center", type=_parse_center, default=None, metavar="X,Y")
    parser.add_argument(
        "--midfreq-chroma-strength", type=float, default=0.0, metavar="VALUE",
        help="Band-pass chroma-noise standard deviation in YCrCb code values (default: 0).",
    )
    args = parser.parse_args()
    if args.input == str(default_input) and not default_input.exists():
        parser.error("default input datasets/generated_images does not exist; pass --input")
    if args.midfreq_chroma_strength < 0:
        parser.error("--midfreq-chroma-strength must be non-negative")
    print(run(args))


if __name__ == "__main__":
    main()
