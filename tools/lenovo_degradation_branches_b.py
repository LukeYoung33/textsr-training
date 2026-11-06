"""Deterministic preview generator for Lenovo degradation branches 3 and 4.

The tool produces visual inspection assets only.  It does not alter its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _noise(image: np.ndarray, rng: np.random.Generator, shot: float, read: float) -> np.ndarray:
    """Approximate signal-dependent sensor noise in display-referred BGR space."""
    normalized = image.astype(np.float32) / 255.0
    sigma = np.sqrt(np.maximum(normalized, 0.0) * shot + read * read)
    return np.clip((normalized + rng.normal(0.0, 1.0, normalized.shape) * sigma) * 255.0, 0, 255).astype(np.uint8)


def _midfreq_chroma_noise(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Add correlated, band-pass chroma-only variation in YCrCb space.

    ``strength`` is the target chroma standard deviation in 8-bit code values.
    Keeping Y unchanged makes this suitable for separating colour artefacts from
    luminance sensor noise in the preview controls.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    band = cv2.GaussianBlur(noise, (0, 0), 1.0) - cv2.GaussianBlur(noise, (0, 0), 4.0)
    band /= max(float(band.std()), 1e-6)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycrcb[..., 1] += band * strength
    ycrcb[..., 2] += np.roll(band, max(1, w // 13), axis=1) * strength
    return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def strong_processing(image: np.ndarray, rng: np.random.Generator, midfreq_chroma_strength: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Blur/noise followed by edge-aware denoise and modest USM halo formation."""
    params = {"blur_sigma": 0.9, "shot_noise": 0.0015, "read_noise": 0.004,
              "bilateral_d": 7, "bilateral_sigma_color": 28, "bilateral_sigma_space": 7,
              "usm_sigma": 1.15, "usm_amount": 1.25, "jpeg_quality": 82,
              "midfreq_chroma_strength": float(midfreq_chroma_strength)}
    blurred = cv2.GaussianBlur(image, (0, 0), params["blur_sigma"], borderType=cv2.BORDER_REFLECT101)
    noisy = _noise(blurred, rng, params["shot_noise"], params["read_noise"])
    denoised = cv2.bilateralFilter(noisy, params["bilateral_d"], params["bilateral_sigma_color"], params["bilateral_sigma_space"])
    base = cv2.GaussianBlur(denoised, (0, 0), params["usm_sigma"], borderType=cv2.BORDER_REFLECT101)
    sharpened = cv2.addWeighted(denoised, 1.0 + params["usm_amount"], base, -params["usm_amount"], 0)
    return _jpeg(_midfreq_chroma_noise(sharpened, rng, midfreq_chroma_strength), params["jpeg_quality"]), params


def secondary_recapture(image: np.ndarray, rng: np.random.Generator, grid_strength: float = 0.0,
                        midfreq_chroma_strength: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate display-downsampling, recapture geometry, camera noise and coding."""
    h, w = image.shape[:2]
    params = {"display_scale": 0.62, "display_jpeg_quality": 88, "perspective_fraction": 0.012,
              "camera_blur_sigma": 0.7, "shot_noise": 0.001, "read_noise": 0.003,
              "final_jpeg_quality": 84, "screen_grid_strength": float(grid_strength),
              "midfreq_chroma_strength": float(midfreq_chroma_strength)}
    small = cv2.resize(image, (max(2, round(w * params["display_scale"])), max(2, round(h * params["display_scale"]))), interpolation=cv2.INTER_AREA)
    displayed = cv2.resize(_jpeg(small, params["display_jpeg_quality"]), (w, h), interpolation=cv2.INTER_CUBIC)
    if grid_strength:
        # Optional weak RGB subpixel modulation; disabled by default because it is not established as dominant.
        modulation = np.ones((h, w, 3), np.float32)
        phase = np.arange(w) % 3
        for channel in range(3):
            modulation[:, phase == channel, channel] += grid_strength
            modulation[:, phase != channel, channel] -= grid_strength / 2.0
        displayed = np.clip(displayed.astype(np.float32) * modulation, 0, 255).astype(np.uint8)
    margin = params["perspective_fraction"]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    offsets = rng.uniform(-margin, margin, (4, 2)).astype(np.float32) * np.float32([w, h])
    dst = src + offsets
    warped = cv2.warpPerspective(displayed, cv2.getPerspectiveTransform(src, dst), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    blurred = cv2.GaussianBlur(warped, (0, 0), params["camera_blur_sigma"], borderType=cv2.BORDER_REFLECT101)
    noisy = _noise(blurred, rng, params["shot_noise"], params["read_noise"])
    return _jpeg(_midfreq_chroma_noise(noisy, rng, midfreq_chroma_strength), params["final_jpeg_quality"]), params


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unsupported or unreadable image: {}".format(path))
    return image


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("output JPEG encoding failed")
    encoded.tofile(str(path))


def _image_paths(input_path: Path, output_root: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    output_resolved = output_root.resolve()
    for path in sorted(input_path.rglob("*")):
        relative_parts = path.relative_to(input_path).parts
        is_preview = any(part.startswith("lenovo_degradation_preview_") for part in relative_parts)
        if path.suffix.lower() in IMAGE_SUFFIXES and not is_preview and output_resolved not in path.resolve().parents:
            yield path


def generate(input_path: Path, output_root: Path, seed: int = 20260910, limit: int | None = None,
             grid_strength: float = 0.0, midfreq_chroma_strength: float = 0.0) -> dict[str, Any]:
    """Generate both branches and return the manifest written at output_root."""
    input_path, output_root = input_path.resolve(), output_root.resolve()
    records: list[dict[str, Any]] = []
    for index, path in enumerate(_image_paths(input_path, output_root)):
        if limit is not None and index >= limit:
            break
        rel = Path(path.name) if input_path.is_file() else path.relative_to(input_path)
        stem = rel.with_suffix("")
        image = _read_image(path)
        item_seed = int.from_bytes(hashlib.sha256((str(rel).replace("\\", "/") + str(seed)).encode()).digest()[:8], "little")
        strong, strong_params = strong_processing(image, np.random.default_rng(item_seed), midfreq_chroma_strength)
        recapture, recapture_params = secondary_recapture(image, np.random.default_rng(item_seed + 1), grid_strength, midfreq_chroma_strength)
        strong_path = output_root / "strong_processing" / stem.with_suffix(".jpg")
        recapture_path = output_root / "secondary_recapture" / stem.with_suffix(".jpg")
        _write_jpeg(strong_path, strong)
        _write_jpeg(recapture_path, recapture)
        records.append({"input": str(path), "relative_input": str(rel), "seed": item_seed,
                        "size": [int(image.shape[1]), int(image.shape[0])],
                        "strong_processing": {"output": str(strong_path), "parameters": strong_params},
                        "secondary_recapture": {"output": str(recapture_path), "parameters": recapture_params}})
    manifest = {"seed": seed, "input": str(input_path), "output_root": str(output_root), "images": records}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/generated_images"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/generated_images/lenovo_degradation_preview_b"))
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--screen-grid-strength", type=float, default=0.0)
    parser.add_argument("--midfreq-chroma-strength", type=float, default=0.0,
                        help="Correlated chroma-only band-pass noise, in 8-bit code values; disabled by default.")
    args = parser.parse_args()
    manifest = generate(args.input, args.output_root, args.seed, args.limit, args.screen_grid_strength,
                        args.midfreq_chroma_strength)
    print("Wrote {} paired previews to {}".format(len(manifest["images"]), args.output_root))


if __name__ == "__main__":
    main()
