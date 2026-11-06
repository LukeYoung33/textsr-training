"""CPU-only Stage A preprocessing screen for corrected Lenovo RGB pairs.

The tool deliberately stops before any model or training interface.  It reads
paired direct-RGB files, keeps source groups together during splitting, selects
bounded active-arm parameters on development groups, freezes them, and evaluates
the same arms on held-out groups. P6 can apply a screenshot-inspired curve to
BT.601 Y while preserving chroma. P10 can run a bounded Y-only Landweber
deblur followed by per-boundary monotone profile cleanup. Packed YUV/NV21
inputs are not accepted.
P11 estimates an equivalent local Gaussian PSF from text edges, applies one
masked Wiener-Hunt deblur, and then applies a local Y-only S-curve.
P12 retains the P11 estimated-kernel deblur and applies a local platform clamp
to remove only supported dark/bright overshoot in a fixed contour band.
P13 retains the P11 estimated-kernel deblur and compares dense spatial
monotone edge projection with a regional shoulder LUT.
P14 retains the P11 estimated-kernel deblur and compares a continuous
contour-monotone projection with a locally supported soft shoulder field.
P16 reuses the P13 shoulder LUT with local overlapping 50x50 patches.
P17 compares P16 against regional Otsu platform statistics in larger patches.
P18 retains those statistics but corrects only signed excess around anchored
continuous background/foreground references, avoiding P17's plateau clamp.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import cv2
import numpy as np


ARM_NAMES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P16", "P17", "P18")
METRIC_NAMES = (
    "masked_l1_rgb",
    "masked_l1_y",
    "psnr_rgb",
    "psnr_y",
    "residual_p95_rgb",
    "luma_bias",
    "edge_mae_y",
    "input_delta_l1_rgb",
    "saturation_fraction",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "version": "lenovo_stage_a_v1",
    "input_contract": "direct_rgb_authoritative_tensor_hwc_float32_0_1",
    "split": {"seed": 20260915, "development_fraction": 0.5},
    "visual_limit": 24,
    "residual_display_max": 0.25,
    "active_arms": ["P0", "P1", "P2", "P3", "P4", "P5"],
    "p5_selection": {
        "mean_rgb_l1_multiplier": 1.02,
        "max_source_group_p95_deterioration": 0.015,
    },
    "arms": {
        "P0": {},
        "P1": {
            "low": [0.02, 0.05],
            "high": [0.95, 0.98],
            "gamma": [1.0],
            "strength": [0.25, 0.5],
        },
        "P2": {
            "sigma": [1.0, 1.5],
            "k": [1.5, 2.0],
            "strength": [0.25, 0.5],
        },
        "P3": {
            "kernel_size": [5, 7],
            "sigma": [0.8, 1.2],
            "regularization": [0.04, 0.08],
            "gain_cap": [1.25, 1.5],
            "strength": [0.25, 0.5],
        },
        "P4": {"composition_strength": [0.25, 0.5]},
        "P5": {
            "sigma": [0.8, 1.2],
            "amount": [1.0, 1.5],
            "delta_cap": [0.12, 0.20],
            "blend": [1.0],
        },
        "P6": {"curves": []},
        "P7": {"curves": []},
        "P8": {"curves": []},
        "P9": {"curves": []},
        "P10": {"candidates": []},
        "P11": {"candidates": []},
        "P12": {"candidates": []},
        "P13": {"candidates": []},
        "P14": {"candidates": []},
        "P16": {"candidates": []},
        "P17": {"candidates": []},
        "P18": {"candidates": []},
    },
}


def _comment_placeholder() -> None:
    """@luna-comment Keep the module comment registry aware of helper scope.

    Input: None.
    Output: None.
    Key Steps: No runtime work; this function is never called.
    Key Params: None.
    State/Dependencies: None.
    """


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """@luna-comment Merge user configuration into safe Stage A defaults.

    Input: Mapping-like default and override dictionaries.
    Output: New recursively merged dictionary without mutating either input.
    Key Steps: Recurse through nested mappings and replace scalar/list values.
    Key Params: User values override defaults; unspecified values stay bounded.
    State/Dependencies: None.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _json_hash(value: Any) -> str:
    """@luna-comment Hash JSON-safe configuration or split state deterministically.

    Input: JSON-serialisable value.
    Output: Lowercase SHA-256 digest string.
    Key Steps: Canonicalise keys and separators before hashing UTF-8 bytes.
    Key Params: Sorted keys and compact separators make replay hashes stable.
    State/Dependencies: Python hashlib and json only.
    """
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    """@luna-comment Hash one input file for manifest identity and replay evidence.

    Input: Existing file path.
    Output: SHA-256 digest of raw file bytes.
    Key Steps: Stream the file in bounded blocks.
    Key Params: One-megabyte blocks limit temporary memory.
    State/Dependencies: File content is read but never modified.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path | None) -> dict[str, Any]:
    """@luna-comment Load and validate the Stage A JSON configuration.

    Input: Optional JSON path containing split, visual, and arm candidate values.
    Output: Deep-merged configuration with all default arms present.
    Key Steps: Parse JSON, merge defaults, and verify arm names and ranges later.
    Key Params: Defaults keep the first screen conservative and low-strength.
    State/Dependencies: JSON file is read only; no global state is changed.
    """
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        override = json.load(handle)
    if not isinstance(override, dict):
        raise ValueError(f"Stage A config must be a JSON object: {path}")
    config = _deep_merge(DEFAULT_CONFIG, override)
    if set(config["arms"]) != set(ARM_NAMES):
        raise ValueError("config arms must contain exactly P0 through P17")
    active_arms = config.get("active_arms")
    if not isinstance(active_arms, list) or not active_arms or active_arms[0] != "P0":
        raise ValueError("active_arms must be a non-empty list beginning with P0")
    if len(set(active_arms)) != len(active_arms) or any(arm not in ARM_NAMES for arm in active_arms):
        raise ValueError("active_arms must contain unique known arms")
    return config


def _resolve_path(value: str, root: Path) -> Path:
    """@luna-comment Resolve a manifest asset path without changing its source.

    Input: Relative or absolute path string and manifest directory.
    Output: Expanded absolute path.
    Key Steps: Expand user marker, then anchor relative paths to manifest root.
    Key Params: Manifest-root anchoring makes copied manifests reproducible.
    State/Dependencies: No filesystem mutation.
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_rgb_tensor(path: Path) -> np.ndarray:
    """@luna-comment Load one authoritative RGB tensor in HWC float32 [0,1].

    Input: RGB image file or .npy array with three channels.
    Output: Contiguous HxWx3 float32 array in [0,1].
    Key Steps: Read BGR image bytes, convert once to RGB, normalise, and validate.
    Key Params: uint8/uint16 scales are supported; packed YUV/NV21 is rejected.
    State/Dependencies: OpenCV/numpy reads only the requested file.
    """
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise FileNotFoundError(f"unable to read direct RGB image: {path}")
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"image must have exactly three channels: {path} shape={value.shape}")
        value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 3 and value.shape[2] != 3:
        value = np.transpose(value, (1, 2, 0))
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"RGB tensor must be HxWx3 or 3xHxW: {path} shape={value.shape}")
    if value.dtype == np.uint8:
        value = value.astype(np.float32) / 255.0
    elif value.dtype == np.uint16:
        value = value.astype(np.float32) / 65535.0
    else:
        value = value.astype(np.float32)
        finite_max = float(np.nanmax(value)) if value.size else float("nan")
        if finite_max > 1.000001:
            if finite_max <= 255.0001:
                value = value / 255.0
            else:
                raise ValueError(f"float RGB values exceed 255: {path}")
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"RGB tensor is empty or non-finite: {path}")
    if float(value.min()) < -1e-6 or float(value.max()) > 1.000001:
        raise ValueError(f"RGB tensor is outside [0,1]: {path}")
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0), dtype=np.float32)


def _load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    """@luna-comment Load an optional aligned binary validity mask.

    Input: Optional image/.npy mask path and expected HxW shape.
    Output: Boolean HxW mask; absent masks become all-valid.
    Key Steps: Read, squeeze a singleton channel, enforce exact alignment.
    Key Params: No implicit resize is allowed because it can hide alignment bugs.
    State/Dependencies: OpenCV/numpy file read only.
    """
    if path is None:
        return np.ones(shape, dtype=bool)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if value is None:
            raise FileNotFoundError(f"unable to read mask: {path}")
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.shape != shape:
        raise ValueError(f"mask shape {value.shape} does not match image {shape}: {path}")
    return np.asarray(value > 0.5, dtype=bool)


def _normalise_split(value: Any) -> str | None:
    """@luna-comment Normalise explicit development/held-out split labels.

    Input: Optional manifest split label.
    Output: `development`, `heldout`, or None for an absent label.
    Key Steps: Accept only documented aliases and reject unknown labels.
    Key Params: Train/dev aliases map to development; val/test aliases map heldout.
    State/Dependencies: None.
    """
    if value is None or str(value).strip() == "":
        return None
    label = str(value).strip().lower().replace("-", "_")
    if label in {"dev", "development", "train"}:
        return "development"
    if label in {"heldout", "held_out", "val", "validation", "test", "evaluation"}:
        return "heldout"
    raise ValueError(f"unsupported split label: {value}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """@luna-comment Load paired direct-RGB records and materialise validation tensors.

    Input: JSONL or JSON-list manifest with sample/source/path fields.
    Output: Records containing private RGB/mask arrays plus reproducibility hashes.
    Key Steps: Reject packed-only contracts, resolve paths, load aligned arrays,
      and enforce unique sample IDs and required source groups.
    Key Params: `lq_rgb_path`, `gt_rgb_path`, and `source_group` are mandatory.
    State/Dependencies: Reads source files; does not call any model or decoder.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            payload = [payload]
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not payload:
        raise ValueError("manifest has no records")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    root = path.parent
    forbidden_keys = {"lq", "gt", "nv21", "yuv", "packed_lq", "packed_gt"}
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest row {index} is not an object")
        if forbidden_keys.intersection(raw) and not {"lq_rgb_path", "gt_rgb_path"}.issubset(raw):
            raise ValueError(f"manifest row {index} is packed/non-RGB and lacks direct RGB paths")
        for key in ("sample_id", "source_group", "lq_rgb_path", "gt_rgb_path"):
            if not str(raw.get(key, "")).strip():
                raise ValueError(f"manifest row {index} missing required field: {key}")
        sample_id = str(raw["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        lq_path = _resolve_path(str(raw["lq_rgb_path"]), root)
        gt_path = _resolve_path(str(raw["gt_rgb_path"]), root)
        lq = load_rgb_tensor(lq_path)
        gt = load_rgb_tensor(gt_path)
        if lq.shape != gt.shape:
            raise ValueError(f"LQ/GT shape mismatch for {sample_id}: {lq.shape} vs {gt.shape}")
        valid_value = raw.get("valid_mask_path")
        text_value = raw.get("text_mask_path")
        valid_path = _resolve_path(str(valid_value), root) if valid_value else None
        text_path = _resolve_path(str(text_value), root) if text_value else None
        record = dict(raw)
        record.update(
            {
                "sample_id": sample_id,
                "source_group": str(raw["source_group"]),
                "split": _normalise_split(raw.get("split")),
                "lq_rgb_path": str(lq_path),
                "gt_rgb_path": str(gt_path),
                "input_hashes": {"lq_rgb": _file_hash(lq_path), "gt_rgb": _file_hash(gt_path)},
                "_lq_rgb": lq,
                "_gt_rgb": gt,
                "_valid_mask": _load_mask(valid_path, lq.shape[:2]),
                "_text_mask": _load_mask(text_path, lq.shape[:2]) if text_path else None,
            }
        )
        records.append(record)
    return records


def assign_source_group_splits(
    records: list[dict[str, Any]], seed: int, development_fraction: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """@luna-comment Assign disjoint source groups to development and held-out sets.

    Input: Loaded records, integer seed, and target development fraction.
    Output: Copied records with normalised split labels plus a split audit object.
    Key Steps: Validate explicit group labels or hash-sort groups deterministically.
    Key Params: Groups never cross splits; at least one group is required per side.
    State/Dependencies: None; source arrays are reused without mutation.
    """
    if not 0.0 < float(development_fraction) < 1.0:
        raise ValueError("development_fraction must be strictly between 0 and 1")
    groups = sorted({str(record["source_group"]) for record in records})
    if len(groups) < 2:
        raise ValueError("at least two source groups are required for held-out evaluation")
    by_group: dict[str, set[str]] = {group: set() for group in groups}
    for record in records:
        if record.get("split") is not None:
            by_group[str(record["source_group"])].add(str(record["split"]))
    explicit = any(by_group[group] for group in groups)
    if explicit:
        if any(record.get("split") is None for record in records):
            raise ValueError("either every record must specify split or none may specify split")
        mixed = [group for group, values in by_group.items() if len(values) != 1]
        if mixed:
            raise ValueError(f"source groups span multiple explicit splits: {mixed}")
        group_split = {group: next(iter(values)) for group, values in by_group.items()}
    else:
        ordered = sorted(groups, key=lambda group: _json_hash(f"{seed}:{group}"))
        dev_count = max(1, min(len(groups) - 1, int(round(len(groups) * development_fraction))))
        group_split = {
            group: "development" if index < dev_count else "heldout"
            for index, group in enumerate(ordered)
        }
    if set(group_split.values()) != {"development", "heldout"}:
        raise ValueError("split must contain both development and heldout source groups")
    copied: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        value["split"] = group_split[str(record["source_group"])]
        copied.append(value)
    split_audit = {
        "seed": int(seed),
        "development_fraction": float(development_fraction),
        "groups": {split: sorted(group for group, value in group_split.items() if value == split) for split in ("development", "heldout")},
        "group_counts": {split: sum(value == split for value in group_split.values()) for split in ("development", "heldout")},
        "record_counts": {split: sum(record["split"] == split for record in copied) for split in ("development", "heldout")},
    }
    split_audit["split_hash"] = _json_hash(split_audit)
    return copied, split_audit


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    """@luna-comment Convert an HWC RGB tensor to BT.601 luma without color decoding.

    Input: HxWx3 float32 RGB tensor.
    Output: HxW float32 luma in the same nominal range.
    Key Steps: Apply fixed RGB luminance weights whose sum is one.
    Key Params: 0.299/0.587/0.114 match existing evaluator metric convention.
    State/Dependencies: None.
    """
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _identity(rgb: np.ndarray, reason: str = "identity") -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Return a detached identity base and explicit fallback metadata.

    Input: Valid HxWx3 RGB tensor and reason label.
    Output: Copy of input and metadata with zero transform delta.
    Key Steps: Copy to prevent downstream mutation of authoritative input.
    Key Params: Reason distinguishes P0 from an invalid-candidate fallback.
    State/Dependencies: None.
    """
    return rgb.copy(), {"fallback": reason != "identity", "fallback_reason": None if reason == "identity" else reason}


def _global_curve(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply a bounded global tonal curve diagnostic.

    Input: RGB tensor and low/high/gamma/strength parameters.
    Output: Clipped RGB tensor and fallback metadata.
    Key Steps: Normalise to [0,1], apply gamma in the bounded interval, blend.
    Key Params: Endpoint compression is diagnostic; strength is low by default.
    State/Dependencies: No GT or global mutable state.
    """
    try:
        low = float(params["low"])
        high = float(params["high"])
        gamma = float(params["gamma"])
        strength = float(params["strength"])
        if not (0.0 <= low < high <= 1.0 and gamma > 0.0 and 0.0 <= strength <= 1.0):
            raise ValueError("invalid global curve parameters")
        if strength == 0.0:
            return _identity(rgb)
        t = np.clip((rgb - low) / (high - low), 0.0, 1.0)
        curve = low + (high - low) * np.power(t, gamma)
        output = np.clip(rgb + strength * (curve - rgb), 0.0, 1.0).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("global curve produced non-finite output")
        return output, {"fallback": False, "fallback_reason": None}
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        return _identity(rgb, f"invalid_global_curve:{error}")


def _local_dering(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Suppress only bounded local luma excursions with soft blending.

    Input: RGB tensor and sigma/k/strength parameters.
    Output: RGB tensor with shared luma correction and fallback metadata.
    Key Steps: Estimate local mean/variance, clip luma to a credible interval,
      add a bounded luma delta to RGB, and gamut-clip the result.
    Key Params: Gaussian sigma, interval multiplier k, and blend strength.
    State/Dependencies: OpenCV Gaussian filtering; no GT dependence.
    """
    try:
        sigma = float(params["sigma"])
        k = float(params["k"])
        strength = float(params["strength"])
        if not (sigma > 0.0 and k >= 0.0 and 0.0 <= strength <= 1.0):
            raise ValueError("invalid local deringing parameters")
        if strength == 0.0:
            return _identity(rgb)
        y = _rgb_to_luma(rgb)
        smooth = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        second = cv2.GaussianBlur(y * y, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        variance = np.maximum(second - smooth * smooth, 1.0e-8)
        interval = k * np.sqrt(variance)
        target = np.clip(y, smooth - interval, smooth + interval)
        delta = strength * (target - y)
        output = np.clip(rgb + delta[..., None], 0.0, 1.0).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("local deringing produced non-finite output")
        return output, {"fallback": False, "fallback_reason": None}
    except (KeyError, TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return _identity(rgb, f"invalid_local_dering:{error}")


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """@luna-comment Build a normalised odd-sized Gaussian point-spread kernel.

    Input: Positive odd kernel size and positive sigma.
    Output: Float32 2-D kernel whose sum is one.
    Key Steps: Evaluate the separable Gaussian and normalise safely.
    Key Params: Kernel size controls support; sigma controls blur width.
    State/Dependencies: Numpy only.
    """
    size = int(size)
    sigma = float(sigma)
    if size < 3 or size % 2 == 0 or sigma <= 0.0:
        raise ValueError("deblur kernel size must be odd and sigma must be positive")
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    one = np.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel = np.outer(one, one)
    total = float(kernel.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("deblur kernel normalisation failed")
    return (kernel / total).astype(np.float32)


def _regularized_deblur(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply reflected-pad Wiener deblur with an explicit gain cap.

    Input: RGB tensor and kernel_size/sigma/regularization/gain_cap/strength.
    Output: Clipped RGB tensor and fallback metadata.
    Key Steps: Build PSF, FFT each padded channel, regularise inverse response,
      cap frequency gain, inverse FFT, crop, and conservatively blend.
    Key Params: K limits noise amplification; gain_cap bounds high-frequency gain.
    State/Dependencies: Numpy FFT and OpenCV reflected padding; CPU only.
    """
    try:
        size = int(params["kernel_size"])
        sigma = float(params["sigma"])
        regularization = float(params["regularization"])
        gain_cap = float(params["gain_cap"])
        strength = float(params["strength"])
        if regularization <= 0.0 or gain_cap <= 0.0 or not 0.0 <= strength <= 1.0:
            raise ValueError("invalid regularized deblur parameters")
        if strength == 0.0:
            return _identity(rgb)
        kernel = _gaussian_kernel(size, sigma)
        radius = size // 2
        padded = cv2.copyMakeBorder(rgb, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
        work_h, work_w = padded.shape[:2]
        psf = np.zeros((work_h, work_w), dtype=np.float32)
        psf[:size, :size] = kernel
        psf = np.roll(psf, (-radius, -radius), axis=(0, 1))
        response = np.fft.fft2(psf)
        inverse = np.conj(response) / (np.abs(response) ** 2 + regularization)
        inverse_magnitude = np.abs(inverse)
        inverse *= np.minimum(1.0, gain_cap / np.maximum(inverse_magnitude, 1.0e-12))
        restored = np.empty_like(padded, dtype=np.float32)
        for channel in range(3):
            transformed = np.fft.fft2(padded[..., channel])
            restored[..., channel] = np.real(np.fft.ifft2(transformed * inverse)).astype(np.float32)
        restored = restored[radius : radius + rgb.shape[0], radius : radius + rgb.shape[1]]
        output = np.clip(rgb + strength * (restored - rgb), 0.0, 1.0).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("regularized deblur produced non-finite output")
        return output, {"fallback": False, "fallback_reason": None}
    except (KeyError, TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return _identity(rgb, f"invalid_regularized_deblur:{error}")


def _dc_preserving_luma_unsharp(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply bounded DC-preserving luma detail without P2/P3/P4 composition.

    Input: Authoritative HxWx3 RGB tensor and sigma/amount/delta_cap/blend values.
    Output: RGB tensor with a clipped luma detail broadcast to all channels and
      metadata including whether output channels were clipped.
    Key Steps: Compute Y, Gaussian blur Y, form high-pass detail, clip the scaled
      delta, broadcast it to RGB, and gamut-clip the result. Constant luma fields
      return an exact detached copy before any clipping operation.
    Key Params: Positive sigma, non-negative amount, positive delta_cap, and blend
      in [0,1] bound the DC-preserving unsharp diagnostic.
    State/Dependencies: OpenCV Gaussian filtering; no P2/P3/P4 calls or GT state.
    """
    try:
        sigma = float(params["sigma"])
        amount = float(params["amount"])
        delta_cap = float(params["delta_cap"])
        blend = float(params["blend"])
        if not (sigma > 0.0 and amount >= 0.0 and delta_cap > 0.0 and 0.0 <= blend <= 1.0):
            raise ValueError("invalid DC-preserving luma-unsharp parameters")
        if not np.isfinite(rgb).all():
            raise ValueError("luma-unsharp input is non-finite")
        y = _rgb_to_luma(rgb)
        if y.size and bool(np.all(y == y.flat[0])):
            return rgb.copy(), {"fallback": False, "fallback_reason": None, "clipping_fraction": 0.0, "constant_identity": True}
        smooth = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        detail = y - smooth
        delta = np.clip(amount * detail, -delta_cap, delta_cap) * blend
        unclipped = rgb + delta[..., None]
        clipping_fraction = float(np.mean((unclipped < 0.0) | (unclipped > 1.0)))
        output = np.clip(unclipped, 0.0, 1.0).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("luma-unsharp produced non-finite output")
        return output, {
            "fallback": False,
            "fallback_reason": None,
            "clipping_fraction": clipping_fraction,
            "constant_identity": False,
        }
    except (KeyError, TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return _identity(rgb, f"invalid_luma_unsharp:{error}")


def _p6_y_s_curve(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply a monotone screenshot-inspired curve to BT.601 Y only.

    Input: Finite HxWx3 encoded RGB in [0,1] and named ``anchors`` [[x,y], ...].
    Output: RGB with the same full-range BT.601 Cb/Cr up to float error, plus
      requested/applied delta and gamut-limiting telemetry.
    Key Steps: Validate a strictly monotone anchor curve, interpolate Y, constrain
      its delta to the RGB gamut interval, add the permitted delta to all channels,
      and remeasure Cb/Cr invariance.
    Key Params: Anchors must span [0,1] and be nondecreasing in Y; gamut bounds
      intentionally prevent per-channel hard clipping from changing chroma.
    State/Dependencies: Numpy only; no GT, model, or persistent state.
    """
    try:
        anchors = np.asarray(params["anchors"], dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 2 or anchors.shape[0] < 2 or not np.isfinite(anchors).all():
            raise ValueError("anchors must be finite Nx2 values")
        x, target = anchors[:, 0], anchors[:, 1]
        if x[0] != 0.0 or x[-1] != 1.0 or np.any(x[1:] <= x[:-1]) or np.any(target[1:] < target[:-1]):
            raise ValueError("anchors must span [0,1] with strictly increasing x and nondecreasing y")
        if np.any(target < 0.0) or np.any(target > 1.0) or rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
            raise ValueError("invalid P6 RGB input or anchor output range")
        y = _rgb_to_luma(rgb)
        requested = np.interp(y, x, target).astype(np.float32) - y
        lower = -np.min(rgb, axis=2)
        upper = 1.0 - np.max(rgb, axis=2)
        applied = np.minimum(np.maximum(requested, lower), upper).astype(np.float32)
        output = (rgb + applied[..., None]).astype(np.float32)
        cb_in = (rgb[..., 2] - y) / 1.772
        cr_in = (rgb[..., 0] - y) / 1.402
        y_out = _rgb_to_luma(output)
        chroma_error = max(
            float(np.max(np.abs((output[..., 2] - y_out) / 1.772 - cb_in))),
            float(np.max(np.abs((output[..., 0] - y_out) / 1.402 - cr_in))),
        )
        if not np.isfinite(output).all() or float(output.min()) < -2.0e-6 or float(output.max()) > 1.0 + 2.0e-6:
            raise ValueError("P6 gamut-constrained output is invalid")
        return output, {
            "fallback": False,
            "fallback_reason": None,
            "clipping_fraction": 0.0,
            "mean_abs_requested_delta": float(np.mean(np.abs(requested))),
            "mean_abs_applied_delta": float(np.mean(np.abs(applied))),
            "p95_abs_applied_delta": float(np.percentile(np.abs(applied), 95.0)),
            "fraction_abs_delta_ge_4_255": float(np.mean(np.abs(applied) >= 4.0 / 255.0)),
            "gamut_limited_fraction": float(np.mean(np.abs(requested - applied) > 1.0e-7)),
            "mean_abs_requested_minus_applied": float(np.mean(np.abs(requested - applied))),
            "chroma_max_abs_error": chroma_error,
        }
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        return _identity(rgb, f"invalid_p6_y_s_curve:{error}")


def _p7_text_edge_y_curve(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply a confidence-gated, feathered text-edge Y curve.

    Input: Finite HxWx3 RGB ROI and frozen named anchors/radius/gain parameters.
    Output: RGB plus scalar segmentation/chroma telemetry and private mask/weight
      arrays for visual evidence; rejected masks return an exact identity copy.
    Key Steps: Otsu-segment blurred Y, infer border-background polarity, remove
      only tiny components, build a symmetric distance feather, curve original Y,
      constrain it to its original 3x3 envelope and RGB gamut, then retain Cb/Cr.
    Key Params: Sigma .6, .60 polarity share, confidence limits, and reflected
      3x3 envelope are fixed by the P7 design; candidate radius/gain are frozen.
    State/Dependencies: OpenCV connected components/distance transforms; no GT.
    """
    try:
        anchors = np.asarray(params["anchors"], dtype=np.float32)
        radius = float(params["radius"])
        gain = float(params.get("gain", 1.0))
        if anchors.ndim != 2 or anchors.shape[1] != 2 or anchors.shape[0] < 2 or not np.isfinite(anchors).all() or not (radius > 0.0 and gain >= 0.0):
            raise ValueError("invalid P7 anchors, radius, or gain")
        x, target = anchors[:, 0], anchors[:, 1]
        if x[0] != 0.0 or x[-1] != 1.0 or np.any(x[1:] <= x[:-1]) or np.any(target[1:] < target[:-1]) or np.any(target < 0.0) or np.any(target > 1.0):
            raise ValueError("P7 anchors must be monotone and span [0,1]")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
            raise ValueError("invalid P7 RGB input")
        y = _rgb_to_luma(rgb)
        smooth = cv2.GaussianBlur(y, (0, 0), sigmaX=0.6, sigmaY=0.6, borderType=cv2.BORDER_REFLECT_101)
        encoded = np.clip(np.rint(smooth * 255.0), 0, 255).astype(np.uint8)
        threshold, high = cv2.threshold(encoded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_mask = high.astype(bool)
        border_width = max(1, round(min(y.shape) * 0.03))
        border = np.zeros_like(high_mask, dtype=bool)
        border[:border_width] = True; border[-border_width:] = True; border[:, :border_width] = True; border[:, -border_width:] = True
        high_border = float(high_mask[border].mean())
        high_roi = float(high_mask.mean())
        if high_border >= 0.60:
            foreground = ~high_mask; polarity = "dark_text_on_light"
        elif (1.0 - high_border) >= 0.60:
            foreground = high_mask; polarity = "light_text_on_dark"
        elif high_roi >= 0.60:
            foreground = ~high_mask; polarity = "dark_text_on_light_roi_majority"
        elif (1.0 - high_roi) >= 0.60:
            foreground = high_mask; polarity = "light_text_on_dark_roi_majority"
        else:
            return _identity(rgb, "ambiguous_polarity")
        original_foreground = int(foreground.sum())
        min_area = max(2, min(8, round(y.shape[0] * y.shape[1] * 0.00002)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(foreground, dtype=bool)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                keep |= labels == label
        removed_fraction = 1.0 - float(keep.sum()) / max(original_foreground, 1)
        low_values, high_values = y[~high_mask], y[high_mask]
        total_variance = float(np.var(y))
        separation = 0.0 if total_variance <= 1.0e-12 or not low_values.size or not high_values.size else float((low_values.size / y.size) * (high_values.size / y.size) * (float(low_values.mean()) - float(high_values.mean())) ** 2 / total_variance)
        median_gap = abs(float(np.median(low_values)) - float(np.median(high_values))) if low_values.size and high_values.size else 0.0
        foreground_fraction = float(keep.mean())
        if not (0.01 <= foreground_fraction <= 0.65 and median_gap >= 0.12 and separation >= 0.45 and removed_fraction <= 0.15):
            output, meta = _identity(rgb, "low_confidence_text_mask")
            meta.update({"segmentation_accepted": False, "polarity": polarity, "otsu_threshold": float(threshold) / 255.0, "foreground_fraction": foreground_fraction, "median_gap": median_gap, "otsu_separation": separation, "removed_foreground_fraction": removed_fraction})
            return output, meta
        inside = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5)
        outside = cv2.distanceTransform((~keep).astype(np.uint8), cv2.DIST_L2, 5)
        distance = np.maximum(np.where(keep, inside, outside) - 0.5, 0.0)
        weight = np.where(distance < radius, 0.5 * (1.0 + np.cos(np.pi * distance / radius)), 0.0).astype(np.float32)
        local_gamma = params.get("local_gamma")
        if local_gamma is None:
            target_y = np.interp(y, x, target).astype(np.float32)
        else:
            gamma = float(local_gamma)
            fg_samples = y[(keep) & (inside >= 1.5)]
            bg_samples = y[(~keep) & (outside >= 1.5) & (outside <= 6.0)]
            if fg_samples.size < 8: fg_samples = y[keep]
            if bg_samples.size < 8: bg_samples = y[~keep]
            if fg_samples.size < 8 or bg_samples.size < 8 or not (gamma > 1.0):
                return _identity(rgb, "insufficient_local_samples")
            foreground_level = float(np.quantile(fg_samples, .2 if polarity.startswith("dark") else .8))
            background_level = float(np.quantile(bg_samples, .8 if polarity.startswith("dark") else .2))
            if not np.isfinite(foreground_level + background_level) or abs(background_level - foreground_level) < .12:
                return _identity(rgb, "invalid_local_luminance_range")
            t = np.clip((y - foreground_level) / (background_level - foreground_level), 0.0, 1.0)
            numerator = np.power(t, gamma); curve_t = numerator / np.maximum(numerator + np.power(1.0 - t, gamma), 1e-12)
            target_y = foreground_level + (background_level - foreground_level) * curve_t
            target_y = np.where((y >= min(foreground_level, background_level)) & (y <= max(foreground_level, background_level)), target_y, y).astype(np.float32)
            if bool(params.get("complete_transition", False)):
                # P9: retain a constant profile gain across the detected transition;
                # the bounded signed-distance search region is only a support limit.
                support = float(params.get("support_radius", 16.0))
                weight = np.where((distance <= support) & (y >= min(foreground_level, background_level)) & (y <= max(foreground_level, background_level)), 1.0, 0.0).astype(np.float32)
        proposed_y = y + weight * gain * (target_y - y)
        local_min = cv2.erode(y, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_REFLECT_101)
        local_max = cv2.dilate(y, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_REFLECT_101)
        bounded_y = np.minimum(np.maximum(proposed_y, local_min), local_max)
        requested = bounded_y - y
        applied = np.minimum(np.maximum(requested, -np.min(rgb, axis=2)), 1.0 - np.max(rgb, axis=2)).astype(np.float32)
        output = (rgb + applied[..., None]).astype(np.float32)
        cb_in, cr_in = (rgb[..., 2] - y) / 1.772, (rgb[..., 0] - y) / 1.402
        y_out = _rgb_to_luma(output)
        chroma_error = max(float(np.max(np.abs((output[..., 2] - y_out) / 1.772 - cb_in))), float(np.max(np.abs((output[..., 0] - y_out) / 1.402 - cr_in))))
        envelope_limited = float(np.mean(np.abs(proposed_y - bounded_y) > 1.0e-7))
        core = keep & (distance >= radius)
        far_background = (~keep) & (distance >= radius)
        return output, {"fallback": False, "fallback_reason": None, "clipping_fraction": 0.0, "segmentation_accepted": True, "polarity": polarity, "otsu_threshold": float(threshold) / 255.0, "foreground_fraction": foreground_fraction, "median_gap": median_gap, "otsu_separation": separation, "removed_foreground_fraction": removed_fraction, "inner_band_fraction": float((keep & (weight > 0)).mean()), "outer_band_fraction": float((~keep & (weight > 0)).mean()), "core_fraction": float(core.mean()), "far_background_fraction": float(far_background.mean()), "edge_band_mean_abs_delta": float(np.mean(np.abs(applied[weight > 0]))), "edge_band_fraction_abs_delta_ge_4_255": float(np.mean(np.abs(applied[weight > 0]) >= 4.0 / 255.0)), "envelope_limited_fraction": envelope_limited, "gamut_limited_fraction": float(np.mean(np.abs(requested - applied) > 1.0e-7)), "chroma_max_abs_error": chroma_error, "outside_band_max_abs_rgb_delta": float(np.max(np.abs(output[weight == 0] - rgb[weight == 0]))) if bool((weight == 0).any()) else 0.0, "local_foreground_y": float(locals().get("foreground_level", 0.0)), "local_background_y": float(locals().get("background_level", 0.0)), "local_gamma": float(local_gamma or 0.0), "_p7_mask": keep, "_p7_weight": weight}
    except (KeyError, TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return _identity(rgb, f"invalid_p7_text_edge_y_curve:{error}")


def _p10_landweber_luma(y: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Run bounded Landweber deblur without quality rollback.

    Input: Finite HxW BT.601 Y plane in [0,1] and deblur parameters.
    Output: The first finite beta candidate (normally beta=1) and telemetry for
      every finite beta candidate, including private arrays for stage evidence.
      Invalid inputs or parameters raise instead of returning an identity image.
    Key Steps: Apply a reflected Gaussian forward operator, take six bounded
      Landweber updates with a Tikhonov pull toward the observation, evaluate
      every configured beta, and retain all finite candidates. Quality and
      reliability measurements are recorded but never choose identity.
    Key Params: ``sigma`` controls the PSF, ``iterations`` defaults to 6,
      ``lambda`` defaults to .08, ``step`` defaults to .8, and beta values
      ``[1,.5,.25]`` are independently observable deblur strengths, not RGB
      weights. Delta/change limits are telemetry thresholds only.
    State/Dependencies: OpenCV Gaussian filtering and numpy only; no GT,
      model, colour-plane, or persistent state is accessed.
    """
    original = np.asarray(y, dtype=np.float32)
    sigma = float(params.get("sigma", 1.2))
    iterations = int(params.get("iterations", 6))
    regularization = float(params.get("lambda", params.get("regularization", 0.08)))
    step = float(params.get("step", 0.8))
    raw_betas = params.get("beta_schedule", params.get("betas", [params.get("beta", 1.0), 0.5, 0.25]))
    if not isinstance(raw_betas, (list, tuple)):
        raw_betas = [raw_betas]
    betas = [float(value) for value in raw_betas]
    max_delta = float(params.get("max_deblur_delta", 0.25))
    max_changed_fraction = float(params.get("max_deblur_changed_fraction", 0.80))
    if original.ndim != 2 or not np.isfinite(original).all():
        raise ValueError("Landweber input must be a finite HxW plane")
    if not (sigma > 0.0 and 1 <= iterations <= 12 and regularization >= 0.0 and 0.0 < step <= 1.5):
        raise ValueError("invalid Landweber sigma/iterations/lambda/step")
    if not betas or any(not math.isfinite(beta) or not 0.0 < beta <= 1.0 for beta in betas):
        raise ValueError("Landweber beta schedule must contain values in (0,1]")
    if any(betas[index + 1] > betas[index] for index in range(len(betas) - 1)):
        raise ValueError("Landweber beta schedule must be non-increasing")
    if not (math.isfinite(max_delta) and max_delta > 0.0 and 0.0 <= max_changed_fraction <= 1.0):
        raise ValueError("invalid Landweber telemetry limits")
    observed = np.clip(original, 0.0, 1.0).astype(np.float32)
    estimate = observed.copy()
    initial_blur = cv2.GaussianBlur(observed, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    initial_energy = float(np.mean(np.abs(observed - initial_blur)))
    for _ in range(iterations):
        reblurred = cv2.GaussianBlur(estimate, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        residual = observed - reblurred
        estimate = estimate + step * (residual - regularization * (estimate - observed))
        estimate = np.clip(estimate, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(estimate).all():
        raise ValueError("Landweber estimate is non-finite")
    raw_delta = estimate - observed
    attempts: list[dict[str, Any]] = []
    beta_outputs: list[dict[str, Any]] = []
    for beta in betas:
        candidate = np.clip(observed + beta * raw_delta, 0.0, 1.0).astype(np.float32)
        delta = candidate - observed
        changed_fraction = float(np.mean(np.abs(delta) > 1.0e-7))
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        finite = bool(np.isfinite(candidate).all())
        if not finite:
            raise ValueError(f"Landweber beta {beta:g} produced a non-finite candidate")
        exceeds_delta = bool(max_abs > max_delta)
        exceeds_changed_fraction = bool(changed_fraction > max_changed_fraction)
        attempts.append({
            "beta": beta,
            "status": "computed_with_telemetry_warning" if exceeds_delta or exceeds_changed_fraction else "computed",
            "finite": finite,
            "max_abs_delta": max_abs,
            "changed_fraction": changed_fraction,
            "max_delta_limit": max_delta,
            "changed_fraction_limit": max_changed_fraction,
            "max_delta_exceeded": exceeds_delta,
            "changed_fraction_exceeded": exceeds_changed_fraction,
            "reliable": not (exceeds_delta or exceeds_changed_fraction),
        })
        beta_outputs.append({"beta": beta, "output": candidate})
    output = beta_outputs[0]["output"]
    output_delta = output - observed
    return output, {
        "fallback": False,
        "fallback_reason": None,
        "deblur_stage_status": "computed",
        "deblur_sigma": sigma,
        "deblur_iterations": iterations,
        "deblur_lambda": regularization,
        "deblur_step": step,
        "deblur_beta_requested": betas[0],
        "deblur_beta_used": betas[0],
        "deblur_beta_fallback": False,
        "deblur_beta_selection_status": "first_finite_candidate",
        "deblur_beta_attempts": attempts,
        "deblur_input_blur_energy": initial_energy,
        "deblur_mean_abs_delta": float(np.mean(np.abs(output_delta))),
        "deblur_p95_abs_delta": float(np.percentile(np.abs(output_delta), 95.0)) if output_delta.size else 0.0,
        "deblur_max_abs_delta": float(np.max(np.abs(output_delta))) if output_delta.size else 0.0,
        "deblur_changed_fraction": float(np.mean(np.abs(output_delta) > 1.0e-7)),
        "_p10_deblur_beta_outputs": beta_outputs,
    }


def _p10_segment_text(y: np.ndarray, params: dict[str, Any], valid_mask: np.ndarray | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """@luna-comment Build a conservative text mask for P10 boundary sampling.

    Input: HxW luma plane and optional mask-confidence parameters.
    Output: Binary foreground mask or None plus polarity, confidence, and
      rejection telemetry; the mask is never inferred from GT.
    Key Steps: Blur and quantise Y for Otsu, infer polarity from the ROI border,
      remove tiny connected components, and enforce separation/coverage rules.
    Key Params: Segmentation sigma .6, border-majority threshold .60, median
      gap .12, Otsu separation .45, and foreground coverage [.01,.65].
    State/Dependencies: OpenCV thresholding/components only; no file writes.
    """
    try:
        if y.ndim != 2 or not np.isfinite(y).all() or y.size == 0:
            return None, {"segmentation_accepted": False, "segmentation_reason": "invalid_luma"}
        sigma = float(params.get("mask_sigma", 0.6))
        border_share = float(params.get("mask_border_share", 0.60))
        if not sigma > 0.0 or not 0.5 <= border_share <= 1.0:
            raise ValueError("invalid P10 mask parameters")
        valid = np.ones_like(y, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if valid.shape != y.shape or int(valid.sum()) < max(256, round(y.size * 0.01)):
            return None, {"segmentation_accepted": False, "segmentation_reason": "insufficient_valid_support"}
        if valid_mask is None:
            smooth = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
            encoded_valid = np.ones_like(valid, dtype=bool)
        else:
            numerator = cv2.GaussianBlur(y * valid.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
            denominator = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
            encoded_valid = valid & (denominator > 1.0e-6)
            smooth = np.zeros_like(y, dtype=np.float32)
            smooth[encoded_valid] = numerator[encoded_valid] / denominator[encoded_valid]
        encoded = np.zeros_like(y, dtype=np.uint8)
        encoded[encoded_valid] = np.clip(np.rint(smooth[encoded_valid] * 255.0), 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(encoded[encoded_valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_mask = (encoded > threshold) & encoded_valid
        border_width = max(1, round(min(y.shape) * 0.03))
        border = np.zeros_like(high_mask, dtype=bool)
        border[:border_width] = True
        border[-border_width:] = True
        border[:, :border_width] = True
        border[:, -border_width:] = True
        border &= encoded_valid
        high_border = float(high_mask[border].mean()) if border.any() else 0.5
        high_roi = float(high_mask[encoded_valid].mean())
        if high_border >= border_share:
            foreground = (~high_mask) & encoded_valid
            polarity = "dark_text_on_light"
        elif 1.0 - high_border >= border_share:
            foreground = high_mask & encoded_valid
            polarity = "light_text_on_dark"
        elif high_roi >= border_share:
            foreground = (~high_mask) & encoded_valid
            polarity = "dark_text_on_light_roi_majority"
        elif 1.0 - high_roi >= border_share:
            foreground = high_mask & encoded_valid
            polarity = "light_text_on_dark_roi_majority"
        else:
            return None, {"segmentation_accepted": False, "segmentation_reason": "ambiguous_polarity", "otsu_threshold": float(threshold) / 255.0}
        original_area = int(foreground.sum())
        min_area = max(2, min(8, round(y.shape[0] * y.shape[1] * 0.00002)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(foreground, dtype=bool)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                keep |= labels == label
        removed_fraction = 1.0 - float(keep.sum()) / max(original_area, 1)
        low_values = y[encoded_valid & ~high_mask]
        high_values = y[high_mask]
        total_variance = float(np.var(y[encoded_valid]))
        separation = 0.0 if total_variance <= 1.0e-12 or not low_values.size or not high_values.size else float((low_values.size / y.size) * (high_values.size / y.size) * (float(low_values.mean()) - float(high_values.mean())) ** 2 / total_variance)
        median_gap = abs(float(np.median(low_values)) - float(np.median(high_values))) if low_values.size and high_values.size else 0.0
        foreground_fraction = float(keep.mean())
        accepted = bool(0.01 <= foreground_fraction <= 0.65 and median_gap >= 0.12 and separation >= 0.45 and removed_fraction <= 0.15)
        meta = {
            "segmentation_accepted": accepted,
            "segmentation_reason": None if accepted else "low_confidence_text_mask",
            "polarity": polarity,
            "otsu_threshold": float(threshold) / 255.0,
            "foreground_fraction": foreground_fraction,
            "median_gap": median_gap,
            "otsu_separation": separation,
            "removed_foreground_fraction": removed_fraction,
            "_p10_encoded": encoded,
            "_p10_low_class": ~high_mask,
            "_p10_high_class": high_mask,
            "_p10_valid": encoded_valid,
        }
        return (keep if accepted else None), meta
    except (TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return None, {"segmentation_accepted": False, "segmentation_reason": f"invalid_p10_text_mask:{error}"}


def _p10_isotonic_nondecreasing(values: np.ndarray) -> np.ndarray:
    """@luna-comment Project a sampled transition onto a nondecreasing profile.

    Input: One-dimensional finite profile values ordered from background to
      foreground and interpreted as normalised transition coordinates.
    Output: Same-length float32 profile with the least-squares nondecreasing
      isotonic projection.
    Key Steps: Pool adjacent violating blocks with their weighted means, then
      expand the block levels in the original sample order.
    Key Params: Equal unit weights make the projection deterministic and avoid
      letting oversampled platform points dominate the cleanup.
    State/Dependencies: Numpy only; no global state or image mutation.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("isotonic input must be finite one-dimensional values")
    levels: list[float] = []
    counts: list[int] = []
    for value in values.astype(np.float64):
        levels.append(float(value))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total = counts[-2] + counts[-1]
            merged = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / total
            levels[-2:] = [merged]
            counts[-2:] = [total]
    projected = np.empty_like(values, dtype=np.float32)
    cursor = 0
    for level, count in zip(levels, counts):
        projected[cursor : cursor + count] = float(level)
        cursor += count
    return projected


def _p10_crossing_offset(values: np.ndarray, offsets: np.ndarray, level: float) -> float | None:
    """@luna-comment Locate the first subpixel crossing of a profile level.

    Input: Finite sampled transition values, matching ordered offsets, and a
      level in [0,1].
    Output: Linear-interpolated offset or None when the level is not crossed.
    Key Steps: Find the first adjacent bracket and interpolate with a safe
      zero-denominator midpoint fallback.
    Key Params: First crossing follows the background-to-foreground normal and
      keeps width/crossing telemetry reproducible under ringing.
    State/Dependencies: Numpy only; no image or global state.
    """
    for index in range(max(0, values.size - 1)):
        left = float(values[index])
        right = float(values[index + 1])
        if (left - level) == 0.0:
            return float(offsets[index])
        if (left - level) * (right - level) <= 0.0:
            denominator = right - left
            fraction = 0.5 if abs(denominator) <= 1.0e-12 else (level - left) / denominator
            return float(offsets[index] + np.clip(fraction, 0.0, 1.0) * (offsets[index + 1] - offsets[index]))
    if values.size and float(values[-1]) == level:
        return float(offsets[-1])
    return None


def _p10_stable_platform(
    values: np.ndarray,
    offsets: np.ndarray,
    side: str,
    params: dict[str, Any],
) -> tuple[float, float, tuple[float, float]] | None:
    """@luna-comment Find the nearest low-variation platform on one edge side.

    Input: Normal-sampled Y profile, ordered offsets, side ``outside`` or
      ``inside``, and platform tolerance parameters.
    Output: Platform median, P90-P10 spread, and the accepted offset interval,
      or None if no stable run of at least two pixels exists.
    Key Steps: Scan contiguous windows outward from the boundary and accept the
      first window whose spread is below absolute/relative contrast tolerance.
    Key Params: Minimum run is 2 px, scan limit is 10 px, absolute tolerance is
      2/255, relative tolerance is .03, and contrast is estimated robustly.
    State/Dependencies: Numpy quantiles only; no writes or external state.
    """
    if side not in {"outside", "inside"} or values.ndim != 1 or values.size != offsets.size:
        raise ValueError("invalid P10 platform profile")
    step = float(params.get("sample_step", 0.5))
    min_run = float(params.get("min_platform_run", 2.0))
    scan_limit = float(params.get("platform_scan_limit", 10.0))
    abs_tolerance = float(params.get("platform_tolerance_abs", 2.0 / 255.0))
    rel_tolerance = float(params.get("platform_tolerance_rel", 0.03))
    robust_contrast = float(np.quantile(values, 0.90) - np.quantile(values, 0.10))
    tolerance = max(abs_tolerance, rel_tolerance * robust_contrast)
    min_samples = max(2, int(math.ceil(min_run / max(step, 1.0e-6))))
    eligible = np.flatnonzero((offsets <= -min_run) & (offsets >= -scan_limit)) if side == "outside" else np.flatnonzero((offsets >= min_run) & (offsets <= scan_limit))
    if eligible.size < min_samples:
        return None
    ordered = eligible[::-1] if side == "outside" else eligible
    for start in range(0, ordered.size - min_samples + 1):
        indices = ordered[start : start + min_samples]
        window = values[indices]
        spread = float(np.quantile(window, 0.90) - np.quantile(window, 0.10))
        if spread <= tolerance:
            return float(np.median(window)), spread, (float(offsets[indices].min()), float(offsets[indices].max()))
    return None


def _p10_measure_profiles(y: np.ndarray, mask: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """@luna-comment Measure reliable per-boundary normal profiles for P10 cleanup.

    Input: Deblurred Y plane, accepted binary text mask, and frozen profile
      geometry/platform parameters.
    Output: Profile records with foreground/background anchors, monotone target
      coordinates, widths/crossings, and aggregate reliability telemetry.
    Key Steps: Derive signed-distance normals, sample each normal with
      ``cv2.remap`` from -16 to +16 px at .5 px, reject missing/returning
      platforms, and prepare monotone or contract-.90 targets.
    Key Params: Half width 16 px, step .5 px, seed stride 3 px, platform run
      >=2 px, contrast >=.12, and cleanup modes none/monotone/contract_090.
    State/Dependencies: OpenCV distance transforms/remapping and numpy only;
      no GT, colour planes, or persistent state.
    """
    try:
        if y.ndim != 2 or mask.shape != y.shape or not np.isfinite(y).all():
            raise ValueError("P10 profile inputs have incompatible shapes")
        half_width = float(params.get("profile_half_width", 16.0))
        step = float(params.get("sample_step", 0.5))
        seed_stride = max(1, int(params.get("boundary_stride", 3)))
        min_contrast = float(params.get("min_profile_contrast", 0.12))
        cleanup = str(params.get("cleanup", "monotone")).lower()
        contract = float(params.get("contract_factor", 0.90))
        max_profiles = max(1, int(params.get("max_profiles", 128)))
        if not (half_width >= 4.0 and step > 0.0 and min_contrast > 0.0 and cleanup in {"none", "monotone", "contract_090"}):
            raise ValueError("invalid P10 profile parameters")
        if cleanup == "contract_090" and not 0.0 < contract < 1.0:
            raise ValueError("P10 contract factor must be in (0,1)")
        offsets = np.arange(-half_width, half_width + step * 0.5, step, dtype=np.float32)
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        signed = inside - outside
        grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
        coordinates = np.argwhere(boundary)
        coordinates = coordinates[::seed_stride]
        if coordinates.shape[0] > max_profiles:
            coordinates = coordinates[:max_profiles]
        profiles: list[dict[str, Any]] = []
        rejected = 0
        for row, column in coordinates:
            magnitude = float(norm[row, column])
            if magnitude <= 1.0e-5:
                rejected += 1
                continue
            normal_x = float(grad_x[row, column] / magnitude)
            normal_y = float(grad_y[row, column] / magnitude)
            map_x = (float(column) + offsets * normal_x).astype(np.float32).reshape(1, -1)
            map_y = (float(row) + offsets * normal_y).astype(np.float32).reshape(1, -1)
            if float(map_x.min()) < 1.0 or float(map_x.max()) > y.shape[1] - 2.0 or float(map_y.min()) < 1.0 or float(map_y.max()) > y.shape[0] - 2.0:
                rejected += 1
                continue
            values = cv2.remap(y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1).astype(np.float32)
            mask_values = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101).reshape(-1) >= 0.5
            outside_idx = offsets <= -2.0
            inside_idx = offsets >= 2.0
            if outside_idx.sum() < 2 or inside_idx.sum() < 2 or float((~mask_values[outside_idx]).mean()) < 0.75 or float(mask_values[inside_idx].mean()) < 0.75:
                rejected += 1
                continue
            background = _p10_stable_platform(values, offsets, "outside", params)
            foreground = _p10_stable_platform(values, offsets, "inside", params)
            if background is None or foreground is None:
                rejected += 1
                continue
            background_level, background_spread, background_interval = background
            foreground_level, foreground_spread, foreground_interval = foreground
            contrast = abs(foreground_level - background_level)
            if not math.isfinite(contrast) or contrast < min_contrast:
                rejected += 1
                continue
            oriented = np.clip((values - background_level) / (foreground_level - background_level), 0.0, 1.0).astype(np.float32)
            projected = np.clip(_p10_isotonic_nondecreasing(oriented), 0.0, 1.0)
            target_t = projected.copy()
            if cleanup == "contract_090":
                target_t = np.clip(0.5 + (projected - 0.5) / contract, 0.0, 1.0).astype(np.float32)
            target_values = (background_level + (foreground_level - background_level) * target_t).astype(np.float32)
            raw_reverse_steps = int(np.sum(np.diff(oriented) < -1.0 / 255.0))
            projected_reverse_steps = int(np.sum(np.diff(projected) < -1.0 / 255.0))
            before_10 = _p10_crossing_offset(oriented, offsets, 0.10)
            before_90 = _p10_crossing_offset(oriented, offsets, 0.90)
            after_10 = _p10_crossing_offset(target_t, offsets, 0.10)
            after_90 = _p10_crossing_offset(target_t, offsets, 0.90)
            before_01 = _p10_crossing_offset(oriented, offsets, 0.01)
            before_99 = _p10_crossing_offset(oriented, offsets, 0.99)
            after_01 = _p10_crossing_offset(target_t, offsets, 0.01)
            after_99 = _p10_crossing_offset(target_t, offsets, 0.99)
            profile = {
                "row": int(row),
                "column": int(column),
                "normal_x": normal_x,
                "normal_y": normal_y,
                "background_y": background_level,
                "foreground_y": foreground_level,
                "contrast": contrast,
                "background_spread": background_spread,
                "foreground_spread": foreground_spread,
                "background_interval": list(background_interval),
                "foreground_interval": list(foreground_interval),
                "raw_reverse_steps": raw_reverse_steps,
                "projected_reverse_steps": projected_reverse_steps,
                "before_10_90_width": None if before_10 is None or before_90 is None else float(before_90 - before_10),
                "after_10_90_width": None if after_10 is None or after_90 is None else float(after_90 - after_10),
                "before_1_99_width": None if before_01 is None or before_99 is None else float(before_99 - before_01),
                "after_1_99_width": None if after_01 is None or after_99 is None else float(after_99 - after_01),
                "before_half_crossing": _p10_crossing_offset(oriented, offsets, 0.5),
                "after_half_crossing": _p10_crossing_offset(target_t, offsets, 0.5),
                "offsets": [float(value) for value in offsets],
                "values": [float(value) for value in values],
                "target_values": [float(value) for value in target_values],
                "_map_x": map_x.reshape(-1),
                "_map_y": map_y.reshape(-1),
                "_delta": target_values - values if cleanup != "none" else np.zeros_like(values),
            }
            profiles.append(profile)
        widths_before = [float(profile["before_10_90_width"]) for profile in profiles if profile["before_10_90_width"] is not None]
        widths_after = [float(profile["after_10_90_width"]) for profile in profiles if profile["after_10_90_width"] is not None]
        widths_1_99_before = [float(profile["before_1_99_width"]) for profile in profiles if profile["before_1_99_width"] is not None]
        widths_1_99_after = [float(profile["after_1_99_width"]) for profile in profiles if profile["after_1_99_width"] is not None]
        crossing_shifts = [abs(float(profile["after_half_crossing"]) - float(profile["before_half_crossing"])) for profile in profiles if profile["before_half_crossing"] is not None and profile["after_half_crossing"] is not None]
        return {
            "profiles": profiles,
            "profile_count": len(profiles),
            "profile_rejected_count": rejected,
            "profile_accept_fraction": float(len(profiles) / max(coordinates.shape[0], 1)),
            "profile_10_90_width_before_median": float(np.median(widths_before)) if widths_before else None,
            "profile_10_90_width_after_median": float(np.median(widths_after)) if widths_after else None,
            "profile_1_99_width_before_median": float(np.median(widths_1_99_before)) if widths_1_99_before else None,
            "profile_1_99_width_after_median": float(np.median(widths_1_99_after)) if widths_1_99_after else None,
            "profile_half_crossing_shift_median": float(np.median(crossing_shifts)) if crossing_shifts else None,
            "profile_reverse_steps_before": int(sum(int(profile["raw_reverse_steps"]) for profile in profiles)),
            "profile_reverse_steps_after": int(sum(int(profile["projected_reverse_steps"]) for profile in profiles)),
            "profile_cleanup_mode": cleanup,
            "profile_contract_factor": contract if cleanup == "contract_090" else None,
        }
    except (TypeError, ValueError, FloatingPointError, cv2.error) as error:
        return {"profiles": [], "profile_count": 0, "profile_rejected_count": 0, "profile_accept_fraction": 0.0, "profile_measurement_error": str(error), "profile_cleanup_mode": str(params.get("cleanup", "monotone"))}


def _p10_apply_profile_deltas(shape: tuple[int, int], profiles: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """@luna-comment Scatter reliable normal-profile corrections into a luma map.

    Input: HxW target shape and P10 profiles containing subpixel maps/deltas.
    Output: Weighted mean delta map and support weights, both HxW float32.
    Key Steps: Bilinearly scatter each normal sample to its four neighbours and
      divide only where profile support exists, preventing hard edge seams.
    Key Params: Profile confidence is currently unit-weight after platform
      validation; overlapping normals are averaged deterministically.
    State/Dependencies: Numpy accumulation only; input profile arrays remain
      read-only and no RGB/chroma state is touched.
    """
    height, width = shape
    delta_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    for profile in profiles:
        map_x = np.asarray(profile["_map_x"], dtype=np.float32)
        map_y = np.asarray(profile["_map_y"], dtype=np.float32)
        delta = np.asarray(profile["_delta"], dtype=np.float32)
        if map_x.shape != map_y.shape or map_x.shape != delta.shape:
            continue
        x0 = np.floor(map_x).astype(np.int32)
        y0 = np.floor(map_y).astype(np.int32)
        wx = map_x - x0
        wy = map_y - y0
        for dx, dy, factor in ((0, 0, (1.0 - wx) * (1.0 - wy)), (1, 0, wx * (1.0 - wy)), (0, 1, (1.0 - wx) * wy), (1, 1, wx * wy)):
            xx = x0 + dx
            yy = y0 + dy
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height) & np.isfinite(delta) & np.isfinite(factor)
            if not bool(valid.any()):
                continue
            np.add.at(delta_sum, (yy[valid], xx[valid]), (delta * factor)[valid])
            np.add.at(weight_sum, (yy[valid], xx[valid]), factor[valid])
    delta_map = np.zeros(shape, dtype=np.float32)
    supported = weight_sum > 1.0e-6
    delta_map[supported] = delta_sum[supported] / weight_sum[supported]
    return delta_map, weight_sum


def _p10_reconstruct_rgb_from_y(
    rgb: np.ndarray, y_input: np.ndarray, y_target: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """@luna-comment Reconstruct RGB from a target Y while preserving Cb/Cr.

    Input: Finite HxWx3 RGB tensor, its BT.601 Y plane, and a finite target Y
      plane with the same shape.
    Output: Gamut-constrained RGB tensor plus requested/applied delta and
      chroma telemetry; no quality decision or identity substitution occurs.
    Key Steps: Form one shared luma delta, clip it to each pixel's available
      RGB interval, reconstruct all channels equally, and measure Cb/Cr error.
    Key Params: BT.601 coefficients and the full-range Cb/Cr denominators are
      fixed by the Stage-A direct-RGB contract.
    State/Dependencies: Numpy only; no GT, model, colour-plane, or global state.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or y_input.shape != rgb.shape[:2] or y_target.shape != y_input.shape:
        raise ValueError("P10 RGB/Y reconstruction shapes are incompatible")
    if not np.isfinite(rgb).all() or not np.isfinite(y_input).all() or not np.isfinite(y_target).all():
        raise ValueError("P10 RGB/Y reconstruction inputs must be finite")
    requested = y_target - y_input
    lower = -np.min(rgb, axis=2)
    upper = 1.0 - np.max(rgb, axis=2)
    applied = np.minimum(np.maximum(requested, lower), upper).astype(np.float32)
    output = (rgb + applied[..., None]).astype(np.float32)
    if not np.isfinite(output).all() or float(output.min()) < -2.0e-6 or float(output.max()) > 1.0 + 2.0e-6:
        raise ValueError("P10 RGB reconstruction is non-finite or outside gamut")
    y_out = _rgb_to_luma(output)
    cb_in = (rgb[..., 2] - y_input) / 1.772
    cr_in = (rgb[..., 0] - y_input) / 1.402
    chroma_error = max(
        float(np.max(np.abs((output[..., 2] - y_out) / 1.772 - cb_in))),
        float(np.max(np.abs((output[..., 0] - y_out) / 1.402 - cr_in))),
    )
    return output, {
        "clipping_fraction": 0.0,
        "mean_abs_requested_delta": float(np.mean(np.abs(requested))),
        "mean_abs_applied_delta": float(np.mean(np.abs(applied))),
        "p95_abs_applied_delta": float(np.percentile(np.abs(applied), 95.0)) if applied.size else 0.0,
        "fraction_abs_delta_ge_4_255": float(np.mean(np.abs(applied) >= 4.0 / 255.0)),
        "gamut_limited_fraction": float(np.mean(np.abs(requested - applied) > 1.0e-7)),
        "mean_abs_requested_minus_applied": float(np.mean(np.abs(requested - applied))),
        "chroma_max_abs_error": chroma_error,
    }


def _p10_text_edge_y_curve(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Run P10 deblur-first Y cleanup without quality rollback.

    Input: Finite HxWx3 RGB ROI and a P10 candidate containing Landweber and
      cleanup settings.
    Output: The first finite beta's cleanup RGB result, plus real deblur and
      cleanup RGB arrays for every beta and status/telemetry for each stage.
      Invalid or non-finite data raise instead of returning an identity image.
    Key Steps: Convert RGB to Y, compute all beta deblur candidates, segment
      text, measure and clean each candidate independently when requested,
      scatter profile corrections, then constrain each shared RGB delta to the
      available gamut interval.
    Key Params: sigma {0.8,1.2,1.6}, six iterations, lambda .08, step .8,
      beta schedule [1,.5,.25], profile support [-16,+16] px, and cleanup
      modes ``none``, ``monotone``, and ``contract_090``.
    State/Dependencies: CPU numpy/OpenCV only; no GT, model, U/V, or global
      mutable state. Missing masks/profiles are stage statuses, not input
      fallbacks; quality and reliability telemetry never suppresses output.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("invalid P10 RGB input")
    y_input = _rgb_to_luma(rgb)
    deblur_y, deblur_meta = _p10_landweber_luma(y_input, params)
    beta_outputs = deblur_meta.get("_p10_deblur_beta_outputs")
    attempts = deblur_meta.get("deblur_beta_attempts")
    if not isinstance(beta_outputs, list) or not beta_outputs or not isinstance(attempts, list) or len(beta_outputs) != len(attempts):
        raise ValueError("P10 deblur did not produce a complete beta attempt set")
    mask, mask_meta = _p10_segment_text(y_input, params)
    cleanup = str(params.get("cleanup", "monotone")).lower()
    stage_outputs: list[dict[str, Any]] = []
    stage_statuses: list[dict[str, Any]] = []
    selected_profile_meta: dict[str, Any] | None = None
    selected_cleanup_rgb: np.ndarray | None = None
    selected_cleanup_telemetry: dict[str, float] | None = None
    for attempt, beta_record in zip(attempts, beta_outputs):
        beta = float(beta_record["beta"])
        beta_y = np.asarray(beta_record["output"], dtype=np.float32)
        if beta_y.shape != y_input.shape or not np.isfinite(beta_y).all():
            raise ValueError(f"P10 beta {beta:g} output is invalid")
        deblur_rgb, deblur_rgb_telemetry = _p10_reconstruct_rgb_from_y(rgb, y_input, beta_y)
        profile_meta: dict[str, Any] = {
            "profiles": [],
            "profile_count": 0,
            "profile_rejected_count": 0,
            "profile_accept_fraction": 0.0,
            "profile_cleanup_mode": cleanup,
        }
        profile_status = "not_requested" if cleanup == "none" else "skipped_no_mask"
        if cleanup == "none":
            cleanup_y = beta_y.copy()
        elif mask is None:
            cleanup_y = beta_y.copy()
        else:
            profile_meta = _p10_measure_profiles(beta_y, mask, params)
            if profile_meta.get("profiles"):
                correction, support = _p10_apply_profile_deltas(y_input.shape, profile_meta["profiles"])
                cleanup_y = np.clip(beta_y + correction, 0.0, 1.0).astype(np.float32)
                profile_meta["profile_support_fraction"] = float(np.mean(support > 1.0e-6))
                profile_meta["profile_correction_mean_abs"] = float(np.mean(np.abs(correction)))
                profile_status = "computed"
            else:
                cleanup_y = beta_y.copy()
                profile_status = "skipped_no_profiles" if not profile_meta.get("profile_measurement_error") else "measurement_error"
                profile_meta["profile_support_fraction"] = 0.0
                profile_meta["profile_correction_mean_abs"] = 0.0
        cleanup_rgb, cleanup_telemetry = _p10_reconstruct_rgb_from_y(rgb, y_input, cleanup_y)
        stage_outputs.append({
            "beta": beta,
            "deblur_status": str(attempt.get("status", "computed")),
            "cleanup_status": profile_status if cleanup != "none" else "bypassed",
            "deblur_output": deblur_rgb,
            "cleanup_output": cleanup_rgb,
            "profile_meta": profile_meta,
        })
        stage_statuses.append({
            "beta": beta,
            "deblur_status": str(attempt.get("status", "computed")),
            "cleanup_status": profile_status if cleanup != "none" else "bypassed",
            "profile_count": int(profile_meta.get("profile_count", 0)),
        })
        if selected_cleanup_rgb is None:
            selected_cleanup_rgb = cleanup_rgb
            selected_cleanup_telemetry = cleanup_telemetry
            selected_profile_meta = profile_meta
    if selected_cleanup_rgb is None or selected_cleanup_telemetry is None or selected_profile_meta is None:
        raise ValueError("P10 produced no finite beta stage")
    selected_stage = stage_outputs[0]
    meta: dict[str, Any] = {
        "fallback": False,
        "fallback_reason": None,
        "cleanup_fallback_to_deblur": False,
        "cleanup_mode": cleanup,
        "cleanup_stage_status": str(selected_stage["cleanup_status"]),
        "p10_stage_status": "computed",
        "p10_beta_stage_statuses": stage_statuses,
        "p10_beta_stage_count": len(stage_statuses),
        "outside_band_max_abs_rgb_delta": 0.0,
        "_p10_mask": mask if mask is not None else np.zeros_like(y_input, dtype=bool),
        "_p10_profiles": selected_profile_meta.get("profiles", []),
        "_p10_beta_stage_outputs": stage_outputs,
    }
    meta.update(selected_cleanup_telemetry)
    meta.update(deblur_meta)
    meta.update(mask_meta)
    for key, value in selected_profile_meta.items():
        if key != "profiles":
            meta[key] = value
    return selected_cleanup_rgb, meta


def _p11_gaussian_cdf(offsets: np.ndarray, sigma: float, center: float) -> np.ndarray:
    """@luna-comment Evaluate a finite Gaussian step response on normal offsets.

    Input: One-dimensional finite offsets in pixels, positive sigma, and edge
      centre in the same pixel coordinate system.
    Output: Float32 values in [0,1] for Phi((offset-center)/sigma).
    Key Steps: Scale offsets by sqrt(2) and evaluate the standard-library
      error function without adding a SciPy dependency.
    Key Params: Sigma is the equivalent isotropic blur width; center accounts
      for subpixel placement of the sampled boundary.
    State/Dependencies: Python math and numpy only.
    """
    if sigma <= 0.0 or not math.isfinite(sigma) or not math.isfinite(center):
        raise ValueError("Gaussian CDF requires finite positive sigma and center")
    values = np.asarray(offsets, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Gaussian CDF offsets must be finite and one-dimensional")
    scale = math.sqrt(2.0) * sigma
    return np.fromiter(
        (0.5 * (1.0 + math.erf((float(value) - center) / scale)) for value in values),
        dtype=np.float32,
        count=values.size,
    )


def _p11_fit_gaussian_edge_profile(profile: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | None:
    """@luna-comment Fit one text-edge normal to an equivalent Gaussian step.

    Input: P10-compatible normal profile with offsets, sampled Y values, stable
      foreground/background platform levels, and normal direction metadata.
    Output: JSON-safe fit record containing sigma, centre, robust residuals,
      bound-hit state, and straight-profile diagnostics, or None if rejected.
    Key Steps: Orient the profile background-to-foreground, reject weak or
      strongly reversing edges, then grid-search sigma and subpixel centre with
      a Huber loss for a+b*Phi.
    Key Params: Sigma bounds default to [.3,4.0] px, centre search is +/-4 px,
      contrast floor is .12, and the Huber transition defaults to .05 luma.
    State/Dependencies: Numpy/math only; the profile and source image remain
      read-only.
    """
    try:
        offsets = np.asarray(profile["offsets"], dtype=np.float32)
        values = np.asarray(profile["values"], dtype=np.float32)
        background = float(profile["background_y"])
        foreground = float(profile["foreground_y"])
        if offsets.ndim != 1 or values.shape != offsets.shape or not np.isfinite(offsets).all() or not np.isfinite(values).all():
            return None
        contrast = abs(foreground - background)
        min_contrast = float(params.get("kernel_min_contrast", 0.12))
        if not math.isfinite(contrast) or contrast < min_contrast:
            return None
        oriented = np.clip((values - background) / (foreground - background), 0.0, 1.0)
        reverse_steps = int(np.sum(np.diff(oriented) < -1.0 / 255.0))
        max_reverse = int(params.get("kernel_max_reverse_steps", 8))
        if reverse_steps > max_reverse:
            return None
        sigma_min = float(params.get("kernel_sigma_min", 0.3))
        sigma_max = float(params.get("kernel_sigma_max", 4.0))
        centre_limit = float(params.get("kernel_center_limit", 4.0))
        centre_step = float(params.get("kernel_center_step", 0.25))
        sigma_step = float(params.get("kernel_sigma_step", 0.05))
        huber_delta = float(params.get("kernel_huber_delta", 0.05))
        if not (0.0 < sigma_min <= sigma_max and centre_limit > 0.0 and centre_step > 0.0 and sigma_step > 0.0 and huber_delta > 0.0):
            raise ValueError("invalid P11 Gaussian-fit bounds")
        sigma_grid = np.arange(sigma_min, sigma_max + sigma_step * 0.5, sigma_step, dtype=np.float32)
        centre_grid = np.arange(-centre_limit, centre_limit + centre_step * 0.5, centre_step, dtype=np.float32)
        if sigma_grid.size == 0 or centre_grid.size == 0:
            raise ValueError("empty P11 Gaussian-fit grid")
        best_loss = float("inf")
        best_sigma = None
        best_centre = None
        best_prediction = None
        for sigma in sigma_grid:
            for centre in centre_grid:
                cdf = _p11_gaussian_cdf(offsets, float(sigma), float(centre))
                prediction = background + (foreground - background) * cdf
                residual = prediction.astype(np.float32) - values
                absolute = np.abs(residual)
                loss = np.where(
                    absolute <= huber_delta,
                    0.5 * np.square(absolute),
                    huber_delta * (absolute - 0.5 * huber_delta),
                )
                mean_loss = float(np.mean(loss))
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_sigma = float(sigma)
                    best_centre = float(centre)
                    best_prediction = prediction
        if best_sigma is None or best_prediction is None:
            return None
        residual = best_prediction.astype(np.float32) - values
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        median_abs = float(np.median(np.abs(residual)))
        max_fit_rmse = float(params.get("kernel_max_fit_rmse", 0.08))
        if not math.isfinite(rmse) or rmse > max_fit_rmse:
            return None
        normal_x = float(profile.get("normal_x", 0.0))
        normal_y = float(profile.get("normal_y", 0.0))
        return {
            "row": int(profile.get("row", -1)),
            "column": int(profile.get("column", -1)),
            "normal_x": normal_x,
            "normal_y": normal_y,
            "contrast": contrast,
            "sigma": best_sigma,
            "center": best_centre,
            "fit_rmse": rmse,
            "fit_median_abs_error": median_abs,
            "fit_huber_loss": best_loss,
            "reverse_steps": reverse_steps,
            "sigma_at_lower_bound": bool(abs(best_sigma - sigma_min) <= sigma_step * 0.51),
            "sigma_at_upper_bound": bool(abs(best_sigma - sigma_max) <= sigma_step * 0.51),
            "background_y": background,
            "foreground_y": foreground,
            "offsets": [float(value) for value in offsets],
            "values": [float(value) for value in values],
            "fit_values": [float(value) for value in best_prediction],
        }
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return None


def _p11_gaussian_psf(sigma: float) -> np.ndarray:
    """@luna-comment Build the normalized isotropic Gaussian PSF for P11.

    Input: Finite positive equivalent blur sigma in pixels.
    Output: Odd-sized float32 PSF covering approximately three sigma per side.
    Key Steps: Select a bounded odd support and evaluate the existing Gaussian
      kernel builder with a unit-sum normalization.
    Key Params: Support radius is ceil(3*sigma), balancing FFT cost and tail
      coverage.
    State/Dependencies: Reuses the pure numpy Gaussian kernel helper.
    """
    sigma = float(sigma)
    if not math.isfinite(sigma) or not 0.3 <= sigma <= 4.0:
        raise ValueError("P11 PSF sigma must be finite and within [0.3,4.0]")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    return _gaussian_kernel(2 * radius + 1, sigma)


def _p11_estimate_equivalent_kernel(
    y: np.ndarray, mask: np.ndarray, params: dict[str, Any]
) -> dict[str, Any]:
    """@luna-comment Estimate one isotropic Gaussian PSF from reliable text edges.

    Input: Finite HxW luma plane, accepted P10 text mask, and P11 fit bounds.
    Output: Kernel status, median sigma/MAD, direction and residual statistics,
      per-profile evidence, and a finite PSF when identifiable.
    Key Steps: Sample local normals with the existing platform detector, retain
      high-contrast near-monotone step fits, and robustly aggregate sigma by
      median while retaining bound-hit telemetry.
    Key Params: Profile geometry is the P10 .5 px/-16..16 px contract; sigma is
      bounded to [.3,4.0] px and fit RMSE is config-limited.
    State/Dependencies: Luma/mask arrays are read-only; no GT or file I/O.
    """
    if y.ndim != 2 or mask.shape != y.shape or not np.isfinite(y).all() or not mask.any():
        return {
            "kernel_status": "kernel_unidentifiable",
            "kernel_reason": "invalid_or_empty_text_mask",
            "kernel_sigma": None,
            "kernel_sigma_mad": None,
            "kernel_segment_count": 0,
            "kernel_profiles_considered": 0,
            "kernel_profiles": [],
            "kernel_direction_stats": {},
            "kernel_fit_residual_stats": {},
            "kernel_bound_hits": {"lower": 0, "upper": 0},
            "kernel_psf": None,
        }
    profile_params = dict(params)
    profile_params["cleanup"] = "none"
    measured = _p10_measure_profiles(y, mask, profile_params)
    candidates = measured.get("profiles", []) if isinstance(measured, dict) else []
    fits = []
    for profile in candidates:
        fit = _p11_fit_gaussian_edge_profile(profile, params)
        if fit is not None:
            fits.append(fit)
    sigma_values = np.asarray([float(item["sigma"]) for item in fits], dtype=np.float64)
    if sigma_values.size == 0:
        return {
            "kernel_status": "kernel_unidentifiable",
            "kernel_reason": "no_profile_passed_gaussian_fit",
            "kernel_sigma": None,
            "kernel_sigma_mad": None,
            "kernel_segment_count": 0,
            "kernel_profiles_considered": len(candidates),
            "kernel_profiles": fits,
            "kernel_direction_stats": {},
            "kernel_fit_residual_stats": {},
            "kernel_bound_hits": {"lower": 0, "upper": 0},
            "kernel_psf": None,
        }
    sigma = float(np.median(sigma_values))
    mad = float(np.median(np.abs(sigma_values - sigma)))
    normal_x = np.asarray([float(item["normal_x"]) for item in fits], dtype=np.float64)
    normal_y = np.asarray([float(item["normal_y"]) for item in fits], dtype=np.float64)
    angles = np.degrees(np.arctan2(normal_y, normal_x))
    horizontal_count = int(np.sum(np.abs(normal_x) >= np.abs(normal_y)))
    vertical_count = len(fits) - horizontal_count
    residuals = np.asarray([float(item["fit_rmse"]) for item in fits], dtype=np.float64)
    psf = _p11_gaussian_psf(sigma)
    return {
        "kernel_status": "computed",
        "kernel_reason": None,
        "kernel_sigma": sigma,
        "kernel_sigma_mad": mad,
        "kernel_segment_count": len(fits),
        "kernel_profiles_considered": len(candidates),
        "kernel_profiles": fits,
        "kernel_direction_stats": {
            "horizontal_normal_count": horizontal_count,
            "vertical_normal_count": vertical_count,
            "angle_mean_deg": float(np.mean(angles)),
            "angle_std_deg": float(np.std(angles)),
            "angle_min_deg": float(np.min(angles)),
            "angle_max_deg": float(np.max(angles)),
        },
        "kernel_fit_residual_stats": {
            "rmse_median": float(np.median(residuals)),
            "rmse_p95": float(np.percentile(residuals, 95.0)),
            "rmse_max": float(np.max(residuals)),
        },
        "kernel_bound_hits": {
            "lower": int(sum(bool(item["sigma_at_lower_bound"]) for item in fits)),
            "upper": int(sum(bool(item["sigma_at_upper_bound"]) for item in fits)),
        },
        "kernel_psf": psf,
    }


def _p11_processing_soft_band(mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """@luna-comment Construct the narrow contour soft band used by P11 stages.

    Input: Boolean HxW text mask and finite equivalent sigma in pixels.
    Output: Float32 [0,1] weight and boolean support map; outside support is
      exactly zero.
    Key Steps: Measure distance to the mask boundary and use full weight through
      2*sigma, then cosine-fade to zero by 3*sigma+1.
    Key Params: The two radii are tied to the measured sigma, preventing global
      or full-ROI processing.
    State/Dependencies: OpenCV distance transforms; no image mutation or I/O.
    """
    sigma = float(sigma)
    if mask.ndim != 2 or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("P11 soft-band inputs are invalid")
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    distance = np.maximum(np.where(mask, inside, outside) - 0.5, 0.0)
    inner = 2.0 * sigma
    outer = 3.0 * sigma + 1.0
    weight = np.where(
        distance <= inner,
        1.0,
        np.where(
            distance < outer,
            0.5 * (1.0 + np.cos(np.pi * (distance - inner) / max(outer - inner, 1.0e-6))),
            0.0,
        ),
    ).astype(np.float32)
    return weight, weight > 0.0


def _p11_wiener_hunt_deblur(
    y: np.ndarray, mask: np.ndarray, sigma: float, regularization: float, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply one reflected-context Wiener-Hunt deblur to local Y.

    Input: Finite HxW Y plane, non-empty text mask, estimated sigma, and positive
      Laplacian regularization.
    Output: Full-size raw deconvolved Y plus crop/context telemetry; no masking
      or RGB reconstruction is performed in this helper.
    Key Steps: Crop text with context, reflect-pad the crop, form FFT responses
      for Gaussian K and a discrete Laplacian L, apply K*/(|K|^2+lambda|L|^2),
      then crop back to the image coordinates.
    Key Params: Context defaults to ceil(3*sigma+1); lambda controls high
      frequency amplification and is never treated as a quality gate.
    State/Dependencies: Numpy FFT/OpenCV padding only; GT is not accessed.
    """
    if y.ndim != 2 or mask.shape != y.shape or not np.isfinite(y).all() or not mask.any():
        raise ValueError("P11 Wiener input must be finite and have a non-empty mask")
    sigma = float(sigma)
    regularization = float(regularization)
    if not (math.isfinite(sigma) and 0.3 <= sigma <= 4.0 and math.isfinite(regularization) and regularization > 0.0):
        raise ValueError("P11 Wiener sigma/lambda is invalid")
    rows, columns = np.nonzero(mask)
    context = max(1, int(math.ceil(3.0 * sigma + 1.0)))
    top = max(0, int(rows.min()) - context)
    bottom = min(y.shape[0], int(rows.max()) + context + 1)
    left = max(0, int(columns.min()) - context)
    right = min(y.shape[1], int(columns.max()) + context + 1)
    crop = np.asarray(y[top:bottom, left:right], dtype=np.float32)
    pad = max(1, int(math.ceil(3.0 * sigma)))
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    kernel = _p11_gaussian_psf(sigma)
    radius = kernel.shape[0] // 2
    psf = np.zeros(padded.shape, dtype=np.float32)
    psf[: kernel.shape[0], : kernel.shape[1]] = kernel
    psf = np.roll(psf, (-radius, -radius), axis=(0, 1))
    laplacian = np.zeros(padded.shape, dtype=np.float32)
    laplacian[:3, :3] = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    laplacian = np.roll(laplacian, (-1, -1), axis=(0, 1))
    response = np.fft.fft2(psf)
    lap_response = np.fft.fft2(laplacian)
    denominator = np.square(np.abs(response)) + regularization * np.square(np.abs(lap_response))
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise ValueError("P11 Wiener denominator is invalid")
    inverse = np.conj(response) / denominator
    restored = np.real(np.fft.ifft2(np.fft.fft2(padded) * inverse)).astype(np.float32)
    restored = restored[pad : pad + crop.shape[0], pad : pad + crop.shape[1]]
    raw = y.copy()
    raw[top:bottom, left:right] = restored
    if not np.isfinite(raw).all():
        raise ValueError("P11 raw Wiener output is non-finite")
    return raw, {
        "wiener_status": "computed",
        "wiener_sigma": sigma,
        "wiener_lambda": regularization,
        "wiener_context": context,
        "wiener_pad": pad,
        "wiener_crop": [top, bottom, left, right],
        "wiener_raw_mean_abs_delta": float(np.mean(np.abs(raw - y))),
        "wiener_raw_p95_abs_delta": float(np.percentile(np.abs(raw - y), 95.0)),
        "_p11_raw_deblur_y": raw,
    }


def _p11_scatter_weighted(
    delta_sum: np.ndarray,
    weight_sum: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    delta: np.ndarray,
    point_weight: np.ndarray,
) -> None:
    """@luna-comment Bilinearly accumulate one weighted normal correction.

    Input: HxW accumulation arrays and matching subpixel maps, luma correction,
      and nonnegative per-sample soft-band weights.
    Output: In-place sums for one later weighted average; invalid samples are
      ignored and never write outside the image.
    Key Steps: Distribute each normal sample to four neighbours and multiply
      both correction and support by the same local soft-band weight.
    Key Params: Bilinear factors avoid pixel seams; overlapping profiles are
      fused once by the final weighted division.
    State/Dependencies: Numpy accumulation only; input arrays are read-only.
    """
    if not (map_x.shape == map_y.shape == delta.shape == point_weight.shape):
        return
    height, width = delta_sum.shape
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    wx = map_x - x0
    wy = map_y - y0
    for dx, dy, factor in (
        (0, 0, (1.0 - wx) * (1.0 - wy)),
        (1, 0, wx * (1.0 - wy)),
        (0, 1, (1.0 - wx) * wy),
        (1, 1, wx * wy),
    ):
        xx = x0 + dx
        yy = y0 + dy
        contribution_weight = factor * point_weight
        valid = (
            (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            & np.isfinite(delta) & np.isfinite(contribution_weight)
            & (contribution_weight > 0.0)
        )
        if bool(valid.any()):
            np.add.at(delta_sum, (yy[valid], xx[valid]), (delta * contribution_weight)[valid])
            np.add.at(weight_sum, (yy[valid], xx[valid]), contribution_weight[valid])


def _p11_local_y_s_curve(
    y: np.ndarray, mask: np.ndarray, soft_weight: np.ndarray, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply a local platform-normalized Y S-curve in the soft band.

    Input: Finite Y plane, accepted text mask, shared P11 soft-band weights, and
      gamma/platform sampling parameters.
    Output: Finite Y plane plus per-profile curve evidence and write coverage;
      distant text interiors/background remain unchanged.
    Key Steps: Measure two-sided local platforms on normal profiles, normalize
      each profile, apply the fixed gamma S-curve, and weighted-scatter each
      correction once at overlapping subpixel samples.
    Key Params: Gamma defaults to 1.5; profile step/half-width and platform
      thresholds reuse the P10 contract.
    State/Dependencies: OpenCV remapping/numpy only; no GT or RGB state.
    """
    if y.ndim != 2 or mask.shape != y.shape or soft_weight.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("P11 local S-curve inputs are incompatible")
    gamma = float(params.get("gamma", 1.5))
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("P11 S-curve gamma must be finite and positive")
    profile_params = dict(params)
    profile_params["cleanup"] = "none"
    measured = _p10_measure_profiles(y, mask, profile_params)
    profiles = measured.get("profiles", []) if isinstance(measured, dict) else []
    delta_sum = np.zeros_like(y, dtype=np.float32)
    weight_sum = np.zeros_like(y, dtype=np.float32)
    profile_evidence: list[dict[str, Any]] = []
    for profile in profiles:
        map_x = np.asarray(profile["_map_x"], dtype=np.float32)
        map_y = np.asarray(profile["_map_y"], dtype=np.float32)
        offsets = np.asarray(profile["offsets"], dtype=np.float32)
        values = np.asarray(profile["values"], dtype=np.float32)
        background = float(profile["background_y"])
        foreground = float(profile["foreground_y"])
        denominator = foreground - background
        if not math.isfinite(denominator) or abs(denominator) < float(params.get("min_profile_contrast", 0.12)):
            continue
        t = np.clip((values - background) / denominator, 0.0, 1.0).astype(np.float32)
        numerator = np.power(t, gamma)
        curve_t = numerator / np.maximum(numerator + np.power(1.0 - t, gamma), 1.0e-12)
        curve_values = (background + denominator * curve_t).astype(np.float32)
        curve_delta = curve_values - values
        sampled_weight = cv2.remap(
            soft_weight,
            map_x.reshape(1, -1),
            map_y.reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1).astype(np.float32)
        active = ((t > 0.0) & (t < 1.0)).astype(np.float32)
        point_weight = sampled_weight * active
        _p11_scatter_weighted(delta_sum, weight_sum, map_x, map_y, curve_delta, point_weight)
        profile_evidence.append({
            "row": int(profile["row"]),
            "column": int(profile["column"]),
            "background_y": background,
            "foreground_y": foreground,
            "contrast": abs(denominator),
            "gamma": gamma,
            "offsets": [float(value) for value in offsets],
            "values": [float(value) for value in values],
            "curve_values": [float(value) for value in curve_values],
            "mean_abs_curve_delta": float(np.mean(np.abs(curve_delta * point_weight))),
            "active_sample_fraction": float(np.mean(point_weight > 0.0)),
        })
    correction = np.zeros_like(y, dtype=np.float32)
    supported = weight_sum > 1.0e-6
    correction[supported] = delta_sum[supported] / weight_sum[supported]
    # Bilinear splatting can place a contribution in an immediately adjacent
    # pixel whose sampled source point was inside the band.  Clamp after the
    # fusion as well, so P11's public write contract is exact: no Y/RGB delta
    # may escape the shared processing band.
    correction[soft_weight <= 0.0] = 0.0
    supported &= soft_weight > 0.0
    output = np.clip(y + correction, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P11 local S-curve output is non-finite")
    return output, {
        "scurve_status": "computed" if profile_evidence else "skipped_no_profiles",
        "scurve_gamma": gamma,
        "scurve_profile_count": len(profile_evidence),
        "scurve_support_fraction": float(np.mean(supported)),
        "scurve_mean_abs_delta": float(np.mean(np.abs(correction))),
        "scurve_p95_abs_delta": float(np.percentile(np.abs(correction), 95.0)),
        "scurve_profile_evidence": profile_evidence,
        "_p11_scurve_correction": correction,
    }


def _p11_kernel_wiener_scurve(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Run P11 kernel estimation, local Wiener, and local Y curve.

    Input: Authoritative finite HxWx3 RGB tensor and one lambda/gamma candidate.
    Output: First finite final RGB result plus six-column stage arrays,
      kernel/PSF/profile telemetry, and explicit skipped-stage statuses.
    Key Steps: Segment text from input Y, estimate an equivalent Gaussian PSF,
      construct one contour soft band, generate S-curve-only output, then run
      each requested Wiener lambda and apply the same local S-curve.
    Key Params: Gamma defaults to 1.5 and candidate lambdas are positive; all
      writes are limited to the measured soft band and no quality fallback is used.
    State/Dependencies: CPU numpy/OpenCV only; GT/model/chroma planes are unused.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("invalid P11 RGB input")
    y = _rgb_to_luma(rgb)
    mask, segmentation = _p10_segment_text(y, params)
    if mask is None:
        scurve_rgb = rgb.copy()
        return scurve_rgb, {
            "fallback": False,
            "fallback_reason": None,
            "p11_status": "kernel_unidentifiable",
            "p11_kernel_status": "kernel_unidentifiable",
            "p11_kernel_reason": segmentation.get("segmentation_reason", "no_text_mask"),
            "p11_kernel_sigma": None,
            "p11_kernel_sigma_mad": None,
            "p11_kernel_segment_count": 0,
            "p11_kernel_profiles_considered": 0,
            "p11_kernel_direction_stats": {},
            "p11_kernel_fit_residual_stats": {},
            "p11_kernel_bound_hits": {"lower": 0, "upper": 0},
            "p11_scurve_status": "skipped_no_mask",
            "p11_scurve_gamma": float(params.get("gamma", 1.5)),
            "p11_write_coverage": 0.0,
            "p11_deblur_status": "kernel_candidate_skipped",
            "p11_deblur_candidates": [],
            "_p11_text_mask": np.zeros_like(y, dtype=bool),
            "_p11_processing_soft_weight": np.zeros_like(y, dtype=np.float32),
            "_p11_edge_segment_map": np.zeros_like(y, dtype=np.uint8),
            "_p11_scurve_only_output": scurve_rgb,
            "_p11_stage_outputs": [],
        }
    kernel = _p11_estimate_equivalent_kernel(y, mask, params)
    sigma = kernel.get("kernel_sigma")
    if sigma is None:
        soft_sigma = float(params.get("scurve_band_sigma", 1.0))
        if not math.isfinite(soft_sigma) or soft_sigma <= 0.0:
            raise ValueError("P11 S-curve fallback band sigma is invalid")
        kernel_status = "kernel_unidentifiable"
    else:
        soft_sigma = float(sigma)
        kernel_status = str(kernel["kernel_status"])
    soft_weight, soft_support = _p11_processing_soft_band(mask, soft_sigma)
    scurve_y, scurve_meta = _p11_local_y_s_curve(y, mask, soft_weight, params)
    scurve_rgb, scurve_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y, scurve_y)
    stage_outputs: list[dict[str, Any]] = []
    lambda_candidates = params.get("lambda_candidates", [params.get("lambda", 0.01), 0.03])
    if not isinstance(lambda_candidates, (list, tuple)):
        lambda_candidates = [lambda_candidates]
    if sigma is not None:
        for raw_lambda in lambda_candidates:
            regularization = float(raw_lambda)
            raw_deblur_y, wiener_meta = _p11_wiener_hunt_deblur(y, mask, soft_sigma, regularization, params)
            masked_deblur_y = (y + soft_weight * (raw_deblur_y - y)).astype(np.float32)
            if not np.isfinite(masked_deblur_y).all():
                raise ValueError("P11 masked deblur output is non-finite")
            deblur_rgb, deblur_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y, masked_deblur_y)
            final_y, local_meta = _p11_local_y_s_curve(masked_deblur_y, mask, soft_weight, params)
            final_rgb, final_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y, final_y)
            stage_outputs.append({
                "lambda": regularization,
                "deblur_status": "computed",
                "deblur_output": deblur_rgb,
                "final_output": final_rgb,
                "raw_deblur_y": raw_deblur_y,
                "masked_deblur_y": masked_deblur_y,
                "wiener_meta": wiener_meta,
                "deblur_rgb_meta": deblur_rgb_meta,
                "final_rgb_meta": final_rgb_meta,
                "scurve_meta": local_meta,
            })
    meta: dict[str, Any] = {
        "fallback": False,
        "fallback_reason": None,
        "p11_status": "computed" if stage_outputs else kernel_status,
        "p11_kernel_status": kernel_status,
        "p11_kernel_reason": kernel.get("kernel_reason"),
        "p11_kernel_sigma": sigma,
        "p11_kernel_sigma_mad": kernel.get("kernel_sigma_mad"),
        "p11_kernel_segment_count": int(kernel.get("kernel_segment_count", 0)),
        "p11_kernel_profiles_considered": int(kernel.get("kernel_profiles_considered", 0)),
        "p11_kernel_direction_stats": kernel.get("kernel_direction_stats", {}),
        "p11_kernel_fit_residual_stats": kernel.get("kernel_fit_residual_stats", {}),
        "p11_kernel_bound_hits": kernel.get("kernel_bound_hits", {}),
        "p11_scurve_status": scurve_meta.get("scurve_status"),
        "p11_scurve_gamma": scurve_meta.get("scurve_gamma"),
        "p11_scurve_profile_count": int(scurve_meta.get("scurve_profile_count", 0)),
        "p11_scurve_support_fraction": float(scurve_meta.get("scurve_support_fraction", 0.0)),
        "p11_scurve_mean_abs_delta": float(scurve_meta.get("scurve_mean_abs_delta", 0.0)),
        "p11_scurve_p95_abs_delta": float(scurve_meta.get("scurve_p95_abs_delta", 0.0)),
        "p11_write_coverage": float(np.mean(soft_support)),
        "p11_deblur_status": "computed" if stage_outputs else "kernel_candidate_skipped",
        "p11_deblur_candidate_count": len(stage_outputs),
        "p11_deblur_candidates": [
            {
                "lambda": float(stage["lambda"]),
                "status": str(stage["deblur_status"]),
                "masked_write_coverage": float(np.mean(soft_support)),
                "masked_mean_abs_delta": float(np.mean(np.abs(stage["masked_deblur_y"] - y))),
                "masked_p95_abs_delta": float(np.percentile(np.abs(stage["masked_deblur_y"] - y), 95.0)),
                "scurve_status": str(stage["scurve_meta"].get("scurve_status")),
                "scurve_profile_count": int(stage["scurve_meta"].get("scurve_profile_count", 0)),
                "scurve_mean_abs_delta": float(stage["scurve_meta"].get("scurve_mean_abs_delta", 0.0)),
            }
            for stage in stage_outputs
        ],
        "mean_abs_requested_delta": float(scurve_rgb_meta.get("mean_abs_requested_delta", 0.0)),
        "mean_abs_applied_delta": float(scurve_rgb_meta.get("mean_abs_applied_delta", 0.0)),
        "p95_abs_applied_delta": float(scurve_rgb_meta.get("p95_abs_applied_delta", 0.0)),
        "fraction_abs_delta_ge_4_255": float(scurve_rgb_meta.get("fraction_abs_delta_ge_4_255", 0.0)),
        "gamut_limited_fraction": float(scurve_rgb_meta.get("gamut_limited_fraction", 0.0)),
        "mean_abs_requested_minus_applied": float(scurve_rgb_meta.get("mean_abs_requested_minus_applied", 0.0)),
        "chroma_max_abs_error": float(scurve_rgb_meta.get("chroma_max_abs_error", 0.0)),
        "segmentation_accepted": bool(segmentation.get("segmentation_accepted", False)),
        "segmentation_reason": segmentation.get("segmentation_reason"),
        "foreground_fraction": float(segmentation.get("foreground_fraction", 0.0)),
        "otsu_threshold": float(segmentation.get("otsu_threshold", 0.0)),
        "otsu_separation": float(segmentation.get("otsu_separation", 0.0)),
        "_p11_text_mask": mask,
        "_p11_processing_soft_weight": soft_weight,
        "_p11_edge_segment_map": np.where(soft_support, 255, 0).astype(np.uint8),
        "_p11_scurve_only_output": scurve_rgb,
        "_p11_kernel_psf": kernel.get("kernel_psf"),
        "_p11_kernel_profiles": kernel.get("kernel_profiles", []),
        "_p11_scurve_profile_evidence": scurve_meta.get("scurve_profile_evidence", []),
        "_p11_stage_outputs": stage_outputs,
    }
    if stage_outputs:
        meta.update(stage_outputs[0]["final_rgb_meta"])
        selected_output = stage_outputs[0]["final_output"]
    else:
        selected_output = scurve_rgb
    return selected_output, meta


def _p12_processing_soft_band(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """@luna-comment Build P12's sigma-independent contour write mask.

    Input: Boolean text mask in the input-image coordinate system.
    Output: Float32 mask ``M`` and a boolean support map with exact zero outside
      three pixels from the input contour.
    Key Steps: Measure the signed distance to the mask contour, keep unit weight
      through one pixel, then cosine-fade to zero at three pixels.
    Key Params: The one/three-pixel radii are fixed by the P12 design and do not
      depend on the estimated blur sigma.
    State/Dependencies: OpenCV distance transforms only; no image or RGB write.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("P12 contour mask must be a boolean two-dimensional array")
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    distance = np.maximum(np.where(mask, inside, outside) - 0.5, 0.0)
    weight = np.where(
        distance <= 1.0,
        1.0,
        np.where(distance < 3.0, 0.5 * (1.0 + np.cos(np.pi * (distance - 1.0) / 2.0)), 0.0),
    ).astype(np.float32)
    if not np.isfinite(weight).all() or float(weight.min()) < -1.0e-7 or float(weight.max()) > 1.0 + 1.0e-7:
        raise ValueError("P12 contour mask is non-finite or outside [0,1]")
    return weight, weight > 0.0


def _p12_scatter_platform_samples(
    low_sum: np.ndarray,
    high_sum: np.ndarray,
    weight_sum: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    low: float,
    high: float,
    point_weight: np.ndarray,
) -> None:
    """@luna-comment Accumulate continuous low/high platform samples.

    Input: HxW platform accumulators, one reliable normal map, its low/high
      LQ-platform values, and nonnegative local sample weights.
    Output: In-place bilinear accumulators; this helper never computes a pixel
      correction and never scatters a deblur residual.
    Key Steps: Validate finite coordinates, distribute each sample to four
      neighbours, and accumulate identical platform levels with support weight.
    Key Params: Bilinear factors provide a continuous platform map before the
      small spatial smoothing pass.
    State/Dependencies: Numpy accumulation only; no GT or RGB state.
    """
    if not (map_x.shape == map_y.shape == point_weight.shape):
        return
    height, width = low_sum.shape
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    wx = map_x - x0
    wy = map_y - y0
    finite = np.isfinite(map_x) & np.isfinite(map_y) & np.isfinite(point_weight) & (point_weight > 0.0)
    for dx, dy, factor in (
        (0, 0, (1.0 - wx) * (1.0 - wy)),
        (1, 0, wx * (1.0 - wy)),
        (0, 1, (1.0 - wx) * wy),
        (1, 1, wx * wy),
    ):
        xx = x0 + dx
        yy = y0 + dy
        contribution = factor * point_weight
        valid = finite & np.isfinite(factor) & (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        if bool(valid.any()):
            np.add.at(low_sum, (yy[valid], xx[valid]), (float(low) * contribution)[valid])
            np.add.at(high_sum, (yy[valid], xx[valid]), (float(high) * contribution)[valid])
            np.add.at(weight_sum, (yy[valid], xx[valid]), contribution[valid])


def _p12_interpolate_platform_maps(
    shape: tuple[int, int],
    profiles: list[dict[str, Any]],
    processing_weight: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """@luna-comment Interpolate LQ-derived low/high platforms in the contour band.

    Input: Image shape, reliable P10 normal profiles measured on input Y,
      fixed P12 contour weight, and interpolation parameters.
    Output: Low/high platform maps, final supported write mask, and support
      telemetry. Unsupported pixels remain false and receive no correction.
    Key Steps: Bilinearly accumulate each profile's constant LQ platforms along
      its normal, smooth numerator/denominator locally, and apply the contour
      weight only after support is known.
    Key Params: A small Gaussian support blur defaults to 1.0 px; the support
      threshold is explicit and does not depend on image quality metrics.
    State/Dependencies: OpenCV/numpy only; profiles are read-only.
    """
    if processing_weight.shape != shape or not np.isfinite(processing_weight).all():
        raise ValueError("P12 platform interpolation mask is invalid")
    low_sum = np.zeros(shape, dtype=np.float32)
    high_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    accepted = 0
    for profile in profiles:
        try:
            map_x = np.asarray(profile["_map_x"], dtype=np.float32).reshape(-1)
            map_y = np.asarray(profile["_map_y"], dtype=np.float32).reshape(-1)
            offsets = np.asarray(profile.get("offsets", []), dtype=np.float32).reshape(-1)
            background = float(profile["background_y"])
            foreground = float(profile["foreground_y"])
            if map_x.size == 0 or map_x.shape != map_y.shape or offsets.shape != map_x.shape:
                continue
            if not all(math.isfinite(value) for value in (background, foreground)):
                continue
            low = min(background, foreground)
            high = max(background, foreground)
            if high - low < float(params.get("min_profile_contrast", 0.12)):
                continue
            local_weight = cv2.remap(
                processing_weight,
                map_x.reshape(1, -1),
                map_y.reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ).reshape(-1).astype(np.float32)
            # A normal's samples outside the fixed contour band have zero
            # weight; inside the band, the profile platform is continuous.
            _p12_scatter_platform_samples(
                low_sum,
                high_sum,
                weight_sum,
                map_x,
                map_y,
                low,
                high,
                local_weight,
            )
            accepted += 1
        except (KeyError, TypeError, ValueError, FloatingPointError, cv2.error):
            continue
    sigma = float(params.get("platform_map_smoothing_sigma", 1.0))
    threshold = float(params.get("platform_support_threshold", 1.0e-4))
    if not math.isfinite(sigma) or sigma <= 0.0 or not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("P12 platform interpolation parameters are invalid")
    smooth_low = cv2.GaussianBlur(low_sum, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    smooth_high = cv2.GaussianBlur(high_sum, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    smooth_weight = cv2.GaussianBlur(weight_sum, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    supported = (smooth_weight > threshold) & (processing_weight > 0.0)
    low_map = np.zeros(shape, dtype=np.float32)
    high_map = np.zeros(shape, dtype=np.float32)
    low_map[supported] = smooth_low[supported] / smooth_weight[supported]
    high_map[supported] = smooth_high[supported] / smooth_weight[supported]
    low_map, high_map = np.minimum(low_map, high_map), np.maximum(low_map, high_map)
    if not np.isfinite(low_map).all() or not np.isfinite(high_map).all():
        raise ValueError("P12 platform maps are non-finite")
    final_weight = np.where(supported, processing_weight, 0.0).astype(np.float32)
    return low_map, high_map, final_weight, {
        "p12_platform_profile_count": int(accepted),
        "p12_platform_support_fraction": float(np.mean(supported)),
        "p12_platform_hole_fraction": float(np.mean((processing_weight > 0.0) & ~supported)),
        "p12_platform_map_smoothing_sigma": sigma,
        "p12_platform_support_threshold": threshold,
    }


def _p12_apply_platform_clamp(
    y_deblur: np.ndarray,
    low_map: np.ndarray,
    high_map: np.ndarray,
    write_weight: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Recover only platform-external dark/bright residuals.

    Input: Finite deblurred Y, supported low/high platform maps, independent
      contour write weight, and a finite alpha in (0,1].
    Output: Clamped Y and exact dark/bright correction telemetry plus private
      residual/correction maps for artifact writing.
    Key Steps: Clip only outside [L,H], multiply the residual by alpha and the
      final mask once, and leave unsupported or in-platform pixels unchanged.
    Key Params: Alpha controls correction strength; no quality gate or identity
      fallback is applied to a finite stage.
    State/Dependencies: Numpy only; no RGB/chroma state or GT access.
    """
    if not (y_deblur.ndim == 2 and low_map.shape == y_deblur.shape == high_map.shape == write_weight.shape):
        raise ValueError("P12 clamp arrays have incompatible shapes")
    if not all(np.isfinite(value).all() for value in (y_deblur, low_map, high_map, write_weight)):
        raise ValueError("P12 clamp arrays must be finite")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P12 clamp alpha must be finite and in (0,1]")
    low = np.minimum(low_map, high_map)
    high = np.maximum(low_map, high_map)
    target = np.clip(y_deblur, low, high).astype(np.float32)
    residual = target - y_deblur
    correction = (alpha * np.clip(write_weight, 0.0, 1.0) * residual).astype(np.float32)
    output = np.clip(y_deblur + correction, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P12 clamp output is non-finite")
    dark = y_deblur < low
    bright = y_deblur > high
    outside = write_weight <= 0.0
    applied_outside = np.abs(correction[outside]) if bool(outside.any()) else np.asarray([], dtype=np.float32)
    return output, {
        "p12_clamp_status": "computed",
        "p12_alpha": alpha,
        "p12_dark_overshoot_pixels": int(np.sum(dark & (write_weight > 0.0))),
        "p12_bright_overshoot_pixels": int(np.sum(bright & (write_weight > 0.0))),
        "p12_dark_correction_mean_abs": float(np.mean(np.abs(correction[dark]))) if bool(dark.any()) else 0.0,
        "p12_bright_correction_mean_abs": float(np.mean(np.abs(correction[bright]))) if bool(bright.any()) else 0.0,
        "p12_dark_correction_sum_abs": float(np.sum(np.abs(correction[dark]))),
        "p12_bright_correction_sum_abs": float(np.sum(np.abs(correction[bright]))),
        "p12_platform_external_fraction": float(np.mean((dark | bright) & (write_weight > 0.0))),
        "p12_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p12_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)),
        "p12_band_outside_max_abs_delta": float(np.max(applied_outside)) if applied_outside.size else 0.0,
        "p12_band_outside_delta_sum_abs": float(np.sum(applied_outside)) if applied_outside.size else 0.0,
        "_p12_platform_residual": residual,
        "_p12_correction": correction,
    }


def _p12_kernel_platform_clamp(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Run P12 estimated-kernel deblur and platform clamp stages.

    Input: Authoritative finite RGB tensor and a fixed P12 candidate matrix.
    Output: First finite clamp result plus all lambda/alpha stage arrays and
      LQ-platform, support, overshoot, and Y/RGB telemetry.
    Key Steps: Segment input Y, estimate the P11-equivalent Gaussian PSF, run
      local Wiener candidates, measure platforms only on input Y, interpolate
      supported maps, and clamp each deblur stage with the fixed contour mask.
    Key Params: Lambda candidates default to [.01,.03], alpha candidates to
      [.5,1.0], and the contour mask is fixed at one-to-three pixels.
    State/Dependencies: CPU numpy/OpenCV only; no GT/model/chroma plane reads.
      P11 helpers are reused but their S-curve/scattered correction is not used.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("invalid P12 RGB input")
    y_input = _rgb_to_luma(rgb)
    mask, segmentation = _p10_segment_text(y_input, params)
    empty_mask = np.zeros_like(y_input, dtype=bool)
    empty_weight = np.zeros_like(y_input, dtype=np.float32)
    base_meta: dict[str, Any] = {
        "fallback": False,
        "fallback_reason": None,
        "p12_status": "kernel_unidentifiable",
        "p12_kernel_status": "kernel_unidentifiable",
        "p12_kernel_reason": None,
        "p12_kernel_sigma": None,
        "p12_kernel_sigma_mad": None,
        "p12_kernel_segment_count": 0,
        "p12_kernel_profiles_considered": 0,
        "p12_platform_reference": "input_lq_y",
        "p12_platform_profile_count": 0,
        "p12_platform_support_fraction": 0.0,
        "p12_platform_hole_fraction": 0.0,
        "p12_mask_coverage": 0.0,
        "p12_deblur_status": "kernel_candidate_skipped",
        "p12_postprocess_status": "skipped_no_platform_support",
        "p12_stage_count": 0,
        "p12_stage_telemetry": [],
        "_p12_text_mask": empty_mask,
        "_p12_processing_weight": empty_weight,
        "_p12_deblur_weight": empty_weight,
        "_p12_platform_low": np.zeros_like(y_input, dtype=np.float32),
        "_p12_platform_high": np.zeros_like(y_input, dtype=np.float32),
        "_p12_platform_support": empty_weight,
        "_p12_platform_holes": empty_weight,
        "_p12_kernel_psf": None,
        "_p12_profiles": [],
        "_p12_stage_outputs": [],
    }
    if mask is None:
        base_meta["p12_kernel_reason"] = segmentation.get("segmentation_reason", "no_text_mask")
        return rgb.copy(), base_meta
    base_meta["_p12_text_mask"] = mask
    processing_weight, _ = _p12_processing_soft_band(mask)
    base_meta["_p12_processing_weight"] = processing_weight
    base_meta["p12_mask_coverage"] = float(np.mean(processing_weight > 0.0))
    kernel = _p11_estimate_equivalent_kernel(y_input, mask, params)
    sigma = kernel.get("kernel_sigma")
    base_meta.update({
        "p12_kernel_status": str(kernel.get("kernel_status", "kernel_unidentifiable")),
        "p12_kernel_reason": kernel.get("kernel_reason"),
        "p12_kernel_sigma": sigma,
        "p12_kernel_sigma_mad": kernel.get("kernel_sigma_mad"),
        "p12_kernel_segment_count": int(kernel.get("kernel_segment_count", 0)),
        "p12_kernel_profiles_considered": int(kernel.get("kernel_profiles_considered", 0)),
        "p12_kernel_direction_stats": kernel.get("kernel_direction_stats", {}),
        "p12_kernel_fit_residual_stats": kernel.get("kernel_fit_residual_stats", {}),
        "p12_kernel_bound_hits": kernel.get("kernel_bound_hits", {}),
        "_p12_kernel_psf": kernel.get("kernel_psf"),
    })
    if sigma is None:
        return rgb.copy(), base_meta
    sigma = float(sigma)
    deblur_weight, _ = _p11_processing_soft_band(mask, sigma)
    base_meta["_p12_deblur_weight"] = deblur_weight
    platform_params = dict(params)
    platform_params["cleanup"] = "none"
    if "platform_min_profile_contrast" in params:
        platform_params["min_profile_contrast"] = params["platform_min_profile_contrast"]
    measured = _p10_measure_profiles(y_input, mask, platform_params)
    profiles = measured.get("profiles", []) if isinstance(measured, dict) else []
    low_map, high_map, final_weight, platform_meta = _p12_interpolate_platform_maps(
        y_input.shape, profiles, processing_weight, params
    )
    base_meta.update(platform_meta)
    base_meta["_p12_platform_low"] = low_map
    base_meta["_p12_platform_high"] = high_map
    base_meta["_p12_platform_support"] = final_weight
    base_meta["_p12_platform_holes"] = ((processing_weight > 0.0) & (final_weight <= 0.0)).astype(np.float32)
    base_meta["_p12_profiles"] = profiles
    lambda_candidates = params.get("lambda_candidates", [params.get("lambda", 0.01), 0.03])
    alpha_candidates = params.get("alpha_candidates", [0.5, 1.0])
    if not isinstance(lambda_candidates, (list, tuple)) or not isinstance(alpha_candidates, (list, tuple)):
        raise ValueError("P12 lambda_candidates and alpha_candidates must be lists")
    lambdas = [float(value) for value in lambda_candidates]
    alphas = [float(value) for value in alpha_candidates]
    if not lambdas or not alphas or any(not math.isfinite(value) or value <= 0.0 for value in lambdas) or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("P12 lambda/alpha candidate values are invalid")
    stage_outputs: list[dict[str, Any]] = []
    for regularization in lambdas:
        raw_deblur_y, wiener_meta = _p11_wiener_hunt_deblur(y_input, mask, sigma, regularization, params)
        masked_deblur_y = (y_input + deblur_weight * (raw_deblur_y - y_input)).astype(np.float32)
        if not np.isfinite(masked_deblur_y).all():
            raise ValueError("P12 masked deblur output is non-finite")
        deblur_rgb, deblur_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, masked_deblur_y)
        for alpha in alphas:
            if profiles and bool(np.any(final_weight > 0.0)):
                clamped_y, clamp_meta = _p12_apply_platform_clamp(
                    masked_deblur_y, low_map, high_map, final_weight, alpha
                )
                post_status = "computed"
            else:
                clamped_y = masked_deblur_y.copy()
                clamp_meta = {
                    "p12_clamp_status": "skipped_no_platform_support",
                    "p12_alpha": alpha,
                    "p12_dark_overshoot_pixels": 0,
                    "p12_bright_overshoot_pixels": 0,
                    "p12_dark_correction_mean_abs": 0.0,
                    "p12_bright_correction_mean_abs": 0.0,
                    "p12_dark_correction_sum_abs": 0.0,
                    "p12_bright_correction_sum_abs": 0.0,
                    "p12_platform_external_fraction": 0.0,
                    "p12_mean_abs_correction": 0.0,
                    "p12_p95_abs_correction": 0.0,
                    "p12_band_outside_max_abs_delta": 0.0,
                    "p12_band_outside_delta_sum_abs": 0.0,
                    "_p12_platform_residual": np.zeros_like(y_input, dtype=np.float32),
                    "_p12_correction": np.zeros_like(y_input, dtype=np.float32),
                }
                post_status = "skipped_no_platform_support"
            clamp_rgb, clamp_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, clamped_y)
            stage = {
                "lambda": regularization,
                "alpha": alpha,
                "deblur_status": "computed",
                "postprocess_status": post_status,
                "deblur_output": deblur_rgb,
                "clamp_output": clamp_rgb,
                "raw_deblur_y": raw_deblur_y,
                "masked_deblur_y": masked_deblur_y,
                "clamped_y": clamped_y,
                "wiener_meta": wiener_meta,
                "deblur_rgb_meta": deblur_rgb_meta,
                "clamp_rgb_meta": clamp_rgb_meta,
                "clamp_meta": clamp_meta,
            }
            stage_outputs.append(stage)
    if not stage_outputs:
        raise ValueError("P12 produced no finite deblur/clamp stage")
    stage_telemetry = []
    for stage in stage_outputs:
        stage_telemetry.append({
            "lambda": float(stage["lambda"]),
            "alpha": float(stage["alpha"]),
            "deblur_status": stage["deblur_status"],
            "postprocess_status": stage["postprocess_status"],
            "p12_deblur_mean_abs_delta": float(stage["wiener_meta"]["wiener_raw_mean_abs_delta"]),
            "p12_deblur_p95_abs_delta": float(stage["wiener_meta"]["wiener_raw_p95_abs_delta"]),
            **{key: value for key, value in stage["clamp_meta"].items() if not key.startswith("_")},
        })
    selected = stage_outputs[0]
    meta = dict(base_meta)
    meta.update({
        "p12_status": "computed",
        "p12_deblur_status": "computed",
        "p12_postprocess_status": str(selected["postprocess_status"]),
        "p12_stage_count": len(stage_outputs),
        "p12_stage_telemetry": stage_telemetry,
        "_p12_stage_outputs": stage_outputs,
        "_p12_output_minus_deblur": selected["clamp_output"] - selected["deblur_output"],
    })
    meta.update(selected["clamp_meta"])
    meta.update(selected["clamp_rgb_meta"])
    meta.update(segmentation)
    return selected["clamp_output"], meta


def _p13_estimate_region_platforms(
    y: np.ndarray, mask: np.ndarray, sigma: float, params: dict[str, Any]
) -> dict[str, Any]:
    """@luna-comment Estimate robust LQ background/foreground platforms for P13.

    Input: Finite HxW input luma, accepted binary text mask, equivalent blur
      sigma, and region/platform thresholds.
    Output: One padded intended text-region mask, robust background/foreground
      levels, quantiles/MAD uncertainty, and finite status telemetry.
    Key Steps: Form a bounded mask bounding box, seed background and text from
      distance-transform interiors, and use medians with quantile spread.
    Key Params: Region padding defaults to max(8, ceil(3*sigma+2)); seed
      distance, minimum seed count, and contrast floor are configurable.
    State/Dependencies: Read-only numpy/OpenCV arrays; no GT or file writes.
    """
    if y.ndim != 2 or mask.shape != y.shape or mask.dtype != np.bool_ or not np.isfinite(y).all():
        raise ValueError("P13 region-platform inputs are invalid")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("P13 region sigma must be finite and positive")
    rows, columns = np.nonzero(mask)
    empty = np.zeros_like(mask, dtype=bool)
    if rows.size == 0:
        return {
            "status": "region_platform_unidentifiable",
            "reason": "empty_text_mask",
            "background_y": None,
            "foreground_y": None,
            "low_y": None,
            "high_y": None,
            "contrast": None,
            "background_count": 0,
            "foreground_count": 0,
            "background_quantiles": [],
            "foreground_quantiles": [],
            "background_mad": None,
            "foreground_mad": None,
            "uncertainty": None,
            "region_bbox": None,
            "region_fraction": 0.0,
            "_region_mask": empty,
            "_background_seed": empty,
            "_foreground_seed": empty,
        }
    padding_default = max(8, int(math.ceil(3.0 * sigma + 2.0)))
    padding = int(params.get("region_padding", padding_default))
    seed_distance = float(params.get("region_seed_distance", 2.0))
    min_seed_count = max(2, int(params.get("region_min_seed_count", 8)))
    min_contrast = float(params.get("region_min_contrast", params.get("min_profile_contrast", 0.12)))
    if padding < 0 or not math.isfinite(seed_distance) or seed_distance <= 0.0 or not math.isfinite(min_contrast) or min_contrast <= 0.0:
        raise ValueError("P13 region-platform parameters are invalid")
    top = max(0, int(rows.min()) - padding)
    bottom = min(y.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(y.shape[1], int(columns.max()) + padding + 1)
    region = np.zeros_like(mask, dtype=bool)
    region[top:bottom, left:right] = True
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    background_seed = region & ~mask & (outside_distance >= seed_distance)
    foreground_seed = region & mask & (inside_distance >= seed_distance)

    def _stats(seed: np.ndarray) -> tuple[float | None, list[float], float | None, int]:
        values = np.asarray(y[seed], dtype=np.float32)
        if values.size == 0:
            return None, [], None, 0
        quantiles = np.quantile(values.astype(np.float64), [0.10, 0.50, 0.90])
        median = float(quantiles[1])
        mad = float(np.median(np.abs(values.astype(np.float64) - median)))
        return median, [float(value) for value in quantiles], mad, int(values.size)

    background, background_quantiles, background_mad, background_count = _stats(background_seed)
    foreground, foreground_quantiles, foreground_mad, foreground_count = _stats(foreground_seed)
    contrast = None if background is None or foreground is None else abs(float(foreground) - float(background))
    uncertainty = None if contrast is None or contrast <= 1.0e-8 else float(max(background_mad or 0.0, foreground_mad or 0.0) / contrast)
    enough_seed = background_count >= min_seed_count and foreground_count >= min_seed_count
    valid_contrast = contrast is not None and math.isfinite(float(contrast)) and float(contrast) >= min_contrast
    status = "computed" if enough_seed and valid_contrast else "region_platform_uncertain"
    reason = None
    if not enough_seed:
        reason = "insufficient_platform_seed"
    elif not valid_contrast:
        reason = "low_region_contrast"
    return {
        "status": status,
        "reason": reason,
        "background_y": background,
        "foreground_y": foreground,
        "low_y": None if contrast is None else min(float(background), float(foreground)),
        "high_y": None if contrast is None else max(float(background), float(foreground)),
        "contrast": contrast,
        "background_count": background_count,
        "foreground_count": foreground_count,
        "background_quantiles": background_quantiles,
        "foreground_quantiles": foreground_quantiles,
        "background_mad": background_mad,
        "foreground_mad": foreground_mad,
        "uncertainty": uncertainty,
        "region_bbox": [top, bottom, left, right],
        "region_fraction": float(np.mean(region)),
        "_region_mask": region,
        "_background_seed": background_seed,
        "_foreground_seed": foreground_seed,
    }


def _p13_find_stable_platform(
    values: np.ndarray,
    mask_values: np.ndarray,
    offsets: np.ndarray,
    side: str,
    initial_radius: float,
    max_radius: float,
    expected_inside: bool,
    contrast: float,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """@luna-comment Search outward for a stable same-class platform on P13.

    Input: One normal's sampled deblur Y values, nearest-neighbour mask values,
      ordered offsets, side selector, radius limits, and regional contrast.
    Output: Nearest stable platform level/spread/interval or None on geometry
      return, adjacent contour, thin stroke, or insufficient support.
    Key Steps: Walk from the edge beyond the initial 3-sigma radius, stop at a
      class transition, and accept the first contiguous stable run.
    Key Params: Stable run defaults to 3 px; spread is bounded by max(2/255,
      0.03*contrast), and search is capped at 32 px.
    State/Dependencies: Pure finite numpy quantile scan; no image mutation.
    """
    if side not in {"outside", "inside"} or values.ndim != 1 or values.shape != offsets.shape or mask_values.shape != offsets.shape:
        raise ValueError("P13 stable-platform inputs are invalid")
    step = float(params.get("sample_step", 0.5))
    min_run = float(params.get("p13_min_platform_run", 3.0))
    abs_tolerance = float(params.get("p13_platform_tolerance_abs", 2.0 / 255.0))
    rel_tolerance = float(params.get("p13_platform_tolerance_rel", 0.03))
    if not (step > 0.0 and min_run > 0.0 and abs_tolerance >= 0.0 and rel_tolerance >= 0.0 and math.isfinite(contrast)):
        raise ValueError("P13 stable-platform parameters are invalid")
    min_samples = max(2, int(math.ceil(min_run / step)))
    initial_radius = max(0.0, float(initial_radius))
    max_radius = max(initial_radius, float(max_radius))
    if side == "outside":
        eligible = np.flatnonzero((offsets <= -initial_radius) & (offsets >= -max_radius))[::-1]
    else:
        eligible = np.flatnonzero((offsets >= initial_radius) & (offsets <= max_radius))
    if eligible.size < min_samples:
        return None
    tolerance = max(abs_tolerance, rel_tolerance * abs(float(contrast)))
    expected = bool(expected_inside)
    run: list[int] = []
    leading_mismatch = 0
    for index in eligible:
        is_expected = bool(mask_values[int(index)]) == expected
        if not is_expected:
            if not run:
                leading_mismatch += 1
                if leading_mismatch <= 1:
                    continue
            break
        run.append(int(index))
        if len(run) < min_samples:
            continue
        window_indices = np.asarray(run[-min_samples:], dtype=np.int32)
        window = values[window_indices].astype(np.float64)
        if not np.isfinite(window).all():
            return None
        spread = float(np.quantile(window, 0.90) - np.quantile(window, 0.10))
        if spread <= tolerance:
            interval = (float(offsets[window_indices].min()), float(offsets[window_indices].max()))
            return {
                "level": float(np.median(window)),
                "spread": spread,
                "interval": [interval[0], interval[1]],
                "sample_count": int(window_indices.size),
                "tolerance": tolerance,
            }
    return None


def _p13_measure_spatial_profiles(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    region_stats: dict[str, Any],
    sigma: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """@luna-comment Measure dense P13 normal curves and bounded monotone targets.

    Input: Finite deblur Y, input text mask, LQ-derived regional B/F stats,
      equivalent sigma, and geometry/platform parameters.
    Output: Accepted profile records with one dense contour label each plus
      rejection/gap, reach, reverse-step, and support telemetry.
    Key Steps: Derive signed-distance normals, sample to 32 px, find complete
      platforms, orient B-to-F, and apply bounded least-squares isotonic
      projection only between actually reached platform intervals.
    Key Params: Initial search radius is ceil(3*sigma+2), maximum radius 32 px,
      sample step .5 px, dense boundary stride 1, and stable run >=3 px.
    State/Dependencies: OpenCV remap/distance labels and numpy only; no GT or
      file writes. Regional statistics are read-only.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or mask.dtype != np.bool_ or not np.isfinite(y_deblur).all():
        raise ValueError("P13 spatial-profile inputs are invalid")
    background = region_stats.get("background_y")
    foreground = region_stats.get("foreground_y")
    contrast = region_stats.get("contrast")
    if background is None or foreground is None or contrast is None or float(contrast) <= 0.0:
        return {
            "profiles": [],
            "profile_count": 0,
            "profile_rejected_count": 0,
            "profile_gap_count": 0,
            "profile_geometry_status": "skipped_no_region_platform",
            "profile_support_fraction": 0.0,
            "profile_reverse_steps_before": 0,
            "profile_reverse_steps_after": 0,
            "profile_coverage_fraction": 0.0,
            "profile_gap_fraction": 1.0,
            "profile_initial_radius": None,
            "profile_max_radius": None,
        }
    sigma = float(sigma)
    step = float(params.get("sample_step", 0.5))
    max_radius = float(params.get("p13_max_radius", 32.0))
    initial_radius = float(math.ceil(3.0 * sigma + 2.0))
    initial_radius = min(initial_radius, max_radius)
    max_profiles = max(1, int(params.get("p13_max_profiles", 4096)))
    if not (math.isfinite(sigma) and sigma > 0.0 and step > 0.0 and max_radius >= initial_radius >= 1.0):
        raise ValueError("P13 spatial-profile geometry is invalid")
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    coordinates = np.argwhere(boundary)
    if coordinates.shape[0] > max_profiles:
        stride = int(math.ceil(coordinates.shape[0] / max_profiles))
        coordinates = coordinates[::stride]
    offsets = np.arange(-max_radius, max_radius + step * 0.5, step, dtype=np.float32)
    signed = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) - cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    profiles: list[dict[str, Any]] = []
    rejected = 0
    gaps = 0
    reverse_before = 0
    reverse_after = 0
    for boundary_index, (row, column) in enumerate(coordinates):
        magnitude = float(norm[row, column])
        if magnitude <= 1.0e-5:
            rejected += 1
            gaps += 1
            continue
        normal_x = float(grad_x[row, column] / magnitude)
        normal_y = float(grad_y[row, column] / magnitude)
        map_x = (float(column) + offsets * normal_x).astype(np.float32)
        map_y = (float(row) + offsets * normal_y).astype(np.float32)
        if float(map_x.min()) < 1.0 or float(map_x.max()) > y_deblur.shape[1] - 2.0 or float(map_y.min()) < 1.0 or float(map_y.max()) > y_deblur.shape[0] - 2.0:
            rejected += 1
            gaps += 1
            continue
        values = cv2.remap(
            y_deblur,
            map_x.reshape(1, -1),
            map_y.reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1).astype(np.float32)
        mask_values = cv2.remap(
            mask.astype(np.float32),
            map_x.reshape(1, -1),
            map_y.reshape(1, -1),
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1) >= 0.5
        outside = _p13_find_stable_platform(values, mask_values, offsets, "outside", initial_radius, max_radius, False, float(contrast), params)
        inside = _p13_find_stable_platform(values, mask_values, offsets, "inside", initial_radius, max_radius, True, float(contrast), params)
        if outside is None or inside is None:
            rejected += 1
            gaps += 1
            continue
        global_tolerance = float(params.get("p13_region_platform_tolerance", max(0.08, 0.25 * float(contrast))))
        if abs(float(outside["level"]) - float(background)) > global_tolerance or abs(float(inside["level"]) - float(foreground)) > global_tolerance:
            rejected += 1
            gaps += 1
            continue
        edge_start = max(float(outside["interval"][0]), float(outside["interval"][1]))
        edge_end = min(float(inside["interval"][0]), float(inside["interval"][1]))
        if not edge_start < edge_end:
            rejected += 1
            gaps += 1
            continue
        oriented = ((values.astype(np.float64) - float(background)) / (float(foreground) - float(background))).astype(np.float32)
        if not np.isfinite(oriented).all():
            rejected += 1
            gaps += 1
            continue
        projected = np.clip(_p10_isotonic_nondecreasing(oriented), 0.0, 1.0).astype(np.float32)
        target_values = values.copy()
        active = (offsets >= edge_start) & (offsets <= edge_end)
        target_values[active] = (float(background) + (float(foreground) - float(background)) * projected[active]).astype(np.float32)
        # A stable run is an actual platform observation, so its nearest
        # endpoints are anchored. Outside those endpoints the curve is not
        # allowed to invent a correction.
        start_index = int(np.argmin(np.abs(offsets - edge_start)))
        end_index = int(np.argmin(np.abs(offsets - edge_end)))
        if 0 <= start_index < target_values.size:
            target_values[start_index] = float(background)
        if 0 <= end_index < target_values.size:
            target_values[end_index] = float(foreground)
        delta = np.where(active, target_values - values, 0.0).astype(np.float32)
        raw_reverse = int(np.sum(np.diff(oriented) < -1.0 / 255.0))
        projected_active = projected[active]
        projected_reverse = int(np.sum(np.diff(projected_active) < -1.0 / 255.0)) if projected_active.size > 1 else 0
        reverse_before += raw_reverse
        reverse_after += projected_reverse
        profiles.append({
            "boundary_index": int(boundary_index),
            "row": int(row),
            "column": int(column),
            "normal_x": normal_x,
            "normal_y": normal_y,
            "background_y": float(background),
            "foreground_y": float(foreground),
            "contrast": float(contrast),
            "outside_platform_y": float(outside["level"]),
            "inside_platform_y": float(inside["level"]),
            "outside_platform_spread": float(outside["spread"]),
            "inside_platform_spread": float(inside["spread"]),
            "outside_platform_interval": list(outside["interval"]),
            "inside_platform_interval": list(inside["interval"]),
            "active_interval": [edge_start, edge_end],
            "raw_reverse_steps": raw_reverse,
            "projected_reverse_steps": projected_reverse,
            "before_10_90_width": None,
            "after_10_90_width": None,
            "offsets": [float(value) for value in offsets],
            "values": [float(value) for value in values],
            "projected_coordinates": [float(value) for value in projected],
            "target_values": [float(value) for value in target_values],
            "_map_x": map_x,
            "_map_y": map_y,
            "_delta": delta,
            "_boundary_label": int(boundary_index + 1),
        })
    return {
        "profiles": profiles,
        "profile_count": len(profiles),
        "profile_rejected_count": rejected,
        "profile_gap_count": gaps,
        "profile_geometry_status": "computed" if profiles else "no_valid_profiles",
        "profile_support_fraction": 0.0,
        "profile_reverse_steps_before": reverse_before,
        "profile_reverse_steps_after": reverse_after,
        "profile_coverage_fraction": float(len(profiles) / max(coordinates.shape[0], 1)),
        "profile_gap_fraction": float(gaps / max(coordinates.shape[0], 1)),
        "profile_initial_radius": initial_radius,
        "profile_max_radius": max_radius,
        "profile_boundary_count": int(coordinates.shape[0]),
        "profile_sample_step": step,
    }


def _p13_apply_curve_field(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    region_mask: np.ndarray,
    profiles: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """@luna-comment Apply one dense nearest-boundary monotone correction field.

    Input: Finite deblur Y, text mask, intended region mask, and accepted dense
      P13 profiles containing a normal and bounded profile delta.
    Output: Y plane, one correction per supported pixel, support map, and exact
      coverage/outside-write telemetry.
    Key Steps: Label each pixel by its nearest contour seed, project its signed
      normal offset, and interpolate only that seed's bounded profile delta.
    Key Params: Distance labels enforce deterministic one-profile-per-pixel
      mapping; no sparse delta averaging or quality fallback is used.
    State/Dependencies: OpenCV distance labels and numpy; arrays are not written
      in place and no RGB/chroma state is accessed.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or region_mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P13 curve-field inputs are invalid")
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    boundary_coordinates = np.argwhere(boundary)
    correction = np.zeros_like(y_deblur, dtype=np.float32)
    support = np.zeros_like(y_deblur, dtype=bool)
    labels = np.zeros_like(y_deblur, dtype=np.int32)
    if boundary_coordinates.size and profiles:
        _, labels = cv2.distanceTransformWithLabels(
            (~boundary).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        for profile in profiles:
            label = int(profile.get("_boundary_label", -1))
            if label <= 0:
                continue
            selected = region_mask & (labels == label)
            if not bool(selected.any()):
                continue
            row = float(profile["row"])
            column = float(profile["column"])
            normal_x = float(profile["normal_x"])
            normal_y = float(profile["normal_y"])
            yy, xx = np.nonzero(selected)
            offsets = (xx.astype(np.float32) - column) * normal_x + (yy.astype(np.float32) - row) * normal_y
            profile_offsets = np.asarray(profile["offsets"], dtype=np.float32)
            profile_delta = np.asarray(profile["_delta"], dtype=np.float32)
            valid = np.isfinite(offsets) & (offsets >= float(profile["active_interval"][0])) & (offsets <= float(profile["active_interval"][1]))
            if bool(valid.any()):
                correction_values = np.interp(offsets[valid], profile_offsets, profile_delta).astype(np.float32)
                correction[yy[valid], xx[valid]] = correction_values
                support[yy[valid], xx[valid]] = True
    output = (y_deblur + correction).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P13 curve-field output is non-finite")
    outside = ~region_mask
    return output, correction, {
        "profile_field_support_fraction": float(np.mean(support)),
        "profile_field_region_fraction": float(np.mean(region_mask)),
        "profile_field_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "profile_field_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "_profile_field_support": support,
        "_profile_field_labels": labels,
    }


def _p13_apply_shoulder_lut(
    y_deblur: np.ndarray,
    region_mask: np.ndarray,
    background: float,
    foreground: float,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply a regional continuous two-shoulder Y LUT for P13.

    Input: Finite deblur Y, intended text-region mask, robust LQ background and
      foreground levels, and alpha in (0,1].
    Output: Finite Y result plus LUT, ROI coverage, dark/bright compression,
      and exact outside-region delta telemetry.
    Key Steps: Keep [L,H] identity and move values below/above it toward the
      corresponding platform with a continuous monotone shoulder transform.
    Key Params: Alpha .5 or 1.0 controls shoulder compression; no gamma or
      histogram extreme is used and no quality fallback is available.
    State/Dependencies: Numpy only; region mask is the sole write scope.
    """
    if y_deblur.ndim != 2 or region_mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all() or not np.isfinite(region_mask).all():
        raise ValueError("P13 LUT inputs are invalid")
    background = float(background)
    foreground = float(foreground)
    alpha = float(alpha)
    if not (math.isfinite(background) and math.isfinite(foreground) and 0.0 < alpha <= 1.0 and abs(foreground - background) > 0.0):
        raise ValueError("P13 LUT parameters are invalid")
    low = min(background, foreground)
    high = max(background, foreground)
    target = y_deblur.copy()
    below = y_deblur < low
    above = y_deblur > high
    target[below] = low + (1.0 - alpha) * (y_deblur[below] - low)
    target[above] = high + (1.0 - alpha) * (y_deblur[above] - high)
    correction = np.where(region_mask, target - y_deblur, 0.0).astype(np.float32)
    output = (y_deblur + correction).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P13 LUT output is non-finite")
    outside = ~region_mask
    lut_x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut_y = lut_x.copy()
    lut_y[lut_x < low] = low + (1.0 - alpha) * (lut_x[lut_x < low] - low)
    lut_y[lut_x > high] = high + (1.0 - alpha) * (lut_x[lut_x > high] - high)
    return output, {
        "p13_b_status": "computed",
        "p13_b_alpha": alpha,
        "p13_b_low": low,
        "p13_b_high": high,
        "p13_b_roi_fraction": float(np.mean(region_mask)),
        "p13_b_dark_compression_pixels": int(np.sum(region_mask & below)),
        "p13_b_bright_compression_pixels": int(np.sum(region_mask & above)),
        "p13_b_dark_compression_sum_abs": float(np.sum(np.abs(correction[region_mask & below]))),
        "p13_b_bright_compression_sum_abs": float(np.sum(np.abs(correction[region_mask & above]))),
        "p13_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p13_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)) if correction.size else 0.0,
        "p13_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p13_b_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "_p13_b_correction": correction,
        "_p13_b_lut_x": lut_x,
        "_p13_b_lut_y": lut_y,
    }


def _p13_kernel_ab(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Run P13 deblur and independent A/B post-processing stages.

    Input: Authoritative finite HxWx3 RGB tensor and one frozen P13 candidate.
    Output: First finite A result plus lambda deblur controls, A spatial
      monotone outputs, B regional LUT outputs, and JSON-safe telemetry.
    Key Steps: Reuse P11 LQ-only kernel/Wiener stages, estimate regional LQ
      platforms, build a dense A curve field, and apply B's regional shoulder
      LUT independently for alpha .5/1.0.
    Key Params: Wiener lambda candidates default [.01,.03], A radius starts at
      ceil(3*sigma+2) and ends at 32 px, and B alpha defaults [.5,1.0].
    State/Dependencies: CPU numpy/OpenCV only; no GT, model, chroma planes,
      quality selection, or identity substitution for a finite stage.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("invalid P13 RGB input")
    y_input = _rgb_to_luma(rgb)
    mask, segmentation = _p10_segment_text(y_input, params)
    empty_mask = np.zeros_like(y_input, dtype=bool)
    empty_weight = np.zeros_like(y_input, dtype=np.float32)
    base_meta: dict[str, Any] = {
        "fallback": False,
        "fallback_reason": None,
        "p13_status": "kernel_unidentifiable",
        "p13_kernel_status": "kernel_unidentifiable",
        "p13_kernel_reason": None,
        "p13_kernel_sigma": None,
        "p13_kernel_sigma_mad": None,
        "p13_kernel_segment_count": 0,
        "p13_kernel_profiles_considered": 0,
        "p13_region_status": "skipped_no_mask",
        "p13_region_reason": None,
        "p13_region_background_y": None,
        "p13_region_foreground_y": None,
        "p13_region_low_y": None,
        "p13_region_high_y": None,
        "p13_region_contrast": None,
        "p13_region_uncertainty": None,
        "p13_region_background_count": 0,
        "p13_region_foreground_count": 0,
        "p13_region_bbox": None,
        "p13_a_status": "skipped_no_mask",
        "p13_a_profile_count": 0,
        "p13_a_rejected_count": 0,
        "p13_a_gap_count": 0,
        "p13_a_coverage_fraction": 0.0,
        "p13_a_gap_fraction": 1.0,
        "p13_a_support_fraction": 0.0,
        "p13_a_reverse_steps_before": 0,
        "p13_a_reverse_steps_after": 0,
        "p13_b_status": "skipped_no_region_platform",
        "p13_b_roi_fraction": 0.0,
        "p13_stage_count": 0,
        "p13_stage_telemetry": [],
        "_p13_text_mask": empty_mask,
        "_p13_region_mask": empty_mask,
        "_p13_region_stats": {},
        "_p13_processing_weight": empty_weight,
        "_p13_stage_outputs": [],
    }
    if mask is None:
        base_meta["p13_kernel_reason"] = segmentation.get("segmentation_reason", "no_text_mask")
        base_meta.update(segmentation)
        return rgb.copy(), base_meta
    base_meta["_p13_text_mask"] = mask
    kernel = _p11_estimate_equivalent_kernel(y_input, mask, params)
    sigma = kernel.get("kernel_sigma")
    base_meta.update({
        "p13_kernel_status": str(kernel.get("kernel_status", "kernel_unidentifiable")),
        "p13_kernel_reason": kernel.get("kernel_reason"),
        "p13_kernel_sigma": sigma,
        "p13_kernel_sigma_mad": kernel.get("kernel_sigma_mad"),
        "p13_kernel_segment_count": int(kernel.get("kernel_segment_count", 0)),
        "p13_kernel_profiles_considered": int(kernel.get("kernel_profiles_considered", 0)),
        "p13_kernel_direction_stats": kernel.get("kernel_direction_stats", {}),
        "p13_kernel_fit_residual_stats": kernel.get("kernel_fit_residual_stats", {}),
        "p13_kernel_bound_hits": kernel.get("kernel_bound_hits", {}),
        "_p13_kernel_psf": kernel.get("kernel_psf"),
    })
    if sigma is None:
        base_meta.update(segmentation)
        return rgb.copy(), base_meta
    sigma = float(sigma)
    region_stats = _p13_estimate_region_platforms(y_input, mask, sigma, params)
    base_meta.update({
        "p13_region_status": region_stats.get("status"),
        "p13_region_reason": region_stats.get("reason"),
        "p13_region_background_y": region_stats.get("background_y"),
        "p13_region_foreground_y": region_stats.get("foreground_y"),
        "p13_region_low_y": region_stats.get("low_y"),
        "p13_region_high_y": region_stats.get("high_y"),
        "p13_region_contrast": region_stats.get("contrast"),
        "p13_region_uncertainty": region_stats.get("uncertainty"),
        "p13_region_background_count": int(region_stats.get("background_count", 0)),
        "p13_region_foreground_count": int(region_stats.get("foreground_count", 0)),
        "p13_region_bbox": region_stats.get("region_bbox"),
        "_p13_region_mask": region_stats.get("_region_mask", empty_mask),
        "_p13_region_stats": region_stats,
    })
    region_mask = region_stats.get("_region_mask", empty_mask)
    deblur_weight, _ = _p11_processing_soft_band(mask, sigma)
    base_meta["_p13_processing_weight"] = deblur_weight
    lambda_candidates = params.get("lambda_candidates", [params.get("lambda", 0.01), 0.03])
    alpha_candidates = params.get("alpha_candidates", [0.5, 1.0])
    if not isinstance(lambda_candidates, (list, tuple)) or not isinstance(alpha_candidates, (list, tuple)):
        raise ValueError("P13 lambda_candidates and alpha_candidates must be lists")
    lambdas = [float(value) for value in lambda_candidates]
    alphas = [float(value) for value in alpha_candidates]
    if not lambdas or not alphas or any(not math.isfinite(value) or value <= 0.0 for value in lambdas) or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("P13 lambda/alpha candidate values are invalid")
    stage_outputs: list[dict[str, Any]] = []
    for regularization in lambdas:
        raw_deblur_y, wiener_meta = _p11_wiener_hunt_deblur(y_input, mask, sigma, regularization, params)
        deblur_y = (y_input + deblur_weight * (raw_deblur_y - y_input)).astype(np.float32)
        if not np.isfinite(deblur_y).all():
            raise ValueError("P13 deblur output is non-finite")
        deblur_rgb, deblur_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, deblur_y)
        profile_meta = _p13_measure_spatial_profiles(deblur_y, mask, region_stats, sigma, params)
        profiles = profile_meta.get("profiles", [])
        if profiles:
            a_y, a_correction, field_meta = _p13_apply_curve_field(deblur_y, mask, region_mask, profiles)
            a_y = np.clip(a_y, 0.0, 1.0).astype(np.float32)
            profile_meta.update(field_meta)
            profile_meta["profile_support_fraction"] = float(field_meta["profile_field_support_fraction"])
            a_status = "computed"
        else:
            a_y = deblur_y.copy()
            a_correction = np.zeros_like(y_input, dtype=np.float32)
            field_meta = {
                "profile_field_support_fraction": 0.0,
                "profile_field_region_fraction": float(np.mean(region_mask)),
                "profile_field_outside_max_abs_delta": 0.0,
                "profile_field_outside_sum_abs_delta": 0.0,
                "_profile_field_support": np.zeros_like(y_input, dtype=bool),
                "_profile_field_labels": np.zeros_like(y_input, dtype=np.int32),
            }
            profile_meta.update(field_meta)
            profile_meta["profile_support_fraction"] = 0.0
            a_status = "skipped_no_valid_profiles"
        a_rgb, a_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, a_y)
        b_stages: list[dict[str, Any]] = []
        for alpha in alphas:
            if region_stats.get("status") == "computed":
                b_y, b_meta = _p13_apply_shoulder_lut(
                    deblur_y,
                    region_mask,
                    float(region_stats["background_y"]),
                    float(region_stats["foreground_y"]),
                    alpha,
                )
                b_status = "computed"
            else:
                b_y = deblur_y.copy()
                b_meta = {
                    "p13_b_status": "skipped_no_region_platform",
                    "p13_b_alpha": alpha,
                    "p13_b_low": None,
                    "p13_b_high": None,
                    "p13_b_roi_fraction": float(np.mean(region_mask)),
                    "p13_b_dark_compression_pixels": 0,
                    "p13_b_bright_compression_pixels": 0,
                    "p13_b_dark_compression_sum_abs": 0.0,
                    "p13_b_bright_compression_sum_abs": 0.0,
                    "p13_b_mean_abs_correction": 0.0,
                    "p13_b_p95_abs_correction": 0.0,
                    "p13_b_outside_max_abs_delta": 0.0,
                    "p13_b_outside_sum_abs_delta": 0.0,
                    "_p13_b_correction": np.zeros_like(y_input, dtype=np.float32),
                    "_p13_b_lut_x": np.linspace(0.0, 1.0, 256, dtype=np.float32),
                    "_p13_b_lut_y": np.linspace(0.0, 1.0, 256, dtype=np.float32),
                }
                b_status = "skipped_no_region_platform"
            b_rgb, b_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, b_y)
            b_stages.append({
                "alpha": alpha,
                "status": b_status,
                "output": b_rgb,
                "y": b_y,
                "rgb_meta": b_rgb_meta,
                "meta": b_meta,
            })
        stage_outputs.append({
            "lambda": regularization,
            "deblur_status": "computed",
            "deblur_output": deblur_rgb,
            "deblur_y": deblur_y,
            "raw_deblur_y": raw_deblur_y,
            "deblur_rgb_meta": deblur_rgb_meta,
            "wiener_meta": wiener_meta,
            "a_status": a_status,
            "a_output": a_rgb,
            "a_y": a_y,
            "a_correction": a_correction,
            "a_rgb_meta": a_rgb_meta,
            "a_meta": profile_meta,
            "b_stages": b_stages,
        })
    if not stage_outputs:
        raise ValueError("P13 produced no finite deblur stage")
    stage_telemetry: list[dict[str, Any]] = []
    for stage in stage_outputs:
        stage_telemetry.append({
            "lambda": float(stage["lambda"]),
            "deblur_status": stage["deblur_status"],
            "p13_a_status": stage["a_status"],
            "p13_a_profile_count": int(stage["a_meta"].get("profile_count", 0)),
            "p13_a_rejected_count": int(stage["a_meta"].get("profile_rejected_count", 0)),
            "p13_a_gap_count": int(stage["a_meta"].get("profile_gap_count", 0)),
            "p13_a_coverage_fraction": float(stage["a_meta"].get("profile_coverage_fraction", 0.0)),
            "p13_a_support_fraction": float(stage["a_meta"].get("profile_support_fraction", 0.0)),
            "p13_a_reverse_steps_before": int(stage["a_meta"].get("profile_reverse_steps_before", 0)),
            "p13_a_reverse_steps_after": int(stage["a_meta"].get("profile_reverse_steps_after", 0)),
            "p13_deblur_mean_abs_delta": float(stage["wiener_meta"].get("wiener_raw_mean_abs_delta", 0.0)),
            "p13_deblur_p95_abs_delta": float(stage["wiener_meta"].get("wiener_raw_p95_abs_delta", 0.0)),
            "b_stages": [
                {
                    "alpha": float(item["alpha"]),
                    "status": item["status"],
                    **{key: value for key, value in item["meta"].items() if not key.startswith("_")},
                }
                for item in stage["b_stages"]
            ],
        })
    selected = stage_outputs[0]
    selected_a_meta = selected["a_meta"]
    selected_b = selected["b_stages"][0] if selected["b_stages"] else None
    meta = dict(base_meta)
    meta.update({
        "p13_status": "computed",
        "p13_kernel_status": "computed",
        "p13_deblur_status": "computed",
        "p13_a_status": selected["a_status"],
        "p13_a_profile_count": int(selected_a_meta.get("profile_count", 0)),
        "p13_a_rejected_count": int(selected_a_meta.get("profile_rejected_count", 0)),
        "p13_a_gap_count": int(selected_a_meta.get("profile_gap_count", 0)),
        "p13_a_coverage_fraction": float(selected_a_meta.get("profile_coverage_fraction", 0.0)),
        "p13_a_gap_fraction": float(selected_a_meta.get("profile_gap_fraction", 0.0)),
        "p13_a_support_fraction": float(selected_a_meta.get("profile_support_fraction", 0.0)),
        "p13_a_reverse_steps_before": int(selected_a_meta.get("profile_reverse_steps_before", 0)),
        "p13_a_reverse_steps_after": int(selected_a_meta.get("profile_reverse_steps_after", 0)),
        "p13_a_initial_radius": selected_a_meta.get("profile_initial_radius"),
        "p13_a_max_radius": selected_a_meta.get("profile_max_radius"),
        "p13_a_field_region_fraction": float(selected_a_meta.get("profile_field_region_fraction", 0.0)),
        "p13_a_field_outside_max_abs_delta": float(selected_a_meta.get("profile_field_outside_max_abs_delta", 0.0)),
        "p13_a_field_outside_sum_abs_delta": float(selected_a_meta.get("profile_field_outside_sum_abs_delta", 0.0)),
        "p13_b_status": selected_b["status"] if selected_b is not None else "skipped_no_stage",
        "p13_b_roi_fraction": float(selected_b["meta"].get("p13_b_roi_fraction", 0.0)) if selected_b is not None else 0.0,
        "p13_stage_count": len(stage_outputs),
        "p13_stage_telemetry": stage_telemetry,
        "_p13_stage_outputs": stage_outputs,
        "_p13_profiles": selected_a_meta.get("profiles", []),
        "_p13_a_correction": selected["a_correction"],
        "_p13_a_support": selected_a_meta.get("_profile_field_support", np.zeros_like(y_input, dtype=bool)),
        "_p13_a_labels": selected_a_meta.get("_profile_field_labels", np.zeros_like(y_input, dtype=np.int32)),
        "_p13_output_minus_deblur": selected["a_output"] - selected["deblur_output"],
    })
    if selected_b is not None:
        meta.update({key: value for key, value in selected_b["meta"].items() if not key.startswith("_")})
    meta.update(segmentation)
    return selected["a_output"], meta


def _p14_gaussian_field(
    values: np.ndarray, seed: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """@luna-comment Smooth sparse platform samples with normalized Gaussian support.

    Input: Finite HxW luma values, boolean platform seed, and positive smoothing
      scale in pixels.
    Output: A finite normalized value field and a [0,1] support field; pixels
      without enough seed support are represented by zero support.
    Key Steps: Blur the value/support numerators independently and divide only
      where the denominator is positive, preserving spatial luminance trends.
    Key Params: The caller derives sigma from the measured blur kernel; no
      rectangular ROI or global platform constant is introduced here.
    State/Dependencies: OpenCV GaussianBlur and read-only numpy arrays.
    """
    if values.ndim != 2 or seed.shape != values.shape or seed.dtype != np.bool_:
        raise ValueError("P14 Gaussian-field inputs are invalid")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(values).all():
        raise ValueError("P14 Gaussian-field parameters are invalid")
    seed_f = seed.astype(np.float32)
    numerator = cv2.GaussianBlur(values.astype(np.float32) * seed_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    denominator = cv2.GaussianBlur(seed_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    field = np.zeros_like(values, dtype=np.float32)
    valid = denominator > 1.0e-8
    field[valid] = numerator[valid] / denominator[valid]
    support = np.clip(denominator, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(field).all() or not np.isfinite(support).all():
        raise ValueError("P14 Gaussian field is non-finite")
    return field, support


def _p14_estimate_local_platform_fields(
    y: np.ndarray, mask: np.ndarray, sigma: float, params: dict[str, Any]
) -> dict[str, Any]:
    """@luna-comment Estimate spatially varying background/foreground platforms.

    Input: Finite input luma, accepted boolean text mask, equivalent blur sigma,
      and P14 field parameters.
    Output: Per-pixel background/foreground/tolerance fields, normalized seed
      supports, component labels, a contour-derived region mask/feather, and
      JSON-safe status/uncertainty telemetry.
    Key Steps: Seed background away from text, seed each connected text
      component independently, smooth numerator/support pairs, and derive
      local MAD tolerance from both seed populations.
    Key Params: Seed distance is max(2,2*sigma); Gaussian field bandwidth is
      max(6,4*sigma); all remaining thresholds are explicit config values.
    State/Dependencies: CPU OpenCV/numpy only; no GT or file writes.
    """
    if y.ndim != 2 or mask.shape != y.shape or mask.dtype != np.bool_ or not np.isfinite(y).all():
        raise ValueError("P14 platform-field inputs are invalid")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("P14 platform-field sigma is invalid")
    shape = y.shape
    empty = np.zeros(shape, dtype=bool)
    zero = np.zeros(shape, dtype=np.float32)
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return {
            "status": "unidentifiable_empty_text",
            "reason": "empty_text_mask",
            "background_field": zero.copy(),
            "foreground_field": zero.copy(),
            "tau_field": zero.copy(),
            "background_support": zero.copy(),
            "foreground_support": zero.copy(),
            "field_support": zero.copy(),
            "region_mask": empty.copy(),
            "region_feather": zero.copy(),
            "component_labels": np.zeros(shape, dtype=np.int32),
            "background_seed": empty.copy(),
            "foreground_seed": empty.copy(),
            "background_seed_count": 0,
            "foreground_seed_count": 0,
            "component_count": 0,
            "background_y": None,
            "foreground_y": None,
            "background_mad": None,
            "foreground_mad": None,
            "uncertainty": None,
            "region_fraction": 0.0,
        }
    # P15 keeps the background seeds well outside the contour halo.  A wider
    # local window is intentional: it follows slow illumination gradients
    # without turning the whole text crop into one constant platform.
    seed_distance = max(4.0, float(math.ceil(4.0 * sigma)))
    field_sigma = max(8.0, 6.0 * sigma)
    region_radius = max(3, int(math.ceil(3.0 * sigma)) + int(math.ceil(2.0 * sigma)))
    region_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * region_radius + 1, 2 * region_radius + 1)
    )
    region_mask = cv2.dilate(mask.astype(np.uint8), region_kernel, iterations=1) > 0
    # The feather is attached to the actual dilated component set. It reaches
    # one over the interior and fades continuously at the region boundary.
    region_distance = cv2.distanceTransform(region_mask.astype(np.uint8), cv2.DIST_L2, 5)
    feather_width = max(1.0, min(float(region_radius), max(1.0, sigma)))
    region_feather = np.clip(region_distance / feather_width, 0.0, 1.0).astype(np.float32)
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    background_seed = region_mask & ~mask & (outside_distance >= seed_distance)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    foreground_seed = mask & (inside_distance >= seed_distance)
    background_field, background_support = _p14_gaussian_field(y, background_seed, field_sigma)
    foreground_field = np.zeros_like(y, dtype=np.float32)
    foreground_support = np.zeros_like(y, dtype=np.float32)
    # Each text component gets its own foreground field. At overlaps, the
    # component with the strongest normalized seed support wins.
    for label in range(1, int(component_count)):
        component = component_labels == label
        component_seed = foreground_seed & component
        if not bool(component_seed.any()):
            continue
        local_field, local_support = _p14_gaussian_field(y, component_seed, field_sigma)
        local_region = cv2.dilate(component.astype(np.uint8), region_kernel, iterations=1) > 0
        take = local_region & (local_support > foreground_support)
        foreground_field[take] = local_field[take]
        foreground_support[take] = local_support[take]
    field_support = np.minimum(background_support, foreground_support).astype(np.float32)
    min_support = float(params.get("p14_min_field_support", 1.0e-3))
    if not math.isfinite(min_support) or min_support < 0.0 or min_support > 1.0:
        raise ValueError("P14 field support threshold is invalid")

    def _robust_stats(seed: np.ndarray) -> tuple[float | None, float | None, int]:
        values = y[seed].astype(np.float64)
        if values.size == 0:
            return None, None, 0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, mad, int(values.size)

    background_y, background_mad, background_count = _robust_stats(background_seed)
    foreground_y, foreground_mad, foreground_count = _robust_stats(foreground_seed)
    deviation = np.zeros_like(y, dtype=np.float32)
    if background_y is not None:
        deviation[background_seed] = np.abs(y[background_seed] - background_field[background_seed])
    if foreground_y is not None:
        foreground_deviation = np.zeros_like(y, dtype=np.float32)
        foreground_deviation[foreground_seed] = np.abs(y[foreground_seed] - foreground_field[foreground_seed])
    else:
        foreground_deviation = np.zeros_like(y, dtype=np.float32)
    bg_dev_field, bg_dev_support = _p14_gaussian_field(deviation, background_seed, field_sigma) if bool(background_seed.any()) else (zero.copy(), zero.copy())
    fg_dev_field, fg_dev_support = _p14_gaussian_field(foreground_deviation, foreground_seed, field_sigma) if bool(foreground_seed.any()) else (zero.copy(), zero.copy())
    local_mad = np.maximum(
        np.where(bg_dev_support > 1.0e-8, bg_dev_field, 0.0),
        np.where(fg_dev_support > 1.0e-8, fg_dev_field, 0.0),
    ).astype(np.float32)
    tau_field = np.clip(2.0 * local_mad, 2.0 / 255.0, 6.0 / 255.0).astype(np.float32)
    valid_global = background_count > 0 and foreground_count > 0 and bool((field_support >= min_support).any())
    global_contrast = None if background_y is None or foreground_y is None else abs(float(foreground_y - background_y))
    status = "computed" if valid_global else "uncertain_field_support"
    reason = None if valid_global else "missing_platform_seed_or_support"
    uncertainty = None
    if global_contrast is not None and global_contrast > 1.0e-8:
        uncertainty = float(max(background_mad or 0.0, foreground_mad or 0.0) / global_contrast)
    return {
        "status": status,
        "reason": reason,
        "background_field": background_field,
        "foreground_field": foreground_field,
        "tau_field": tau_field,
        "background_support": background_support,
        "foreground_support": foreground_support,
        "field_support": field_support,
        "region_mask": region_mask,
        "region_feather": region_feather,
        "component_labels": component_labels.astype(np.int32),
        "background_seed": background_seed,
        "foreground_seed": foreground_seed,
        "background_seed_count": background_count,
        "foreground_seed_count": foreground_count,
        "component_count": max(int(component_count) - 1, 0),
        "background_y": background_y,
        "foreground_y": foreground_y,
        "background_mad": background_mad,
        "foreground_mad": foreground_mad,
        "uncertainty": uncertainty,
        "global_contrast": global_contrast,
        "field_sigma": field_sigma,
        "seed_distance": seed_distance,
        "region_radius": region_radius,
        "region_fraction": float(np.mean(region_mask)),
        "field_support_fraction": float(np.mean(field_support >= min_support)),
        "_min_support": min_support,
    }


def _p14_side_interval(
    offsets: np.ndarray,
    mask_values: np.ndarray,
    distance_values: np.ndarray,
    side: str,
) -> tuple[int, int] | None:
    """@luna-comment Find one contour side interval without crossing a rival edge.

    Input: Ordered signed normal offsets, sampled text-class values, distance
      to the nearest contour, and either outside or inside side.
    Output: Inclusive index interval beginning at the contour and ending at the
      first class/medial-axis/image-boundary stop, or None if unavailable.
    Key Steps: Walk outward from the contour, require the expected class, and
      stop at the first competing-contour or distance-turning point.
    Key Params: A single one-sample class mismatch is tolerated at the raster
      contour; no full-width fixed normal interval is forced.
    State/Dependencies: Pure finite arrays; no image mutation.
    """
    if offsets.ndim != 1 or mask_values.shape != offsets.shape or distance_values.shape != offsets.shape:
        raise ValueError("P14 side-interval arrays are invalid")
    zero_candidates = np.flatnonzero(np.abs(offsets) <= 0.5 + 1.0e-6)
    if zero_candidates.size == 0:
        return None
    zero_index = int(zero_candidates[np.argmin(np.abs(offsets[zero_candidates]))])
    expected = side == "inside"
    if side == "outside":
        indices = list(range(zero_index, -1, -1))
    elif side == "inside":
        indices = list(range(zero_index, offsets.size))
    else:
        raise ValueError("P14 side must be outside or inside")
    chosen: list[int] = []
    previous_distance = -1.0
    turning_index: int | None = None
    for position, index in enumerate(indices):
        class_value = bool(mask_values[index])
        if class_value != expected and not chosen:
            # The nearest raster sample can still belong to the other side of
            # a one-pixel contour. Skip that leading sample and start at the
            # first expected-class sample; a later mismatch terminates the
            # single-side interval.
            continue
        if class_value != expected:
            break
        current_distance = float(distance_values[index])
        if previous_distance >= 0.0 and current_distance + 0.35 < previous_distance:
            turning_index = index
            break
        chosen.append(index)
        previous_distance = max(previous_distance, current_distance)
        if position > 1 and current_distance > 0.0 and current_distance <= previous_distance + 0.35:
            # Keep the current sample; the next decreasing sample marks the
            # medial/competing boundary.
            continue
    if len(chosen) < 2:
        return None
    if turning_index is not None and turning_index in chosen:
        chosen = chosen[: chosen.index(turning_index) + 1]
    if not chosen:
        return None
    return min(chosen), max(chosen)


def _p14_sample_field(field: np.ndarray, x: float, y: float) -> float:
    """@luna-comment Bilinearly sample one finite scalar platform field.

    Input: HxW float field and floating-point image coordinate in pixel units.
    Output: Finite interpolated scalar, using reflected border only for the
      one-pixel sampling envelope required by the diagnostic.
    Key Steps: Call OpenCV remap with one coordinate and validate finiteness.
    Key Params: Linear interpolation preserves smooth regional platform drift.
    State/Dependencies: Read-only numpy/OpenCV field.
    """
    if field.ndim != 2 or not np.isfinite(field).all() or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("P14 scalar-field sample is invalid")
    value = cv2.remap(
        field.astype(np.float32),
        np.asarray([[float(x)]], dtype=np.float32),
        np.asarray([[float(y)]], dtype=np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    result = float(value[0, 0])
    if not math.isfinite(result):
        raise ValueError("P14 scalar-field sample is non-finite")
    return result


def _p14_measure_contour_profiles(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    fields: dict[str, Any],
    sigma: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """@luna-comment Measure contour-limited normal curves for P14 branch A.

    Input: Finite deblur luma, input text mask, local platform fields, kernel
      sigma, and profile geometry parameters.
    Output: Ordered accepted contour profiles with side-limited offsets and
      bounded monotone target deltas, plus explicit gap/coverage telemetry.
    Key Steps: Enumerate ordered contours, derive signed-distance normals, clip
      outside/inside sides independently at image/rival/medial limits, and
      isotonic-project only the resulting single contour interval.
    Key Params: Search radius is min(32,ceil(4*sigma+4)); contour interpolation
      tolerates only angle <30 degrees and gaps <=4 pixels.
    State/Dependencies: OpenCV remap/contours and numpy; no GT or writes.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or mask.dtype != np.bool_ or not np.isfinite(y_deblur).all():
        raise ValueError("P14 contour-profile inputs are invalid")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("P14 contour-profile sigma is invalid")
    background_field = np.asarray(fields.get("background_field"), dtype=np.float32)
    foreground_field = np.asarray(fields.get("foreground_field"), dtype=np.float32)
    field_support = np.asarray(fields.get("field_support"), dtype=np.float32)
    if background_field.shape != mask.shape or foreground_field.shape != mask.shape or field_support.shape != mask.shape:
        raise ValueError("P14 platform fields have incompatible shapes")
    min_support = float(fields.get("_min_support", params.get("p14_min_field_support", 1.0e-3)))
    default_radius = float(math.ceil(4.0 * sigma + 4.0))
    if bool(params.get("p16_mode", False)) or bool(params.get("p17_mode", False)):
        default_radius = max(default_radius, float(params.get("p16_background_radius", 24.0)))
    max_radius = min(32.0, default_radius)
    configured_max = float(params.get("p14_max_radius", max_radius))
    if not math.isfinite(configured_max) or configured_max <= 0.0:
        raise ValueError("P14 maximum profile radius is invalid")
    max_radius = min(max_radius, configured_max)
    step = float(params.get("p14_sample_step", params.get("sample_step", 0.5)))
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("P14 profile sample step is invalid")
    stride = max(1, int(params.get("p14_boundary_stride", 1)))
    max_profiles = max(1, int(params.get("p14_max_profiles", 4096)))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    signed = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) - cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_boundary = cv2.distanceTransform((cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) == 0).astype(np.uint8), cv2.DIST_L2, 5)
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    offsets = np.arange(-max_radius, max_radius + step * 0.5, step, dtype=np.float32)
    profiles: list[dict[str, Any]] = []
    rejected = 0
    gaps = 0
    reverse_before = 0
    reverse_after = 0
    boundary_total = 0
    for contour_id, contour in enumerate(contours):
        points = contour.reshape(-1, 2)
        if points.shape[0] < 2:
            continue
        for contour_order in range(0, points.shape[0], stride):
            column, row = (int(value) for value in points[contour_order])
            boundary_total += 1
            magnitude = float(norm[row, column])
            if magnitude <= 1.0e-5:
                rejected += 1
                gaps += 1
                continue
            normal_x = float(grad_x[row, column] / magnitude)
            normal_y = float(grad_y[row, column] / magnitude)
            map_x = float(column) + offsets * normal_x
            map_y = float(row) + offsets * normal_y
            valid_coord = (map_x >= 0.0) & (map_x <= y_deblur.shape[1] - 1.0) & (map_y >= 0.0) & (map_y <= y_deblur.shape[0] - 1.0)
            if int(np.sum(valid_coord)) < 4:
                rejected += 1
                gaps += 1
                continue
            values = cv2.remap(y_deblur, map_x.reshape(1, -1).astype(np.float32), map_y.reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1).astype(np.float32)
            mask_values = cv2.remap(mask.astype(np.float32), map_x.reshape(1, -1).astype(np.float32), map_y.reshape(1, -1).astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101).reshape(-1) >= 0.5
            distance_values = cv2.remap(distance_to_boundary.astype(np.float32), map_x.reshape(1, -1).astype(np.float32), map_y.reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(-1)
            outside = _p14_side_interval(offsets, mask_values, distance_values, "outside")
            inside = _p14_side_interval(offsets, mask_values, distance_values, "inside")
            if outside is None or inside is None:
                rejected += 1
                gaps += 1
                continue
            # The active interval must span both measured sides, from the
            # farthest valid outside sample through the contour to the
            # farthest valid inside sample.  The previous max(outside)/
            # min(inside) choice retained only the narrow overlap at the
            # contour and therefore almost never changed a pixel.
            start_index = min(outside[0], outside[1])
            end_index = max(inside[0], inside[1])
            if not start_index < end_index:
                rejected += 1
                gaps += 1
                continue
            b0 = _p14_sample_field(background_field, float(column), float(row))
            f0 = _p14_sample_field(foreground_field, float(column), float(row))
            support0 = _p14_sample_field(field_support, float(column), float(row))
            contrast = f0 - b0
            min_contrast = float(params.get("p14_min_profile_contrast", params.get("min_profile_contrast", 0.12)))
            if not math.isfinite(contrast) or abs(contrast) < min_contrast or support0 < min_support:
                rejected += 1
                gaps += 1
                continue
            active = np.arange(offsets.size) >= start_index
            active &= np.arange(offsets.size) <= end_index
            active &= valid_coord
            active_indices = np.flatnonzero(active)
            if active_indices.size < 3:
                rejected += 1
                gaps += 1
                continue
            oriented = ((values.astype(np.float64) - b0) / contrast).astype(np.float32)
            if not np.isfinite(oriented).all():
                rejected += 1
                gaps += 1
                continue
            raw_active = oriented[active]
            projected_active = np.clip(_p10_isotonic_nondecreasing(np.clip(raw_active, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
            target_values = values.copy()
            target_values[active] = (b0 + contrast * projected_active).astype(np.float32)
            delta = np.where(active, target_values - values, 0.0).astype(np.float32)
            raw_reverse = int(np.sum(np.diff(raw_active) < -1.0 / 255.0)) if raw_active.size > 1 else 0
            projected_reverse = int(np.sum(np.diff(projected_active) < -1.0 / 255.0)) if projected_active.size > 1 else 0
            reverse_before += raw_reverse
            reverse_after += projected_reverse
            # Tangent angle is used by the field stage to prevent interpolation
            # across corners and neighboring contours.
            prior = points[(contour_order - 1) % points.shape[0]].astype(np.float32)
            following = points[(contour_order + 1) % points.shape[0]].astype(np.float32)
            tangent = following - prior
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= 1.0e-6:
                tangent_x, tangent_y = 1.0, 0.0
            else:
                tangent_x, tangent_y = float(tangent[0] / tangent_norm), float(tangent[1] / tangent_norm)
            profiles.append({
                "contour_id": int(contour_id),
                "contour_order": int(contour_order),
                "row": row,
                "column": column,
                "normal_x": normal_x,
                "normal_y": normal_y,
                "tangent_x": tangent_x,
                "tangent_y": tangent_y,
                "background_y": b0,
                "foreground_y": f0,
                "contrast": contrast,
                "field_support": support0,
                "active_offset": [float(offsets[start_index]), float(offsets[end_index])],
                "active_width_px": float(offsets[end_index] - offsets[start_index]),
                "active_valid_sample_count": int(np.sum(active)) if "active" in locals() else 0,
                "active_raw_min": float(np.min(raw_active)),
                "active_raw_max": float(np.max(raw_active)),
                "active_raw_peak_count": int(np.sum(raw_active > 1.0 + 1.0 / 255.0)),
                "active_raw_valley_count": int(np.sum(raw_active < -1.0 / 255.0)),
                "raw_reverse_steps": raw_reverse,
                "projected_reverse_steps": projected_reverse,
                "offsets": [float(value) for value in offsets],
                "values": [float(value) for value in values],
                "projected_coordinates": [0.0 for _ in range(offsets.size)],
                "target_values": [float(value) for value in target_values],
                "_map_x": map_x.astype(np.float32),
                "_map_y": map_y.astype(np.float32),
                "_delta": delta,
                "_active": active,
            })
            if len(profiles) >= max_profiles:
                break
        if len(profiles) >= max_profiles:
            break
    # Replace the intentionally compact public placeholder with the true
    # projected coordinate array for each accepted profile.
    for profile in profiles:
        values = np.asarray(profile["values"], dtype=np.float32)
        b0 = float(profile["background_y"])
        contrast = float(profile["contrast"])
        active = np.asarray(profile["_active"], dtype=bool)
        projected = np.clip(_p10_isotonic_nondecreasing(np.clip((values[active] - b0) / contrast, 0.0, 1.0)), 0.0, 1.0)
        full = np.zeros_like(values, dtype=np.float32)
        full[active] = projected
        profile["projected_coordinates"] = [float(value) for value in full]
    return {
        "profiles": profiles,
        "profile_count": len(profiles),
        "profile_rejected_count": rejected,
        "profile_gap_count": gaps,
        "profile_geometry_status": "computed" if profiles else "no_valid_profiles",
        "profile_reverse_steps_before": reverse_before,
        "profile_reverse_steps_after": reverse_after,
        "profile_coverage_fraction": float(len(profiles) / max(boundary_total, 1)),
        "profile_gap_fraction": float(gaps / max(boundary_total, 1)),
        "profile_boundary_count": boundary_total,
        "profile_initial_radius": 0.0,
        "profile_max_radius": max_radius,
        "profile_interpolation_gap_limit": 4.0,
        "profile_interpolation_angle_limit": 30.0,
    }


def _p14_apply_curve_field(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    region_mask: np.ndarray,
    profiles: list[dict[str, Any]],
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """@luna-comment Apply the continuous P14 spatial-monotone correction field.

    Input: Finite deblur luma, text mask, contour-derived region mask, accepted
      P14 profiles, and field interpolation parameters.
    Output: Finite target luma, correction/support maps, and exact region/outside
      telemetry. Unsupported long contour gaps remain unchanged.
    Key Steps: Assign each pixel to a nearest contour profile, reject tangent
      mismatches or gaps over four pixels, interpolate its profile delta, and
      apply an independent endpoint fade so profile weights cannot cancel.
    Key Params: Same-contour angle limit is 30 degrees, gap limit is 4 pixels,
      and endpoint feather defaults to one pixel.
    State/Dependencies: OpenCV distance labels and numpy; no RGB state.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or region_mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P14 curve-field inputs are invalid")
    correction = np.zeros_like(y_deblur, dtype=np.float32)
    support = np.zeros_like(y_deblur, dtype=bool)
    if not profiles:
        return y_deblur.copy(), correction, {
            "p14_a_field_support_fraction": 0.0,
            "p14_a_field_region_fraction": float(np.mean(region_mask)),
            "p14_a_field_outside_max_abs_delta": 0.0,
            "p14_a_field_outside_sum_abs_delta": 0.0,
            "p14_a_interpolated_profile_count": 0,
            "p14_a_long_gap_pixels": 0,
            "p14_a_angle_rejected_count": 0,
            "_p14_a_support": support,
            "_p14_a_correction": correction,
        }
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    boundary_coordinates = np.argwhere(boundary)
    if boundary_coordinates.size == 0:
        return y_deblur.copy(), correction, {
            "p14_a_field_support_fraction": 0.0,
            "p14_a_field_region_fraction": float(np.mean(region_mask)),
            "p14_a_field_outside_max_abs_delta": 0.0,
            "p14_a_field_outside_sum_abs_delta": 0.0,
            "p14_a_interpolated_profile_count": 0,
            "p14_a_long_gap_pixels": 0,
            "p14_a_angle_rejected_count": 0,
            "_p14_a_support": support,
            "_p14_a_correction": correction,
        }
    # Labels from DIST_LABEL_PIXEL enumerate zero pixels (the contour) in row
    # major order. Mapping by boundary order keeps this deterministic and does
    # not average sparse profile deltas.
    _, labels = cv2.distanceTransformWithLabels((~boundary).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    signed = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) - cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)
    coord_to_profile: dict[tuple[int, int], dict[str, Any]] = {
        (int(profile["row"]), int(profile["column"])): profile for profile in profiles
    }
    profile_rows = np.asarray([int(profile["row"]) for profile in profiles], dtype=np.float32)
    profile_cols = np.asarray([int(profile["column"]) for profile in profiles], dtype=np.float32)
    angle_limit_cos = math.cos(math.radians(float(params.get("p14_interpolation_angle_deg", 30.0))))
    gap_limit = float(params.get("p14_interpolation_gap_px", 4.0))
    endpoint_feather = float(params.get("p14_endpoint_feather_px", 1.0))
    if not (math.isfinite(angle_limit_cos) and 0.0 < gap_limit <= 32.0 and math.isfinite(endpoint_feather) and endpoint_feather > 0.0):
        raise ValueError("P14 curve-field interpolation parameters are invalid")
    interpolated_count = 0
    long_gap_pixels = 0
    angle_rejected = 0
    yy, xx = np.nonzero(region_mask)
    for row, column in zip(yy.tolist(), xx.tolist()):
        label = int(labels[row, column])
        candidate: dict[str, Any] | None = None
        if 0 < label <= boundary_coordinates.shape[0]:
            boundary_row, boundary_column = (int(value) for value in boundary_coordinates[label - 1])
            candidate = coord_to_profile.get((boundary_row, boundary_column))
            if candidate is None:
                # P15 deliberately does not use the nearest profile from an
                # arbitrary contour.  A missing exact contour sample is a
                # gap, not permission to cross onto a neighboring stroke.
                long_gap_pixels += 1
        if candidate is None:
            continue
        profile = candidate
        normal_x = float(profile["normal_x"])
        normal_y = float(profile["normal_y"])
        local_nx = float(grad_x[boundary_row, boundary_column])
        local_ny = float(grad_y[boundary_row, boundary_column])
        local_norm = math.hypot(local_nx, local_ny)
        if local_norm > 1.0e-6:
            if abs((local_nx / local_norm) * normal_x + (local_ny / local_norm) * normal_y) < angle_limit_cos:
                angle_rejected += 1
                continue
        profile_row = float(profile["row"])
        profile_column = float(profile["column"])
        offset = (float(column) - profile_column) * normal_x + (float(row) - profile_row) * normal_y
        active_start, active_end = (float(value) for value in profile["active_offset"])
        if offset < active_start or offset > active_end:
            continue
        profile_offsets = np.asarray(profile["offsets"], dtype=np.float32)
        profile_delta = np.asarray(profile["_delta"], dtype=np.float32)
        value = float(np.interp(offset, profile_offsets, profile_delta))
        distance_to_start = max(0.0, offset - active_start)
        distance_to_end = max(0.0, active_end - offset)
        fade = min(1.0, distance_to_start / endpoint_feather, distance_to_end / endpoint_feather)
        fade = 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, fade))))
        correction[row, column] = float(value * fade)
        support[row, column] = fade > 0.0
    output = (y_deblur + correction).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P14 curve-field output is non-finite")
    outside = ~region_mask
    return output, correction, {
        "p14_a_field_support_fraction": float(np.mean(support)),
        "p14_a_field_region_fraction": float(np.mean(region_mask)),
        "p14_a_field_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_a_field_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_a_interpolated_profile_count": interpolated_count,
        "p14_a_long_gap_pixels": long_gap_pixels,
        "p14_a_angle_rejected_count": angle_rejected,
        "_p14_a_support": support,
        "_p14_a_correction": correction,
    }


def _p14_apply_local_soft_lut(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    fields: dict[str, Any],
    sigma: float,
    alpha: float,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply P14's locally supported, smoothly feathered shoulders.

    Input: Finite deblur luma, text mask, local LQ platform fields, kernel sigma,
      shoulder strength alpha, and radius/support parameters.
    Output: Finite Y result plus correction, local L/H/tau maps, support mask,
      and compression/edge telemetry.
    Key Steps: Form local L/H from B/F and tau, apply the quadratic shoulder
      residual, multiply contour and region feathers, and leave far background
      exactly unchanged.
    Key Params: Full contour radius is max(3,ceil(3*sigma)); zero radius adds
      max(2,ceil(2*sigma)); s=max(2/255,tau) stabilizes tiny overshoots.
    State/Dependencies: OpenCV distance transforms and numpy only.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P14 soft-LUT inputs are invalid")
    sigma = float(sigma)
    alpha = float(alpha)
    if not math.isfinite(sigma) or sigma <= 0.0 or not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P14 soft-LUT parameters are invalid")
    background_field = np.asarray(fields.get("background_field"), dtype=np.float32)
    foreground_field = np.asarray(fields.get("foreground_field"), dtype=np.float32)
    tau_field = np.asarray(fields.get("tau_field"), dtype=np.float32)
    field_support = np.asarray(fields.get("field_support"), dtype=np.float32)
    region_mask = np.asarray(fields.get("region_mask"), dtype=bool)
    region_feather = np.asarray(fields.get("region_feather"), dtype=np.float32)
    if any(value.shape != y_deblur.shape for value in (background_field, foreground_field, tau_field, field_support, region_mask, region_feather)):
        raise ValueError("P14 soft-LUT fields have incompatible shapes")
    min_support = float(fields.get("_min_support", params.get("p14_min_field_support", 1.0e-3)))
    if not math.isfinite(min_support) or min_support < 0.0 or min_support > 1.0:
        raise ValueError("P14 soft-LUT support threshold is invalid")
    low_platform = np.minimum(background_field, foreground_field)
    high_platform = np.maximum(background_field, foreground_field)
    if bool(params.get("p15_ablation_legacy_tau", False)):
        tau = np.maximum(2.0 / 255.0, tau_field).astype(np.float32)
    else:
        tau = np.clip(tau_field, 2.0 / 255.0, 6.0 / 255.0).astype(np.float32)
    low = low_platform - tau
    high = high_platform + tau
    # The shoulder stabilizer is independent of the local tolerance.  Using
    # tau here made broad-uncertainty regions almost identity transforms.
    s = 2.0 / 255.0
    d_low = np.maximum(low - y_deblur, 0.0)
    d_high = np.maximum(y_deblur - high, 0.0)
    delta = alpha * (np.square(d_low) / (d_low + s) - np.square(d_high) / (d_high + s))
    delta = np.where(field_support >= min_support, delta, 0.0).astype(np.float32)
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    # The two distance transforms are defined on opposite sides of the
    # contour.  Taking their minimum everywhere collapses the distance to
    # zero on both sides and makes the feather behave like a thin halo.  P15
    # selects the distance belonging to the pixel's actual class.
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
    r_full = max(4.0, float(math.ceil(4.0 * sigma)))
    r_zero = r_full + max(2.0, float(math.ceil(2.0 * sigma)))
    contour_weight = np.ones_like(y_deblur, dtype=np.float32)
    mid = (contour_distance > r_full) & (contour_distance < r_zero)
    contour_weight[contour_distance >= r_zero] = 0.0
    if bool(mid.any()):
        ratio = (contour_distance[mid] - r_full) / max(r_zero - r_full, 1.0e-8)
        contour_weight[mid] = (0.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    # Support is a validity gate only.  Multiplying the correction by local
    # density was the main reason P14 became visually indistinguishable from
    # deblur in sparse regions.
    support_gate = field_support >= min_support
    weight = contour_weight * region_feather
    weight = np.where(region_mask & support_gate, weight, 0.0).astype(np.float32)
    correction = (weight * delta).astype(np.float32)
    output = (y_deblur + correction).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("P14 soft-LUT output is non-finite")
    outside = ~region_mask
    dark = (d_low > 0.0) & (weight > 0.0)
    bright = (d_high > 0.0) & (weight > 0.0)
    return output, {
        "p14_b_status": "computed",
        "p14_b_alpha": alpha,
        "p14_b_r_full": r_full,
        "p14_b_r_zero": r_zero,
        "p14_b_roi_fraction": float(np.mean(region_mask)),
        "p14_b_weight_coverage_fraction": float(np.mean(weight > 0.0)),
        "p14_b_supported_fraction": float(np.mean(field_support >= min_support)),
        "p14_b_dark_compression_pixels": int(np.sum(dark)),
        "p14_b_bright_compression_pixels": int(np.sum(bright)),
        "p14_b_dark_compression_sum_abs": float(np.sum(np.abs(correction[dark]))),
        "p14_b_bright_compression_sum_abs": float(np.sum(np.abs(correction[bright]))),
        "p14_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p14_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)) if correction.size else 0.0,
        "p14_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_b_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p14_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p15_b_tau_mode": "legacy_uncapped" if bool(params.get("p15_ablation_legacy_tau", False)) else "clipped_2_6_per_255",
        "p15_b_support_weight_mode": "gate_only",
        "_p14_b_correction": correction,
        "_p14_b_low": low,
        "_p14_b_high": high,
        "_p14_b_tau": tau,
        "_p14_b_weight": weight,
        "_p14_b_delta_raw": delta,
    }


def _p16_apply_patch_shoulder_lut(
    y_deblur: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply the legacy P13 shoulder LUT using overlapping local patches.

    Input: Deblurred luma, text mask, shoulder strength, and P16 patch limits.
    Output: Luma result and JSON-safe patch/borrow/fusion telemetry.
    Key Steps: Estimate robust local background/foreground medians in 50x50
      patches, borrow missing platform scalars from the nearest valid patch,
      apply the exact P13 LUT per patch, and overlap-add its residual with a
      two-dimensional sin-squared window. Only the 16/24 px contour band writes.
    Key Params: Patch stride defaults to 25 px; background distance is 12 px;
      alpha .5/1.0 is the unchanged P13 LUT strength.
    State/Dependencies: Numpy/OpenCV only; no GT, RGB chroma, or quality gate.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P16 patch LUT inputs are invalid")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P16 patch LUT alpha is invalid")
    height, width = y_deblur.shape
    patch_size = max(3, int(params.get("p16_patch_size", 50)))
    stride = max(1, int(params.get("p16_patch_stride", 25)))
    bg_distance = max(1.0, float(params.get("p16_background_distance", 12.0)))
    fg_distance = max(0.0, float(params.get("p16_foreground_distance", 2.0)))
    full_radius = max(1.0, float(params.get("p16_write_full_radius", 16.0)))
    zero_radius = max(full_radius + 1.0, float(params.get("p16_write_zero_radius", 24.0)))
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
    starts_y = sorted(set(list(range(0, max(height - patch_size, 0) + 1, stride)) + [max(height - patch_size, 0)]))
    starts_x = sorted(set(list(range(0, max(width - patch_size, 0) + 1, stride)) + [max(width - patch_size, 0)]))
    valid: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for top in starts_y:
        bottom = min(height, top + patch_size)
        for left in starts_x:
            right = min(width, left + patch_size)
            patch_mask = mask[top:bottom, left:right]
            patch_y = y_deblur[top:bottom, left:right]
            patch_inside = inside_distance[top:bottom, left:right]
            patch_outside = outside_distance[top:bottom, left:right]
            bg_values = patch_y[(~patch_mask) & (patch_outside >= bg_distance)]
            fg_values = patch_y[patch_mask & (patch_inside >= fg_distance)]
            item = {
                "top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right),
                "center_y": float((top + bottom - 1) * 0.5), "center_x": float((left + right - 1) * 0.5),
                "background_count": int(bg_values.size), "foreground_count": int(fg_values.size),
                "background_y": float(np.median(bg_values)) if bg_values.size else None,
                "foreground_y": float(np.median(fg_values)) if fg_values.size else None,
                "background_borrowed": False, "foreground_borrowed": False,
                "background_borrow_distance": 0.0, "foreground_borrow_distance": 0.0,
            }
            if bg_values.size and fg_values.size and abs(float(np.median(fg_values)) - float(np.median(bg_values))) > 1.0e-6:
                valid.append(item)
            records.append(item)
    for item in records:
        for field, count_field, borrowed_field, distance_field in (
            ("background_y", "background_count", "background_borrowed", "background_borrow_distance"),
            ("foreground_y", "foreground_count", "foreground_borrowed", "foreground_borrow_distance"),
        ):
            if item[field] is not None:
                continue
            if not valid:
                continue
            nearest = min(valid, key=lambda other: (other["center_y"] - item["center_y"]) ** 2 + (other["center_x"] - item["center_x"]) ** 2)
            item[field] = nearest[field]
            item[borrowed_field] = True
            item[distance_field] = float(math.sqrt((nearest["center_y"] - item["center_y"]) ** 2 + (nearest["center_x"] - item["center_x"]) ** 2))
        if item["background_y"] is None or item["foreground_y"] is None:
            continue
        if abs(float(item["foreground_y"]) - float(item["background_y"])) <= 1.0e-6:
            continue
        item["usable"] = True
    residual_sum = np.zeros_like(y_deblur, dtype=np.float32)
    weight_sum = np.zeros_like(y_deblur, dtype=np.float32)
    bright = np.zeros_like(y_deblur, dtype=bool)
    dark = np.zeros_like(y_deblur, dtype=bool)
    for item in records:
        if not item.get("usable", False):
            continue
        top, left, bottom, right = (int(item[key]) for key in ("top", "left", "bottom", "right"))
        patch = y_deblur[top:bottom, left:right]
        low = min(float(item["background_y"]), float(item["foreground_y"]))
        high = max(float(item["background_y"]), float(item["foreground_y"]))
        target = patch.copy()
        below = patch < low
        above = patch > high
        target[below] = low + (1.0 - alpha) * (patch[below] - low)
        target[above] = high + (1.0 - alpha) * (patch[above] - high)
        ph, pw = patch.shape
        wy = np.sin(np.pi * (np.arange(ph, dtype=np.float32) + 0.5) / max(float(ph), 1.0)) ** 2
        wx = np.sin(np.pi * (np.arange(pw, dtype=np.float32) + 0.5) / max(float(pw), 1.0)) ** 2
        window = wy[:, None] * wx[None, :]
        residual_sum[top:bottom, left:right] += (target - patch) * window
        weight_sum[top:bottom, left:right] += window
        dark[top:bottom, left:right] |= below
        bright[top:bottom, left:right] |= above
    fused = np.divide(residual_sum, np.maximum(weight_sum, 1.0e-8), out=np.zeros_like(y_deblur), where=weight_sum > 0.0)
    write_weight = np.ones_like(y_deblur, dtype=np.float32)
    mid = (contour_distance > full_radius) & (contour_distance < zero_radius)
    write_weight[contour_distance >= zero_radius] = 0.0
    if bool(mid.any()):
        ratio = (contour_distance[mid] - full_radius) / max(zero_radius - full_radius, 1.0e-8)
        write_weight[mid] = (0.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    correction = (fused * write_weight).astype(np.float32)
    output = (y_deblur + correction).astype(np.float32)
    outside = write_weight <= 0.0
    if not np.isfinite(output).all():
        raise ValueError("P16 patch LUT output is non-finite")
    return output, {
        "p16_b_status": "computed" if valid else "skipped_no_valid_patch",
        "p16_b_alpha": alpha,
        "p16_b_patch_size": patch_size,
        "p16_b_patch_stride": stride,
        "p16_b_patch_count": len(records),
        "p16_b_valid_patch_count": len(valid),
        "p16_b_background_borrow_count": int(sum(bool(item["background_borrowed"]) for item in records if item.get("usable", False))),
        "p16_b_foreground_borrow_count": int(sum(bool(item["foreground_borrowed"]) for item in records if item.get("usable", False))),
        "p16_b_write_full_radius": full_radius,
        "p16_b_write_zero_radius": zero_radius,
        "p16_b_write_fraction": float(np.mean(write_weight > 0.0)),
        "p16_b_weight_fraction": float(np.mean(weight_sum > 0.0)),
        "p16_b_dark_compression_pixels": int(np.sum(dark & (write_weight > 0.0))),
        "p16_b_bright_compression_pixels": int(np.sum(bright & (write_weight > 0.0))),
        "p16_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p16_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)),
        "p16_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p16_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p16_b_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "_p16_b_correction": correction,
        "_p16_b_write_weight": write_weight,
        "_p16_b_weight_sum": weight_sum,
        "_p16_b_patch_records": records,
        "_p16_b_patch_grid": np.asarray([[float(item.get("background_y") or 0.0), float(item.get("foreground_y") or 0.0)] for item in records], dtype=np.float32),
    }


def _p17_robust_otsu_platforms(
    patch_y: np.ndarray, foreground_is_high: bool, params: dict[str, Any]
) -> tuple[float | None, float | None, dict[str, Any]]:
    """@luna-comment Estimate P17 platforms from Otsu classes without seed masks.

    Input: One finite deblurred-Y patch, established global polarity, and P17 bounds.
    Output: Low/high robust class centres or an explicit invalid-distribution reason.
    Key Steps: Blur only for histogram statistics, split with Otsu, robustly centre
      both classes, and reject weak occupancy or class separation.
    Key Params: Both Otsu classes need 12 percent occupancy; centres use p35--p65.
    State/Dependencies: Does not inspect a text mask or distance field.
    """
    sigma = max(0.0, float(params.get("p17_stats_gaussian_sigma", 0.6)))
    stats_y = cv2.GaussianBlur(patch_y.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma > 0.0 else patch_y.astype(np.float32)
    encoded = np.clip(np.rint(stats_y * 255.0), 0, 255).astype(np.uint8)
    threshold, classes = cv2.threshold(encoded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    high_class = classes > 0
    low_values = stats_y[~high_class]
    high_values = stats_y[high_class]
    fraction_low = float(low_values.size / max(stats_y.size, 1))
    fraction_high = float(high_values.size / max(stats_y.size, 1))
    record: dict[str, Any] = {
        "otsu_threshold_y": float(threshold / 255.0),
        "otsu_low_fraction": fraction_low,
        "otsu_high_fraction": fraction_high,
        "global_foreground_is_high": bool(foreground_is_high),
        "validity_reason": None,
    }
    min_fraction = float(params.get("p17_min_class_fraction", 0.12))
    if min(fraction_low, fraction_high) < min_fraction:
        record["validity_reason"] = "class_fraction_below_minimum"
        return None, None, record

    def robust_center(values: np.ndarray) -> float:
        lo, hi = np.percentile(values, [35.0, 65.0])
        return float(np.median(values[(values >= lo) & (values <= hi)]))

    low_center = robust_center(low_values)
    high_center = robust_center(high_values)
    dynamic_range = float(np.max(stats_y) - np.min(stats_y))
    separation = abs(high_center - low_center)
    minimum_separation = max(12.0 / 255.0, 0.08 * dynamic_range)
    record.update({
        "otsu_low_center_y": low_center,
        "otsu_high_center_y": high_center,
        "otsu_dynamic_range": dynamic_range,
        "otsu_separation": separation,
        "otsu_minimum_separation": minimum_separation,
        "foreground_y": high_center if foreground_is_high else low_center,
        "background_y": low_center if foreground_is_high else high_center,
    })
    if separation < minimum_separation:
        record["validity_reason"] = "class_separation_below_minimum"
        return None, None, record
    return min(low_center, high_center), max(low_center, high_center), record


def _p17_apply_regional_otsu_lut(
    y_deblur: np.ndarray, mask: np.ndarray, alpha: float, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Apply P16's shoulder LUT with P17 regional Otsu platforms.

    Input: Deblurred Y, text mask used only for global polarity/write band, alpha, and P17 configuration.
    Output: Y result plus per-patch Otsu telemetry and correction maps.
    Key Steps: Compute independent regional Otsu platforms, Hann overlap-add residuals,
      and gate the fused correction through P16's 16--24 pixel contour band.
    Key Params: 96/32 and 128/48 statistic domains are fixed candidate variants.
    State/Dependencies: No mask or distance values participate in patch platform statistics.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P17 Otsu LUT inputs are invalid")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P17 Otsu LUT alpha is invalid")
    height, width = y_deblur.shape
    patch_size = max(3, int(params.get("p17_patch_size", 96)))
    stride = max(1, int(params.get("p17_patch_stride", 32)))
    full_radius = max(1.0, float(params.get("p16_write_full_radius", 16.0)))
    zero_radius = max(full_radius + 1.0, float(params.get("p16_write_zero_radius", 24.0)))
    foreground_is_high = bool(np.median(y_deblur[mask]) >= np.median(y_deblur[~mask])) if bool(mask.any()) and bool((~mask).any()) else True
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
    starts_y = sorted(set(list(range(0, max(height - patch_size, 0) + 1, stride)) + [max(height - patch_size, 0)]))
    starts_x = sorted(set(list(range(0, max(width - patch_size, 0) + 1, stride)) + [max(width - patch_size, 0)]))
    residual_sum = np.zeros_like(y_deblur, dtype=np.float32)
    weight_sum = np.zeros_like(y_deblur, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for top in starts_y:
        bottom = min(height, top + patch_size)
        for left in starts_x:
            right = min(width, left + patch_size)
            patch = y_deblur[top:bottom, left:right]
            low, high, record = _p17_robust_otsu_platforms(patch, foreground_is_high, params)
            record.update({"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right), "usable": low is not None and high is not None})
            records.append(record)
            if low is None or high is None:
                continue
            target = patch.copy()
            below, above = patch < low, patch > high
            target[below] = low + (1.0 - alpha) * (patch[below] - low)
            target[above] = high + (1.0 - alpha) * (patch[above] - high)
            ph, pw = patch.shape
            wy = np.sin(np.pi * (np.arange(ph, dtype=np.float32) + 0.5) / float(ph)) ** 2
            wx = np.sin(np.pi * (np.arange(pw, dtype=np.float32) + 0.5) / float(pw)) ** 2
            window = wy[:, None] * wx[None, :]
            residual_sum[top:bottom, left:right] += (target - patch) * window
            weight_sum[top:bottom, left:right] += window
    fused = np.divide(residual_sum, np.maximum(weight_sum, 1.0e-8), out=np.zeros_like(y_deblur), where=weight_sum > 0.0)
    write_weight = np.ones_like(y_deblur, dtype=np.float32)
    middle = (contour_distance > full_radius) & (contour_distance < zero_radius)
    write_weight[contour_distance >= zero_radius] = 0.0
    if bool(middle.any()):
        ratio = (contour_distance[middle] - full_radius) / max(zero_radius - full_radius, 1.0e-8)
        write_weight[middle] = (0.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    correction = (fused * write_weight).astype(np.float32)
    output = (y_deblur + correction).astype(np.float32)
    outside = write_weight <= 0.0
    if not np.isfinite(output).all():
        raise ValueError("P17 Otsu LUT output is non-finite")
    valid = [record for record in records if record["usable"]]
    invalid_reasons: dict[str, int] = {}
    for record in records:
        reason = record.get("validity_reason")
        if reason:
            invalid_reasons[str(reason)] = invalid_reasons.get(str(reason), 0) + 1
    return output, {
        "p17_b_status": "computed" if valid else "skipped_no_valid_patch",
        "p17_b_alpha": alpha, "p17_b_patch_size": patch_size, "p17_b_patch_stride": stride,
        "p17_b_patch_count": len(records), "p17_b_valid_patch_count": len(valid),
        "p17_b_invalid_reason_counts": invalid_reasons,
        "p17_b_global_foreground_is_high": foreground_is_high,
        "p17_b_write_full_radius": full_radius, "p17_b_write_zero_radius": zero_radius,
        "p17_b_write_fraction": float(np.mean(write_weight > 0.0)),
        "p17_b_weight_fraction": float(np.mean(weight_sum > 0.0)),
        "p17_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p17_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)),
        "p17_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p17_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "p17_b_outside_sum_abs_delta": float(np.sum(np.abs(correction[outside]))) if bool(outside.any()) else 0.0,
        "_p17_b_correction": correction, "_p17_b_write_weight": write_weight,
        "_p17_b_weight_sum": weight_sum, "_p17_b_patch_records": records,
    }


def _p18_apply_anchored_excess(
    y_deblur: np.ndarray, mask: np.ndarray, alpha: float, params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Correct only side-band excess relative to continuous anchors.

    Input: Finite deblurred Y, a text mask used only for polarity/components and
      contour anchoring, alpha, and P18 parameters.
    Output: Finite corrected Y plus Otsu, anchor, write-mask, and continuity telemetry.
    Key Steps: Reuse regional Otsu ranges, interpolate exterior 24--48 px anchors,
      derive isolated adaptive component foreground anchors, and correct only
      sign-appropriate excess beyond a dead zone with a bounded tanh response.
    Key Params: P18 keeps P17's 96/32 Otsu domain; A uses 16--24 and B 10--28 taper.
    State/Dependencies: No cross-component foreground borrowing and no output fallback.
    """
    if y_deblur.ndim != 2 or mask.shape != y_deblur.shape or not np.isfinite(y_deblur).all():
        raise ValueError("P18 anchored-excess inputs are invalid")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("P18 anchored-excess alpha is invalid")
    height, width = y_deblur.shape
    patch_size = max(3, int(params.get("p17_patch_size", 96)))
    stride = max(1, int(params.get("p17_patch_stride", 32)))
    full_radius = max(1.0, float(params.get("p18_write_full_radius", 16.0)))
    zero_radius = max(full_radius + 1.0, float(params.get("p18_write_zero_radius", 24.0)))
    exterior_near = max(zero_radius, float(params.get("p18_exterior_anchor_near", 24.0)))
    exterior_far = max(exterior_near + 1.0, float(params.get("p18_exterior_anchor_far", 48.0)))
    foreground_is_high = bool(np.median(y_deblur[mask]) >= np.median(y_deblur[~mask])) if bool(mask.any()) and bool((~mask).any()) else True
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)

    # The Otsu field supplies the local two-class separation used by the P18 dead zone.
    starts_y = sorted(set(list(range(0, max(height - patch_size, 0) + 1, stride)) + [max(height - patch_size, 0)]))
    starts_x = sorted(set(list(range(0, max(width - patch_size, 0) + 1, stride)) + [max(width - patch_size, 0)]))
    range_sum = np.zeros_like(y_deblur, dtype=np.float32)
    range_weight = np.zeros_like(y_deblur, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for top in starts_y:
        bottom = min(height, top + patch_size)
        for left in starts_x:
            right = min(width, left + patch_size)
            patch = y_deblur[top:bottom, left:right]
            low, high, record = _p17_robust_otsu_platforms(patch, foreground_is_high, params)
            usable = low is not None and high is not None
            record.update({"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right), "usable": usable})
            records.append(record)
            if not usable:
                continue
            ph, pw = patch.shape
            wy = np.sin(np.pi * (np.arange(ph, dtype=np.float32) + .5) / float(ph)) ** 2
            wx = np.sin(np.pi * (np.arange(pw, dtype=np.float32) + .5) / float(pw)) ** 2
            window = wy[:, None] * wx[None, :]
            range_sum[top:bottom, left:right] += float(high - low) * window
            range_weight[top:bottom, left:right] += window
    range_field = np.divide(range_sum, np.maximum(range_weight, 1.0e-8), out=np.zeros_like(y_deblur), where=range_weight > 0.0)

    # Normalized Gaussian interpolation makes B(x) continuous while being based solely
    # on the same-side exterior annulus.  Empty support is a skip, never a fabricated value.
    exterior_anchor = (~mask) & (outside_distance >= exterior_near) & (outside_distance <= exterior_far)
    bg_seed = np.where(exterior_anchor, y_deblur, 0.0).astype(np.float32)
    bg_support = exterior_anchor.astype(np.float32)
    bg_sigma = max(1.0, float(params.get("p18_background_reference_sigma", 12.0)))
    bg_num = cv2.GaussianBlur(bg_seed, (0, 0), bg_sigma, borderType=cv2.BORDER_REFLECT)
    bg_den = cv2.GaussianBlur(bg_support, (0, 0), bg_sigma, borderType=cv2.BORDER_REFLECT)
    background_field = np.divide(bg_num, np.maximum(bg_den, 1.0e-8), out=np.zeros_like(y_deblur), where=bg_den > 1.0e-5)
    background_valid = bg_den > float(params.get("p18_min_anchor_support", 1.0e-4))

    labels_count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    foreground_field = np.zeros_like(y_deblur, dtype=np.float32)
    foreground_valid = np.zeros_like(mask, dtype=bool)
    component_records: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        component = labels == label
        component_max = float(np.max(inside_distance[component])) if bool(component.any()) else 0.0
        if component_max < 3.0:
            component_records.append({"label": label, "max_inside_distance": component_max, "usable": False, "reason": "thin_component_exterior_only"})
            continue
        core_distance = min(6.0, .5 * component_max)
        core = component & (inside_distance >= core_distance)
        if not bool(core.any()):
            component_records.append({"label": label, "max_inside_distance": component_max, "usable": False, "reason": "no_adaptive_core"})
            continue
        values = y_deblur[core]
        lo, hi = np.percentile(values, [35.0, 65.0])
        value = float(np.median(values[(values >= lo) & (values <= hi)]))
        foreground_field[component] = value
        foreground_valid[component] = True
        component_records.append({"label": label, "max_inside_distance": component_max, "core_distance": core_distance, "foreground_y": value, "usable": True})

    dead_zone = np.maximum(2.0 / 255.0, .10 * range_field).astype(np.float32)
    cap = max(1.0e-6, float(params.get("p18_tanh_cap", .12)))
    exterior = ~mask
    correction_raw = np.zeros_like(y_deblur, dtype=np.float32)
    if foreground_is_high:
        dark_excess = np.maximum(background_field - y_deblur - dead_zone, 0.0)
        bright_excess = np.maximum(y_deblur - foreground_field - dead_zone, 0.0)
        correction_raw[exterior & background_valid] = cap * np.tanh(dark_excess[exterior & background_valid] / cap)
        interior_ok = mask & foreground_valid
        correction_raw[interior_ok] = -cap * np.tanh(bright_excess[interior_ok] / cap)
    else:
        bright_excess = np.maximum(y_deblur - background_field - dead_zone, 0.0)
        dark_excess = np.maximum(foreground_field - y_deblur - dead_zone, 0.0)
        correction_raw[exterior & background_valid] = -cap * np.tanh(bright_excess[exterior & background_valid] / cap)
        interior_ok = mask & foreground_valid
        correction_raw[interior_ok] = cap * np.tanh(dark_excess[interior_ok] / cap)

    write_weight = np.ones_like(y_deblur, dtype=np.float32)
    middle = (contour_distance > full_radius) & (contour_distance < zero_radius)
    write_weight[contour_distance >= zero_radius] = 0.0
    if bool(middle.any()):
        ratio = (contour_distance[middle] - full_radius) / max(zero_radius - full_radius, 1.0e-8)
        write_weight[middle] = (.5 * (1.0 + np.cos(np.pi * ratio))).astype(np.float32)
    correction = (alpha * correction_raw * write_weight).astype(np.float32)
    output = np.clip(y_deblur + correction, 0.0, 1.0).astype(np.float32)
    outside_write = write_weight <= 0.0
    bins: dict[str, float] = {}
    for low, high in ((0., 4.), (4., 10.), (10., 16.), (16., zero_radius)):
        selection = (contour_distance >= low) & (contour_distance < high)
        bins[f"{int(low)}_{int(high)}"] = float(np.mean(np.abs(correction[selection]))) if bool(selection.any()) else 0.0
    status = "computed" if bool(exterior_anchor.any()) and bool(range_weight.any()) else "skipped_missing_exterior_anchor"
    if status != "computed":
        correction.fill(0.0)
        output = y_deblur.copy()
    if not np.isfinite(output).all():
        raise ValueError("P18 anchored-excess output is non-finite")
    return output, {
        "p18_b_status": status, "p18_b_alpha": alpha,
        "p18_b_patch_size": patch_size, "p18_b_patch_stride": stride,
        "p18_b_patch_count": len(records), "p18_b_valid_patch_count": int(sum(item["usable"] for item in records)),
        "p18_b_global_foreground_is_high": foreground_is_high,
        "p18_b_write_full_radius": full_radius, "p18_b_write_zero_radius": zero_radius,
        "p18_b_exterior_anchor_near": exterior_near, "p18_b_exterior_anchor_far": exterior_far,
        "p18_b_exterior_anchor_fraction": float(np.mean(exterior_anchor)),
        "p18_b_background_reference_coverage": float(np.mean(background_valid)),
        "p18_b_foreground_reference_coverage": float(np.mean(foreground_valid)),
        "p18_b_distance_bin_mean_abs_correction": bins,
        "p18_b_mean_abs_correction": float(np.mean(np.abs(correction))),
        "p18_b_p95_abs_correction": float(np.percentile(np.abs(correction), 95.0)),
        "p18_b_max_abs_delta": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "p18_b_outside_max_abs_delta": float(np.max(np.abs(correction[outside_write]))) if bool(outside_write.any()) else 0.0,
        "_p18_b_correction": correction, "_p18_b_write_weight": write_weight,
        "_p18_b_background_field": background_field, "_p18_b_foreground_field": foreground_field,
        "_p18_b_background_anchor": exterior_anchor, "_p18_b_foreground_anchor": foreground_valid,
        "_p18_b_anchor_coverage": (background_valid | foreground_valid),
        "_p18_b_range_field": range_field, "_p18_b_patch_records": records,
        "_p18_b_component_records": component_records,
        "_p18_b_continuity_profile": {"background_field_mean": float(np.mean(background_field[background_valid])) if bool(background_valid.any()) else None, "write_band_mean_abs": bins},
    }


def _p14_kernel_ab(rgb: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Run P14 local-kernel deblur and continuous A/B branches.

    Input: Authoritative finite HxWx3 RGB tensor and one frozen P14 candidate.
    Output: First finite A output plus both lambda deblur controls, continuous
      spatial-monotone A outputs, local soft-shoulder B outputs, and telemetry.
    Key Steps: Reuse P11 segmentation/kernel/Wiener, estimate LQ platform fields,
      measure side-clipped contours, and export both post-deblur branches.
    Key Params: Lambda defaults [.01,.03], B alpha defaults [.5,1.0], and all
      contour/field limits are candidate-configured; no quality fallback exists.
    State/Dependencies: CPU numpy/OpenCV only; GT and chroma planes are unused.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError("invalid P14 RGB input")
    y_input = _rgb_to_luma(rgb)
    mask, segmentation = _p10_segment_text(y_input, params)
    empty_mask = np.zeros_like(y_input, dtype=bool)
    empty_weight = np.zeros_like(y_input, dtype=np.float32)
    base_meta: dict[str, Any] = {
        "fallback": False,
        "fallback_reason": None,
        "p14_status": "kernel_unidentifiable",
        "p14_kernel_status": "kernel_unidentifiable",
        "p14_kernel_reason": None,
        "p14_kernel_sigma": None,
        "p14_kernel_sigma_mad": None,
        "p14_kernel_segment_count": 0,
        "p14_kernel_profiles_considered": 0,
        "p14_field_status": "skipped_no_mask",
        "p14_field_reason": None,
        "p14_field_sigma": None,
        "p14_field_seed_distance": None,
        "p14_field_background_y": None,
        "p14_field_foreground_y": None,
        "p14_field_uncertainty": None,
        "p14_field_support_fraction": 0.0,
        "p14_field_region_fraction": 0.0,
        "p14_a_status": "skipped_no_mask",
        "p14_a_profile_count": 0,
        "p14_a_rejected_count": 0,
        "p14_a_gap_count": 0,
        "p14_a_coverage_fraction": 0.0,
        "p14_a_gap_fraction": 1.0,
        "p14_a_support_fraction": 0.0,
        "p14_a_reverse_steps_before": 0,
        "p14_a_reverse_steps_after": 0,
        "p14_b_status": "skipped_no_field",
        "p14_b_roi_fraction": 0.0,
        "p14_stage_count": 0,
        "p14_stage_telemetry": [],
        "_p14_text_mask": empty_mask,
        "_p14_region_mask": empty_mask,
        "_p14_fields": {},
        "_p14_stage_outputs": [],
    }
    if mask is None:
        base_meta["p14_kernel_reason"] = segmentation.get("segmentation_reason", "no_text_mask")
        base_meta.update(segmentation)
        return rgb.copy(), base_meta
    base_meta["_p14_text_mask"] = mask
    kernel = _p11_estimate_equivalent_kernel(y_input, mask, params)
    sigma = kernel.get("kernel_sigma")
    base_meta.update({
        "p14_kernel_status": str(kernel.get("kernel_status", "kernel_unidentifiable")),
        "p14_kernel_reason": kernel.get("kernel_reason"),
        "p14_kernel_sigma": sigma,
        "p14_kernel_sigma_mad": kernel.get("kernel_sigma_mad"),
        "p14_kernel_segment_count": int(kernel.get("kernel_segment_count", 0)),
        "p14_kernel_profiles_considered": int(kernel.get("kernel_profiles_considered", 0)),
        "p14_kernel_direction_stats": kernel.get("kernel_direction_stats", {}),
        "p14_kernel_fit_residual_stats": kernel.get("kernel_fit_residual_stats", {}),
        "p14_kernel_bound_hits": kernel.get("kernel_bound_hits", {}),
        "_p14_kernel_psf": kernel.get("kernel_psf"),
    })
    if sigma is None:
        base_meta.update(segmentation)
        return rgb.copy(), base_meta
    sigma = float(sigma)
    fields = _p14_estimate_local_platform_fields(y_input, mask, sigma, params)
    if bool(params.get("p16_mode", False)):
        inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        contour_distance = np.where(mask, inside_distance, outside_distance).astype(np.float32)
        p16_region = contour_distance <= float(params.get("p16_a_write_radius", 24.0))
        fields["region_mask"] = p16_region
        fields["region_feather"] = np.ones_like(y_input, dtype=np.float32)
    base_meta.update({
        "p14_field_status": fields.get("status"),
        "p14_field_reason": fields.get("reason"),
        "p14_field_sigma": fields.get("field_sigma"),
        "p14_field_seed_distance": fields.get("seed_distance"),
        "p14_field_background_y": fields.get("background_y"),
        "p14_field_foreground_y": fields.get("foreground_y"),
        "p14_field_uncertainty": fields.get("uncertainty"),
        "p14_field_support_fraction": fields.get("field_support_fraction", 0.0),
        "p14_field_region_fraction": fields.get("region_fraction", 0.0),
        "_p14_fields": fields,
        "_p14_region_mask": fields.get("region_mask", empty_mask),
    })
    deblur_weight, _ = _p11_processing_soft_band(mask, sigma)
    base_meta["_p14_processing_weight"] = deblur_weight
    lambda_candidates = params.get("lambda_candidates", [params.get("lambda", 0.01), 0.03])
    alpha_candidates = params.get("alpha_candidates", [0.5, 1.0])
    if not isinstance(lambda_candidates, (list, tuple)) or not isinstance(alpha_candidates, (list, tuple)):
        raise ValueError("P14 lambda_candidates and alpha_candidates must be lists")
    lambdas = [float(value) for value in lambda_candidates]
    alphas = [float(value) for value in alpha_candidates]
    if not lambdas or not alphas or any(not math.isfinite(value) or value <= 0.0 for value in lambdas) or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in alphas):
        raise ValueError("P14 lambda/alpha candidate values are invalid")
    stage_outputs: list[dict[str, Any]] = []
    for regularization in lambdas:
        raw_deblur_y, wiener_meta = _p11_wiener_hunt_deblur(y_input, mask, sigma, regularization, params)
        deblur_y = (y_input + deblur_weight * (raw_deblur_y - y_input)).astype(np.float32)
        if not np.isfinite(deblur_y).all():
            raise ValueError("P14 deblur output is non-finite")
        deblur_rgb, deblur_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, deblur_y)
        profile_meta = _p14_measure_contour_profiles(deblur_y, mask, fields, sigma, params)
        profiles = profile_meta.get("profiles", [])
        a_y, a_correction, field_meta = _p14_apply_curve_field(
            deblur_y, mask, fields.get("region_mask", empty_mask), profiles, params
        )
        profile_meta.update(field_meta)
        profile_meta["profile_support_fraction"] = float(field_meta.get("p14_a_field_support_fraction", 0.0))
        a_y = np.clip(a_y, 0.0, 1.0).astype(np.float32)
        a_rgb, a_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, a_y)
        b_stages: list[dict[str, Any]] = []
        b_ablation_stages: list[dict[str, Any]] = []
        for alpha in alphas:
            if bool(params.get("p18_mode", False)):
                b_y, b_meta = _p18_apply_anchored_excess(deblur_y, mask, alpha, params)
            elif bool(params.get("p17_mode", False)):
                b_y, b_meta = _p17_apply_regional_otsu_lut(deblur_y, mask, alpha, params)
            elif bool(params.get("p16_mode", False)):
                b_y, b_meta = _p16_apply_patch_shoulder_lut(deblur_y, mask, alpha, params)
            else:
                b_y, b_meta = _p14_apply_local_soft_lut(deblur_y, mask, fields, sigma, alpha, params)
            b_rgb, b_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, b_y)
            b_stages.append({
                "alpha": alpha,
                "status": b_meta.get("p18_b_status", b_meta.get("p17_b_status", b_meta.get("p16_b_status", b_meta.get("p14_b_status", "computed")))),
                "output": b_rgb,
                "y": b_y,
                "rgb_meta": b_rgb_meta,
                "meta": b_meta,
            })
            # Fixed 112_v2 ablation: keep the pre-P15 uncapped tau but retain
            # the P15 support-gate-only behavior.  It is diagnostic only and
            # never participates in the nine-column main matrix.
            ablation_params = dict(params)
            ablation_params["p15_ablation_legacy_tau"] = True
            ab_y, ab_meta = _p14_apply_local_soft_lut(deblur_y, mask, fields, sigma, alpha, ablation_params)
            ab_rgb, ab_rgb_meta = _p10_reconstruct_rgb_from_y(rgb, y_input, ab_y)
            b_ablation_stages.append({
                "alpha": alpha,
                "status": ab_meta.get("p14_b_status", "computed"),
                "output": ab_rgb,
                "y": ab_y,
                "rgb_meta": ab_rgb_meta,
                "meta": ab_meta,
            })
        stage_outputs.append({
            "lambda": regularization,
            "deblur_status": "computed",
            "deblur_output": deblur_rgb,
            "deblur_y": deblur_y,
            "raw_deblur_y": raw_deblur_y,
            "deblur_rgb_meta": deblur_rgb_meta,
            "wiener_meta": wiener_meta,
            "a_status": "computed" if profiles else "computed_no_profiles",
            "a_output": a_rgb,
            "a_y": a_y,
            "a_correction": a_correction,
            "a_rgb_meta": a_rgb_meta,
            "a_meta": profile_meta,
            "b_stages": b_stages,
            "b_ablation_stages": b_ablation_stages,
        })
    if not stage_outputs:
        raise ValueError("P14 produced no finite deblur stage")
    stage_telemetry: list[dict[str, Any]] = []
    for stage in stage_outputs:
        stage_telemetry.append({
            "lambda": float(stage["lambda"]),
            "deblur_status": stage["deblur_status"],
            "p14_a_status": stage["a_status"],
            "p14_a_profile_count": int(stage["a_meta"].get("profile_count", 0)),
            "p14_a_rejected_count": int(stage["a_meta"].get("profile_rejected_count", 0)),
            "p14_a_gap_count": int(stage["a_meta"].get("profile_gap_count", 0)),
            "p14_a_coverage_fraction": float(stage["a_meta"].get("profile_coverage_fraction", 0.0)),
            "p14_a_support_fraction": float(stage["a_meta"].get("profile_support_fraction", 0.0)),
            "p14_a_reverse_steps_before": int(stage["a_meta"].get("profile_reverse_steps_before", 0)),
            "p14_a_reverse_steps_after": int(stage["a_meta"].get("profile_reverse_steps_after", 0)),
            "p14_deblur_mean_abs_delta": float(stage["wiener_meta"].get("wiener_raw_mean_abs_delta", 0.0)),
            "p14_deblur_p95_abs_delta": float(stage["wiener_meta"].get("wiener_raw_p95_abs_delta", 0.0)),
            "b_stages": [
                {
                    "alpha": float(item["alpha"]),
                    "status": item["status"],
                    **{key: value for key, value in item["meta"].items() if not str(key).startswith("_")},
                }
                for item in stage["b_stages"]
            ],
        })
    selected = stage_outputs[0]
    selected_a_meta = selected["a_meta"]
    selected_b = selected["b_stages"][0] if selected["b_stages"] else None
    meta = dict(base_meta)
    meta.update({
        "p14_status": "computed",
        "p14_kernel_status": "computed",
        "p14_deblur_status": "computed",
        "p14_a_status": selected["a_status"],
        "p14_a_profile_count": int(selected_a_meta.get("profile_count", 0)),
        "p14_a_rejected_count": int(selected_a_meta.get("profile_rejected_count", 0)),
        "p14_a_gap_count": int(selected_a_meta.get("profile_gap_count", 0)),
        "p14_a_coverage_fraction": float(selected_a_meta.get("profile_coverage_fraction", 0.0)),
        "p14_a_gap_fraction": float(selected_a_meta.get("profile_gap_fraction", 0.0)),
        "p14_a_support_fraction": float(selected_a_meta.get("profile_support_fraction", 0.0)),
        "p14_a_reverse_steps_before": int(selected_a_meta.get("profile_reverse_steps_before", 0)),
        "p14_a_reverse_steps_after": int(selected_a_meta.get("profile_reverse_steps_after", 0)),
        "p14_a_field_region_fraction": float(selected_a_meta.get("p14_a_field_region_fraction", 0.0)),
        "p14_a_field_outside_max_abs_delta": float(selected_a_meta.get("p14_a_field_outside_max_abs_delta", 0.0)),
        "p14_a_field_outside_sum_abs_delta": float(selected_a_meta.get("p14_a_field_outside_sum_abs_delta", 0.0)),
        "p14_a_interpolated_profile_count": int(selected_a_meta.get("p14_a_interpolated_profile_count", 0)),
        "p14_a_long_gap_pixels": int(selected_a_meta.get("p14_a_long_gap_pixels", 0)),
        "p14_a_angle_rejected_count": int(selected_a_meta.get("p14_a_angle_rejected_count", 0)),
        "p14_b_status": selected_b["status"] if selected_b is not None else "skipped_no_stage",
        "p14_b_roi_fraction": float(selected_b["meta"].get("p14_b_roi_fraction", 0.0)) if selected_b is not None else 0.0,
        "p14_stage_count": len(stage_outputs),
        "p14_stage_telemetry": stage_telemetry,
        "_p14_stage_outputs": stage_outputs,
        "_p14_profiles": selected_a_meta.get("profiles", []),
        "_p14_a_correction": selected["a_correction"],
        "_p14_a_support": selected_a_meta.get("_p14_a_support", np.zeros_like(y_input, dtype=bool)),
        "_p14_output_minus_deblur": selected["a_output"] - selected["deblur_output"],
    })
    if selected_b is not None:
        meta.update({key: value for key, value in selected_b["meta"].items() if not str(key).startswith("_")})
    meta.update(segmentation)
    return selected["a_output"], meta


def preprocess_image(rgb: np.ndarray, arm: str, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment Dispatch one frozen Stage-A preprocessing arm.

    Input: Authoritative RGB tensor, arm name, and frozen parameter dictionary.
    Output: Processed RGB tensor plus stage/fallback metadata.
    Key Steps: Run one primitive or compose P3 then P2 for P4, retaining reasons.
    Key Params: Arm-specific bounded values; P0 is exact identity and P5 is a
      standalone luma-unsharp path, P6 is the gamut-constrained Y-only curve,
    P10 is an independent deblur-first profile-cleanup path, P11 is an
      estimated-kernel local Wiener plus local Y S-curve path, P12 is the same
      kernel/deblur path followed by LQ-platform residual clamping, P13 is the
      independent spatial-monotone versus regional-shoulder comparison, and
      P14 is the side-clipped continuous-monotone versus local-soft comparison.
    State/Dependencies: Pure CPU transformation; no model, GT, or global state.
    """
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported Stage A arm: {arm}")
    if arm == "P0":
        return _identity(rgb)
    if arm == "P1":
        return _global_curve(rgb, params)
    if arm == "P2":
        return _local_dering(rgb, params)
    if arm == "P3":
        return _regularized_deblur(rgb, params)
    if arm == "P5":
        return _dc_preserving_luma_unsharp(rgb, params)
    if arm == "P6":
        return _p6_y_s_curve(rgb, params)
    if arm == "P7":
        return _p7_text_edge_y_curve(rgb, params)
    if arm == "P8":
        return _p7_text_edge_y_curve(rgb, params)
    if arm == "P9":
        return _p7_text_edge_y_curve(rgb, params)
    if arm == "P10":
        return _p10_text_edge_y_curve(rgb, params)
    if arm == "P11":
        return _p11_kernel_wiener_scurve(rgb, params)
    if arm == "P12":
        return _p12_kernel_platform_clamp(rgb, params)
    if arm == "P13":
        return _p13_kernel_ab(rgb, params)
    if arm == "P14":
        return _p14_kernel_ab(rgb, params)
    if arm == "P16":
        p16_params = dict(params)
        p16_params["p16_mode"] = True
        return _p14_kernel_ab(rgb, p16_params)
    if arm == "P17":
        p17_params = dict(params)
        p17_params["p17_mode"] = True
        return _p14_kernel_ab(rgb, p17_params)
    if arm == "P18":
        p18_params = dict(params)
        p18_params["p18_mode"] = True
        return _p14_kernel_ab(rgb, p18_params)
    p3, meta3 = _regularized_deblur(rgb, dict(params["p3"]))
    p2, meta2 = _local_dering(p3, dict(params["p2"]))
    try:
        composition_strength = float(params["composition_strength"])
        if not 0.0 <= composition_strength <= 1.0:
            raise ValueError("invalid P4 composition strength")
        output = np.clip(rgb + composition_strength * (p2 - rgb), 0.0, 1.0).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("P4 composition produced non-finite output")
        reasons = [value for value in (meta3.get("fallback_reason"), meta2.get("fallback_reason")) if value]
        return output, {"fallback": bool(reasons), "fallback_reason": ";".join(reasons) or None}
    except (KeyError, TypeError, ValueError, FloatingPointError) as error:
        return _identity(rgb, f"invalid_p4_composition:{error}")


def _cartesian_parameters(spec: dict[str, Any], keys: Iterable[str]) -> list[dict[str, Any]]:
    """@luna-comment Expand ordered scalar candidate lists into parameter records.

    Input: Candidate specification and ordered keys.
    Output: Deterministically ordered list of dictionaries.
    Key Steps: Validate non-empty lists and enumerate Cartesian products.
    Key Params: Key order is fixed by each arm to make tie-breaking reproducible.
    State/Dependencies: None.
    """
    names = list(keys)
    values: list[list[Any]] = []
    for name in names:
        value = spec.get(name)
        if not isinstance(value, list) or not value:
            raise ValueError(f"candidate list for {name} must be non-empty")
        values.append(value)
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def candidate_parameters(config: dict[str, Any], arm: str, selected: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """@luna-comment Build the bounded development-search grid for one arm.

    Input: Merged config, arm name, and previously selected P2/P3 values for P4.
    Output: Deterministic candidate parameter dictionaries.
    Key Steps: Expand primitive grids, pass frozen P6 curves through unchanged,
      pass the fixed P11/P12 matrices through unchanged, and compose P4 from
      frozen selected primitives.
    Key Params: Grid sizes stay small enough for CPU development screening.
    State/Dependencies: None.
    """
    spec = config["arms"][arm]
    if arm == "P0":
        return [{}]
    if arm == "P1":
        return _cartesian_parameters(spec, ("low", "high", "gamma", "strength"))
    if arm == "P2":
        return _cartesian_parameters(spec, ("sigma", "k", "strength"))
    if arm == "P3":
        return _cartesian_parameters(spec, ("kernel_size", "sigma", "regularization", "gain_cap", "strength"))
    if arm == "P5":
        return _cartesian_parameters(spec, ("sigma", "amount", "delta_cap", "blend"))
    if arm == "P6":
        curves = spec.get("curves")
        if not isinstance(curves, list) or not curves:
            raise ValueError("P6 curves must be a non-empty list")
        return [dict(curve) for curve in curves if isinstance(curve, dict)]
    if arm == "P7":
        curves = spec.get("curves")
        if not isinstance(curves, list) or not curves or any(not isinstance(curve, dict) for curve in curves):
            raise ValueError("P7 curves must be a non-empty object list")
        return [dict(curve) for curve in curves]
    if arm == "P8":
        curves = spec.get("curves")
        if not isinstance(curves, list) or not curves or any(not isinstance(curve, dict) for curve in curves):
            raise ValueError("P8 curves must be a non-empty object list")
        return [dict(curve) for curve in curves]
    if arm == "P9":
        curves = spec.get("curves")
        if not isinstance(curves, list) or not curves:
            raise ValueError("P9 curves must be a non-empty object list")
        return [dict(curve) for curve in curves]
    if arm == "P10":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P10 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if arm == "P11":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P11 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if arm == "P12":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P12 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if arm == "P13":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P13 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if arm == "P14":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P14 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if arm == "P16":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P16 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if arm == "P17":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P17 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if arm == "P18":
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or not candidates or any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("P18 candidates must be a non-empty object list")
        return [dict(candidate) for candidate in candidates]
    if selected is None or "P2" not in selected or "P3" not in selected:
        raise ValueError("P4 candidates require selected P2 and P3 parameters")
    return [
        {
            "p2": dict(selected["P2"]),
            "p3": dict(selected["P3"]),
            "composition_strength": value,
        }
        for value in spec.get("composition_strength", [])
    ]


def compute_metrics(prediction: np.ndarray, gt: np.ndarray, valid: np.ndarray, lq: np.ndarray) -> dict[str, float]:
    """@luna-comment Compute paired RGB/luma residual and edge proxy metrics.

    Input: Predicted/base RGB, aligned GT RGB, valid HxW mask, and original LQ.
    Output: Finite scalar metrics using the existing evaluator naming convention.
    Key Steps: Measure masked pixel error, luma error, p95 residual, gradients,
      input delta, and gamut saturation fraction.
    Key Params: BT.601 luma weights and PSNR floor 1e-12 match evaluator behavior.
    State/Dependencies: OpenCV Sobel; no persistent state.
    """
    if prediction.shape != gt.shape or prediction.shape != lq.shape:
        raise ValueError("prediction, GT, and LQ shapes must match")
    if valid.shape != prediction.shape[:2] or not bool(valid.any()):
        raise ValueError("valid mask must match image and contain pixels")
    diff = prediction - gt
    luma_diff = _rgb_to_luma(diff)
    abs_diff = np.abs(diff[valid])
    abs_y = np.abs(luma_diff[valid])
    mse_rgb = float(np.square(diff[valid]).mean())
    mse_y = float(np.square(luma_diff[valid]).mean())
    pred_y = _rgb_to_luma(prediction)
    gt_y = _rgb_to_luma(gt)
    pred_grad_x = cv2.Sobel(pred_y, cv2.CV_32F, 1, 0, ksize=3)
    pred_grad_y = cv2.Sobel(pred_y, cv2.CV_32F, 0, 1, ksize=3)
    gt_grad_x = cv2.Sobel(gt_y, cv2.CV_32F, 1, 0, ksize=3)
    gt_grad_y = cv2.Sobel(gt_y, cv2.CV_32F, 0, 1, ksize=3)
    edge_delta = np.sqrt(np.square(pred_grad_x - gt_grad_x) + np.square(pred_grad_y - gt_grad_y))
    input_delta = prediction - lq
    return {
        "masked_l1_rgb": float(abs_diff.mean()),
        "masked_l1_y": float(abs_y.mean()),
        "psnr_rgb": float(-10.0 * np.log10(max(mse_rgb, 1.0e-12))),
        "psnr_y": float(-10.0 * np.log10(max(mse_y, 1.0e-12))),
        "residual_p95_rgb": float(np.percentile(abs_diff, 95.0)),
        "luma_bias": float(luma_diff[valid].mean()),
        "edge_mae_y": float(edge_delta[valid].mean()),
        "input_delta_l1_rgb": float(np.abs(input_delta[valid]).mean()),
        "saturation_fraction": float(np.mean((prediction <= 1.0e-6) | (prediction >= 1.0 - 1.0e-6))),
    }


def compute_edge_region_error(prediction: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """@luna-comment Measure deterministic GT-defined edge-region RGB error.

    Input: Candidate RGB output, aligned GT RGB, and valid-pixel mask.
    Output: Mean absolute RGB error over a predefined edge region.
    Key Steps: Compute BT.601 GT Sobel magnitude, select valid pixels at the
      fixed 75th-percentile threshold, and average RGB L1; if no positive
      gradient exists, use all valid pixels as a deterministic finite fallback.
    Key Params: Sobel kernel size 3 and percentile 75 are fixed so selection
      cannot depend on candidate output or iteration order.
    State/Dependencies: Numpy/OpenCV only; no global state or file writes.
    """
    if prediction.shape != gt.shape or prediction.shape[:2] != valid.shape:
        raise ValueError("edge-region inputs have incompatible shapes")
    valid_bool = np.asarray(valid, dtype=bool)
    if not bool(valid_bool.any()):
        raise ValueError("edge-region error requires at least one valid pixel")
    gt_y = _rgb_to_luma(gt)
    grad_x = cv2.Sobel(gt_y, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt_y, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    threshold = float(np.percentile(magnitude[valid_bool], 75.0))
    edge = valid_bool & (magnitude >= threshold) if threshold > 0.0 else valid_bool & (magnitude > 0.0)
    if not bool(edge.any()):
        edge = valid_bool
    return float(np.mean(np.abs(prediction[edge] - gt[edge])))


def _score_candidate(records: list[dict[str, Any]], arm: str, params: dict[str, Any]) -> dict[str, Any]:
    """@luna-comment Score one candidate using development records only.

    Input: Development records, arm name, and one candidate parameter dictionary.
    Output: Mean metrics, fallback count, and lexicographic selection score;
      P5 carries edge/p95 evidence and P6 carries source-balanced L1 evidence.
    Key Steps: Process every dev record and aggregate valid paired metrics by
      record and, for P6, by source group.
    Key Params: RGB L1 is primary; Y L1 and residual p95 break ties.
    State/Dependencies: Calls only pure preprocessing and metric helpers.
    """
    rows: list[dict[str, Any]] = []
    fallback_count = 0
    edge_by_group: dict[str, list[float]] = {}
    residual_p95_by_group: dict[str, list[float]] = {}
    p6_y_by_group: dict[str, list[float]] = {}
    p6_rgb_by_group: dict[str, list[float]] = {}
    for record in records:
        output, meta = preprocess_image(record["_lq_rgb"], arm, params)
        fallback_count += int(meta.get("fallback", False))
        metrics = compute_metrics(output, record["_gt_rgb"], record["_valid_mask"], record["_lq_rgb"])
        rows.append(metrics)
        if arm == "P6":
            group = str(record["source_group"])
            p6_y_by_group.setdefault(group, []).append(float(metrics["masked_l1_y"]))
            p6_rgb_by_group.setdefault(group, []).append(float(metrics["masked_l1_rgb"]))
        if arm in ("P0", "P5"):
            group = str(record["source_group"])
            residual_p95_by_group.setdefault(group, []).append(float(metrics["residual_p95_rgb"]))
            if arm == "P5":
                edge_by_group.setdefault(group, []).append(
                    compute_edge_region_error(output, record["_gt_rgb"], record["_valid_mask"])
                )
    means = {name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES}
    score = [means["masked_l1_rgb"], means["masked_l1_y"], means["residual_p95_rgb"]]
    result: dict[str, Any] = {
        "arm": arm,
        "params": params,
        "metrics_mean": means,
        "fallback_count": fallback_count,
        "score": score,
        "candidate_hash": _json_hash(params),
    }
    if arm in ("P0", "P5"):
        result["source_group_p95_rgb"] = {
            group: float(np.percentile(values, 95.0))
            for group, values in sorted(residual_p95_by_group.items())
        }
    if arm == "P5":
        group_edge_means = {
            group: float(np.mean(values)) for group, values in sorted(edge_by_group.items())
        }
        result["edge_region_error_by_source_group"] = group_edge_means
        result["edge_region_error"] = float(np.mean(list(group_edge_means.values())))
    if arm == "P6":
        result["source_balanced_masked_l1_y"] = float(np.mean([np.mean(values) for values in p6_y_by_group.values()]))
        result["source_balanced_masked_l1_rgb"] = float(np.mean([np.mean(values) for values in p6_rgb_by_group.values()]))
    return result


def _select_p5_candidate(
    p5_rows: list[dict[str, Any]],
    p0_row: dict[str, Any],
    selection_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """@luna-comment Apply P5 admissibility guards and edge-first selection.

    Input: P5 development candidate rows, the P0 development score row, and
      optional ``p5_selection`` thresholds from the merged config.
    Output: One admissible frozen P5 candidate row.
    Key Steps: Reject fallbacks, enforce mean RGB L1 <= P0*1.01 and every
      source-group p95 RGB deterioration <= 0.002, then sort by source-balanced
      edge-region error, RGB L1, and candidate hash. Raise explicitly if empty.
    Key Params: The mean RGB multiplier and source-group p95 deterioration cap
      are explicit config values, defaulting to the approved Replan 4 tier.
    State/Dependencies: Pure JSON-safe score records; no file or global state.
    """
    if selection_config is not None and not isinstance(selection_config, dict):
        raise ValueError("P5 selection thresholds must be a JSON object")
    selection = selection_config or {}
    try:
        mean_multiplier = float(selection.get("mean_rgb_l1_multiplier", 1.02))
        max_p95_deterioration = float(selection.get("max_source_group_p95_deterioration", 0.015))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid P5 selection thresholds: {error}") from error
    if not (math.isfinite(mean_multiplier) and mean_multiplier >= 1.0):
        raise ValueError("P5 mean_rgb_l1_multiplier must be finite and >= 1.0")
    if not (math.isfinite(max_p95_deterioration) and max_p95_deterioration >= 0.0):
        raise ValueError("P5 max_source_group_p95_deterioration must be finite and >= 0")
    p0_mean = float(p0_row["metrics_mean"]["masked_l1_rgb"])
    p0_group_p95 = {str(group): float(value) for group, value in p0_row.get("source_group_p95_rgb", {}).items()}
    if not math.isfinite(p0_mean) or not p0_group_p95 or any(not math.isfinite(value) for value in p0_group_p95.values()):
        raise ValueError("P0 baseline is missing finite source-group p95 evidence")
    admissible: list[dict[str, Any]] = []
    for row in p5_rows:
        reasons: list[str] = []
        if int(row.get("fallback_count", 0)) != 0:
            reasons.append("fallback_candidate")
        candidate_mean = float(row["metrics_mean"]["masked_l1_rgb"])
        if not math.isfinite(candidate_mean) or candidate_mean > p0_mean * mean_multiplier:
            reasons.append("mean_rgb_l1_over_configured_p0_limit")
        candidate_group_p95 = {str(group): float(value) for group, value in row.get("source_group_p95_rgb", {}).items()}
        missing_groups = sorted(set(p0_group_p95) - set(candidate_group_p95))
        extra_groups = sorted(set(candidate_group_p95) - set(p0_group_p95))
        if missing_groups or extra_groups:
            reasons.append("source_group_p95_rgb_coverage_mismatch")
        deltas = {
            group: candidate_group_p95[group] - p0_group_p95[group]
            for group in sorted(set(p0_group_p95).intersection(candidate_group_p95))
        }
        max_deterioration = max(deltas.values(), default=float("nan") if missing_groups else 0.0)
        if any(not math.isfinite(value) for value in candidate_group_p95.values()):
            reasons.append("non_finite_source_group_p95_rgb")
        if not math.isfinite(max_deterioration) or max_deterioration > max_p95_deterioration:
            reasons.append("source_group_p95_rgb_deterioration_over_configured_limit")
        row["source_group_p95_rgb_delta_vs_p0"] = deltas
        row["max_source_group_p95_rgb_delta_vs_p0"] = max_deterioration
        row["mean_rgb_l1_multiplier"] = mean_multiplier
        row["mean_rgb_l1_limit_vs_p0"] = p0_mean * mean_multiplier
        row["max_source_group_p95_deterioration"] = max_p95_deterioration
        row["admissible"] = not reasons
        row["admissibility_reasons"] = reasons
        if not reasons:
            admissible.append(row)
    if not admissible:
        raise ValueError("no admissible P5 candidate after fallback and deterioration safeguards")
    return min(
        admissible,
        key=lambda row: (
            float(row["edge_region_error"]),
            float(row["metrics_mean"]["masked_l1_rgb"]),
            str(row["candidate_hash"]),
        ),
    )


def fit_parameters(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """@luna-comment Select active P1-P14 parameters from development groups and freeze them.

    Input: Development-only records and merged Stage A config.
    Output: Selected parameter map and complete candidate score records.
    Key Steps: Fit P1/P2/P3 independently, fit P4 from frozen primitives, then
      apply P5 safeguards when active, mark P6-P9 with their diagnostic
      selectors, score P10 candidates on development records only, and retain
      the P11/P12/P13/P14 matrices in declared order without GT-based selection.
    Key Params: ``active_arms`` isolates P0/P6 without altering historical grids.
    State/Dependencies: GT is read only from development records.
    """
    if not records:
        raise ValueError("cannot select parameters with no development records")
    active_arms = list(config["active_arms"])
    selected: dict[str, dict[str, Any]] = {"P0": {}}
    candidate_rows: list[dict[str, Any]] = []
    for arm in ("P1", "P2", "P3"):
        if arm not in active_arms:
            continue
        arm_rows = [_score_candidate(records, arm, params) for params in candidate_parameters(config, arm)]
        candidate_rows.extend(arm_rows)
        winner = min(arm_rows, key=lambda row: (tuple(row["score"]), row["candidate_hash"]))
        selected[arm] = dict(winner["params"])
    if "P4" in active_arms:
        p4_rows = [_score_candidate(records, "P4", params) for params in candidate_parameters(config, "P4", selected)]
        candidate_rows.extend(p4_rows)
        selected["P4"] = dict(min(p4_rows, key=lambda row: (tuple(row["score"]), row["candidate_hash"]))["params"])
    if "P5" in active_arms:
        p0_row = _score_candidate(records, "P0", {})
        p5_rows = [_score_candidate(records, "P5", params) for params in candidate_parameters(config, "P5")]
        candidate_rows.extend(p5_rows)
        selected["P5"] = dict(_select_p5_candidate(p5_rows, p0_row, config.get("p5_selection"))["params"])
    if "P6" in active_arms:
        p6_rows = [_score_candidate(records, "P6", params) for params in candidate_parameters(config, "P6")]
        candidate_rows.extend(p6_rows)
        selected["P6"] = dict(min(
            p6_rows,
            key=lambda row: (
                float(row["source_balanced_masked_l1_y"]),
                float(row["source_balanced_masked_l1_rgb"]),
                str(row["params"].get("name", "")),
            ),
        )["params"])
    if "P7" in active_arms:
        p7_rows = [_score_candidate(records, "P7", params) for params in candidate_parameters(config, "P7")]
        candidate_rows.extend(p7_rows)
        selected["P7"] = dict(min(p7_rows, key=lambda row: (float(row["metrics_mean"]["masked_l1_y"]), float(row["metrics_mean"]["masked_l1_rgb"]), ("narrow", "reference", "strong").index(str(row["params"].get("name", "strong"))) if str(row["params"].get("name", "strong")) in ("narrow", "reference", "strong") else 99))["params"])
    if "P8" in active_arms:
        p8_rows = [_score_candidate(records, "P8", params) for params in candidate_parameters(config, "P8")]
        candidate_rows.extend(p8_rows)
        selected["P8"] = dict(min(p8_rows, key=lambda row: (float(row["metrics_mean"]["masked_l1_y"]), float(row["metrics_mean"]["masked_l1_rgb"]), float(row["params"]["local_gamma"])))["params"])
    if "P9" in active_arms:
        p9_rows = [_score_candidate(records, "P9", params) for params in candidate_parameters(config, "P9")]
        candidate_rows.extend(p9_rows)
        selected["P9"] = dict(min(p9_rows, key=lambda row: (float(row["metrics_mean"]["masked_l1_y"]), float(row["metrics_mean"]["masked_l1_rgb"]), float(row["params"]["local_gamma"])))["params"])
    if "P10" in active_arms:
        p10_rows = [_score_candidate(records, "P10", params) for params in candidate_parameters(config, "P10")]
        candidate_rows.extend(p10_rows)
        selected["P10"] = dict(min(
            p10_rows,
            key=lambda row: (tuple(row["score"]), str(row["params"].get("name", "")), row["candidate_hash"]),
        )["params"])
    if "P11" in active_arms:
        p11_candidates = candidate_parameters(config, "P11")
        candidate_rows.extend(
            {
                "arm": "P11",
                "params": params,
                "candidate_hash": _json_hash(params),
                "selection": "fixed_declared_matrix",
                "gt_used_for_selection": False,
            }
            for params in p11_candidates
        )
        selected["P11"] = dict(p11_candidates[0])
    if "P12" in active_arms:
        p12_candidates = candidate_parameters(config, "P12")
        candidate_rows.extend(
            {
                "arm": "P12",
                "params": params,
                "candidate_hash": _json_hash(params),
                "selection": "fixed_declared_matrix",
                "gt_used_for_selection": False,
            }
            for params in p12_candidates
        )
        selected["P12"] = dict(p12_candidates[0])
    if "P13" in active_arms:
        p13_candidates = candidate_parameters(config, "P13")
        candidate_rows.extend(
            {
                "arm": "P13",
                "params": params,
                "candidate_hash": _json_hash(params),
                "selection": "fixed_declared_matrix",
                "gt_used_for_selection": False,
            }
            for params in p13_candidates
        )
        selected["P13"] = dict(p13_candidates[0])
    if "P14" in active_arms:
        p14_candidates = candidate_parameters(config, "P14")
        candidate_rows.extend(
            {
                "arm": "P14",
                "params": params,
                "candidate_hash": _json_hash(params),
                "selection": "fixed_declared_matrix",
                "gt_used_for_selection": False,
            }
            for params in p14_candidates
        )
        selected["P14"] = dict(p14_candidates[0])
    if "P16" in active_arms:
        p16_candidates = candidate_parameters(config, "P16")
        candidate_rows.extend(
            {
                "arm": "P16",
                "params": params,
                "candidate_hash": _json_hash(params),
                "selection": "fixed_declared_matrix",
                "gt_used_for_selection": False,
            }
            for params in p16_candidates
        )
        selected["P16"] = dict(p16_candidates[0])
    if "P17" in active_arms:
        p17_candidates = candidate_parameters(config, "P17")
        candidate_rows.extend(
            {"arm": "P17", "params": params, "candidate_hash": _json_hash(params), "selection": "fixed_declared_matrix", "gt_used_for_selection": False}
            for params in p17_candidates
        )
        selected["P17"] = dict(p17_candidates[0])
    if "P18" in active_arms:
        p18_candidates = candidate_parameters(config, "P18")
        candidate_rows.extend(
            {"arm": "P18", "params": params, "candidate_hash": _json_hash(params), "selection": "fixed_declared_matrix", "gt_used_for_selection": False}
            for params in p18_candidates
        )
        selected["P18"] = dict(p18_candidates[0])
    return selected, candidate_rows


def _safe_name(value: str) -> str:
    """@luna-comment Convert a sample identifier into a stable visual directory name.

    Input: Arbitrary manifest sample identifier.
    Output: Filesystem-safe compact name retaining readable identity.
    Key Steps: Keep alphanumerics and selected punctuation; replace other bytes.
    Key Params: Slash and whitespace become double underscores for source clarity.
    State/Dependencies: None.
    """
    result = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return result.strip("._") or "sample"


def _write_rgb(path: Path, value: np.ndarray) -> None:
    """@luna-comment Write one float RGB tensor as an 8-bit PNG artifact.

    Input: HxWx3 RGB float tensor in [0,1] and destination path.
    Output: None; creates the parent directory and writes BGR bytes for OpenCV.
    Key Steps: Clip, round, channel-swap, and verify cv2.imwrite success.
    Key Params: Fixed 8-bit display encoding is recorded by the run manifest.
    State/Dependencies: Writes only the requested artifact path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write RGB artifact: {path}")


def _write_residual(path: Path, prediction: np.ndarray, gt: np.ndarray, scale: float) -> None:
    """@luna-comment Write a fixed-scale absolute residual heatmap.

    Input: Predicted/base RGB, GT RGB, and display scale in RGB units.
    Output: BGR heatmap PNG with deterministic scaling.
    Key Steps: Collapse absolute RGB residual, scale/clamp, and apply JET map.
    Key Params: Fixed scale prevents per-image visual exaggeration.
    State/Dependencies: OpenCV write only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    residual = np.mean(np.abs(prediction - gt), axis=2)
    encoded = np.clip(np.rint(residual / max(float(scale), 1.0e-8) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.applyColorMap(encoded, cv2.COLORMAP_JET)):
        raise IOError(f"failed to write residual artifact: {path}")


def _write_output_minus_lq(path: Path, output: np.ndarray, lq: np.ndarray, scale: float) -> None:
    """@luna-comment Write a fixed-scale signed output-minus-LQ heatmap.

    Input: Processed RGB output, authoritative LQ RGB, and display scale.
    Output: BGR PNG centred at neutral gray for zero change.
    Key Steps: Average signed RGB delta, map zero to 128, clamp to fixed scale,
      and apply a deterministic JET colour map.
    Key Params: Fixed scale makes P5 changes comparable across selected samples.
    State/Dependencies: OpenCV write only; no source image mutation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    delta = np.mean(output - lq, axis=2)
    encoded = np.clip(np.rint(127.5 + delta / max(float(scale), 1.0e-8) * 127.5), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.applyColorMap(encoded, cv2.COLORMAP_JET)):
        raise IOError(f"failed to write output-minus-LQ artifact: {path}")


def _output_minus_deblur_stats(output: np.ndarray, deblur: np.ndarray) -> dict[str, float | int]:
    """@luna-comment Quantify each final RGB output against its deblur control.

    Input: Two finite HxWx3 RGB tensors with matching shape.
    Output: Absolute mean-channel residual counts and distribution statistics.
    Key Params: Thresholds are one and four 8-bit levels.
    State/Dependencies: Pure numpy; no image writes or quality selection.
    """
    if output.shape != deblur.shape or output.ndim != 3 or not np.isfinite(output).all() or not np.isfinite(deblur).all():
        raise ValueError("output/deblur stats inputs are invalid")
    residual = np.mean(np.abs(output.astype(np.float32) - deblur.astype(np.float32)), axis=2)
    return {
        "nonzero": int(np.sum(residual > 1.0e-8)),
        "gt_1_over_255": int(np.sum(residual > 1.0 / 255.0)),
        "gt_4_over_255": int(np.sum(residual > 4.0 / 255.0)),
        "max": float(np.max(residual)) if residual.size else 0.0,
        "p95": float(np.percentile(residual, 95.0)) if residual.size else 0.0,
    }


def _write_edge_profile(path: Path, output: np.ndarray, gt: np.ndarray, lq: np.ndarray) -> None:
    """@luna-comment Write a deterministic luma edge profile JSON artifact.

    Input: P5 output, GT RGB, and LQ RGB tensors.
    Output: JSON profile through the GT strongest-gradient row.
    Key Steps: Select the first maximum GT Sobel magnitude, then serialize GT,
      LQ, and output luma values across that row for direct edge inspection.
    Key Params: BT.601 luma and Sobel kernel size 3 are fixed.
    State/Dependencies: Writes only the requested JSON artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_y = _rgb_to_luma(gt)
    grad_x = cv2.Sobel(gt_y, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt_y, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    row = int(np.argmax(magnitude) // magnitude.shape[1])
    payload = {
        "row": row,
        "gt_luma": [float(value) for value in _rgb_to_luma(gt)[row]],
        "lq_luma": [float(value) for value in _rgb_to_luma(lq)[row]],
        "output_luma": [float(value) for value in _rgb_to_luma(output)[row]],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_p7_mask_evidence(path: Path, rgb: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> None:
    """@luna-comment Write P7 text-mask, feather, and contour-overlay artifacts.

    Input: Destination directory, original RGB, boolean foreground mask, and [0,1]
      feather weight map from an accepted P7 segmentation.
    Output: Three deterministic PNGs for mask correctness and edge-band review.
    Key Steps: Encode mask/weight and draw the text contour over an RGB copy.
    Key Params: Mask uses 0/255; weight uses linear 8-bit display encoding.
    State/Dependencies: OpenCV output writes only.
    """
    path.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask.astype(np.uint8) * 255)
    if not cv2.imwrite(str(path / "p7_text_mask.png"), mask_u8):
        raise IOError("failed to write P7 text mask")
    if not cv2.imwrite(str(path / "p7_edge_weight.png"), np.clip(np.rint(weight * 255.0), 0, 255).astype(np.uint8)):
        raise IOError("failed to write P7 edge weight")
    overlay = cv2.cvtColor(np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    if not cv2.imwrite(str(path / "p7_mask_overlay.png"), overlay):
        raise IOError("failed to write P7 mask overlay")


def _write_p10_mask_evidence(path: Path, rgb: np.ndarray, mask: np.ndarray, profiles: list[dict[str, Any]]) -> None:
    """@luna-comment Write P10 mask and sampled-boundary visual evidence.

    Input: Destination visual directory, source RGB ROI, accepted P10 mask, and
      reliable normal-profile records.
    Output: Text-mask, boundary-seed overlay, and profile-count PNG artifacts.
    Key Steps: Encode the mask, mark profile seed points, and draw contours on
      an RGB copy without changing the source tensor.
    Key Params: Seed locations come from the frozen normal sampler; profile
      acceptance is represented by green points and red mask contours.
    State/Dependencies: OpenCV writes only the requested visual directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    mask_u8 = mask.astype(np.uint8) * 255
    if not cv2.imwrite(str(path / "p10_text_mask.png"), mask_u8):
        raise IOError("failed to write P10 text mask")
    overlay = cv2.cvtColor(np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    for profile in profiles:
        row = int(profile["row"])
        column = int(profile["column"])
        if 0 <= row < overlay.shape[0] and 0 <= column < overlay.shape[1]:
            cv2.circle(overlay, (column, row), 1, (0, 255, 0), -1)
    if not cv2.imwrite(str(path / "p10_boundary_overlay.png"), overlay):
        raise IOError("failed to write P10 boundary overlay")
    profile_count = np.zeros(mask.shape, dtype=np.uint8)
    for profile in profiles:
        row = int(profile["row"])
        column = int(profile["column"])
        if 0 <= row < profile_count.shape[0] and 0 <= column < profile_count.shape[1]:
            profile_count[row, column] = 255
    if not cv2.imwrite(str(path / "p10_profile_seeds.png"), profile_count):
        raise IOError("failed to write P10 profile seeds")


def _write_p10_profile_evidence(path: Path, meta: dict[str, Any]) -> None:
    """@luna-comment Serialize P10 profiles, beta attempts, and stage status.

    Input: Destination JSON path and P10 metadata containing private profile
      arrays plus aggregate scalar measurements.
    Output: JSON-safe profile artifact retaining sampled values, target values,
      anchors, crossings, reverse-step counts, beta attempt statuses, and
      aggregate stage telemetry.
    Key Steps: Strip private remap/scatter arrays from the public record and
      convert numpy scalars/arrays into ordinary JSON values.
    Key Params: Profile values are retained at the configured .5 px sampling
      step so visual review can reproduce the local transition calculation.
    State/Dependencies: Writes only the requested JSON file; no image mutation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles: list[dict[str, Any]] = []
    for profile in meta.get("_p10_profiles", []):
        profiles.append({key: value for key, value in profile.items() if not key.startswith("_")})
    payload = {key: value for key, value in meta.items() if not key.startswith("_") and key != "fallback"}
    payload["profiles"] = profiles
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_gray(path: Path, value: np.ndarray) -> None:
    """@luna-comment Write a finite luma plane as an 8-bit grayscale artifact.

    Input: Destination path and HxW luma values nominally in [0,1].
    Output: None; creates the parent and verifies the OpenCV write.
    Key Steps: Clip, quantise, and encode the plane without channel conversion.
    Key Params: Eight-bit output is only for diagnostics; processing stays float.
    State/Dependencies: Writes only the requested artifact path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite grayscale artifact: {path}")
    encoded = np.clip(np.rint(np.asarray(value) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), encoded):
        raise IOError(f"failed to write grayscale artifact: {path}")


def _write_p11_kernel_evidence(path: Path, meta: dict[str, Any]) -> None:
    """@luna-comment Serialize P11 kernel, PSF, mask, and profile diagnostics.

    Input: Destination sample directory and P11 metadata with private arrays.
    Output: Kernel JSON, PSF image, masks, and per-profile fit JSON.
    Key Steps: Strip private numpy fields from the public JSON and write the
      finite estimated kernel/fit records and processing support maps.
    Key Params: Artifact names are stable for direct six-column visual review.
    State/Dependencies: OpenCV/JSON writes only; source arrays are not mutated.
    """
    path.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in meta.items() if not key.startswith("_")}
    public["kernel_profiles"] = meta.get("_p11_kernel_profiles", [])
    public["scurve_profile_evidence"] = meta.get("_p11_scurve_profile_evidence", [])
    public.pop("p11_deblur_candidates", None)
    (path / "p11_kernel.json").write_text(
        json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    psf = meta.get("_p11_kernel_psf")
    if isinstance(psf, np.ndarray):
        _write_gray(path / "p11_psf.png", psf / max(float(psf.max()), 1.0e-12))
    text_mask = meta.get("_p11_text_mask")
    if isinstance(text_mask, np.ndarray):
        _write_gray(path / "p11_text_mask.png", text_mask.astype(np.float32))
    soft_weight = meta.get("_p11_processing_soft_weight")
    if isinstance(soft_weight, np.ndarray):
        _write_gray(path / "p11_processing_mask.png", soft_weight)
    edge_map = meta.get("_p11_edge_segment_map")
    if isinstance(edge_map, np.ndarray):
        _write_gray(path / "p11_edge_segment_map.png", edge_map.astype(np.float32) / 255.0)


def _write_p11_visual_columns(path: Path, meta: dict[str, Any], lq: np.ndarray) -> None:
    """@luna-comment Write the six P11 stage columns and diagnostic raw planes.

    Input: Sample visual directory, one P11 metadata record, and input RGB.
    Output: 00_lq, 01 local S-curve, two deblur, and two deblur-plus-S-curve
      PNGs when their finite kernel candidates exist; skipped candidates have
      no identity image substituted.
    Key Steps: Write the common input/S-curve once, map lambda candidates to
      columns 02-05, and retain raw/masked Y planes for analysis.
    Key Params: Lambda .01 and .03 map to the fixed six-column names; all
      outputs are produced from actual P11 stage arrays.
    State/Dependencies: Uses existing RGB/gray writers and only writes path.
    """
    path.mkdir(parents=True, exist_ok=True)
    _write_rgb(path / "00_lq.png", lq)
    scurve_only = meta.get("_p11_scurve_only_output")
    if isinstance(scurve_only, np.ndarray):
        _write_rgb(path / "01_scurve_only_g1p5.png", scurve_only)
    _write_p11_kernel_evidence(path, meta)
    for stage in meta.get("_p11_stage_outputs", []):
        regularization = float(stage["lambda"])
        token = "0p01" if abs(regularization - 0.01) <= 1.0e-8 else "0p03" if abs(regularization - 0.03) <= 1.0e-8 else str(regularization).replace(".", "p")
        deblur_output = stage.get("deblur_output")
        final_output = stage.get("final_output")
        if not isinstance(deblur_output, np.ndarray) or not isinstance(final_output, np.ndarray):
            continue
        _write_rgb(path / f"02_deblur_lam{token}.png" if token == "0p01" else path / f"03_deblur_lam{token}.png", deblur_output)
        _write_rgb(
            path / f"04_deblur_lam{token}_scurve_g1p5.png" if token == "0p01" else path / f"05_deblur_lam{token}_scurve_g1p5.png",
            final_output,
        )
        raw_y = stage.get("raw_deblur_y")
        masked_y = stage.get("masked_deblur_y")
        if isinstance(raw_y, np.ndarray):
            _write_gray(path / f"p11_raw_deblur_lam{token}_y.png", raw_y)
        if isinstance(masked_y, np.ndarray):
            _write_gray(path / f"p11_masked_deblur_lam{token}_y.png", masked_y)


def _p12_profile_json(profile: dict[str, Any]) -> dict[str, Any]:
    """@luna-comment Remove private sampling arrays from one P12 profile record.

    Input: One P10 normal profile measured from input LQ Y.
    Output: JSON-safe public platform evidence without private coordinate arrays.
    Key Steps: Drop underscore-prefixed arrays and convert numpy scalars/lists.
    Key Params: Public platform levels and profile geometry remain available for
      review; the dense maps are emitted as PNG artifacts separately.
    State/Dependencies: Pure Python/numpy conversion; no file writes.
    """
    result: dict[str, Any] = {}
    for key, value in profile.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            result[str(key)] = value.tolist()
        elif isinstance(value, (np.floating, np.integer, np.bool_)):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _write_p12_evidence(path: Path, meta: dict[str, Any]) -> None:
    """@luna-comment Write P12 kernel, platform, mask, and support evidence.

    Input: Sample visual directory and P12 metadata containing private arrays.
    Output: JSON telemetry plus PSF, low/high platform, support, hole, and mask
      grayscale artifacts. All maps use the input-image coordinate system.
    Key Steps: Strip private numpy fields for JSON and write finite diagnostic
      planes with stable names used by the fixed observation review.
    Key Params: Unsupported platform holes remain visible and are not filled by
      an identity or quality fallback.
    State/Dependencies: OpenCV/JSON writes only; source arrays are unchanged.
    """
    path.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in meta.items() if not str(key).startswith("_")}
    public["platform_profiles"] = [_p12_profile_json(profile) for profile in meta.get("_p12_profiles", [])]
    public.pop("p12_stage_telemetry", None)
    (path / "p12_platform.json").write_text(
        json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    psf = meta.get("_p12_kernel_psf")
    if isinstance(psf, np.ndarray):
        _write_gray(path / "p12_psf.png", psf / max(float(psf.max()), 1.0e-12))
    mask = meta.get("_p12_text_mask")
    if isinstance(mask, np.ndarray):
        _write_gray(path / "p12_text_mask.png", mask.astype(np.float32))
    processing = meta.get("_p12_processing_weight")
    if isinstance(processing, np.ndarray):
        _write_gray(path / "p12_processing_mask.png", processing)
    deblur = meta.get("_p12_deblur_weight")
    if isinstance(deblur, np.ndarray):
        _write_gray(path / "p12_deblur_mask.png", deblur)
    low = meta.get("_p12_platform_low")
    high = meta.get("_p12_platform_high")
    support = meta.get("_p12_platform_support")
    holes = meta.get("_p12_platform_holes")
    if isinstance(low, np.ndarray):
        _write_gray(path / "p12_platform_low.png", low)
    if isinstance(high, np.ndarray):
        _write_gray(path / "p12_platform_high.png", high)
    if isinstance(support, np.ndarray):
        _write_gray(path / "p12_platform_support.png", support)
    if isinstance(holes, np.ndarray):
        _write_gray(path / "p12_platform_holes.png", holes)


def _write_p12_visual_columns(path: Path, meta: dict[str, Any], lq: np.ndarray, residual_scale: float) -> None:
    """@luna-comment Write the fixed P12 deblur/clamp visual columns.

    Input: Sample visual directory, P12 metadata, input RGB, and residual display
      scale. Output: 00 LQ, 01/02 deblur, and 03-06 lambda/alpha clamp PNGs.
    Key Steps: Write each finite stage from its actual arrays, preserving the
      declared matrix order and never creating a P12 base/selected image.
    Key Params: Lambda `.01/.03` map to columns 01/02; alpha `.5/1.0` maps to
      columns 03-06. Diagnostic residual maps are emitted per clamp stage.
    State/Dependencies: Existing RGB/grayscale writers; no GT-based choice.
    """
    path.mkdir(parents=True, exist_ok=True)
    _write_rgb(path / "00_lq.png", lq)
    _write_p12_evidence(path, meta)
    stage_outputs = meta.get("_p12_stage_outputs", [])
    lambda_column = {0.01: "01", 0.03: "02"}
    clamp_column = {(0.01, 0.5): "03", (0.01, 1.0): "04", (0.03, 0.5): "05", (0.03, 1.0): "06"}
    seen_deblur: set[float] = set()
    for stage in stage_outputs:
        regularization = float(stage["lambda"])
        alpha = float(stage["alpha"])
        token = "0p01" if abs(regularization - 0.01) <= 1.0e-8 else "0p03" if abs(regularization - 0.03) <= 1.0e-8 else str(regularization).replace(".", "p")
        deblur_output = stage.get("deblur_output")
        clamp_output = stage.get("clamp_output")
        if not isinstance(deblur_output, np.ndarray) or not isinstance(clamp_output, np.ndarray):
            continue
        if regularization not in seen_deblur:
            column = lambda_column.get(regularization)
            if column is not None:
                _write_rgb(path / f"{column}_deblur_lam{token}.png", deblur_output)
            seen_deblur.add(regularization)
        alpha_token = "0p5" if abs(alpha - 0.5) <= 1.0e-8 else "1p0" if abs(alpha - 1.0) <= 1.0e-8 else str(alpha).replace(".", "p")
        column = clamp_column.get((regularization, alpha))
        if column is not None:
            _write_rgb(path / f"{column}_deblur_lam{token}_clamp_a{alpha_token}.png", clamp_output)
        prefix = f"p12_lam{token}_alpha{alpha_token}"
        _write_output_minus_lq(path / f"{prefix}_output_minus_deblur.png", clamp_output, deblur_output, residual_scale)
        clamp_meta = stage.get("clamp_meta", {})
        residual = np.asarray(clamp_meta.get("_p12_platform_residual", np.zeros(lq.shape[:2], dtype=np.float32)), dtype=np.float32)
        correction = np.asarray(clamp_meta.get("_p12_correction", np.zeros(lq.shape[:2], dtype=np.float32)), dtype=np.float32)
        scale = max(float(residual_scale), 1.0e-6)
        _write_gray(path / f"{prefix}_dark_overshoot.png", np.clip(np.maximum(residual, 0.0) / scale, 0.0, 1.0))
        _write_gray(path / f"{prefix}_bright_overshoot.png", np.clip(np.maximum(-residual, 0.0) / scale, 0.0, 1.0))
        _write_gray(path / f"{prefix}_correction.png", np.clip(np.abs(correction) / scale, 0.0, 1.0))


def _p13_profile_json(profile: dict[str, Any]) -> dict[str, Any]:
    """@luna-comment Convert one P13 profile to JSON-safe evidence.

    Input: P13 profile containing public scalar/list fields and private numpy
      coordinate/delta arrays.
    Output: Public profile record retaining platform, range, and curve samples.
    Key Steps: Remove private field names and unwrap numpy scalar/array values.
    Key Params: Private maps stay in image artifacts so profile JSON remains
      inspectable without serialising implementation-only coordinate buffers.
    State/Dependencies: Pure conversion; no file writes or mutation.
    """
    result: dict[str, Any] = {}
    for key, value in profile.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            result[str(key)] = value.tolist()
        elif isinstance(value, (np.floating, np.integer, np.bool_)):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _write_p13_evidence(path: Path, meta: dict[str, Any]) -> None:
    """@luna-comment Write P13 regional, profile, LUT, and support diagnostics.

    Input: Sample visual directory and P13 metadata with private luma/mask maps.
    Output: JSON platform/profile telemetry plus grayscale masks, PSF, and A
      curve-field support/label artifacts.
    Key Steps: Strip private arrays from JSON, preserve regional quantiles and
      profiles, and encode every finite map in input-image coordinates.
    Key Params: Artifact names are stable for the fixed nine-column review;
      unsupported/gap areas remain visible rather than identity-labelled.
    State/Dependencies: OpenCV/JSON writers only; source tensors are unchanged.
    """
    path.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in meta.items() if not str(key).startswith("_")}
    region_stats = meta.get("_p13_region_stats", {})
    if isinstance(region_stats, dict):
        public["region_platforms"] = {key: value for key, value in region_stats.items() if not str(key).startswith("_")}
    public["spatial_profiles"] = [_p13_profile_json(profile) for profile in meta.get("_p13_profiles", [])]
    public.pop("p13_stage_telemetry", None)
    (path / "p13_platform.json").write_text(
        json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    psf = meta.get("_p13_kernel_psf")
    if isinstance(psf, np.ndarray):
        _write_gray(path / "p13_psf.png", psf / max(float(psf.max()), 1.0e-12))
    mask = meta.get("_p13_text_mask")
    if isinstance(mask, np.ndarray):
        _write_gray(path / "p13_text_mask.png", mask.astype(np.float32))
    region = meta.get("_p13_region_mask")
    if isinstance(region, np.ndarray):
        _write_gray(path / "p13_region_mask.png", region.astype(np.float32))
    weight = meta.get("_p13_processing_weight")
    if isinstance(weight, np.ndarray):
        _write_gray(path / "p13_deblur_mask.png", weight)
    support = meta.get("_p13_a_support")
    if isinstance(support, np.ndarray):
        _write_gray(path / "p13_a_support.png", support.astype(np.float32))
    labels = meta.get("_p13_a_labels")
    if isinstance(labels, np.ndarray):
        scale = max(float(np.max(labels)), 1.0)
        _write_gray(path / "p13_a_nearest_boundary_labels.png", labels.astype(np.float32) / scale)
    for stage in meta.get("_p13_stage_outputs", []):
        regularization = float(stage["lambda"])
        token = "0p01" if abs(regularization - 0.01) <= 1.0e-8 else "0p03" if abs(regularization - 0.03) <= 1.0e-8 else str(regularization).replace(".", "p")
        stage_public = {
            "lambda": regularization,
            "a_status": stage.get("a_status"),
            "a_meta": {key: value for key, value in stage.get("a_meta", {}).items() if not str(key).startswith("_") and key != "profiles"},
            "a_profiles": [_p13_profile_json(profile) for profile in stage.get("a_meta", {}).get("profiles", [])],
            "b_stages": [],
        }
        for item in stage.get("b_stages", []):
            item_meta = item.get("meta", {})
            stage_public["b_stages"].append({
                "alpha": float(item.get("alpha", 0.0)),
                "status": item.get("status"),
                "meta": {key: value for key, value in item_meta.items() if not str(key).startswith("_")},
            })
            lut_x = item_meta.get("_p13_b_lut_x")
            lut_y = item_meta.get("_p13_b_lut_y")
            if isinstance(lut_x, np.ndarray) and isinstance(lut_y, np.ndarray):
                alpha_token = "0p5" if abs(float(item["alpha"]) - 0.5) <= 1.0e-8 else "1p0" if abs(float(item["alpha"]) - 1.0) <= 1.0e-8 else str(item["alpha"]).replace(".", "p")
                (path / f"p13_lam{token}_b_lut_a{alpha_token}.json").write_text(
                    json.dumps({"x": lut_x.tolist(), "y": lut_y.tolist()}, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        (path / f"p13_lam{token}_profiles.json").write_text(
            json.dumps(stage_public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _write_p13_visual_columns(path: Path, meta: dict[str, Any], lq: np.ndarray, residual_scale: float) -> None:
    """@luna-comment Write P13's fixed nine visual columns and stage deltas.

    Input: Sample visual directory, P13 metadata, LQ RGB tensor, and residual
      display scale.
    Output: 00 LQ, 01/02 deblur controls, 03/04 spatial A outputs, and 05-08
      regional B LUT outputs, with actual finite stage diagnostics only.
    Key Steps: Map lambda and alpha to declared filenames, write output-minus-
      deblur maps, and retain profile/shoulder correction images.
    Key Params: Lambda `.01/.03` and alpha `.5/1.0` define the fixed columns;
      no generic P13 base or automatic candidate image is created.
    State/Dependencies: Existing RGB/grayscale writers; no GT-based choice.
    """
    path.mkdir(parents=True, exist_ok=True)
    _write_rgb(path / "00_lq.png", lq)
    _write_p13_evidence(path, meta)
    lambda_deblur = {0.01: "01", 0.03: "02"}
    lambda_a = {0.01: "03", 0.03: "04"}
    column_b = {(0.01, 0.5): "05", (0.03, 0.5): "06", (0.01, 1.0): "07", (0.03, 1.0): "08"}
    for stage in meta.get("_p13_stage_outputs", []):
        regularization = float(stage["lambda"])
        token = "0p01" if abs(regularization - 0.01) <= 1.0e-8 else "0p03" if abs(regularization - 0.03) <= 1.0e-8 else str(regularization).replace(".", "p")
        deblur_output = stage.get("deblur_output")
        a_output = stage.get("a_output")
        if isinstance(deblur_output, np.ndarray):
            column = lambda_deblur.get(regularization)
            if column is not None:
                _write_rgb(path / f"{column}_deblur_lam{token}.png", deblur_output)
        if isinstance(a_output, np.ndarray):
            column = lambda_a.get(regularization)
            if column is not None:
                _write_rgb(path / f"{column}_lam{token}_A_spatial_monotone.png", a_output)
            _write_output_minus_lq(path / f"p13_lam{token}_A_output_minus_deblur.png", a_output, deblur_output, residual_scale)
            a_correction = np.asarray(stage.get("a_correction", np.zeros(lq.shape[:2], dtype=np.float32)), dtype=np.float32)
            _write_gray(path / f"p13_lam{token}_A_correction.png", np.clip(np.abs(a_correction) / max(residual_scale, 1.0e-6), 0.0, 1.0))
        for item in stage.get("b_stages", []):
            alpha = float(item["alpha"])
            output = item.get("output")
            if not isinstance(output, np.ndarray) or not isinstance(deblur_output, np.ndarray):
                continue
            column = column_b.get((regularization, alpha))
            alpha_token = "0p5" if abs(alpha - 0.5) <= 1.0e-8 else "1p0" if abs(alpha - 1.0) <= 1.0e-8 else str(alpha).replace(".", "p")
            if column is not None:
                _write_rgb(path / f"{column}_lam{token}_B_lut_a{alpha_token}.png", output)
            prefix = f"p13_lam{token}_B_lut_a{alpha_token}"
            _write_output_minus_lq(path / f"{prefix}_output_minus_deblur.png", output, deblur_output, residual_scale)
            correction = np.asarray(item.get("meta", {}).get("_p13_b_correction", np.zeros(lq.shape[:2], dtype=np.float32)), dtype=np.float32)
            _write_gray(path / f"{prefix}_correction.png", np.clip(np.abs(correction) / max(residual_scale, 1.0e-6), 0.0, 1.0))


def _write_p14_evidence(path: Path, meta: dict[str, Any], lq: np.ndarray) -> None:
    """@luna-comment Write P14 local fields, contour profiles, and stage maps.

    Input: Sample visual directory, P14 metadata, and the input RGB tensor.
    Output: JSON-safe field/profile telemetry and grayscale field, feather,
      support, correction, and per-stage shoulder diagnostics.
    Key Steps: Strip private numpy buffers from public JSON, encode finite maps,
      and preserve every declared lambda/alpha stage for visual inspection.
    Key Params: Fixed artifact names distinguish P14 A continuous-monotone
      evidence from B local-soft evidence; no selected/base image is written.
    State/Dependencies: Existing JSON/grayscale writers; source is unchanged.
    """
    path.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in meta.items() if not str(key).startswith("_")}
    fields = meta.get("_p14_fields", {})
    if isinstance(fields, dict):
        public["local_platform_fields"] = {
            key: value for key, value in fields.items() if not str(key).startswith("_") and not isinstance(value, np.ndarray)
        }
    public["spatial_profiles"] = [_p13_profile_json(profile) for profile in meta.get("_p14_profiles", [])]
    public.pop("p14_stage_telemetry", None)
    (path / "p14_platform.json").write_text(
        json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    psf = meta.get("_p14_kernel_psf")
    if isinstance(psf, np.ndarray):
        _write_gray(path / "p14_psf.png", psf / max(float(psf.max()), 1.0e-12))
    mask = meta.get("_p14_text_mask")
    if isinstance(mask, np.ndarray):
        _write_gray(path / "p14_text_mask.png", mask.astype(np.float32))
    if isinstance(fields, dict):
        for key, filename in (
            ("background_field", "p14_background_field.png"),
            ("foreground_field", "p14_foreground_field.png"),
            ("tau_field", "p14_tau_field.png"),
            ("field_support", "p14_field_support.png"),
            ("region_mask", "p14_region_mask.png"),
            ("region_feather", "p14_region_feather.png"),
            ("background_seed", "p14_background_seed.png"),
            ("foreground_seed", "p14_foreground_seed.png"),
        ):
            value = fields.get(key)
            if isinstance(value, np.ndarray):
                if key in {"background_field", "foreground_field", "tau_field"}:
                    display = np.clip(value, 0.0, 1.0)
                else:
                    display = value.astype(np.float32)
                _write_gray(path / filename, display)
    support = meta.get("_p14_a_support")
    if isinstance(support, np.ndarray):
        _write_gray(path / "p14_a_support.png", support.astype(np.float32))
    correction = meta.get("_p14_a_correction")
    if isinstance(correction, np.ndarray):
        _write_gray(path / "p14_a_correction.png", np.clip(np.abs(correction) / 0.25, 0.0, 1.0))
    for stage in meta.get("_p14_stage_outputs", []):
        regularization = float(stage["lambda"])
        token = "0p01" if abs(regularization - 0.01) <= 1.0e-8 else "0p03" if abs(regularization - 0.03) <= 1.0e-8 else str(regularization).replace(".", "p")
        stage_public = {
            "lambda": regularization,
            "a_status": stage.get("a_status"),
            "a_meta": {key: value for key, value in stage.get("a_meta", {}).items() if not str(key).startswith("_") and key != "profiles"},
            "a_profiles": [_p13_profile_json(profile) for profile in stage.get("a_meta", {}).get("profiles", [])],
            "b_stages": [],
        }
        a_correction = np.asarray(stage.get("a_correction", np.zeros(lq.shape[:2], dtype=np.float32)), dtype=np.float32)
        _write_gray(path / f"p14_lam{token}_A_correction.png", np.clip(np.abs(a_correction) / 0.25, 0.0, 1.0))
        for item in stage.get("b_stages", []):
            item_meta = item.get("meta", {})
            stage_public["b_stages"].append({
                "alpha": float(item.get("alpha", 0.0)),
                "status": item.get("status"),
                "meta": {key: value for key, value in item_meta.items() if not str(key).startswith("_")},
            })
            alpha = float(item.get("alpha", 0.0))
            alpha_token = "0p5" if abs(alpha - 0.5) <= 1.0e-8 else "1p0" if abs(alpha - 1.0) <= 1.0e-8 else str(alpha).replace(".", "p")
            for key, suffix in (("_p14_b_low", "low"), ("_p14_b_high", "high"), ("_p14_b_tau", "tau"), ("_p14_b_weight", "weight"), ("_p14_b_delta_raw", "delta_raw"), ("_p14_b_correction", "correction")):
                value = item_meta.get(key)
                if isinstance(value, np.ndarray):
                    scale = 0.25 if suffix in {"delta_raw", "correction"} else 1.0
                    _write_gray(path / f"p14_lam{token}_B_local_soft_a{alpha_token}_{suffix}.png", np.clip(np.abs(value) / scale if suffix in {"delta_raw", "correction"} else value, 0.0, 1.0))
        (path / f"p14_lam{token}_profiles.json").write_text(
            json.dumps(stage_public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )


def _write_p14_visual_columns(path: Path, meta: dict[str, Any], lq: np.ndarray, residual_scale: float) -> None:
    """@luna-comment Write P14's fixed nine-column A/B visual matrix.

    Input: Sample visual directory, P14 metadata, input RGB tensor, and fixed
      residual display scale.
    Output: 00 LQ, 01/02 deblur controls, 03/04 A continuous-monotone outputs,
      and 05-08 B local-soft alpha outputs, plus difference diagnostics.
    Key Steps: Map declared lambda/alpha pairs to stable filenames and write
      only actual finite arrays from every P14 stage.
    Key Params: Lambda .01/.03 and alpha .5/1.0 define the nine main columns;
      no P14 base/selected artifact is created.
    State/Dependencies: Existing RGB/grayscale writers; no GT-based choice.
    """
    path.mkdir(parents=True, exist_ok=True)
    _write_rgb(path / "00_lq.png", lq)
    _write_p14_evidence(path, meta, lq)
    lambda_deblur = {0.01: "01", 0.03: "02"}
    lambda_a = {0.01: "03", 0.03: "04"}
    column_b = {(0.01, 0.5): "05", (0.03, 0.5): "06", (0.01, 1.0): "07", (0.03, 1.0): "08"}
    for stage in meta.get("_p14_stage_outputs", []):
        regularization = float(stage["lambda"])
        token = "0p01" if abs(regularization - 0.01) <= 1.0e-8 else "0p03" if abs(regularization - 0.03) <= 1.0e-8 else str(regularization).replace(".", "p")
        deblur_output = stage.get("deblur_output")
        a_output = stage.get("a_output")
        if isinstance(deblur_output, np.ndarray):
            column = lambda_deblur.get(regularization)
            if column is not None:
                _write_rgb(path / f"{column}_deblur_lam{token}.png", deblur_output)
        if isinstance(a_output, np.ndarray) and isinstance(deblur_output, np.ndarray):
            column = lambda_a.get(regularization)
            if column is not None:
                _write_rgb(path / f"{column}_lam{token}_A_continuous_monotone.png", a_output)
            _write_output_minus_lq(path / f"p14_lam{token}_A_output_minus_deblur.png", a_output, deblur_output, residual_scale)
            (path / f"p14_lam{token}_A_diff_stats.json").write_text(
                json.dumps(_output_minus_deblur_stats(a_output, deblur_output), sort_keys=True) + "\n", encoding="utf-8"
            )
        for item in stage.get("b_stages", []):
            alpha = float(item["alpha"])
            output = item.get("output")
            if not isinstance(output, np.ndarray) or not isinstance(deblur_output, np.ndarray):
                continue
            alpha_token = "0p5" if abs(alpha - 0.5) <= 1.0e-8 else "1p0" if abs(alpha - 1.0) <= 1.0e-8 else str(alpha).replace(".", "p")
            column = column_b.get((regularization, alpha))
            if column is not None:
                _write_rgb(path / f"{column}_lam{token}_B_local_soft_a{alpha_token}.png", output)
            prefix = f"p14_lam{token}_B_local_soft_a{alpha_token}"
            _write_output_minus_lq(path / f"{prefix}_output_minus_deblur.png", output, deblur_output, residual_scale)
            (path / f"{prefix}_diff_stats.json").write_text(
                json.dumps(_output_minus_deblur_stats(output, deblur_output), sort_keys=True) + "\n", encoding="utf-8"
            )
        for item in stage.get("b_ablation_stages", []):
            output = item.get("output")
            if not isinstance(output, np.ndarray) or not isinstance(deblur_output, np.ndarray):
                continue
            alpha = float(item["alpha"])
            alpha_token = "0p5" if abs(alpha - 0.5) <= 1.0e-8 else "1p0" if abs(alpha - 1.0) <= 1.0e-8 else str(alpha).replace(".", "p")
            prefix = f"p14_lam{token}_B_ablation_legacy_tau_a{alpha_token}"
            _write_rgb(path / f"{prefix}.png", output)
            _write_output_minus_lq(path / f"{prefix}_output_minus_deblur.png", output, deblur_output, residual_scale)
            (path / f"{prefix}_diff_stats.json").write_text(
                json.dumps(_output_minus_deblur_stats(output, deblur_output), sort_keys=True) + "\n", encoding="utf-8"
            )


def _write_p16_visual_columns(path: Path, meta: dict[str, Any], lq: np.ndarray, residual_scale: float) -> None:
    """@luna-comment Write P16's fixed matrix and patch-local diagnostics.

    Input: P16 metadata, LQ RGB image, and residual display scale.
    Output: LQ, two deblur controls, A controls, and four patch-LUT images.
    Key Steps: Reuse P14 stage outputs while exposing P16-specific patch maps,
      per-patch platform records, write mask, and output/deblur residuals.
    Key Params: Lambda .01/.03 and alpha .5/1.0 define the nine columns.
    State/Dependencies: Existing image/JSON writers; no selection or rollback.
    """
    path.mkdir(parents=True, exist_ok=True)
    _write_rgb(path / "00_lq.png", lq)
    public = {key: value for key, value in meta.items() if not str(key).startswith("_")}
    (path / "p16_platform.json").write_text(json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    mask = meta.get("_p14_text_mask")
    if isinstance(mask, np.ndarray):
        _write_gray(path / "p16_text_mask.png", mask.astype(np.float32))
    lambda_deblur = {0.01: "01", 0.03: "02"}
    lambda_a = {0.01: "03", 0.03: "04"}
    column_b = {(0.01, 0.5): "05", (0.03, 0.5): "06", (0.01, 1.0): "07", (0.03, 1.0): "08"}
    for stage in meta.get("_p14_stage_outputs", []):
        regularization = float(stage["lambda"])
        token = "0p01" if abs(regularization - 0.01) <= 1.0e-8 else "0p03" if abs(regularization - 0.03) <= 1.0e-8 else str(regularization).replace(".", "p")
        deblur_output = stage.get("deblur_output")
        if isinstance(deblur_output, np.ndarray) and regularization in lambda_deblur:
            _write_rgb(path / f"{lambda_deblur[regularization]}_deblur_lam{token}.png", deblur_output)
        a_output = stage.get("a_output")
        if isinstance(a_output, np.ndarray) and isinstance(deblur_output, np.ndarray):
            if regularization in lambda_a:
                _write_rgb(path / f"{lambda_a[regularization]}_lam{token}_A_continuous_monotone.png", a_output)
            _write_output_minus_lq(path / f"p16_lam{token}_A_output_minus_deblur.png", a_output, deblur_output, residual_scale)
        for item in stage.get("b_stages", []):
            output = item.get("output")
            if not isinstance(output, np.ndarray) or not isinstance(deblur_output, np.ndarray):
                continue
            alpha = float(item["alpha"])
            alpha_token = "0p5" if abs(alpha - 0.5) <= 1.0e-8 else "1p0" if abs(alpha - 1.0) <= 1.0e-8 else str(alpha).replace(".", "p")
            column = column_b.get((regularization, alpha))
            if column is not None:
                _write_rgb(path / f"{column}_lam{token}_B_patch_lut_a{alpha_token}.png", output)
            prefix = f"p16_lam{token}_B_patch_lut_a{alpha_token}"
            _write_output_minus_lq(path / f"{prefix}_output_minus_deblur.png", output, deblur_output, residual_scale)
            item_meta = item.get("meta", {})
            (path / f"{prefix}_diff_stats.json").write_text(json.dumps(_output_minus_deblur_stats(output, deblur_output), sort_keys=True) + "\n", encoding="utf-8")
            for key, filename in (("_p16_b_correction", f"{prefix}_correction.png"), ("_p16_b_write_weight", f"{prefix}_write_mask.png"), ("_p16_b_weight_sum", f"{prefix}_weight_sum.png")):
                value = item_meta.get(key)
                if isinstance(value, np.ndarray):
                    scale = residual_scale if key.endswith("correction") else max(float(np.max(value)), 1.0e-8)
                    _write_gray(path / filename, np.clip(np.abs(value) / max(scale, 1.0e-8) if key.endswith("correction") else value / scale, 0.0, 1.0))
            patch_records = item_meta.get("_p16_b_patch_records", [])
            (path / f"{prefix}_patches.json").write_text(json.dumps({key: value for key, value in item_meta.items() if not str(key).startswith("_")} | {"patches": patch_records}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_p17_visual_columns(path: Path, meta: dict[str, Any], lq: np.ndarray, residual_scale: float, variant_name: str) -> None:
    """@luna-comment Write P17 deblur and Otsu-LUT evidence for visual review.

    Input: P17 stage metadata, LQ RGB, residual display scale, and candidate name.
    Output: Deblur/Otsu outputs, correction/write/weight maps, and patch telemetry.
    Key Steps: Export each P17 candidate below an isolated folder without altering
      the frozen P16 layout or overwriting another candidate's evidence.
    Key Params: Candidate lambda and alpha values are retained in stage metadata.
    State/Dependencies: Existing image and JSON writers only.
    """
    path = path / f"p17_{_safe_name(variant_name)}"
    path.mkdir(parents=True, exist_ok=True)
    _write_rgb(path / "00_lq.png", lq)
    public = {key: value for key, value in meta.items() if not str(key).startswith("_")}
    (path / "p17_platform.json").write_text(json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for stage in meta.get("_p14_stage_outputs", []):
        regularization = float(stage["lambda"])
        token = "0p01" if abs(regularization - .01) <= 1.0e-8 else "0p03" if abs(regularization - .03) <= 1.0e-8 else str(regularization).replace(".", "p")
        deblur = stage.get("deblur_output")
        if isinstance(deblur, np.ndarray):
            _write_rgb(path / f"p17_lam{token}_deblur.png", deblur)
        for item in stage.get("b_stages", []):
            output, item_meta = item.get("output"), item.get("meta", {})
            if not isinstance(output, np.ndarray) or not isinstance(deblur, np.ndarray):
                continue
            alpha = float(item["alpha"])
            alpha_token = "1p0" if abs(alpha - 1.0) <= 1.0e-8 else str(alpha).replace(".", "p")
            prefix = f"p17_lam{token}_B_regional_otsu_a{alpha_token}"
            _write_rgb(path / f"{prefix}.png", output)
            _write_output_minus_lq(path / f"{prefix}_output_minus_deblur.png", output, deblur, residual_scale)
            for key, suffix in (("_p17_b_correction", "correction"), ("_p17_b_write_weight", "write_mask"), ("_p17_b_weight_sum", "weight_sum")):
                value = item_meta.get(key)
                if isinstance(value, np.ndarray):
                    scale = residual_scale if suffix == "correction" else max(float(np.max(value)), 1.0e-8)
                    _write_gray(path / f"{prefix}_{suffix}.png", np.clip(np.abs(value) / scale if suffix == "correction" else value / scale, 0.0, 1.0))
            patches = item_meta.get("_p17_b_patch_records", [])
            (path / f"{prefix}_patches.json").write_text(json.dumps({key: value for key, value in item_meta.items() if not str(key).startswith("_")} | {"patches": patches}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_p18_visual_columns(path: Path, meta: dict[str, Any], lq: np.ndarray, residual_scale: float, variant_name: str) -> None:
    """@luna-comment Write isolated P18 anchored-excess evidence and diagnostics."""
    path = path / f"p18_{_safe_name(variant_name)}"
    path.mkdir(parents=True, exist_ok=True)
    _write_rgb(path / "00_lq.png", lq)
    public = {key: value for key, value in meta.items() if not str(key).startswith("_")}
    (path / "p18_platform.json").write_text(json.dumps(public, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    for stage in meta.get("_p14_stage_outputs", []):
        token = "0p03" if abs(float(stage["lambda"]) - .03) <= 1.0e-8 else str(stage["lambda"]).replace(".", "p")
        deblur = stage.get("deblur_output")
        if isinstance(deblur, np.ndarray):
            _write_rgb(path / f"p18_lam{token}_deblur.png", deblur)
        for item in stage.get("b_stages", []):
            output, item_meta = item.get("output"), item.get("meta", {})
            if not isinstance(output, np.ndarray) or not isinstance(deblur, np.ndarray):
                continue
            prefix = f"p18_lam{token}_B_anchored_excess_a1p0"
            _write_rgb(path / f"{prefix}.png", output)
            _write_output_minus_lq(path / f"{prefix}_output_minus_deblur.png", output, deblur, residual_scale)
            for key, suffix in (("_p18_b_correction", "correction"), ("_p18_b_write_weight", "write_mask"), ("_p18_b_background_field", "background_field"), ("_p18_b_foreground_field", "foreground_field"), ("_p18_b_background_anchor", "background_anchor"), ("_p18_b_foreground_anchor", "foreground_anchor"), ("_p18_b_anchor_coverage", "anchor_coverage")):
                value = item_meta.get(key)
                if isinstance(value, np.ndarray):
                    scale = residual_scale if suffix == "correction" else max(float(np.max(np.abs(value))), 1.0e-8)
                    _write_gray(path / f"{prefix}_{suffix}.png", np.clip(np.abs(value) / scale, 0.0, 1.0))
            diagnostic = {key: value for key, value in item_meta.items() if not str(key).startswith("_")}
            diagnostic["patches"] = item_meta.get("_p18_b_patch_records", [])
            diagnostic["components"] = item_meta.get("_p18_b_component_records", [])
            diagnostic["continuity_profile"] = item_meta.get("_p18_b_continuity_profile", {})
            (path / f"{prefix}_diagnostics.json").write_text(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _select_visual_records(
    records: list[dict[str, Any]], selected: dict[str, dict[str, Any]], config: dict[str, Any], diagnostic_arm: str
) -> tuple[set[str], dict[str, Any]]:
    """@luna-comment Select predefined and held-out adverse-tail visual samples.

    Input: Split records, frozen parameters, and visual selection config.
    Output: Sample IDs plus an audit describing predefined IDs and adverse groups.
    Key Steps: Use configured IDs or sorted deterministic defaults, score held-out
      diagnostic-arm-vs-P0 RGB deltas by source group, and include the highest adverse tail.
    Key Params: Counts come from the arm-specific visual config; ties sort by source-group ID.
    State/Dependencies: CPU preprocessing/metrics only; no writes or mutation.
    """
    visual_cfg = config.get(f"{diagnostic_arm.lower()}_visual", config.get("p5_visual", {}))
    configured_ids = [str(value) for value in visual_cfg.get("predefined_sample_ids", [])]
    predefined_count = max(0, int(visual_cfg.get("predefined_sample_count", 4)))
    if configured_ids:
        predefined_ids = sorted({sample_id for sample_id in configured_ids if any(str(record["sample_id"]) == sample_id for record in records)})
    else:
        predefined_ids = sorted(str(record["sample_id"]) for record in records)[:predefined_count]
    group_deltas: dict[str, list[float]] = {}
    for record in records:
        if str(record["split"]) != "heldout":
            continue
        p0, _ = preprocess_image(record["_lq_rgb"], "P0", selected["P0"])
        diagnostic, _ = preprocess_image(record["_lq_rgb"], diagnostic_arm, selected[diagnostic_arm])
        p0_metrics = compute_metrics(p0, record["_gt_rgb"], record["_valid_mask"], record["_lq_rgb"])
        diagnostic_metrics = compute_metrics(diagnostic, record["_gt_rgb"], record["_valid_mask"], record["_lq_rgb"])
        group_deltas.setdefault(str(record["source_group"]), []).append(
            float(diagnostic_metrics["masked_l1_rgb"] - p0_metrics["masked_l1_rgb"])
        )
    group_means = {group: float(np.mean(values)) for group, values in group_deltas.items()}
    tail_count = max(0, int(visual_cfg.get("adverse_tail_group_count", 4)))
    ordered_groups = sorted(group_means, key=lambda group: (-group_means[group], group))
    adverse_groups = ordered_groups[:tail_count]
    adverse_ids = {
        str(record["sample_id"])
        for record in records
        if str(record["split"]) == "heldout" and str(record["source_group"]) in adverse_groups
    }
    visual_ids = set(predefined_ids).union(adverse_ids)
    return visual_ids, {
        "predefined_sample_ids": predefined_ids,
        "adverse_tail_source_groups": adverse_groups,
        "adverse_tail_rgb_l1_delta": {group: group_means[group] for group in adverse_groups},
        "visual_sample_ids": sorted(visual_ids),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """@luna-comment Write newline-delimited JSON records with stable encoding.

    Input: Destination path and iterable of JSON-safe row dictionaries.
    Output: None; one record per line.
    Key Steps: Create parent, serialise without NaN, and flush line boundaries.
    Key Params: UTF-8 and sorted keys make artifacts diffable.
    State/Dependencies: Writes only the requested artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def evaluate_frozen_arms(
    records: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    output_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """@luna-comment Evaluate configured arms with immutable selected parameters and artifacts.

    Input: Development/held-out records, frozen parameter map, output root, config.
    Output: Per-sample rows, fallback rows, and timing/identity/artifact evidence.
    Key Steps: Process only ``active_arms``, compute metrics, write
      predefined/adverse-tail visuals, and serialize P6/P10/P13 chroma,
      deblur, profile, beta-attempt, per-stage, and gamut telemetry. P10 visual
      records include each finite beta's deblur and cleanup outputs; P13/P14
      emit their fixed nine-column A/B matrices.
    Key Params: Visual selection and fixed residual display scale come from config.
    State/Dependencies: CPU-only pure transforms plus output writes.
    """
    rows: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    clipping_rates: list[dict[str, Any]] = []
    active_arms = list(config["active_arms"])
    timing: dict[str, dict[str, float | int]] = {arm: {"samples": 0, "total_ms": 0.0} for arm in active_arms}
    identity_checks: list[dict[str, Any]] = []
    diagnostic_arm = "P18" if "P18" in active_arms else "P17" if "P17" in active_arms else "P16" if "P16" in active_arms else "P14" if "P14" in active_arms else "P13" if "P13" in active_arms else "P12" if "P12" in active_arms else "P11" if "P11" in active_arms else "P10" if "P10" in active_arms else "P9" if "P9" in active_arms else "P8" if "P8" in active_arms else "P7" if "P7" in active_arms else "P6" if "P6" in active_arms else "P5" if "P5" in active_arms else active_arms[-1]
    visual_ids, visual_selection = _select_visual_records(records, selected, config, diagnostic_arm)
    selection_hash = _json_hash(selected)
    for record in records:
        sample_id = str(record["sample_id"])
        for arm in active_arms:
            started = time.perf_counter()
            output, meta = preprocess_image(record["_lq_rgb"], arm, selected[arm])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            metrics = compute_metrics(output, record["_gt_rgb"], record["_valid_mask"], record["_lq_rgb"])
            row = {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "split": str(record["split"]),
                "category": record.get("category"),
                "arm": arm,
                "lq_rgb_path": str(record["lq_rgb_path"]),
                "gt_rgb_path": str(record["gt_rgb_path"]),
                "params": selected[arm],
                "params_hash": _json_hash(selected[arm]),
                "frozen_selection_hash": selection_hash,
                "fallback": bool(meta.get("fallback", False)),
                "fallback_reason": meta.get("fallback_reason"),
                "clipping_fraction": float(meta.get("clipping_fraction", 0.0)),
                "mean_abs_requested_delta": float(meta.get("mean_abs_requested_delta", 0.0)),
                "mean_abs_applied_delta": float(meta.get("mean_abs_applied_delta", 0.0)),
                "p95_abs_applied_delta": float(meta.get("p95_abs_applied_delta", 0.0)),
                "fraction_abs_delta_ge_4_255": float(meta.get("fraction_abs_delta_ge_4_255", 0.0)),
                "gamut_limited_fraction": float(meta.get("gamut_limited_fraction", 0.0)),
                "mean_abs_requested_minus_applied": float(meta.get("mean_abs_requested_minus_applied", 0.0)),
                "chroma_max_abs_error": float(meta.get("chroma_max_abs_error", 0.0)),
                "segmentation_accepted": bool(meta.get("segmentation_accepted", False)),
                "segmentation_reason": meta.get("fallback_reason") if arm == "P7" and not bool(meta.get("segmentation_accepted", False)) else None,
                "otsu_threshold": float(meta.get("otsu_threshold", 0.0)),
                "foreground_fraction": float(meta.get("foreground_fraction", 0.0)),
                "median_gap": float(meta.get("median_gap", 0.0)),
                "otsu_separation": float(meta.get("otsu_separation", 0.0)),
                "removed_foreground_fraction": float(meta.get("removed_foreground_fraction", 0.0)),
                "inner_band_fraction": float(meta.get("inner_band_fraction", 0.0)),
                "outer_band_fraction": float(meta.get("outer_band_fraction", 0.0)),
                "core_fraction": float(meta.get("core_fraction", 0.0)),
                "far_background_fraction": float(meta.get("far_background_fraction", 0.0)),
                "edge_band_mean_abs_delta": float(meta.get("edge_band_mean_abs_delta", 0.0)),
                "edge_band_fraction_abs_delta_ge_4_255": float(meta.get("edge_band_fraction_abs_delta_ge_4_255", 0.0)),
                "envelope_limited_fraction": float(meta.get("envelope_limited_fraction", 0.0)),
                "outside_band_max_abs_rgb_delta": float(meta.get("outside_band_max_abs_rgb_delta", 0.0)),
                "cleanup_mode": meta.get("cleanup_mode"),
                "cleanup_fallback_to_deblur": bool(meta.get("cleanup_fallback_to_deblur", False)),
                "cleanup_stage_status": meta.get("cleanup_stage_status"),
                "p10_stage_status": meta.get("p10_stage_status"),
                "p10_beta_stage_count": int(meta.get("p10_beta_stage_count", 0)),
                "p10_beta_stage_statuses": meta.get("p10_beta_stage_statuses", []),
                "deblur_sigma": float(meta.get("deblur_sigma", 0.0)),
                "deblur_iterations": int(meta.get("deblur_iterations", 0)),
                "deblur_lambda": float(meta.get("deblur_lambda", 0.0)),
                "deblur_step": float(meta.get("deblur_step", 0.0)),
                "deblur_beta_requested": float(meta.get("deblur_beta_requested", 0.0)),
                "deblur_beta_used": None if meta.get("deblur_beta_used") is None else float(meta["deblur_beta_used"]),
                "deblur_beta_fallback": bool(meta.get("deblur_beta_fallback", False)),
                "deblur_input_blur_energy": float(meta.get("deblur_input_blur_energy", 0.0)),
                "deblur_mean_abs_delta": float(meta.get("deblur_mean_abs_delta", 0.0)),
                "deblur_p95_abs_delta": float(meta.get("deblur_p95_abs_delta", 0.0)),
                "deblur_max_abs_delta": float(meta.get("deblur_max_abs_delta", 0.0)),
                "deblur_changed_fraction": float(meta.get("deblur_changed_fraction", 0.0)),
                "deblur_beta_attempts": meta.get("deblur_beta_attempts", []),
                "profile_count": int(meta.get("profile_count", 0)),
                "profile_rejected_count": int(meta.get("profile_rejected_count", 0)),
                "profile_accept_fraction": float(meta.get("profile_accept_fraction", 0.0)),
                "profile_support_fraction": float(meta.get("profile_support_fraction", 0.0)),
                "profile_correction_mean_abs": float(meta.get("profile_correction_mean_abs", 0.0)),
                "profile_10_90_width_before_median": meta.get("profile_10_90_width_before_median"),
                "profile_10_90_width_after_median": meta.get("profile_10_90_width_after_median"),
                "profile_1_99_width_before_median": meta.get("profile_1_99_width_before_median"),
                "profile_1_99_width_after_median": meta.get("profile_1_99_width_after_median"),
                "profile_half_crossing_shift_median": meta.get("profile_half_crossing_shift_median"),
                "profile_reverse_steps_before": int(meta.get("profile_reverse_steps_before", 0)),
                "profile_reverse_steps_after": int(meta.get("profile_reverse_steps_after", 0)),
                "p11_status": meta.get("p11_status"),
                "p11_kernel_status": meta.get("p11_kernel_status"),
                "p11_kernel_reason": meta.get("p11_kernel_reason"),
                "p11_kernel_sigma": meta.get("p11_kernel_sigma"),
                "p11_kernel_sigma_mad": meta.get("p11_kernel_sigma_mad"),
                "p11_kernel_segment_count": int(meta.get("p11_kernel_segment_count", 0)),
                "p11_kernel_profiles_considered": int(meta.get("p11_kernel_profiles_considered", 0)),
                "p11_kernel_direction_stats": meta.get("p11_kernel_direction_stats", {}),
                "p11_kernel_fit_residual_stats": meta.get("p11_kernel_fit_residual_stats", {}),
                "p11_kernel_bound_hits": meta.get("p11_kernel_bound_hits", {}),
                "p11_scurve_status": meta.get("p11_scurve_status"),
                "p11_scurve_gamma": meta.get("p11_scurve_gamma"),
                "p11_scurve_profile_count": int(meta.get("p11_scurve_profile_count", 0)),
                "p11_scurve_support_fraction": float(meta.get("p11_scurve_support_fraction", 0.0)),
                "p11_scurve_mean_abs_delta": float(meta.get("p11_scurve_mean_abs_delta", 0.0)),
                "p11_scurve_p95_abs_delta": float(meta.get("p11_scurve_p95_abs_delta", 0.0)),
                "p11_write_coverage": float(meta.get("p11_write_coverage", 0.0)),
                "p11_deblur_status": meta.get("p11_deblur_status"),
                "p11_deblur_candidate_count": int(meta.get("p11_deblur_candidate_count", 0)),
                "p11_deblur_candidates": meta.get("p11_deblur_candidates", []),
                "p12_status": meta.get("p12_status"),
                "p12_kernel_status": meta.get("p12_kernel_status"),
                "p12_kernel_reason": meta.get("p12_kernel_reason"),
                "p12_kernel_sigma": meta.get("p12_kernel_sigma"),
                "p12_kernel_sigma_mad": meta.get("p12_kernel_sigma_mad"),
                "p12_kernel_segment_count": int(meta.get("p12_kernel_segment_count", 0)),
                "p12_kernel_profiles_considered": int(meta.get("p12_kernel_profiles_considered", 0)),
                "p12_platform_reference": meta.get("p12_platform_reference"),
                "p12_platform_profile_count": int(meta.get("p12_platform_profile_count", 0)),
                "p12_platform_support_fraction": float(meta.get("p12_platform_support_fraction", 0.0)),
                "p12_platform_hole_fraction": float(meta.get("p12_platform_hole_fraction", 0.0)),
                "p12_mask_coverage": float(meta.get("p12_mask_coverage", 0.0)),
                "p12_deblur_status": meta.get("p12_deblur_status"),
                "p12_postprocess_status": meta.get("p12_postprocess_status"),
                "p12_stage_count": int(meta.get("p12_stage_count", 0)),
                "p12_stage_telemetry": meta.get("p12_stage_telemetry", []),
                "p12_alpha": meta.get("p12_alpha"),
                "p12_dark_overshoot_pixels": int(meta.get("p12_dark_overshoot_pixels", 0)),
                "p12_bright_overshoot_pixels": int(meta.get("p12_bright_overshoot_pixels", 0)),
                "p12_dark_correction_mean_abs": float(meta.get("p12_dark_correction_mean_abs", 0.0)),
                "p12_bright_correction_mean_abs": float(meta.get("p12_bright_correction_mean_abs", 0.0)),
                "p12_dark_correction_sum_abs": float(meta.get("p12_dark_correction_sum_abs", 0.0)),
                "p12_bright_correction_sum_abs": float(meta.get("p12_bright_correction_sum_abs", 0.0)),
                "p12_platform_external_fraction": float(meta.get("p12_platform_external_fraction", 0.0)),
                "p12_mean_abs_correction": float(meta.get("p12_mean_abs_correction", 0.0)),
                "p12_p95_abs_correction": float(meta.get("p12_p95_abs_correction", 0.0)),
                "p12_band_outside_max_abs_delta": float(meta.get("p12_band_outside_max_abs_delta", 0.0)),
                "p12_band_outside_delta_sum_abs": float(meta.get("p12_band_outside_delta_sum_abs", 0.0)),
                "p13_status": meta.get("p13_status"),
                "p13_kernel_status": meta.get("p13_kernel_status"),
                "p13_kernel_reason": meta.get("p13_kernel_reason"),
                "p13_kernel_sigma": meta.get("p13_kernel_sigma"),
                "p13_kernel_sigma_mad": meta.get("p13_kernel_sigma_mad"),
                "p13_kernel_segment_count": int(meta.get("p13_kernel_segment_count", 0)),
                "p13_kernel_profiles_considered": int(meta.get("p13_kernel_profiles_considered", 0)),
                "p13_region_status": meta.get("p13_region_status"),
                "p13_region_reason": meta.get("p13_region_reason"),
                "p13_region_background_y": meta.get("p13_region_background_y"),
                "p13_region_foreground_y": meta.get("p13_region_foreground_y"),
                "p13_region_low_y": meta.get("p13_region_low_y"),
                "p13_region_high_y": meta.get("p13_region_high_y"),
                "p13_region_contrast": meta.get("p13_region_contrast"),
                "p13_region_uncertainty": meta.get("p13_region_uncertainty"),
                "p13_region_background_count": int(meta.get("p13_region_background_count", 0)),
                "p13_region_foreground_count": int(meta.get("p13_region_foreground_count", 0)),
                "p13_region_bbox": meta.get("p13_region_bbox"),
                "p13_a_status": meta.get("p13_a_status"),
                "p13_a_profile_count": int(meta.get("p13_a_profile_count", 0)),
                "p13_a_rejected_count": int(meta.get("p13_a_rejected_count", 0)),
                "p13_a_gap_count": int(meta.get("p13_a_gap_count", 0)),
                "p13_a_coverage_fraction": float(meta.get("p13_a_coverage_fraction", 0.0)),
                "p13_a_gap_fraction": float(meta.get("p13_a_gap_fraction", 0.0)),
                "p13_a_support_fraction": float(meta.get("p13_a_support_fraction", 0.0)),
                "p13_a_reverse_steps_before": int(meta.get("p13_a_reverse_steps_before", 0)),
                "p13_a_reverse_steps_after": int(meta.get("p13_a_reverse_steps_after", 0)),
                "p13_a_initial_radius": meta.get("p13_a_initial_radius"),
                "p13_a_max_radius": meta.get("p13_a_max_radius"),
                "p13_a_field_region_fraction": float(meta.get("p13_a_field_region_fraction", 0.0)),
                "p13_a_field_outside_max_abs_delta": float(meta.get("p13_a_field_outside_max_abs_delta", 0.0)),
                "p13_a_field_outside_sum_abs_delta": float(meta.get("p13_a_field_outside_sum_abs_delta", 0.0)),
                "p13_b_status": meta.get("p13_b_status"),
                "p13_b_alpha": meta.get("p13_b_alpha"),
                "p13_b_low": meta.get("p13_b_low"),
                "p13_b_high": meta.get("p13_b_high"),
                "p13_b_roi_fraction": float(meta.get("p13_b_roi_fraction", 0.0)),
                "p13_b_dark_compression_pixels": int(meta.get("p13_b_dark_compression_pixels", 0)),
                "p13_b_bright_compression_pixels": int(meta.get("p13_b_bright_compression_pixels", 0)),
                "p13_b_dark_compression_sum_abs": float(meta.get("p13_b_dark_compression_sum_abs", 0.0)),
                "p13_b_bright_compression_sum_abs": float(meta.get("p13_b_bright_compression_sum_abs", 0.0)),
                "p13_b_mean_abs_correction": float(meta.get("p13_b_mean_abs_correction", 0.0)),
                "p13_b_p95_abs_correction": float(meta.get("p13_b_p95_abs_correction", 0.0)),
                "p13_b_outside_max_abs_delta": float(meta.get("p13_b_outside_max_abs_delta", 0.0)),
                "p13_b_outside_sum_abs_delta": float(meta.get("p13_b_outside_sum_abs_delta", 0.0)),
                "p13_stage_count": int(meta.get("p13_stage_count", 0)),
                "p13_stage_telemetry": meta.get("p13_stage_telemetry", []),
                "p14_status": meta.get("p14_status"),
                "p14_kernel_status": meta.get("p14_kernel_status"),
                "p14_kernel_reason": meta.get("p14_kernel_reason"),
                "p14_kernel_sigma": meta.get("p14_kernel_sigma"),
                "p14_kernel_sigma_mad": meta.get("p14_kernel_sigma_mad"),
                "p14_kernel_segment_count": int(meta.get("p14_kernel_segment_count", 0)),
                "p14_kernel_profiles_considered": int(meta.get("p14_kernel_profiles_considered", 0)),
                "p14_field_status": meta.get("p14_field_status"),
                "p14_field_reason": meta.get("p14_field_reason"),
                "p14_field_sigma": meta.get("p14_field_sigma"),
                "p14_field_seed_distance": meta.get("p14_field_seed_distance"),
                "p14_field_background_y": meta.get("p14_field_background_y"),
                "p14_field_foreground_y": meta.get("p14_field_foreground_y"),
                "p14_field_uncertainty": meta.get("p14_field_uncertainty"),
                "p14_field_support_fraction": float(meta.get("p14_field_support_fraction", 0.0)),
                "p14_field_region_fraction": float(meta.get("p14_field_region_fraction", 0.0)),
                "p14_a_status": meta.get("p14_a_status"),
                "p14_a_profile_count": int(meta.get("p14_a_profile_count", 0)),
                "p14_a_rejected_count": int(meta.get("p14_a_rejected_count", 0)),
                "p14_a_gap_count": int(meta.get("p14_a_gap_count", 0)),
                "p14_a_coverage_fraction": float(meta.get("p14_a_coverage_fraction", 0.0)),
                "p14_a_gap_fraction": float(meta.get("p14_a_gap_fraction", 0.0)),
                "p14_a_support_fraction": float(meta.get("p14_a_support_fraction", 0.0)),
                "p14_a_reverse_steps_before": int(meta.get("p14_a_reverse_steps_before", 0)),
                "p14_a_reverse_steps_after": int(meta.get("p14_a_reverse_steps_after", 0)),
                "p14_a_field_region_fraction": float(meta.get("p14_a_field_region_fraction", 0.0)),
                "p14_a_field_outside_max_abs_delta": float(meta.get("p14_a_field_outside_max_abs_delta", 0.0)),
                "p14_a_field_outside_sum_abs_delta": float(meta.get("p14_a_field_outside_sum_abs_delta", 0.0)),
                "p14_a_interpolated_profile_count": int(meta.get("p14_a_interpolated_profile_count", 0)),
                "p14_a_long_gap_pixels": int(meta.get("p14_a_long_gap_pixels", 0)),
                "p14_b_status": meta.get("p14_b_status"),
                "p14_b_roi_fraction": float(meta.get("p14_b_roi_fraction", 0.0)),
                "p14_b_alpha": meta.get("p14_b_alpha"),
                "p14_b_r_full": meta.get("p14_b_r_full"),
                "p14_b_r_zero": meta.get("p14_b_r_zero"),
                "p14_b_weight_coverage_fraction": float(meta.get("p14_b_weight_coverage_fraction", 0.0)),
                "p14_b_supported_fraction": float(meta.get("p14_b_supported_fraction", 0.0)),
                "p14_b_dark_compression_pixels": int(meta.get("p14_b_dark_compression_pixels", 0)),
                "p14_b_bright_compression_pixels": int(meta.get("p14_b_bright_compression_pixels", 0)),
                "p14_b_dark_compression_sum_abs": float(meta.get("p14_b_dark_compression_sum_abs", 0.0)),
                "p14_b_bright_compression_sum_abs": float(meta.get("p14_b_bright_compression_sum_abs", 0.0)),
                "p14_b_mean_abs_correction": float(meta.get("p14_b_mean_abs_correction", 0.0)),
                "p14_b_p95_abs_correction": float(meta.get("p14_b_p95_abs_correction", 0.0)),
                "p14_b_outside_max_abs_delta": float(meta.get("p14_b_outside_max_abs_delta", 0.0)),
                "p14_b_outside_sum_abs_delta": float(meta.get("p14_b_outside_sum_abs_delta", 0.0)),
                "p14_stage_count": int(meta.get("p14_stage_count", 0)),
                "p14_stage_telemetry": meta.get("p14_stage_telemetry", []),
                "timing_ms": float(elapsed_ms),
                **metrics,
            }
            rows.append(row)
            clipping_rates.append({
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "split": str(record["split"]),
                "arm": arm,
                "clipping_fraction": float(row["clipping_fraction"]),
                "mean_abs_requested_delta": row["mean_abs_requested_delta"],
                "mean_abs_applied_delta": row["mean_abs_applied_delta"],
                "p95_abs_applied_delta": row["p95_abs_applied_delta"],
                "fraction_abs_delta_ge_4_255": row["fraction_abs_delta_ge_4_255"],
                "gamut_limited_fraction": row["gamut_limited_fraction"],
                "mean_abs_requested_minus_applied": row["mean_abs_requested_minus_applied"],
                "chroma_max_abs_error": row["chroma_max_abs_error"],
                "segmentation_accepted": row["segmentation_accepted"],
                "foreground_fraction": row["foreground_fraction"],
                "otsu_separation": row["otsu_separation"],
                "edge_band_mean_abs_delta": row["edge_band_mean_abs_delta"],
                "edge_band_fraction_abs_delta_ge_4_255": row["edge_band_fraction_abs_delta_ge_4_255"],
                "envelope_limited_fraction": row["envelope_limited_fraction"],
                "outside_band_max_abs_rgb_delta": row["outside_band_max_abs_rgb_delta"],
                "cleanup_mode": row["cleanup_mode"],
                "cleanup_fallback_to_deblur": row["cleanup_fallback_to_deblur"],
                "cleanup_stage_status": row["cleanup_stage_status"],
                "p10_stage_status": row["p10_stage_status"],
                "p10_beta_stage_count": row["p10_beta_stage_count"],
                "p10_beta_stage_statuses": row["p10_beta_stage_statuses"],
                "deblur_sigma": row["deblur_sigma"],
                "deblur_beta_requested": row["deblur_beta_requested"],
                "deblur_beta_used": row["deblur_beta_used"],
                "deblur_beta_fallback": row["deblur_beta_fallback"],
                "deblur_input_blur_energy": row["deblur_input_blur_energy"],
                "deblur_mean_abs_delta": row["deblur_mean_abs_delta"],
                "deblur_p95_abs_delta": row["deblur_p95_abs_delta"],
                "deblur_max_abs_delta": row["deblur_max_abs_delta"],
                "deblur_changed_fraction": row["deblur_changed_fraction"],
                "deblur_beta_attempts": row["deblur_beta_attempts"],
                "profile_count": row["profile_count"],
                "profile_rejected_count": row["profile_rejected_count"],
                "profile_accept_fraction": row["profile_accept_fraction"],
                "profile_support_fraction": row["profile_support_fraction"],
                "profile_correction_mean_abs": row["profile_correction_mean_abs"],
                "profile_10_90_width_before_median": row["profile_10_90_width_before_median"],
                "profile_10_90_width_after_median": row["profile_10_90_width_after_median"],
                "profile_1_99_width_before_median": row["profile_1_99_width_before_median"],
                "profile_1_99_width_after_median": row["profile_1_99_width_after_median"],
                "profile_half_crossing_shift_median": row["profile_half_crossing_shift_median"],
                "profile_reverse_steps_before": row["profile_reverse_steps_before"],
                "profile_reverse_steps_after": row["profile_reverse_steps_after"],
                "p11_status": row["p11_status"],
                "p11_kernel_status": row["p11_kernel_status"],
                "p11_kernel_sigma": row["p11_kernel_sigma"],
                "p11_kernel_sigma_mad": row["p11_kernel_sigma_mad"],
                "p11_kernel_segment_count": row["p11_kernel_segment_count"],
                "p11_kernel_profiles_considered": row["p11_kernel_profiles_considered"],
                "p11_kernel_direction_stats": row["p11_kernel_direction_stats"],
                "p11_kernel_fit_residual_stats": row["p11_kernel_fit_residual_stats"],
                "p11_kernel_bound_hits": row["p11_kernel_bound_hits"],
                "p11_scurve_status": row["p11_scurve_status"],
                "p11_scurve_gamma": row["p11_scurve_gamma"],
                "p11_scurve_profile_count": row["p11_scurve_profile_count"],
                "p11_scurve_support_fraction": row["p11_scurve_support_fraction"],
                "p11_scurve_mean_abs_delta": row["p11_scurve_mean_abs_delta"],
                "p11_write_coverage": row["p11_write_coverage"],
                "p11_deblur_status": row["p11_deblur_status"],
                "p11_deblur_candidate_count": row["p11_deblur_candidate_count"],
                "p11_deblur_candidates": row["p11_deblur_candidates"],
                "p12_status": row["p12_status"],
                "p12_kernel_status": row["p12_kernel_status"],
                "p12_kernel_reason": row["p12_kernel_reason"],
                "p12_kernel_sigma": row["p12_kernel_sigma"],
                "p12_kernel_sigma_mad": row["p12_kernel_sigma_mad"],
                "p12_kernel_segment_count": row["p12_kernel_segment_count"],
                "p12_kernel_profiles_considered": row["p12_kernel_profiles_considered"],
                "p12_platform_reference": row["p12_platform_reference"],
                "p12_platform_profile_count": row["p12_platform_profile_count"],
                "p12_platform_support_fraction": row["p12_platform_support_fraction"],
                "p12_platform_hole_fraction": row["p12_platform_hole_fraction"],
                "p12_mask_coverage": row["p12_mask_coverage"],
                "p12_deblur_status": row["p12_deblur_status"],
                "p12_postprocess_status": row["p12_postprocess_status"],
                "p12_stage_count": row["p12_stage_count"],
                "p12_stage_telemetry": row["p12_stage_telemetry"],
                "p12_alpha": row["p12_alpha"],
                "p12_dark_overshoot_pixels": row["p12_dark_overshoot_pixels"],
                "p12_bright_overshoot_pixels": row["p12_bright_overshoot_pixels"],
                "p12_dark_correction_mean_abs": row["p12_dark_correction_mean_abs"],
                "p12_bright_correction_mean_abs": row["p12_bright_correction_mean_abs"],
                "p12_dark_correction_sum_abs": row["p12_dark_correction_sum_abs"],
                "p12_bright_correction_sum_abs": row["p12_bright_correction_sum_abs"],
                "p12_platform_external_fraction": row["p12_platform_external_fraction"],
                "p12_mean_abs_correction": row["p12_mean_abs_correction"],
                "p12_p95_abs_correction": row["p12_p95_abs_correction"],
                "p12_band_outside_max_abs_delta": row["p12_band_outside_max_abs_delta"],
                "p12_band_outside_delta_sum_abs": row["p12_band_outside_delta_sum_abs"],
                "p13_status": row["p13_status"],
                "p13_region_status": row["p13_region_status"],
                "p13_region_background_y": row["p13_region_background_y"],
                "p13_region_foreground_y": row["p13_region_foreground_y"],
                "p13_region_uncertainty": row["p13_region_uncertainty"],
                "p13_a_status": row["p13_a_status"],
                "p13_a_profile_count": row["p13_a_profile_count"],
                "p13_a_rejected_count": row["p13_a_rejected_count"],
                "p13_a_gap_count": row["p13_a_gap_count"],
                "p13_a_coverage_fraction": row["p13_a_coverage_fraction"],
                "p13_a_support_fraction": row["p13_a_support_fraction"],
                "p13_a_reverse_steps_before": row["p13_a_reverse_steps_before"],
                "p13_a_reverse_steps_after": row["p13_a_reverse_steps_after"],
                "p13_b_status": row["p13_b_status"],
                "p13_b_alpha": row["p13_b_alpha"],
                "p13_b_low": row["p13_b_low"],
                "p13_b_high": row["p13_b_high"],
                "p13_b_roi_fraction": row["p13_b_roi_fraction"],
                "p13_b_dark_compression_pixels": row["p13_b_dark_compression_pixels"],
                "p13_b_bright_compression_pixels": row["p13_b_bright_compression_pixels"],
                "p13_b_mean_abs_correction": row["p13_b_mean_abs_correction"],
                "p13_b_p95_abs_correction": row["p13_b_p95_abs_correction"],
                "p13_b_outside_max_abs_delta": row["p13_b_outside_max_abs_delta"],
                "p13_b_outside_sum_abs_delta": row["p13_b_outside_sum_abs_delta"],
                "p13_stage_count": row["p13_stage_count"],
                "p13_stage_telemetry": row["p13_stage_telemetry"],
                "p14_status": row["p14_status"],
                "p14_field_status": row["p14_field_status"],
                "p14_field_support_fraction": row["p14_field_support_fraction"],
                "p14_field_region_fraction": row["p14_field_region_fraction"],
                "p14_a_status": row["p14_a_status"],
                "p14_a_profile_count": row["p14_a_profile_count"],
                "p14_a_gap_count": row["p14_a_gap_count"],
                "p14_a_coverage_fraction": row["p14_a_coverage_fraction"],
                "p14_a_support_fraction": row["p14_a_support_fraction"],
                "p14_a_reverse_steps_before": row["p14_a_reverse_steps_before"],
                "p14_a_reverse_steps_after": row["p14_a_reverse_steps_after"],
                "p14_b_status": row["p14_b_status"],
                "p14_b_alpha": row["p14_b_alpha"],
                "p14_b_roi_fraction": row["p14_b_roi_fraction"],
                "p14_b_weight_coverage_fraction": row["p14_b_weight_coverage_fraction"],
                "p14_b_dark_compression_pixels": row["p14_b_dark_compression_pixels"],
                "p14_b_bright_compression_pixels": row["p14_b_bright_compression_pixels"],
                "p14_b_mean_abs_correction": row["p14_b_mean_abs_correction"],
                "p14_b_p95_abs_correction": row["p14_b_p95_abs_correction"],
                "p14_b_outside_max_abs_delta": row["p14_b_outside_max_abs_delta"],
                "p14_b_outside_sum_abs_delta": row["p14_b_outside_sum_abs_delta"],
                "p14_stage_count": row["p14_stage_count"],
                "p14_stage_telemetry": row["p14_stage_telemetry"],
            })
            timing[arm]["samples"] = int(timing[arm]["samples"]) + 1
            timing[arm]["total_ms"] = float(timing[arm]["total_ms"]) + elapsed_ms
            if meta.get("fallback"):
                fallbacks.append({key: row[key] for key in ("sample_id", "source_group", "split", "arm", "fallback_reason", "timing_ms")})
            if arm == "P0":
                max_abs = float(np.max(np.abs(output - record["_lq_rgb"])))
                identity_checks.append({"sample_id": sample_id, "split": record["split"], "max_abs_rgb": max_abs, "pass": max_abs <= 1.0e-7})
            if sample_id in visual_ids and not (output_root / "visuals" / _safe_name(sample_id) / "lq_rgb.png").is_file():
                sample_root = output_root / "visuals" / _safe_name(sample_id)
                _write_rgb(sample_root / "lq_rgb.png", record["_lq_rgb"])
                _write_rgb(sample_root / "gt_rgb.png", record["_gt_rgb"])
            if sample_id in visual_ids:
                sample_root = output_root / "visuals" / _safe_name(sample_id)
                if arm not in ("P11", "P12", "P13", "P14", "P16", "P17"):
                    _write_rgb(sample_root / f"{arm.lower()}_base.png", output)
                    _write_residual(sample_root / f"{arm.lower()}_residual.png", output, record["_gt_rgb"], float(config.get("residual_display_max", 0.25)))
                    if arm == diagnostic_arm:
                        _write_output_minus_lq(sample_root / f"{arm.lower()}_output_minus_lq.png", output, record["_lq_rgb"], float(config.get("residual_display_max", 0.25)))
                        _write_edge_profile(sample_root / f"{arm.lower()}_edge_profile.json", output, record["_gt_rgb"], record["_lq_rgb"])
                if arm == "P6":
                    for variant in candidate_parameters(config, "P6"):
                        variant_name = _safe_name(str(variant.get("name", "unnamed")))
                        variant_output, _ = preprocess_image(record["_lq_rgb"], "P6", variant)
                        _write_rgb(sample_root / f"p6_{variant_name}_base.png", variant_output)
                        _write_output_minus_lq(sample_root / f"p6_{variant_name}_output_minus_lq.png", variant_output, record["_lq_rgb"], float(config.get("residual_display_max", 0.25)))
                if arm == "P10":
                    for variant in candidate_parameters(config, "P10"):
                        variant_name = _safe_name(str(variant.get("name", "unnamed")))
                        variant_output, variant_meta = preprocess_image(record["_lq_rgb"], "P10", variant)
                        _write_rgb(sample_root / f"p10_{variant_name}_base.png", variant_output)
                        _write_residual(sample_root / f"p10_{variant_name}_residual.png", variant_output, record["_gt_rgb"], float(config.get("residual_display_max", 0.25)))
                        _write_output_minus_lq(sample_root / f"p10_{variant_name}_output_minus_lq.png", variant_output, record["_lq_rgb"], float(config.get("residual_display_max", 0.25)))
                        _write_p10_profile_evidence(sample_root / f"p10_{variant_name}_profiles.json", variant_meta)
                        for stage in variant_meta.get("_p10_beta_stage_outputs", []):
                            beta_token = str(stage.get("beta", "unknown")).replace(".", "_")
                            for phase in ("deblur", "cleanup"):
                                phase_output = stage.get(f"{phase}_output")
                                if not isinstance(phase_output, np.ndarray):
                                    raise ValueError(f"P10 {phase} stage output is missing for beta {stage.get('beta')}")
                                _write_rgb(sample_root / f"p10_{variant_name}_beta_{beta_token}_{phase}.png", phase_output)
                                _write_output_minus_lq(
                                    sample_root / f"p10_{variant_name}_beta_{beta_token}_{phase}_output_minus_lq.png",
                                    phase_output,
                                    record["_lq_rgb"],
                                    float(config.get("residual_display_max", 0.25)),
                                )
                        if bool(variant_meta.get("segmentation_accepted", False)):
                            _write_p10_mask_evidence(sample_root, record["_lq_rgb"], variant_meta["_p10_mask"], variant_meta.get("_p10_profiles", []))
                if arm == "P11":
                    for variant in candidate_parameters(config, "P11"):
                        _, variant_meta = preprocess_image(record["_lq_rgb"], "P11", variant)
                        _write_p11_visual_columns(sample_root, variant_meta, record["_lq_rgb"])
                if arm == "P12":
                    for variant in candidate_parameters(config, "P12"):
                        _, variant_meta = preprocess_image(record["_lq_rgb"], "P12", variant)
                        _write_p12_visual_columns(
                            sample_root,
                            variant_meta,
                            record["_lq_rgb"],
                            float(config.get("residual_display_max", 0.25)),
                        )
                if arm == "P13":
                    for variant in candidate_parameters(config, "P13"):
                        _, variant_meta = preprocess_image(record["_lq_rgb"], "P13", variant)
                        _write_p13_visual_columns(
                            sample_root,
                            variant_meta,
                            record["_lq_rgb"],
                            float(config.get("residual_display_max", 0.25)),
                        )
                if arm == "P14":
                    for variant in candidate_parameters(config, "P14"):
                        _, variant_meta = preprocess_image(record["_lq_rgb"], "P14", variant)
                        _write_p14_visual_columns(
                            sample_root,
                            variant_meta,
                            record["_lq_rgb"],
                            float(config.get("residual_display_max", 0.25)),
                            )
                if arm == "P16":
                    for variant in candidate_parameters(config, "P16"):
                        variant = dict(variant)
                        variant["p16_mode"] = True
                        _, variant_meta = preprocess_image(record["_lq_rgb"], "P16", variant)
                        _write_p16_visual_columns(
                            sample_root,
                            variant_meta,
                            record["_lq_rgb"],
                            float(config.get("residual_display_max", 0.25)),
                        )
                if arm == "P17":
                    for variant in candidate_parameters(config, "P17"):
                        _, variant_meta = preprocess_image(record["_lq_rgb"], "P17", variant)
                        _write_p17_visual_columns(
                            sample_root, variant_meta, record["_lq_rgb"],
                            float(config.get("residual_display_max", 0.25)), str(variant["name"]),
                        )
                if arm == "P18":
                    for variant in candidate_parameters(config, "P18"):
                        _, variant_meta = preprocess_image(record["_lq_rgb"], "P18", variant)
                        _write_p18_visual_columns(
                            sample_root, variant_meta, record["_lq_rgb"],
                            float(config.get("residual_display_max", 0.25)), str(variant["name"]),
                        )
                if arm in ("P7", "P8", "P9") and bool(meta.get("segmentation_accepted", False)):
                    _write_p7_mask_evidence(sample_root, record["_lq_rgb"], meta["_p7_mask"], meta["_p7_weight"])
    timing["overall"] = {
        "samples": len(records),
        "arms": len(active_arms),
        "total_ms": float(sum(float(timing[arm]["total_ms"]) for arm in active_arms)),
    }
    return rows, fallbacks, {
        "per_arm": timing,
        "identity_checks": identity_checks,
        "frozen_selection_hash": selection_hash,
        "clipping_rates": clipping_rates,
        "visual_selection": visual_selection,
    }


def aggregate_group_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """@luna-comment Aggregate per-sample metrics by split, arm, and source group.

    Input: Per-sample rows from frozen evaluation.
    Output: One JSON-safe summary row per split/arm/source-group combination.
    Key Steps: Group rows, then compute mean/median/p95 for each metric.
    Key Params: Aggregation never pools records across source groups.
    State/Dependencies: Numpy only; no file writes.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["split"]), str(row["arm"]), str(row["source_group"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (split, arm, source_group), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "split": split,
            "arm": arm,
            "source_group": source_group,
            "sample_count": len(group_rows),
            "fallback_count": int(sum(bool(row["fallback"]) for row in group_rows)),
        }
        for name in METRIC_NAMES:
            values = np.asarray([float(row[name]) for row in group_rows], dtype=np.float64)
            summary[f"{name}_mean"] = float(values.mean())
            summary[f"{name}_median"] = float(np.median(values))
            summary[f"{name}_p95"] = float(np.percentile(values, 95.0))
        summaries.append(summary)
    return summaries


def run_screen(manifest: Path, output_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """@luna-comment Run Stage A selection, frozen evaluation, and evidence export.

    Input: JSONL/JSON manifest, output directory, and optional config path.
    Output: Completion report with counts, hashes, selected arms, and artifact paths.
    Key Steps: Load direct RGB records, split groups, fit on dev, evaluate frozen
      arms, aggregate groups, and write all reproducibility artifacts.
    Key Params: Seed/fraction come from config; held-out never refits parameters.
    State/Dependencies: CPU numpy/OpenCV only; no model/training invocation.
    """
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    records = load_manifest(manifest)
    split_cfg = config.get("split", {})
    seed = int(split_cfg.get("seed", 20260915))
    development_fraction = float(split_cfg.get("development_fraction", 0.5))
    records, split_audit = assign_source_group_splits(records, seed, development_fraction)
    development = [record for record in records if record["split"] == "development"]
    heldout = [record for record in records if record["split"] == "heldout"]
    started = time.perf_counter()
    selected, candidate_rows = fit_parameters(development, config)
    selection_elapsed_ms = (time.perf_counter() - started) * 1000.0
    rows, fallbacks, timing = evaluate_frozen_arms(records, selected, output_root, config)
    group_summary = aggregate_group_metrics(rows)
    manifest_hash = _file_hash(manifest.expanduser().resolve())
    config_hash = _json_hash(config)
    selection_hash = _json_hash(selected)
    _write_jsonl(output_root / "per_sample_metrics.jsonl", rows)
    _write_jsonl(output_root / "fallbacks.jsonl", fallbacks)
    _write_jsonl(output_root / "development_candidate_scores.jsonl", candidate_rows)
    _write_jsonl(output_root / "clipping_rates.jsonl", timing["clipping_rates"])
    if "P6" in selected:
        p6_curves = []
        for curve in candidate_parameters(config, "P6"):
            anchors = np.asarray(curve["anchors"], dtype=np.float32)
            lut_x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            lut_y = np.interp(lut_x, anchors[:, 0], anchors[:, 1]).astype(np.float32)
            p6_curves.append({"params": curve, "params_sha256": _json_hash(curve), "lut_256": [float(value) for value in lut_y]})
        (output_root / "p6_curves.json").write_text(
            json.dumps({"selected": selected["P6"], "curves": p6_curves}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if "P10" in selected:
        p10_candidates = []
        for candidate in candidate_parameters(config, "P10"):
            p10_candidates.append({"params": candidate, "params_sha256": _json_hash(candidate)})
        (output_root / "p10_candidates.json").write_text(
            json.dumps({"selected": selected["P10"], "candidates": p10_candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if "P11" in selected:
        p11_candidates = []
        for candidate in candidate_parameters(config, "P11"):
            p11_candidates.append(
                {
                    "params": candidate,
                    "params_sha256": _json_hash(candidate),
                    "selection": "fixed_declared_matrix",
                    "gt_used_for_selection": False,
                }
            )
        (output_root / "p11_candidates.json").write_text(
            json.dumps({"selected": selected["P11"], "candidates": p11_candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if "P12" in selected:
        p12_candidates = []
        for candidate in candidate_parameters(config, "P12"):
            p12_candidates.append(
                {
                    "params": candidate,
                    "params_sha256": _json_hash(candidate),
                    "selection": "fixed_declared_matrix",
                    "gt_used_for_selection": False,
                }
            )
        (output_root / "p12_candidates.json").write_text(
            json.dumps({"selected": selected["P12"], "candidates": p12_candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if "P13" in selected:
        p13_candidates = []
        for candidate in candidate_parameters(config, "P13"):
            p13_candidates.append(
                {
                    "params": candidate,
                    "params_sha256": _json_hash(candidate),
                    "selection": "fixed_declared_matrix",
                    "gt_used_for_selection": False,
                }
            )
        (output_root / "p13_candidates.json").write_text(
            json.dumps({"selected": selected["P13"], "candidates": p13_candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if "P14" in selected:
        p14_candidates = []
        for candidate in candidate_parameters(config, "P14"):
            p14_candidates.append(
                {
                    "params": candidate,
                    "params_sha256": _json_hash(candidate),
                    "selection": "fixed_declared_matrix",
                    "gt_used_for_selection": False,
                }
            )
        (output_root / "p14_candidates.json").write_text(
            json.dumps({"selected": selected["P14"], "candidates": p14_candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if "P16" in selected:
        p16_candidates = []
        for candidate in candidate_parameters(config, "P16"):
            p16_candidates.append({
                "params": candidate,
                "params_sha256": _json_hash(candidate),
                "selection": "fixed_declared_matrix",
                "gt_used_for_selection": False,
            })
        (output_root / "p16_candidates.json").write_text(
            json.dumps({"selected": selected["P16"], "candidates": p16_candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if "P17" in selected:
        p17_candidates = [
            {"params": candidate, "params_sha256": _json_hash(candidate), "selection": "fixed_declared_matrix", "gt_used_for_selection": False}
            for candidate in candidate_parameters(config, "P17")
        ]
        (output_root / "p17_candidates.json").write_text(
            json.dumps({"selected": selected["P17"], "candidates": p17_candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (output_root / "group_summary.json").write_text(json.dumps({"groups": group_summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "split_audit.json").write_text(json.dumps(split_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "visual_selection.json").write_text(json.dumps(timing["visual_selection"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "timing.json").write_text(json.dumps({"selection_ms": selection_elapsed_ms, **timing}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    parameter_manifest = {
        "status": "FROZEN",
        "tool": "tools/lenovo_stage_a_preprocess_screen.py",
        "version": config.get("version"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_contract": config.get("input_contract"),
        "manifest": str(manifest.expanduser().resolve()),
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "split_audit": split_audit,
        "selection_split": "development",
        "heldout_split": "heldout",
        "selected_parameters": selected,
        "selected_parameters_sha256": selection_hash,
        "candidate_grid": config["arms"],
        "candidate_scores": candidate_rows,
        "frozen_before_heldout": True,
        "source_group_disjoint": not bool(set(split_audit["groups"]["development"]) & set(split_audit["groups"]["heldout"])),
        "records": {"total": len(records), "development": len(development), "heldout": len(heldout)},
        "artifacts": {
            "per_sample_metrics": "per_sample_metrics.jsonl",
            "group_summary": "group_summary.json",
            "development_candidate_scores": "development_candidate_scores.jsonl",
            "fallbacks": "fallbacks.jsonl",
            "clipping_rates": "clipping_rates.jsonl",
            "split_audit": "split_audit.json",
            "visual_selection": "visual_selection.json",
            "timing": "timing.json",
            "visuals": "visuals/",
            "p6_curves": "p6_curves.json" if "P6" in selected else None,
            "p10_candidates": "p10_candidates.json" if "P10" in selected else None,
            "p11_candidates": "p11_candidates.json" if "P11" in selected else None,
            "p12_candidates": "p12_candidates.json" if "P12" in selected else None,
            "p13_candidates": "p13_candidates.json" if "P13" in selected else None,
            "p14_candidates": "p14_candidates.json" if "P14" in selected else None,
        },
    }
    (output_root / "parameter_manifest.json").write_text(json.dumps(parameter_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    identity = timing["identity_checks"]
    (output_root / "identity_checks.json").write_text(json.dumps({"checks": identity, "all_pass": all(row["pass"] for row in identity)}, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "COMPLETE",
        "output_root": str(output_root),
        "records": len(records),
        "development_records": len(development),
        "heldout_records": len(heldout),
        "source_groups": len(split_audit["groups"]["development"]) + len(split_audit["groups"]["heldout"]),
        "selected_parameters_sha256": selection_hash,
        "fallback_count": len(fallbacks),
        "identity_all_pass": all(row["pass"] for row in identity),
        "visual_sample_count": len(timing["visual_selection"]["visual_sample_ids"]),
        "input_contract": config.get("input_contract"),
    }


def build_parser() -> argparse.ArgumentParser:
    """@luna-comment Build the Stage A CLI parser and its bounded user surface.

    Input: None.
    Output: ArgumentParser requiring a manifest and output root.
    Key Steps: Register manifest, output, and optional config arguments.
    Key Params: Config owns seed/fraction and candidate grids for reproducibility.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL/JSON direct-RGB pair manifest")
    parser.add_argument("--output-root", type=Path, required=True, help="new output directory for Stage A artifacts")
    parser.add_argument("--config", type=Path, help="optional JSON config; defaults to conservative Stage A grid")
    return parser


def main() -> None:
    """@luna-comment Parse CLI arguments, run the screen, and print path-hit evidence.

    Input: Process arguments from the Stage A command line.
    Output: JSON completion summary on stdout; artifacts are written by run_screen.
    Key Steps: Parse, execute, and emit a stable status/contract record.
    Key Params: No implicit dataset or checkpoint defaults are used.
    State/Dependencies: Calls run_screen; non-zero exceptions remain visible.
    """
    args = build_parser().parse_args()
    report = run_screen(args.manifest, args.output_root, args.config)
    print(json.dumps({"PATH_HIT": True, **report}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
