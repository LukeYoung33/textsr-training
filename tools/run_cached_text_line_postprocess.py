"""Run the frozen V5 cached text-line postprocess experiment offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.lenovo_stage_a_preprocess_screen import _p10_segment_text, _rgb_to_luma, candidate_parameters, load_config

MODE = "cached_maskraw_hard_foreground_outer_frame_feather_v5"
RING_WEIGHTS = (0.015625, 0.0625, 0.140625, 0.25, 0.390625, 0.5625, 0.765625, 1.0)
KERNEL_3X3 = np.ones((3, 3), np.uint8)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_rgb(yuv: np.ndarray) -> np.ndarray:
    pixels = np.rint(yuv.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(pixels, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0


def _png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _rgb_png(path: Path, yuv: np.ndarray) -> None:
    rgb = _to_rgb(yuv)
    _png(path, cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _params() -> dict[str, Any]:
    config = load_config(ROOT / "options" / ("lenovo_stage_a_preprocess_screen_p11_kernel_" + "wiener_scurve.json"))
    return next(item for item in candidate_parameters(config, "P11") if item["name"] == "lambda_0p03")


def _rings(mask: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    current, output = mask.copy(), []
    for _ in range(12):
        expanded = cv2.dilate(current.astype(np.uint8), KERNEL_3X3).astype(bool) & valid
        output.append(expanded & ~current)
        current = expanded
    return output


def _evidence(valid: np.ndarray) -> dict[str, np.ndarray]:
    empty = np.zeros_like(valid, dtype=bool)
    result = {"valid": valid, "low": empty, "high": empty, "raw": empty, "foreground": empty,
              "weight": np.zeros_like(valid, np.float32)}
    result.update({f"ring_{index}": empty for index in range(1, 13)})
    return result


def fill_v5(prediction: np.ndarray, frame_crop: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """@luna-comment
    Input: normalized cached prediction/frame YUV crops [3,H,W] and valid support.
    Output: V5 reconstructed crop, raw-mask/ring evidence, and line telemetry.
    Key Steps: P10 raw mask, hard foreground/protection writes, outer cache-frame recovery.
    Key Params: frozen contrast clamp and the R5--R12 weight schedule.
    State/Dependencies: only supplied cache arrays and frozen P10 parameters are read.
    """
    if prediction.shape != frame_crop.shape or prediction.ndim != 3 or valid.shape != prediction.shape[1:]:
        raise ValueError("V5 crop/support shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(frame_crop).all():
        raise ValueError("V5 non-finite cache array")
    evidence = _evidence(valid)
    telemetry: dict[str, Any] = {"status": "identity_skip", "valid_pixels": int(valid.sum())}
    raw, meta = _p10_segment_text(_rgb_to_luma(_to_rgb(prediction)), _params(), valid_mask=valid)
    if raw is None or not meta.get("segmentation_accepted", False):
        telemetry["identity_skip_reason"] = "p10_rejected"
        return frame_crop.copy(), evidence, telemetry
    low = np.asarray(meta["_p10_low_class"], bool)
    high = np.asarray(meta["_p10_high_class"], bool)
    domain = np.asarray(meta["_p10_valid"], bool)
    raw = np.asarray(raw, bool)
    if not (np.array_equal(domain, valid) and np.array_equal(low | high, valid) and not (low & high).any() and not (raw & ~valid).any()):
        raise ValueError("V5 P10 partition contract failed")
    polarity = str(meta.get("polarity", ""))
    if polarity.startswith("dark_text_on_light"):
        foreground_class, background_class = low, high
    elif polarity.startswith("light_text_on_dark"):
        foreground_class, background_class = high, low
    else:
        telemetry["identity_skip_reason"] = "unknown_polarity"
        return frame_crop.copy(), evidence, telemetry
    if not raw.any() or not foreground_class.any() or not background_class.any():
        telemetry["identity_skip_reason"] = "empty_class"
        return frame_crop.copy(), evidence, telemetry
    foreground_anchor = np.array([np.median(prediction[channel, foreground_class]) for channel in range(3)], np.float32)
    background_anchor = np.array([np.median(prediction[channel, background_class]) for channel in range(3)], np.float32)
    delta_y = float(foreground_anchor[0] - background_anchor[0])
    if not np.isfinite(foreground_anchor).all() or not np.isfinite(background_anchor).all() or abs(delta_y) < .12:
        telemetry["identity_skip_reason"] = "anchor_rejected"
        return frame_crop.copy(), evidence, telemetry
    contrast = min(.55, max(.16, 1.15 * abs(delta_y)))
    hard_foreground = foreground_anchor.copy()
    hard_foreground[0] = np.clip(background_anchor[0] + np.sign(delta_y) * contrast, 0, 1)
    hard_foreground[1:] = np.clip(hard_foreground[1:], 0, 1)
    background_anchor = np.clip(background_anchor, 0, 1)
    rings = _rings(raw, valid)
    output = frame_crop.copy()
    output[:, raw] = hard_foreground[:, None]
    for ring in rings[:4]:
        output[:, ring] = background_anchor[:, None]
    weights = np.zeros_like(valid, np.float32)
    for ring, weight in zip(rings[4:], RING_WEIGHTS):
        output[:, ring] = (1 - weight) * background_anchor[:, None] + weight * frame_crop[:, ring]
        weights[ring] = weight
    evidence.update({"low": low, "high": high, "raw": raw, "foreground": raw, "weight": weights})
    evidence.update({f"ring_{index}": ring for index, ring in enumerate(rings, 1)})
    telemetry.update({"status": "processed", "polarity": polarity, "raw_mask_pixels": int(raw.sum()),
                      "foreground_anchor_yuv": foreground_anchor.tolist(), "background_anchor_yuv": background_anchor.tolist(),
                      "hard_foreground_yuv": hard_foreground.tolist(), "foreground_background_delta_y": delta_y,
                      "contrast_target": contrast, "foreground_feather_pixels": 0, "protection_feather_pixels": 0,
                      "cache_frame_donor_only": True, "ring_pixels": {str(i): int(r.sum()) for i, r in enumerate(rings, 1)},
                      "ring_weights": {str(i): w for i, w in zip(range(5, 13), RING_WEIGHTS)}})
    return output, evidence, telemetry


def verify_cache(cache_root: Path) -> dict[str, Any]:
    """@luna-comment
    Input: cache root with manifest, frame arrays, and per-line artifact hashes.
    Output: verified cache manifest or a hard pre-execution failure.
    Key Steps: check manifest/hash receipt, each artifact hash, and array shape/finiteness.
    Key Params: artifact names are frozen by the V5 manifest.
    State/Dependencies: immutable cache artifacts only.
    """
    manifest = json.loads((cache_root / "cache_manifest.json").read_text())
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if hashlib.sha256(canonical).hexdigest() != (cache_root / "cache_manifest.sha256").read_text().strip() or not (cache_root / "cache_build_receipt.json").is_file():
        raise ValueError("cache manifest/receipt invalid")
    for frame in manifest["frames"]:
        directory = cache_root / frame["directory"]
        frame_yuv = np.load(directory / "frame_yuv444.npy")
        if frame_yuv.ndim != 3 or frame_yuv.shape[0] != 3 or not np.isfinite(frame_yuv).all():
            raise ValueError("invalid frame cache array")
        for line in frame["lines"]:
            directory = cache_root / line["directory"]
            for name, receipt in json.loads((directory / "artifact.sha256.json").read_text()).items():
                if _sha256(directory / name) != receipt["sha256"]:
                    raise ValueError(f"cache hash mismatch: {name}")
            prediction, support = np.load(directory / "prediction_yuv444.npy"), np.load(directory / "valid_support.npy")
            if prediction.ndim != 3 or prediction.shape[0] != 3 or prediction.shape[1:] != support.shape or not np.isfinite(prediction).all():
                raise ValueError("invalid line cache array")
    return manifest


def _delta_image(delta: np.ndarray) -> np.ndarray:
    scale = max(float(np.abs(delta).max()), 1e-6)
    return np.rint(np.clip(delta / scale * .5 + .5, 0, 1) * 255).astype(np.uint8)


def run(manifest_path: Path, cache_root: Path, output_root: Path) -> None:
    """@luna-comment
    Input: frozen V5 manifest, verified cache root, and a fresh experiment directory.
    Output: full PNG result, V5 debug/telemetry, summary, receipt, and command log.
    Key Steps: verify cache, process ascending lines, then apply hard first-line ownership.
    Key Params: V5 mode and output schema are both manifest-constrained.
    State/Dependencies: cache-only execution; model and OCR are never invoked.
    """
    experiment = json.loads(manifest_path.read_text())
    if experiment["authorized_postprocess"] != MODE:
        raise ValueError("manifest mode mismatch")
    cache_manifest = verify_cache(cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"experiment_id": experiment["experiment_id"], "mode": MODE, "images": []}
    for frame in cache_manifest["frames"]:
        frame_dir = cache_root / frame["directory"]
        cached_frame = np.load(frame_dir / "frame_yuv444.npy")
        result, taken, routes = cached_frame.copy(), np.zeros(cached_frame.shape[1:], bool), []
        for line in sorted(frame["lines"], key=lambda item: item["line_index"]):
            line_dir = cache_root / line["directory"]
            prediction = np.load(line_dir / "prediction_yuv444.npy")
            valid = np.load(line_dir / "valid_support.npy").astype(bool)
            left, top, right, bottom = json.loads((line_dir / "geometry.json").read_text())["crop_xyxy"]
            cache_crop = cached_frame[:, top:bottom, left:right]
            reconstructed, evidence, telemetry = fill_v5(prediction, cache_crop, valid)
            prior = taken[top:bottom, left:right]
            write = valid & ~prior
            result[:, top:bottom, left:right][:, write] = reconstructed[:, write]
            taken[top:bottom, left:right][write] = True
            debug = output_root / "debug" / frame["image_id"] / f"line_{line['line_index']:04d}"
            files = [("valid_support.png", "valid"), ("otsu_low_class.png", "low"), ("otsu_high_class.png", "high"), ("mask_raw.png", "raw"), ("foreground_hard_support.png", "foreground")]
            files += [(f"protection_ring_{i}.png", f"ring_{i}") for i in range(1, 5)]
            files += [(f"frame_feather_ring_{i}.png", f"ring_{i}") for i in range(5, 13)]
            for name, key in files:
                _png(debug / name, np.where(evidence[key], 255, 0).astype(np.uint8))
            _png(debug / "frame_feather_weight.png", np.rint(evidence["weight"] * 255).astype(np.uint8))
            foreground_yuv = np.asarray(telemetry.get("hard_foreground_yuv", [0, .5, .5]), np.float32)
            _png(debug / "foreground_reconstruction_y.png", np.full(valid.shape, int(round(foreground_yuv[0] * 255)), np.uint8))
            _rgb_png(debug / "foreground_reconstruction_uv.png", np.broadcast_to(foreground_yuv[:, None, None], prediction.shape))
            _rgb_png(debug / "cache_frame_crop.png", cache_crop)
            _rgb_png(debug / "cached_prediction.png", prediction)
            _rgb_png(debug / "reconstructed_line.png", reconstructed)
            for channel, name in enumerate(("y", "u", "v")):
                _png(debug / f"delta_{name}_signed.png", _delta_image(reconstructed[channel] - prediction[channel]))
            telemetry.update({"line_index": line["line_index"], "overlap_conflict_pixels": int((valid & prior).sum()), "write_pixels": int(write.sum())})
            (debug / "line_telemetry.json").write_text(json.dumps(telemetry, ensure_ascii=False, indent=2))
            routes.append(telemetry)
        _rgb_png(output_root / f"{frame['image_id']}.png", result)
        (output_root / f"{frame['image_id']}.routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2))
        summary["images"].append({"image_id": frame["image_id"], "lines": len(routes)})
    summary["manifest_sha256"] = _sha256(manifest_path)
    (output_root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_root / "run_receipt.json").write_text(json.dumps({"manifest_sha256": summary["manifest_sha256"], "mode": MODE}, indent=2))
    (output_root / "command.log").write_text("offline V5 cached postprocess; OCR/model not executed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.manifest, arguments.cache_root, arguments.output_dir)
