"""Audit the V5 evaluator's RGB-input contract on two fixed validation samples.

This is deliberately an isolated diagnostic.  It neither changes the existing
V5 evaluator nor writes inside prior post-evaluation directories.  It compares
the evaluator's historical NV21/OpenCV reconstruction with the dataset RGB
pair consumed by ``LenovoTextLineDegradationModel.feed_data(is_train=False)``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realesrgan.data  # noqa: F401 - register the validation dataset.
from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv, y4uv_to_yuv444
from realesrgan.models.lenovo_text_line_degradation_model import LenovoTextLineDegradationModel
from tools.basic_v2_eval_network import build_eval_network
from tools.dump_basic_v2_validation_rgb_pairs import (
    _load_dataset,
    _nv21_to_rgb,
    _write_rgb,
    _yuv444_to_rgb,
)


DEFAULT_LINES = ("000_000000/v2", "045_000264/v3")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--line-id",
        action="append",
        dest="line_ids",
        help="Repeat exactly twice to override the two bounded audit samples.",
    )
    return parser.parse_args()


def _to_rgb(value: torch.Tensor) -> np.ndarray:
    """Convert a CHW RGB dataset tensor to a contiguous HWC float array."""
    return value.detach().cpu().float().permute(1, 2, 0).numpy().copy()


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _masked_stats(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    count = int(mask.sum())
    if count == 0:
        return {"pixels": 0, "mean_y_255": None, "mean_rgb_255": None}
    return {
        "pixels": count,
        "mean_y_255": float(_luma(rgb)[mask].mean() * 255.0),
        "mean_rgb_255": float(rgb[mask].mean() * 255.0),
    }


def _image_delta(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    diff = left - right
    return {
        "max_abs_rgb_255": float(np.abs(diff).max() * 255.0),
        "mean_abs_rgb_255_valid": float(np.abs(diff[valid]).mean() * 255.0),
        "mean_y_delta_255_valid": float(_luma(diff)[valid].mean() * 255.0),
    }


def _estimated_text_mask(gt_rgb: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Make a clearly labeled, audit-only text estimate from GT luminance.

    This fixed validation manifest has no structure asset, so its dataset
    ``text_mask_rgb`` is the full valid canvas.  A minority-side Otsu split is
    adequate for inspecting the two user-selected text lines, but is never a
    replacement for a supplied text annotation.
    """
    gray = np.clip(np.rint(_luma(gt_rgb) * 255.0), 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(gray[valid], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = valid & (gray <= threshold)
    bright = valid & (gray > threshold)
    text = dark if dark.sum() <= bright.sum() else bright
    return text, {
        "method": "GT luminance Otsu; choose minority polarity",
        "threshold_y_255": float(threshold),
        "polarity": "dark" if text is dark else "bright",
        "pixels": int(text.sum()),
    }


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    value = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8).copy()
    cv2.rectangle(value, (0, 0), (min(value.shape[1] - 1, 240), 23), (0, 0, 0), -1)
    cv2.putText(value, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return value.astype(np.float32) / 255.0


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.where(mask, 255, 0).astype(np.uint8)):
        raise IOError(f"failed to write mask: {path}")


def _batch(item: dict[str, Any]) -> dict[str, Any]:
    """Add a batch dimension to dataset tensors for the real non-train feed path."""
    return {
        key: (value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value])
        for key, value in item.items()
    }


def _actual_feed_data_contract(
    item: dict[str, Any], device: torch.device, inference_packed: torch.Tensor
) -> dict[str, Any]:
    """Exercise remote training packing with a deliberately injected fixed LQ.

    The remote training source currently has no non-training branch.  Its
    degradation callable is therefore replaced only on this throwaway probe,
    allowing the exact feed_data packing after the degradation output to be
    compared with the evaluator's direct RGB packing.
    """
    probe = LenovoTextLineDegradationModel.__new__(LenovoTextLineDegradationModel)
    probe.__dict__["device"] = device
    fixed_lq = item["lq_rgb"].unsqueeze(0).to(device).float()
    probe.__dict__["lenovo_degradation"] = lambda _gt, _generator: (fixed_lq, [{"mode": "isolated_fixed_lq"}])
    probe.__dict__["_generator"] = lambda _device: None
    LenovoTextLineDegradationModel.feed_data(probe, _batch(item))
    expected_gt = rgb_to_y4uv(item["gt_rgb"].unsqueeze(0).to(device).float().clamp(0.0, 1.0))
    return {
        "feed_data_lq_vs_direct_inference_packed_max_abs": float((probe.lq - inference_packed).abs().max().item()),
        "feed_data_gt_packed_max_abs": float((probe.gt - expected_gt).abs().max().item()),
        "feed_data_lq_rgb_max_abs": float((probe.lq_rgb - item["lq_rgb"].unsqueeze(0).to(device)).abs().max().item()),
        "feed_data_gt_rgb_max_abs": float((probe.gt_rgb - item["gt_rgb"].unsqueeze(0).to(device)).abs().max().item()),
        "contract": "remote training feed_data with lq_rgb injected as LenovoDegradationPipeline output",
    }


def _network_output(network: torch.nn.Module, rgb: np.ndarray, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    packed = rgb_to_y4uv(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)).to(device)
    return _yuv444_to_rgb(y4uv_to_yuv444(network(packed))), packed


def _find_items(dataset: Any, requested: tuple[str, ...]) -> list[tuple[int, dict[str, Any]]]:
    # The HTML visual viewer keys a tile as ``<GT filename stem>/<variant>``.
    # Derive that key from the actual LQ path, never from a guessed line_id.
    wanted = set(requested)
    matches: dict[str, tuple[int, dict[str, Any]]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        lq_path = Path(str(item["lq_path"]))
        variant = lq_path.parent.name.replace("variant_", "v", 1)
        viewer_key = f"{lq_path.stem}/{variant}"
        if viewer_key in wanted:
            matches[viewer_key] = (index, item)
            if len(matches) == len(wanted):
                break
    missing = sorted(wanted.difference(matches))
    if missing:
        raise RuntimeError(f"requested audit line_id values were not in the validation set: {missing}")
    return [matches[viewer_key] for viewer_key in requested]


def audit(config: Path, checkpoint: Path, output_root: Path, device_name: str, line_ids: tuple[str, ...]) -> dict[str, Any]:
    """Run exactly two old/direct input comparisons with one supplied G2 checkpoint."""
    if len(line_ids) != 2 or len(set(line_ids)) != 2:
        raise ValueError("this bounded audit requires exactly two distinct --line-id values")
    config, checkpoint, output_root = (path.expanduser().resolve() for path in (config, checkpoint, output_root))
    if not config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("config and checkpoint must exist")
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = _load_dataset(config)
    network, checkpoint_sha256 = build_eval_network(checkpoint, device, config)
    runtime_model_source = Path(inspect.getfile(LenovoTextLineDegradationModel)).resolve()
    runtime_model_sha256 = hashlib.sha256(runtime_model_source.read_bytes()).hexdigest()
    records: list[dict[str, Any]] = []

    with torch.inference_mode():
        for index, item in _find_items(dataset, line_ids):
            line_id = str(item["line_id"])
            sample_root = output_root / line_id.replace("/", "__")
            old_lq = _nv21_to_rgb(item["lq"])
            direct_lq = _to_rgb(item["lq_rgb"])
            old_gt = _yuv444_to_rgb(item["gt"])
            direct_gt = _to_rgb(item["gt_rgb"])
            valid = item["valid_mask_rgb"][0].numpy() > 0.5
            dataset_text = item["text_mask_rgb"][0].numpy() > 0.5
            text, text_estimate = _estimated_text_mask(direct_gt, valid)
            background = valid & ~text
            background_far = cv2.erode(background.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
            text_inner = cv2.erode((valid & text).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
            old_output, _ = _network_output(network, old_lq, device)
            direct_output, direct_packed = _network_output(network, direct_lq, device)
            direct_roundtrip = _yuv444_to_rgb(y4uv_to_yuv444(rgb_to_y4uv(item["lq_rgb"].unsqueeze(0))))
            feed_contract = _actual_feed_data_contract(item, device, direct_packed)

            images = {
                "old_lq_nv21_opencv": old_lq,
                "direct_lq_rgb": direct_lq,
                "old_gt_yuv444_decoder": old_gt,
                "direct_gt_rgb": direct_gt,
                "old_output": old_output,
                "direct_output": direct_output,
                "direct_lq_y4uv_roundtrip": direct_roundtrip,
            }
            for name, image in images.items():
                _write_rgb(sample_root / f"{name}.png", image)
            masks = {
                "valid": valid,
                "text_dataset_unusable": valid & dataset_text,
                "text_estimated_otsu": text,
                "background": background,
                "background_far_erode9": background_far,
                "text_inner_erode3": text_inner,
            }
            for name, mask in masks.items():
                _write_mask(sample_root / f"mask_{name}.png", mask)
            montage = np.concatenate([
                _label(old_lq, "OLD LQ: NV21/OpenCV"),
                _label(direct_lq, "DIRECT LQ: dataset RGB"),
                _label(old_output, "OUTPUT: old LQ"),
                _label(direct_output, "OUTPUT: direct LQ"),
                _label(direct_gt, "GT: dataset RGB"),
            ], axis=1)
            _write_rgb(sample_root / "comparison.png", montage)
            regions = masks
            sample = {
                "index": index,
                "line_id": line_id,
                "lq_path": str(item["lq_path"]),
                "gt_path": str(item["gt_path"]),
                "valid_hw": [int(value) for value in item["valid_hw"]],
                "dataset_text_mask_pixels": int((valid & dataset_text).sum()),
                "text_region_for_audit_only": text_estimate,
                "brightness_y_255": {name: {region: _masked_stats(image, mask) for region, mask in regions.items()} for name, image in images.items()},
                "deltas": {
                    "old_lq_minus_direct_lq": _image_delta(old_lq, direct_lq, valid),
                    "old_gt_minus_direct_gt": _image_delta(old_gt, direct_gt, valid),
                    "old_output_minus_direct_output": _image_delta(old_output, direct_output, valid),
                    "direct_lq_y4uv_roundtrip_minus_direct_lq": _image_delta(direct_roundtrip, direct_lq, valid),
                },
                "feed_data_nontrain_contract": feed_contract,
                "artifacts": {name: str((sample_root / f"{name}.png").relative_to(output_root)) for name in images} | {"comparison": str((sample_root / "comparison.png").relative_to(output_root))},
            }
            sample["artifacts"]["masks"] = {
                name: str((sample_root / f"mask_{name}.png").relative_to(output_root)) for name in masks
            }
            (sample_root / "sample_report.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(sample)
            print(f"[color-audit] {index}: {line_id}", flush=True)

    report = {
        "status": "COMPLETE",
        "scope": "isolated two-sample G2 color-contract audit; does not replace existing evaluations",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "requested_line_ids": list(line_ids),
        "corrected_evaluator_input_contract": "item['lq_rgb'] -> rgb_to_y4uv -> network; GT reference item['gt_rgb']",
        "historical_evaluator_input_contract": "item['lq'] -> OpenCV COLOR_YUV2RGB_NV21 -> rgb_to_y4uv -> network",
        "remote_training_packing_contract": "remote feed_data always calls Lenovo degradation; this audit injects lq_rgb into a throwaway probe and verifies subsequent rgb_to_y4uv packing",
        "runtime_model_source": str(runtime_model_source),
        "runtime_model_sha256": runtime_model_sha256,
        "boundary": "The remote implementation lacks the local non-training branch. BasicSR nondist_validation call-state and V4 evaluator remain out of scope; this script changes neither.",
        "records": records,
    }
    (output_root / "color_contract_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = _args()
    report = audit(args.config, args.checkpoint, args.output_root, args.device, tuple(args.line_ids or DEFAULT_LINES))
    print(json.dumps({"status": report["status"], "output_root": str(args.output_root), "samples": len(report["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
