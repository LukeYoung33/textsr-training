"""Inspect IQ560 JSONL manifests and run a bounded BASIC V2 smoke fit.

The script never edits source data. When the configured IQ560 root is not mounted,
it runs a clearly marked synthetic contract fixture using the same manifest schema,
so model/data/gradient plumbing remains testable on a local workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if not (_REPO_ROOT / "realesrgan").is_dir() and (Path.cwd() / "realesrgan").is_dir():
    _REPO_ROOT = Path.cwd()
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

try:
    from realesrgan.archs.basic_text_line_restore_v2_arch import BasicTextLineRestoreNetV2
    from realesrgan.losses.basic_text_line_losses_v2 import MaskedL1LossV2Basic
except ModuleNotFoundError:
    # Remote smoke fallback: avoid importing the repository-wide registry when
    # optional legacy dependencies are not installed in the remote environment.
    from torch import nn

    class BasicTextLineRestoreNetV2(nn.Module):
        """@shanlan-comment Minimal identical V2 forward for isolated remote smoke."""
        def __init__(self, canvas_hw=(96, 320), features=8):
            super().__init__()
            self.canvas_hw = tuple(canvas_hw)
            self.body = nn.Sequential(nn.Conv2d(3, features, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(features, features, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(features, 3, 3, padding=1))

        def forward(self, input_yuv420):
            height, width = self.canvas_hw
            y = input_yuv420[:, :, :height, :]
            uv = input_yuv420[:, :, height:, :].reshape(input_yuv420.shape[0], 1, height // 2, width // 2, 2)
            uv = uv.permute(0, 1, 4, 2, 3).reshape(input_yuv420.shape[0], 2, height // 2, width // 2)
            uv = torch.nn.functional.interpolate(uv, size=(height, width), mode="bilinear", align_corners=False)
            base = torch.cat([y, uv], dim=1)
            return base + self.body(base)

    class MaskedL1LossV2Basic(nn.Module):
        """@shanlan-comment Minimal valid-mask-only loss for isolated remote smoke."""
        def __init__(self, require_valid_mask=True):
            super().__init__()
            self.require_valid_mask = require_valid_mask

        def forward(self, pred, target, valid_mask=None, **kwargs):
            if valid_mask is None and self.require_valid_mask:
                raise ValueError("valid_mask is required")
            valid_mask = torch.ones_like(pred[:, :1]) if valid_mask is None else valid_mask
            return ((pred - target).abs() * valid_mask).sum() / (valid_mask.sum() * pred.shape[1] + 1e-6)

    REMOTE_FALLBACK_MODEL = True
else:
    REMOTE_FALLBACK_MODEL = False

DEFAULT_ROOT = Path("/AI_DNS/guang.yang/TextSR/data/IQ_data_grouped_560/training_lists_full_20260812")
DEFAULT_OUTPUT = Path("results/basic_v2_iq560_smoke_20260815")
CANVAS_HW = (96, 320)


def _sha256_file(path: Path) -> str:
    """@shanlan-comment Hash a manifest for reproducibility without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_manifests(root: Path) -> Dict[str, Any]:
    """@shanlan-comment Inspect train/val JSONL counts, fields, sources, and path availability."""
    result: Dict[str, Any] = {"root": str(root), "available": root.is_dir(), "splits": {}}
    if not root.is_dir():
        return result
    for split in ("train", "val"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            result["splits"][split] = {"exists": False}
            continue
        records = []
        fields = set()
        datasets: Dict[str, int] = {}
        missing_assets = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                records.append(record)
                fields.update(record.keys())
                name = str(record.get("dataset", record.get("source", "unknown")))
                datasets[name] = datasets.get(name, 0) + 1
                for key in ("lq", "gt", "valid_mask", "text_mask", "ocr"):
                    if record.get(key) and not Path(record[key]).is_file():
                        missing_assets += 1
        result["splits"][split] = {
            "exists": True,
            "records": len(records),
            "fields": sorted(fields),
            "datasets": datasets,
            "missing_asset_references": missing_assets,
            "sha256": _sha256_file(path),
            "first_record": records[0] if records else None,
        }
    return result


def _load_image(path: str, channels: int = 3) -> np.ndarray:
    """@shanlan-comment Decode an image or grayscale mask into float32 HWC data."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if channels == 1:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)[..., None] / (255.0 if image.max() > 1.0 else 1.0)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / (255.0 if image.max() > 1.0 else 1.0)


def _prepare_rgb(rgb: np.ndarray, canvas_hw: Tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Pad RGB data to the fixed canvas without resizing or changing pixel scale."""
    height, width = canvas_hw
    source = rgb[:height, :width]
    padded = np.zeros((height, width, 3), dtype=np.float32)
    padded[:source.shape[0], :source.shape[1]] = source
    return padded


def _rgb_to_nv21(rgb: np.ndarray, canvas_hw: Tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Encode padded RGB data as canonical packed NV21."""
    height, width = canvas_hw
    padded = _prepare_rgb(rgb, canvas_hw)
    u8 = np.clip(np.rint(padded * 255.0), 0, 255).astype(np.uint8)
    yuv = cv2.cvtColor(u8, cv2.COLOR_RGB2YUV)
    y = yuv[..., 0]
    u = cv2.resize(yuv[..., 1], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[..., 2], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    vu = np.empty((height // 2, width), np.uint8)
    vu[:, 0::2] = v
    vu[:, 1::2] = u
    return np.concatenate([y, vu], axis=0).astype(np.float32) / 255.0


def _rgb_to_yuv444(rgb: np.ndarray, canvas_hw: Tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Convert RGB data to padded planar YUV444 CHW tensor data."""
    padded = _prepare_rgb(rgb, canvas_hw)
    u8 = np.clip(np.rint(padded * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2YUV).astype(np.float32).transpose(2, 0, 1) / 255.0


def _mask_canvas(path: str | None, source_hw: Tuple[int, int], canvas_hw: Tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Load and pad a valid/text mask, defaulting to an explicit valid ROI."""
    height, width = canvas_hw
    if path and Path(path).is_file():
        mask = _load_image(path, channels=1)[..., 0]
        mask = mask[:height, :width]
    else:
        mask = np.ones((min(height, source_hw[0]), min(width, source_hw[1])), dtype=np.float32)
    output = np.zeros((height, width), dtype=np.float32)
    output[:mask.shape[0], :mask.shape[1]] = (mask > 0.5).astype(np.float32)
    return output


def _ocr_line_polygons(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """@shanlan-comment Parse OCR line polygons and text from the explicit JSON annotation."""
    path = Path(record["ocr"])
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons = data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes") or []
    texts = data.get("rec_texts") or data.get("text_word") or []
    lines: List[Dict[str, Any]] = []
    for index, raw_polygon in enumerate(polygons):
        try:
            polygon = np.asarray(raw_polygon, dtype=np.float32).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if polygon.shape[0] < 4 or not np.isfinite(polygon).all():
            continue
        x0, y0 = polygon.min(axis=0)
        x1, y1 = polygon.max(axis=0)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        text = str(texts[index]).strip() if index < len(texts) else ""
        lines.append({"polygon": polygon, "text": text, "ocr_line_index": index,
                      "bbox": (float(x0), float(y0), float(x1), float(y1))})
    return lines


def _line_orientation_and_axis(polygon: np.ndarray) -> Tuple[str, np.ndarray]:
    """@shanlan-comment Classify a polygon by its minimum-area rectangle and return its long axis."""
    rectangle = cv2.minAreaRect(polygon.astype(np.float32))
    points = cv2.boxPoints(rectangle).astype(np.float32)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    axis = edges[int(np.argmax(lengths))]
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    if axis[0] < 0 or (abs(float(axis[0])) < 1e-6 and axis[1] < 0):
        axis = -axis
    angle = abs(float(np.degrees(np.arctan2(axis[1], axis[0]))))
    if angle <= 20.0 or angle >= 160.0:
        orientation = "HORIZONTAL"
    elif 70.0 <= angle <= 110.0:
        orientation = "VERTICAL"
    else:
        orientation = "OBLIQUE"
    return orientation, axis


def _even_bounds(x0: float, y0: float, x1: float, y1: float, image_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """@shanlan-comment Expand a polygon crop outward to even in-bounds YUV420 coordinates."""
    height, width = image_hw
    left = max(0, int(np.floor(x0)) - int(np.floor(x0)) % 2)
    top = max(0, int(np.floor(y0)) - int(np.floor(y0)) % 2)
    right = min(width, int(np.ceil(x1)) + int(np.ceil(x1)) % 2)
    bottom = min(height, int(np.ceil(y1)) + int(np.ceil(y1)) % 2)
    if (right - left) % 2 and right < width:
        right += 1
    if (bottom - top) % 2 and bottom < height:
        bottom += 1
    if right <= left or bottom <= top:
        raise ValueError("empty OCR polygon crop")
    return left, top, right, bottom


def _line_windows(line: Dict[str, Any], canvas_hw: Tuple[int, int], image_hw: Tuple[int, int]) -> List[Dict[str, Any]]:
    """@shanlan-comment Build context-aware horizontal slices or vertical short segments from one OCR polygon."""
    polygon = line["polygon"]
    orientation, axis = _line_orientation_and_axis(polygon)
    x0, y0, x1, y1 = line["bbox"]
    short_side = max(1.0, min(x1 - x0, y1 - y0))
    margin = max(4.0, 0.15 * short_side)
    x0 -= margin; y0 -= margin; x1 += margin; y1 += margin
    canvas_h, canvas_w = canvas_hw
    span = (x1 - x0) if orientation == "HORIZONTAL" else (y1 - y0)
    limit = canvas_w if orientation != "VERTICAL" else canvas_h
    overlap = int(round(0.25 * limit))
    if orientation == "VERTICAL":
        start, end = y0, y1
    else:
        start, end = x0, x1
    if min(x1 - x0, y1 - y0) > canvas_h:
        return []
    stride = max(1, limit - overlap)
    windows = []
    cursor = start
    while cursor < end:
        segment_end = min(end, cursor + limit)
        left, top, right, bottom = x0, y0, x1, y1
        if orientation == "VERTICAL":
            top, bottom = cursor, segment_end
        else:
            left, right = cursor, segment_end
        try:
            crop = _even_bounds(left, top, right, bottom, image_hw)
        except ValueError:
            break
        if crop[2] - crop[0] <= canvas_w and crop[3] - crop[1] <= canvas_h:
            windows.append({"crop": crop, "orientation": orientation, "axis": axis,
                            "tile_index": len(windows), "tile_count": 0})
        if segment_end >= end:
            break
        cursor += stride
    for item in windows:
        item["tile_count"] = len(windows)
    return windows


def _polygon_mask_on_crop(polygon: np.ndarray, crop: Tuple[int, int, int, int], canvas_hw: Tuple[int, int]) -> np.ndarray:
    """@shanlan-comment Rasterize one OCR polygon in its padded fixed canvas coordinates."""
    left, top, right, bottom = crop
    canvas_h, canvas_w = canvas_hw
    local = polygon - np.array([left, top], dtype=np.float32)
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    clipped = local.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, canvas_w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, canvas_h - 1)
    cv2.fillPoly(mask, [np.round(clipped).astype(np.int32)], 255)
    return mask.astype(np.float32) / 255.0


def _build_ocr_line_samples(record: Dict[str, Any], canvas_hw: Tuple[int, int], max_samples: int | None = None) -> List[Dict[str, Any]]:
    """@shanlan-comment Convert real OCR polygons into one-line fixed-canvas samples and slice metadata."""
    lq = _load_image(record["lq"])
    gt = _load_image(record["gt"])
    source_hw = lq.shape[:2]
    valid_source = _load_image(record["valid_mask"], channels=1)[..., 0] if Path(record["valid_mask"]).is_file() else np.ones(source_hw, np.float32)
    samples: List[Dict[str, Any]] = []
    for line in _ocr_line_polygons(record):
        for window in _line_windows(line, canvas_hw, source_hw):
            left, top, right, bottom = window["crop"]
            lq_crop = lq[top:bottom, left:right]
            gt_crop = gt[top:bottom, left:right]
            valid_crop = (valid_source[top:bottom, left:right] > 0.5).astype(np.float32)
            h, w = canvas_hw
            lq_canvas = np.zeros((h, w, 3), np.float32); lq_canvas[:lq_crop.shape[0], :lq_crop.shape[1]] = lq_crop
            gt_canvas = np.zeros((h, w, 3), np.float32); gt_canvas[:gt_crop.shape[0], :gt_crop.shape[1]] = gt_crop
            valid_canvas = np.zeros((h, w), np.float32); valid_canvas[:valid_crop.shape[0], :valid_crop.shape[1]] = valid_crop
            text_mask = _polygon_mask_on_crop(line["polygon"], window["crop"], canvas_hw)
            samples.append({"lq_rgb": lq_canvas, "gt_rgb": gt_canvas, "valid": valid_canvas,
                            "text": text_mask, "sample_id": f"{record['stem']}:ocr{line['ocr_line_index']}:tile{window['tile_index']}",
                            "orientation": window["orientation"], "tile_index": window["tile_index"],
                            "tile_count": window["tile_count"], "crop_xyxy": window["crop"],
                            "text_area": float(text_mask.sum())})
            if max_samples and len(samples) >= max_samples:
                return samples
    return samples


class IQ560SmokeDataset(Dataset):
    """Read real manifest assets when mounted, or deterministic synthetic records."""

    def __init__(self, records: Iterable[Dict[str, Any]], synthetic: bool, canvas_hw: Tuple[int, int] = CANVAS_HW):
        self.records = list(records)
        self.synthetic = synthetic
        self.canvas_hw = canvas_hw
        self.line_samples = [] if synthetic else [sample for record in self.records for sample in _build_ocr_line_samples(record, canvas_hw, max_samples=8)]
        if not synthetic and not self.line_samples:
            raise RuntimeError("No OCR polygon-derived line samples were generated")

    def __len__(self) -> int:
        return len(self.records) if self.synthetic else len(self.line_samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        h, w = self.canvas_hw
        if not self.synthetic:
            sample = self.line_samples[index]
            return {
                "lq": torch.from_numpy(_rgb_to_nv21(sample["lq_rgb"], self.canvas_hw)[None]),
                "gt": torch.from_numpy(_rgb_to_yuv444(sample["gt_rgb"], self.canvas_hw)),
                "lq_rgb": torch.from_numpy(np.transpose(sample["lq_rgb"], (2, 0, 1)).copy()),
                "gt_rgb": torch.from_numpy(np.transpose(sample["gt_rgb"], (2, 0, 1)).copy()),
                "valid_mask": torch.from_numpy(sample["valid"][None]),
                "text_mask": torch.from_numpy(sample["text"][None]),
                "sample_id": sample["sample_id"],
                "orientation": sample["orientation"],
                "tile_index": sample["tile_index"],
                "tile_count": sample["tile_count"],
                "crop_xyxy": sample["crop_xyxy"],
                "text_area": sample["text_area"],
            }
        record = self.records[index]
        if self.synthetic:
            rng = np.random.default_rng(20260815 + index)
            gt_rgb = np.zeros((h, w, 3), np.float32)
            gt_rgb[..., 0] = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :]
            gt_rgb[..., 1] = np.linspace(0.1, 0.8, h, dtype=np.float32)[:, None]
            gt_rgb[..., 2] = 0.25
            lq_rgb = np.clip(gt_rgb + rng.normal(0.0, 0.025, gt_rgb.shape).astype(np.float32), 0.0, 1.0)
            valid = np.zeros((h, w), np.float32); valid[: h - 8, : w - 16] = 1.0
            text = np.zeros((h, w), np.float32); text[h // 3:h // 2, 24:w - 32] = 1.0
        else:
            lq_rgb = _load_image(record["lq"])
            gt_rgb = _load_image(record["gt"])
            source_hw = lq_rgb.shape[:2]
            valid = _mask_canvas(record.get("valid_mask"), source_hw, self.canvas_hw)
            text = _mask_canvas(record.get("text_mask"), source_hw, self.canvas_hw)
        return {
            "lq": torch.from_numpy(_rgb_to_nv21(lq_rgb, self.canvas_hw)[None]),
            "gt": torch.from_numpy(_rgb_to_yuv444(gt_rgb, self.canvas_hw)),
            "lq_rgb": torch.from_numpy(np.transpose(_prepare_rgb(lq_rgb, self.canvas_hw), (2, 0, 1)).copy()),
            "gt_rgb": torch.from_numpy(np.transpose(_prepare_rgb(gt_rgb, self.canvas_hw), (2, 0, 1)).copy()),
            "valid_mask": torch.from_numpy(valid[None]),
            "text_mask": torch.from_numpy(text[None]),
            "sample_id": str(record.get("stem", record.get("dataset", index))),
        }


def _dump_tensor_image(path: Path, tensor: torch.Tensor, mode: str) -> None:
    """@shanlan-comment Dump RGB/YUV/mask tensor data as PNG evidence."""
    array = tensor.detach().cpu().float().numpy()
    if mode == "rgb":
        image = np.clip(array.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif mode == "yuv444":
        yuv = np.clip(array.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    elif mode == "nv21":
        packed = array.squeeze()
        height = packed.shape[0] * 2 // 3
        y = np.clip(packed[:height] * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    else:
        image = np.clip(array.squeeze() * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), image)


def _select_geometry_records(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """@shanlan-comment Select real records covering horizontal, long, and vertical OCR geometry."""
    selected: List[Dict[str, Any]] = []
    buckets: Dict[str, Dict[str, Any]] = {}
    for record in records:
        lines = _ocr_line_polygons(record)
        for line in lines:
            orientation, _ = _line_orientation_and_axis(line["polygon"])
            x0, y0, x1, y1 = line["bbox"]
            width, height = x1 - x0, y1 - y0
            if orientation == "VERTICAL":
                key = "VERTICAL"
            elif orientation == "HORIZONTAL" and width > CANVAS_HW[1]:
                key = "LONG_HORIZONTAL"
            else:
                key = "HORIZONTAL"
            buckets.setdefault(key, record)
    for key in ("HORIZONTAL", "LONG_HORIZONTAL", "VERTICAL"):
        if key in buckets:
            selected.append(buckets[key])
    if len(selected) < limit:
        seen = {record.get("stem") for record in selected}
        for record in records:
            if record.get("stem") not in seen:
                selected.append(record)
                seen.add(record.get("stem"))
            if len(selected) >= limit:
                break
    return selected[:limit]


def run_smoke(records: List[Dict[str, Any]], synthetic: bool, output_dir: Path, steps: int):
    """@shanlan-comment Run bounded forward/backward smoke training and dump evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = output_dir / "evidence"
    evidence.mkdir(exist_ok=True)
    dataset_records = records[: max(steps, 2)] if synthetic else _select_geometry_records(records, max(steps, 3))
    dataset = IQ560SmokeDataset(dataset_records, synthetic=synthetic)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = BasicTextLineRestoreNetV2(CANVAS_HW, features=8).cpu()
    criterion = MaskedL1LossV2Basic(require_valid_mask=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses: List[float] = []
    gradient_rows: List[Dict[str, Any]] = []
    first = next(iter(loader))
    _dump_tensor_image(evidence / "raw_lq_rgb.png", first["lq_rgb"][0], "rgb")
    _dump_tensor_image(evidence / "raw_gt_rgb.png", first["gt_rgb"][0], "rgb")
    _dump_tensor_image(evidence / "model_input_nv21_y.png", first["lq"][0], "nv21")
    _dump_tensor_image(evidence / "valid_mask.png", first["valid_mask"][0], "mask")
    _dump_tensor_image(evidence / "text_mask.png", first["text_mask"][0], "mask")
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["lq"])
        loss = criterion(output, batch["gt"], valid_mask=batch["valid_mask"], text_mask=batch["text_mask"])
        loss.backward()
        grad_values = [parameter.grad.detach().abs().max().item() for parameter in model.parameters() if parameter.grad is not None]
        finite_grad = all(math.isfinite(value) for value in grad_values)
        if not finite_grad or max(grad_values, default=0.0) <= 0.0:
            raise RuntimeError(f"gradient check failed at step {step}: {grad_values}")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        gradient_rows.append({"step": step, "loss": losses[-1], "max_grad": max(grad_values), "finite": finite_grad})
        if step == 0:
            _dump_tensor_image(evidence / "model_output_yuv444.png", output[0], "yuv444")
            _dump_tensor_image(evidence / "model_target_yuv444.png", batch["gt"][0], "yuv444")
    (output_dir / "gradient_trace.json").write_text(json.dumps(gradient_rows, indent=2), encoding="utf-8")
    return {"synthetic": synthetic, "steps": len(losses), "losses": losses, "gradient_trace": gradient_rows,
            "evidence_dir": str(evidence), "dataset_samples": len(dataset),
            "nonempty_text_samples": sum(float(item.get("text_area", 0.0)) > 0 for item in dataset.line_samples) if not synthetic else len(dataset),
            "geometry_samples": [{"sample_id": item.get("sample_id"), "orientation": item.get("orientation"),
                                  "tile_index": item.get("tile_index"), "tile_count": item.get("tile_count"),
                                  "crop_xyxy": item.get("crop_xyxy"), "text_area": item.get("text_area")}
                                 for item in dataset.line_samples[: max(steps, 3)]] if not synthetic else []}


def main(argv=None) -> int:
    """@shanlan-comment Inspect IQ560 manifests, run real-or-fallback smoke training, and write JSON evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    inspection = inspect_manifests(args.manifest_root)
    records: List[Dict[str, Any]] = []
    if inspection["available"] and inspection["splits"].get("train", {}).get("exists"):
        with (args.manifest_root / "train.jsonl").open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()][: max(args.steps, 2)]
        synthetic = False
    else:
        records = [{"dataset": "synthetic_schema_fixture", "stem": f"fixture_{i}"} for i in range(max(args.steps, 2))]
        synthetic = True
    result = run_smoke(records, synthetic, args.output_dir, args.steps)
    payload = {"inspection": inspection, "smoke": result, "canvas_hw": list(CANVAS_HW), "status": "passed"}
    (args.output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
