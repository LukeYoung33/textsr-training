"""Remote58 full-frame JSONL adapter for the fixed BASIC V2 contract.

Remote58's ``real_finetune`` manifests describe one text line per RGB frame,
not one pre-expanded crop per record.  This adapter resolves the paired trees,
fits each frame into an even fixed canvas while preserving aspect ratio, and
emits the same packed-NV21/YUV444 and mask fields as ``BasicTextLineDatasetV2``.
It never writes to the source tree.
"""

from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from basicsr.utils.registry import DATASET_REGISTRY


DATASET_VERSION = "basic_v2_remote58_full_frame_v1"


def _allocate_quotas(lengths: List[int], weights: List[float], epoch_size: int) -> List[int]:
    """@luna-comment Allocate exact integer source quotas by largest remainder."""
    if not lengths or len(lengths) != len(weights):
        raise ValueError("source lengths and weights must be non-empty and aligned")
    if epoch_size < len(lengths):
        raise ValueError("epoch_size must allocate at least one record per source")
    total_weight = float(sum(weights))
    raw = [float(epoch_size) * weight / total_weight for weight in weights]
    quotas = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda item: (-(raw[item] - quotas[item]), item))[: epoch_size - sum(quotas)]:
        quotas[index] += 1
    if any(quota <= 0 for quota in quotas):
        raise ValueError("epoch_size and weights leave a source with zero quota")
    return quotas


def _interleave_source_entries(source_records: List[List[Dict[str, Any]]], quotas: List[int]) -> List[Dict[str, Any]]:
    """@luna-comment Build a deterministic round-robin schedule with exact quotas."""
    entries: List[Dict[str, Any]] = []
    for round_index in range(max(quotas)):
        for source_index, quota in enumerate(quotas):
            if round_index < quota:
                records = source_records[source_index]
                entries.append({"source_index": source_index, "record": records[round_index % len(records)]})
    return entries


def _read_rgb(path: Path) -> np.ndarray:
    """@luna-comment Decode one source image into contiguous RGB float data."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _even_scaled_hw(source_hw: Tuple[int, int], canvas_hw: Tuple[int, int], max_upscale: float) -> Tuple[int, int, float]:
    """@luna-comment Compute an aspect-preserving even size inside the canvas."""
    source_h, source_w = (int(source_hw[0]), int(source_hw[1]))
    canvas_h, canvas_w = (int(canvas_hw[0]), int(canvas_hw[1]))
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"source image shape must be positive, got {source_hw}")
    scale = min(canvas_h / float(source_h), canvas_w / float(source_w), float(max_upscale))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError(f"invalid resize scale {scale}")
    height = max(2, int(round(source_h * scale)))
    width = max(2, int(round(source_w * scale)))
    # The packed NV21 path requires even dimensions.  Rounding up is safe
    # because the scale was selected against the fixed canvas.
    if height % 2:
        height += 1
    if width % 2:
        width += 1
    if height > canvas_h or width > canvas_w:
        scale = min((canvas_h - canvas_h % 2) / float(source_h),
                    (canvas_w - canvas_w % 2) / float(source_w),
                    float(max_upscale))
        height = max(2, int(round(source_h * scale)) // 2 * 2)
        width = max(2, int(round(source_w * scale)) // 2 * 2)
    if height > canvas_h or width > canvas_w:
        raise ValueError(f"cannot fit source {source_hw} into canvas {canvas_hw}")
    return height, width, float(scale)


def _resize_and_pad(image: np.ndarray, canvas_hw: Tuple[int, int], max_upscale: float, interpolation: int) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """@luna-comment Resize an image into the top-left of a zero canvas."""
    target_h, target_w, scale = _even_scaled_hw(image.shape[:2], canvas_hw, max_upscale)
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    output = np.zeros((int(canvas_hw[0]), int(canvas_hw[1]), image.shape[2]), dtype=np.float32)
    output[:target_h, :target_w] = resized.astype(np.float32, copy=False)
    return output, (target_h, target_w), scale


def _resize_stretch(image: np.ndarray, canvas_hw: Tuple[int, int], interpolation: int) -> Tuple[np.ndarray, Tuple[int, int], List[float]]:
    """@luna-comment Directly resize an image to the full canvas without padding."""
    canvas_h, canvas_w = (int(canvas_hw[0]), int(canvas_hw[1]))
    source_h, source_w = (int(image.shape[0]), int(image.shape[1]))
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"source image shape must be positive, got {image.shape[:2]}")
    resized = cv2.resize(image, (canvas_w, canvas_h), interpolation=interpolation)
    scale = [canvas_h / float(source_h), canvas_w / float(source_w)]
    return np.ascontiguousarray(resized, dtype=np.float32), (canvas_h, canvas_w), scale


def _height_crop_pad_geometry(source_hw: Tuple[int, int], canvas_hw: Tuple[int, int], random_crop: bool) -> Tuple[int, int, float]:
    """Return the shared preserve-ratio height-resize geometry for paired assets."""
    source_h, source_w = (int(source_hw[0]), int(source_hw[1]))
    canvas_h, canvas_w = (int(canvas_hw[0]), int(canvas_hw[1]))
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"source image shape must be positive, got {source_hw}")
    scale = canvas_h / float(source_h)
    resized_w = max(2, int(round(source_w * scale)))
    if resized_w % 2:
        resized_w += 1
    if resized_w <= canvas_w:
        return resized_w, 0, scale
    maximum_left = resized_w - canvas_w
    left = int(np.random.randint(maximum_left + 1)) if random_crop else maximum_left // 2
    return resized_w, left, scale


def _resize_height_crop_pad(image: np.ndarray, canvas_hw: Tuple[int, int], geometry: Tuple[int, int, float], interpolation: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize to canvas height, then crop or right-pad width with shared geometry."""
    canvas_h, canvas_w = (int(canvas_hw[0]), int(canvas_hw[1]))
    resized_w, left, _ = geometry
    resized = cv2.resize(image, (resized_w, canvas_h), interpolation=interpolation)
    output_shape = (canvas_h, canvas_w) + (() if image.ndim == 2 else (image.shape[2],))
    output = np.zeros(output_shape, dtype=resized.dtype)
    if resized_w <= canvas_w:
        output[:, :resized_w] = resized
        return output, (canvas_h, resized_w)
    output[:] = resized[:, left:left + canvas_w]
    return output, (canvas_h, canvas_w)


def _text_line_crop_box(mask: np.ndarray, threshold: float, line_gap: int, select_random: bool,
                        vertical_context: float, horizontal_context: float, boundary_jitter: float) -> Optional[Tuple[int, int, int, int]]:
    """Select one text-row band from a full-frame text mask and expand it safely."""
    foreground = mask > threshold
    row_mass = foreground.sum(axis=1)
    active_rows = np.flatnonzero(row_mass > 0)
    if active_rows.size == 0:
        return None
    bands: List[Tuple[int, int]] = []
    start = previous = int(active_rows[0])
    for row in active_rows[1:]:
        row = int(row)
        if row - previous > line_gap:
            bands.append((start, previous + 1))
            start = row
        previous = row
    bands.append((start, previous + 1))
    # Tiny mask fragments are normally punctuation or annotation noise.  Keep
    # a fallback so that sparse, legitimate one-pixel text is still usable.
    substantive = [band for band in bands if int(row_mass[band[0]:band[1]].sum()) >= 8]
    bands = substantive or bands
    if select_random:
        top, bottom = bands[int(np.random.randint(len(bands)))]
    else:
        top, bottom = max(bands, key=lambda band: int(row_mass[band[0]:band[1]].sum()))
    cols = np.flatnonzero(foreground[top:bottom].any(axis=0))
    if cols.size == 0:
        return None
    left, right = int(cols[0]), int(cols[-1]) + 1
    line_h, line_w = bottom - top, right - left
    pad_y = max(2, int(round(line_h * vertical_context)))
    pad_x = max(2, int(round(line_w * horizontal_context)))
    if boundary_jitter > 0.0:
        jitter_y, jitter_x = int(round(line_h * boundary_jitter)), int(round(line_w * boundary_jitter))
        top += int(np.random.randint(-jitter_y, jitter_y + 1)) if jitter_y else 0
        bottom += int(np.random.randint(-jitter_y, jitter_y + 1)) if jitter_y else 0
        left += int(np.random.randint(-jitter_x, jitter_x + 1)) if jitter_x else 0
        right += int(np.random.randint(-jitter_x, jitter_x + 1)) if jitter_x else 0
    height, width = mask.shape[:2]
    top, bottom = max(0, top - pad_y), min(height, bottom + pad_y)
    left, right = max(0, left - pad_x), min(width, right + pad_x)
    if bottom - top < 2 or right - left < 2:
        return None
    return top, bottom, left, right


def _ocr_annotation_crop(annotation: Dict[str, Any], image_hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Validate and clip one crop-local OCR annotation bounding box."""
    bbox = annotation.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([left, top, right, bottom]).all():
        return None
    height, width = image_hw
    left, right = max(0, int(np.floor(left))), min(width, int(np.ceil(right)))
    top, bottom = max(0, int(np.floor(top))), min(height, int(np.ceil(bottom)))
    return (left, top, right, bottom) if right - left >= 2 and bottom - top >= 2 else None


def _normalized_ocr_annotations(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return legacy annotations or normalized Mojito OCR lines as one contract."""
    annotations = document.get("ocr", {}).get("annotations", [])
    if isinstance(annotations, list) and annotations:
        return annotations
    lines = document.get("lines", [])
    if not isinstance(lines, list):
        return []
    result: List[Dict[str, Any]] = []
    for ordinal, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        bbox = line.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        result.append({
            "bbox": bbox,
            "instance_id": str(line.get("line_id", ordinal)),
            "text": line.get("text", ""),
            "visible": True,
            "ignore": bool(line.get("ignore", False)),
        })
    return result


def _expand_ocr_line_records(records: List[Dict[str, Any]], root: Path, split: str,
                             resolve_path) -> List[Dict[str, Any]]:
    """Expand full frames, or reuse the OCR crop of an already materialized line."""
    expanded: List[Dict[str, Any]] = []
    for record in records:
        if "ocr_crop_xyxy" in record and "line_envelope_xyxy_in_parent" in record:
            # The matrix manifests already contain one aligned line per row.
            # Re-expanding its OCR JSON would decode every line during dataset
            # construction and could duplicate a line with multiple annotations.
            bbox = record["ocr_crop_xyxy"]
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                    or float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1])):
                raise ValueError(f"invalid materialized OCR crop: {record.get('sample_id')}")
            expanded.append(dict(record))
            continue
        ocr_path = resolve_path(record, "ocr", "ocr", root, split)
        if not ocr_path.is_file():
            raise FileNotFoundError(f"missing OCR annotation for {record['image']}: {ocr_path}")
        document = json.loads(ocr_path.read_text(encoding="utf-8"))
        annotations = _normalized_ocr_annotations(document)
        lq_path = resolve_path(record, "lq", "lq", root, split)
        image = cv2.imread(str(lq_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(str(lq_path))
        for ordinal, annotation in enumerate(annotations):
            if not isinstance(annotation, dict) or annotation.get("ignore") or annotation.get("visible") is False:
                continue
            bbox = _ocr_annotation_crop(annotation, image.shape[:2])
            if bbox is None:
                continue
            instance_id = str(annotation.get("instance_id") or f"line_{ordinal:03d}")
            line_record = dict(record)
            line_record["image"] = f"{record['image']}::{instance_id}"
            line_record["ocr_annotation"] = annotation
            line_record["ocr_crop_xyxy"] = bbox
            expanded.append(line_record)
    if not expanded:
        raise ValueError("OCR line expansion produced no visible, in-bounds annotations")
    return expanded


def _rgb_to_nv21(image: np.ndarray) -> torch.Tensor:
    """@luna-comment Convert an even RGB canvas to normalized packed NV21."""
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"NV21 requires even canvas dimensions, got {(height, width)}")
    rgb_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    yuv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YUV)
    u = cv2.resize(yuv[..., 1], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(yuv[..., 2], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    vu = np.empty((height // 2, width), dtype=np.uint8)
    vu[:, 0::2], vu[:, 1::2] = v, u
    packed = np.concatenate((yuv[..., 0], vu), axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(packed[None])


def _rgb_to_yuv444(image: np.ndarray) -> torch.Tensor:
    """@luna-comment Convert an RGB canvas to planar normalized YUV444."""
    rgb_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    yuv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
    return torch.from_numpy(yuv).permute(2, 0, 1)


@DATASET_REGISTRY.register()
class BasicTextLineRemote58Dataset(Dataset):
    """Read Remote58 full-frame records and emit one fixed BASIC V2 sample."""

    def __init__(self, opt: Dict[str, Any]):
        """@luna-comment Resolve one or more manifests and freeze exact epoch quotas."""
        self.opt = dict(opt)
        self.split = str(self.opt.get("split") or ("val" if self.opt.get("phase") != "train" else "train"))
        self.ocr_line_crop = bool(self.opt.get("ocr_line_crop", False))
        source_options = self.opt.get("sources")
        if source_options is None:
            source_options = [{
                "root": self.opt.get("root") or self.opt.get("dataset_root"),
                "manifest": self.opt.get("manifest") or self.opt.get("manifest_jsonl"),
                "weight": 1.0,
            }]
        if not isinstance(source_options, list) or not source_options:
            raise ValueError("sources must be a non-empty list")
        self.source_specs: List[Dict[str, Any]] = []
        source_records: List[List[Dict[str, Any]]] = []
        weights: List[float] = []
        for source_index, source_option in enumerate(source_options):
            if not isinstance(source_option, dict):
                raise ValueError("each sources entry must be a mapping")
            root_value = source_option.get("root") or source_option.get("dataset_root")
            manifest_value = source_option.get("manifest") or source_option.get("manifest_jsonl")
            root = Path(root_value) if root_value else None
            manifest = Path(manifest_value) if manifest_value else (root / f"{source_option.get('split', self.split)}.jsonl" if root else None)
            if manifest is None:
                raise ValueError("each source requires root or manifest")
            if not manifest.is_absolute() and root is not None:
                manifest = root / manifest
            if not manifest.is_file():
                raise FileNotFoundError(str(manifest))
            source_root = root if root is not None else manifest.parent
            source_split = str(source_option.get("split") or self.split)
            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            # The 20260920 valid-crop manifest stores assets under ``files``.
            # Normalize it once into the established paired-record contract.
            for record in records:
                files = record.get("files")
                if isinstance(files, dict):
                    record.setdefault("image", record.get("sample_id") or Path(str(files.get("gt", ""))).name)
                    for key in ("lq", "gt", "ocr", "text_mask", "valid_mask"):
                        if files.get(key): record.setdefault(key, files[key])
            if self.ocr_line_crop:
                records = _expand_ocr_line_records(records, source_root, source_split, self._path)
            if not records:
                raise ValueError(f"empty Remote58 manifest: {manifest}")
            names = [str(record.get("image", "")) for record in records]
            if any(not name for name in names):
                raise ValueError("every Remote58 record requires image")
            if len(names) != len(set(names)):
                raise ValueError(f"Remote58 manifest contains duplicate image records: {manifest}")
            for record in records:
                lq_path = self._path(record, "lq", "lq", source_root, source_split)
                gt_path = self._path(record, "gt", "gt", source_root, source_split)
                if not lq_path.is_file() or not gt_path.is_file():
                    missing = [str(path) for path in (lq_path, gt_path) if not path.is_file()]
                    raise FileNotFoundError(f"missing paired asset(s) for {record['image']}: {missing}")
            source_name = str(source_option.get("name") or source_root.name or f"source_{source_index}")
            self.source_specs.append({
                "name": source_name,
                "root": source_root,
                "split": source_split,
                "manifest": manifest,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "records": records,
            })
            source_records.append(records)
            weight = float(source_option.get("weight", 1.0))
            if weight <= 0.0 or not np.isfinite(weight):
                raise ValueError("source weights must be finite and positive")
            weights.append(weight)
        lengths = [len(records) for records in source_records]
        requested_epoch_size = self.opt.get("epoch_size")
        epoch_size = int(requested_epoch_size) if requested_epoch_size is not None else (max(lengths) * len(lengths) if len(lengths) > 1 else lengths[0])
        self.source_quotas = _allocate_quotas(lengths, weights, epoch_size)
        for source_index, (source, quota) in enumerate(zip(self.source_specs, self.source_quotas)):
            source["weight"] = weights[source_index]
            source["epoch_quota"] = quota
        self._entries = _interleave_source_entries(source_records, self.source_quotas)
        for entry in self._entries:
            source = self.source_specs[entry["source_index"]]
            entry.update({"root": source["root"], "split": source["split"], "name": source["name"], "manifest": source["manifest"]})
        self.records = [entry["record"] for entry in self._entries]
        self.root = self.source_specs[0]["root"] if len(self.source_specs) == 1 else None
        self.manifest = self.source_specs[0]["manifest"] if len(self.source_specs) == 1 else None
        self.manifest_sha256 = self.source_specs[0]["manifest_sha256"] if len(self.source_specs) == 1 else {source["name"]: source["manifest_sha256"] for source in self.source_specs}
        self._resolved_pairs = [(self._path(entry["record"], "lq", "lq", entry["root"], entry["split"]), self._path(entry["record"], "gt", "gt", entry["root"], entry["split"])) for entry in self._entries]
        canvas = self.opt.get("canvas_hw", [96, 320])
        if len(canvas) != 2 or any(int(value) <= 0 or int(value) % 2 for value in canvas):
            raise ValueError("canvas_hw must contain positive even dimensions")
        self.canvas_hw = (int(canvas[0]), int(canvas[1]))
        self.max_upscale = float(self.opt.get("max_upscale", 1.0))
        if self.max_upscale <= 0.0 or not np.isfinite(self.max_upscale):
            raise ValueError("max_upscale must be finite and positive")
        self.resize_interpolation = int(self.opt.get("resize_interpolation", cv2.INTER_AREA))
        self.resize_mode = str(self.opt.get("resize_mode", "aspect_fit")).strip().lower()
        if self.resize_mode not in ("aspect_fit", "stretch", "height_crop_pad"):
            raise ValueError("resize_mode must be 'aspect_fit', 'stretch', or 'height_crop_pad'")
        self.structure_threshold = float(self.opt.get("structure_threshold", 0.0))
        self.text_line_crop = bool(self.opt.get("text_line_crop", False))
        self.text_line_gap = max(0, int(self.opt.get("text_line_gap", 6)))
        self.text_line_vertical_context = max(0.0, float(self.opt.get("text_line_vertical_context", 0.12)))
        self.text_line_horizontal_context = max(0.0, float(self.opt.get("text_line_horizontal_context", 0.03)))
        self.text_line_boundary_jitter = max(0.0, float(self.opt.get("text_line_boundary_jitter", 0.05)))
        self.homography_prob = float(self.opt.get("train_homography_prob", 0.0)) if self.split == "train" else 0.0
        self.homography_jitter = float(self.opt.get("train_homography_jitter", 0.0))
        self.decoded_source_cache_size = max(0, int(self.opt.get("decoded_source_cache_size", 0)))
        self.source_grouped_sampling = bool(self.opt.get("source_grouped_sampling", False))
        self._decoded_source_cache: OrderedDict[Tuple[str, str, str, str], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = OrderedDict()
        cache_root = self.opt.get("ocr_line_crop_cache_root")
        self.ocr_line_crop_cache_root = Path(str(cache_root)) if cache_root else None
        if self.ocr_line_crop_cache_root is not None and not self.ocr_line_crop:
            raise ValueError("ocr_line_crop_cache_root requires ocr_line_crop=true")
        self.dataset_version = DATASET_VERSION

    def __len__(self) -> int:
        return len(self._entries)

    def _path(self, record: Dict[str, Any], key: str, tree: str, root: Optional[Path] = None, split: Optional[str] = None) -> Path:
        """@luna-comment Resolve explicit or split-relative image paths."""
        root = root or self.root
        split = split or self.split
        if root is None:
            raise ValueError("source root is required to resolve a relative image path")
        value = record.get(key)
        if value:
            path = Path(str(value))
            if path.is_absolute():
                return path
            # A basename follows the Remote58 tree convention.  A path with
            # directories is already root-relative (for example lq/train/x).
            return root / tree / split / path.name if len(path.parts) == 1 else root / path
        return root / tree / split / str(record["image"])

    def _structure_path(self, record: Dict[str, Any], root: Path, split: str) -> Optional[Path]:
        """@luna-comment Resolve a structure basename or explicit root-relative path."""
        value = record.get("structure")
        if not value:
            return root / "structure" / split / str(record["image"])
        path = Path(str(value))
        if path.is_absolute():
            return path
        return root / "structure" / split / path.name if len(path.parts) == 1 else root / path

    def grouped_sampling_indices(self) -> List[List[int]]:
        """Group expanded OCR-line indices by source image for optional cache locality."""
        if not self.source_grouped_sampling:
            return []
        grouped: OrderedDict[str, List[int]] = OrderedDict()
        for index, entry in enumerate(self._entries):
            record = entry["record"]
            key = str(self._path(record, "lq", "lq", entry["root"], entry["split"]))
            grouped.setdefault(key, []).append(index)
        return list(grouped.values())

    def _load_source_arrays(self, lq_path: Path, gt_path: Path, text_path: Path,
                            valid_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Decode a source frame once per worker and retain a bounded LRU cache."""
        key = (str(lq_path), str(gt_path), str(text_path), str(valid_path))
        cached = self._decoded_source_cache.pop(key, None) if self.decoded_source_cache_size else None
        if cached is not None:
            self._decoded_source_cache[key] = cached
            return cached
        lq_source, gt_source = _read_rgb(lq_path), _read_rgb(gt_path)
        text_source = cv2.imread(str(text_path), cv2.IMREAD_GRAYSCALE) if text_path.is_file() else None
        valid_source = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE) if valid_path.is_file() else None
        result = (lq_source, gt_source, text_source, valid_source)
        if self.decoded_source_cache_size:
            self._decoded_source_cache[key] = result
            while len(self._decoded_source_cache) > self.decoded_source_cache_size:
                self._decoded_source_cache.popitem(last=False)
        return result

    def _load_cached_line_envelope(self, index: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int, int, int], Tuple[int, int]]:
        """Load one lossless OCR-line envelope prepared for this dataset index."""
        if self.ocr_line_crop_cache_root is None:
            raise RuntimeError("OCR line crop cache is not configured")
        path = self.ocr_line_crop_cache_root / "lines" / f"{index:06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing OCR line crop cache entry: {path}")
        with np.load(path, allow_pickle=False) as archive:
            envelope = tuple(int(value) for value in archive["envelope_xyxy"].tolist())
            full_hw = tuple(int(value) for value in archive["full_hw"].tolist())
            text = archive["text"] if bool(archive["has_text"][0]) else None
            valid = archive["valid"] if bool(archive["has_valid"][0]) else None
            lq = archive["lq"].astype(np.float32) / 255.0
            gt = archive["gt"].astype(np.float32) / 255.0
        return lq, gt, text, valid, envelope, full_hw

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """@luna-comment
        Input: One indexed manifest entry with aligned LQ/GT, optional masks, and OCR geometry.
        Output: Canvas-sized training tensors and metadata without changing source assets.
        Key Steps: Load source or cached line, jitter the OCR crop within context, optionally warp the pair, then resize and pack it.
        Key Params: Crop context, bounded edge jitter, homography probability/scale, and canvas geometry come from dataset options.
        State/Dependencies: Reads image assets and uses random draws for training augmentation; validation paths are deterministic.
        """
        entry = self._entries[index]
        record = entry["record"]
        lq_path = self._path(record, "lq", "lq", entry["root"], entry["split"])
        gt_path = self._path(record, "gt", "gt", entry["root"], entry["split"])
        text_path = self._path(record, "text_mask", "text_mask", entry["root"], entry["split"])
        valid_path = self._path(record, "valid_mask", "valid_mask", entry["root"], entry["split"])
        cache_envelope = None
        if self.ocr_line_crop_cache_root is None:
            lq_source, gt_source, text_source, valid_source = self._load_source_arrays(lq_path, gt_path, text_path, valid_path)
            full_hw = lq_source.shape[:2]
        else:
            lq_source, gt_source, text_source, valid_source, cache_envelope, full_hw = self._load_cached_line_envelope(index)
        if lq_source.shape != gt_source.shape:
            raise ValueError(f"LQ/GT shape mismatch for {record['image']}: {lq_source.shape} vs {gt_source.shape}")
        if text_source is not None and text_source.shape != lq_source.shape[:2]: raise ValueError("text mask shape mismatch")
        if valid_source is not None and valid_source.shape != lq_source.shape[:2]: raise ValueError("valid mask shape mismatch")
        source_crop = (0, 0, full_hw[1], full_hw[0])
        if self.ocr_line_crop:
            ocr_crop = record.get("ocr_crop_xyxy")
            if not isinstance(ocr_crop, (tuple, list)) or len(ocr_crop) != 4:
                raise ValueError(f"OCR line record is missing a valid crop for {record['image']}")
            ocr_left, ocr_top, ocr_right, ocr_bottom = (int(value) for value in ocr_crop)
            left, top, right, bottom = ocr_left, ocr_top, ocr_right, ocr_bottom
            line_h, line_w = bottom - top, right - left
            pad_y = max(2, int(round(line_h * self.text_line_vertical_context)))
            pad_x = max(2, int(round(line_w * self.text_line_horizontal_context)))
            if self.split == "train" and self.text_line_boundary_jitter > 0.0:
                jitter_y, jitter_x = int(round(line_h * self.text_line_boundary_jitter)), int(round(line_w * self.text_line_boundary_jitter))
                top = ocr_top - pad_y + (int(np.random.randint(-jitter_y, jitter_y + 1)) if jitter_y else 0)
                bottom = ocr_bottom + pad_y + (int(np.random.randint(-jitter_y, jitter_y + 1)) if jitter_y else 0)
                left = ocr_left - pad_x + (int(np.random.randint(-jitter_x, jitter_x + 1)) if jitter_x else 0)
                right = ocr_right + pad_x + (int(np.random.randint(-jitter_x, jitter_x + 1)) if jitter_x else 0)
            else:
                top, bottom = ocr_top - pad_y, ocr_bottom + pad_y
                left, right = ocr_left - pad_x, ocr_right + pad_x
            height, width = full_hw
            # Jitter may vary the extension but never crops into the OCR box
            # or extends beyond the configured context envelope.
            top = min(ocr_top, max(ocr_top - pad_y, top))
            bottom = max(ocr_bottom, min(ocr_bottom + pad_y, bottom))
            left = min(ocr_left, max(ocr_left - pad_x, left))
            right = max(ocr_right, min(ocr_right + pad_x, right))
            top, bottom = max(0, top), min(height, bottom)
            left, right = max(0, left), min(width, right)
            if bottom - top < 2 or right - left < 2:
                raise ValueError(f"OCR crop collapsed for {record['image']}")
            source_crop = (left, top, right, bottom)
            if cache_envelope is not None:
                cache_left, cache_top, cache_right, cache_bottom = cache_envelope
                if left < cache_left or top < cache_top or right > cache_right or bottom > cache_bottom:
                    raise ValueError(f"cached OCR envelope does not contain requested crop for {record['image']}")
                left, top, right, bottom = left - cache_left, top - cache_top, right - cache_left, bottom - cache_top
            lq_source = lq_source[top:bottom, left:right]
            gt_source = gt_source[top:bottom, left:right]
            text_source = text_source[top:bottom, left:right] if text_source is not None else None
            valid_source = valid_source[top:bottom, left:right] if valid_source is not None else None
        elif self.text_line_crop and text_source is not None:
            crop = _text_line_crop_box(
                text_source, self.structure_threshold, self.text_line_gap,
                self.split == "train", self.text_line_vertical_context,
                self.text_line_horizontal_context,
                self.text_line_boundary_jitter if self.split == "train" else 0.0,
            )
            if crop is not None:
                top, bottom, left, right = crop
                source_crop = (left, top, right, bottom)
                lq_source = lq_source[top:bottom, left:right]
                gt_source = gt_source[top:bottom, left:right]
                text_source = text_source[top:bottom, left:right]
                if valid_source is not None:
                    valid_source = valid_source[top:bottom, left:right]
        if self.homography_prob > 0 and np.random.random() < self.homography_prob:
            h, w = lq_source.shape[:2]; j = self.homography_jitter * min(h, w)
            src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            dst = src + np.random.uniform(-j, j, src.shape).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            lq_source = cv2.warpPerspective(lq_source, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            gt_source = cv2.warpPerspective(gt_source, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if text_source is not None: text_source = cv2.warpPerspective(text_source, matrix, (w,h), flags=cv2.INTER_NEAREST)
            if valid_source is not None: valid_source = cv2.warpPerspective(valid_source, matrix, (w,h), flags=cv2.INTER_NEAREST)
        transformed_text = None
        transformed_valid = None
        width_truncated = False
        if self.resize_mode == "stretch":
            lq, valid_hw, scale = _resize_stretch(lq_source, self.canvas_hw, self.resize_interpolation)
            gt, gt_valid_hw, _ = _resize_stretch(gt_source, self.canvas_hw, self.resize_interpolation)
        elif self.resize_mode == "height_crop_pad":
            geometry = _height_crop_pad_geometry(lq_source.shape[:2], self.canvas_hw, self.split == "train")
            lq, valid_hw = _resize_height_crop_pad(lq_source, self.canvas_hw, geometry, self.resize_interpolation)
            gt, gt_valid_hw = _resize_height_crop_pad(gt_source, self.canvas_hw, geometry, self.resize_interpolation)
            scale = geometry[2]
            width_truncated = geometry[0] > self.canvas_hw[1]
            if text_source is not None:
                transformed_text, _ = _resize_height_crop_pad(text_source, self.canvas_hw, geometry, cv2.INTER_NEAREST)
            if valid_source is not None:
                transformed_valid, _ = _resize_height_crop_pad(valid_source, self.canvas_hw, geometry, cv2.INTER_NEAREST)
        else:
            lq, valid_hw, scale = _resize_and_pad(lq_source, self.canvas_hw, self.max_upscale, self.resize_interpolation)
            gt, gt_valid_hw, _ = _resize_and_pad(gt_source, self.canvas_hw, self.max_upscale, self.resize_interpolation)
        if valid_hw != gt_valid_hw:
            raise ValueError(f"resized LQ/GT shape mismatch for {record['image']}")
        valid = np.zeros(self.canvas_hw, dtype=np.float32)
        valid[:valid_hw[0], :valid_hw[1]] = 1.0
        if transformed_valid is not None:
            valid *= (transformed_valid > 0).astype(np.float32)
        elif valid_source is not None:
            vm = cv2.resize(valid_source, (valid_hw[1], valid_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
            valid[:valid_hw[0], :valid_hw[1]] *= vm.astype(np.float32)
        structure_path = self._structure_path(record, entry["root"], entry["split"])
        if structure_path is not None and not structure_path.is_file():
            if record.get("structure"):
                raise FileNotFoundError(f"missing structure asset for {record['image']}: {structure_path}")
            structure_path = None
        if transformed_text is not None:
            text = (transformed_text > self.structure_threshold).astype(np.float32)
        elif text_source is not None:
            structure = cv2.resize(text_source, (valid_hw[1], valid_hw[0]), interpolation=cv2.INTER_NEAREST)
            text = np.zeros(self.canvas_hw, dtype=np.float32); text[:valid_hw[0], :valid_hw[1]] = (structure > self.structure_threshold).astype(np.float32)
        elif structure_path is not None and structure_path.is_file():
            structure = cv2.imread(str(structure_path), cv2.IMREAD_GRAYSCALE)
            if structure is None or structure.shape != lq_source.shape[:2]:
                raise ValueError(f"structure shape mismatch for {record['image']}")
            if self.resize_mode == "stretch":
                structure = cv2.resize(structure, (self.canvas_hw[1], self.canvas_hw[0]), interpolation=cv2.INTER_NEAREST)
                text = (structure > self.structure_threshold).astype(np.float32)
            else:
                structure = cv2.resize(structure, (valid_hw[1], valid_hw[0]), interpolation=cv2.INTER_NEAREST)
                text = np.zeros(self.canvas_hw, dtype=np.float32)
                text[:valid_hw[0], :valid_hw[1]] = (structure > self.structure_threshold).astype(np.float32)
        else:
            text = valid.copy()
        text *= valid
        # A manifest-provided text mask is a verified structure annotation just
        # as much as a legacy ``structure`` asset.  The 20260920 valid-crop
        # manifests use the former, so checking only ``structure_path`` would
        # incorrectly reject every batch once structure losses are enabled.
        structure_available = text_source is not None or structure_path is not None
        cp_eligible = bool(text[:valid_hw[0], :valid_hw[1]].sum() > 0 and valid_hw[1] >= 16)
        return {
            "lq": _rgb_to_nv21(lq), "gt": _rgb_to_yuv444(gt),
            "lq_rgb": torch.from_numpy(lq).permute(2, 0, 1),
            "gt_rgb": torch.from_numpy(gt).permute(2, 0, 1),
            "input_yuv420": _rgb_to_nv21(lq), "target_yuv444": _rgb_to_yuv444(gt),
            "valid_mask": torch.from_numpy(valid[None]), "text_mask": torch.from_numpy(text[None]),
            # Keep full-resolution masks for training-only structure losses.
            # The legacy six-channel masks remain the native-loss contract.
            "valid_mask_rgb": torch.from_numpy(valid[None]), "text_mask_rgb": torch.from_numpy(text[None]),
            "structure_available": structure_available,
            "blend_mask": torch.from_numpy(text[None]),
            "lq_path": str(lq_path), "gt_path": str(gt_path),
            "line_id": str(record["image"]), "frame_id": str(record["image"]),
            "valid_hw": list(valid_hw), "crop_xyxy_even": (0, 0, valid_hw[1], valid_hw[0]),
            "source_crop_xyxy": source_crop,
            "pad_ltrb": (0, 0, self.canvas_hw[1] - valid_hw[1], self.canvas_hw[0] - valid_hw[0]),
            "tile_index": 0, "tile_count": 1, "orientation": "HORIZONTAL",
            "cp_eligible": cp_eligible, "cp_skip_reason": "" if cp_eligible else "EMPTY_OR_SHORT_TEXT",
            "polygon_valid": True, "is_crop_truncated": width_truncated,
            "transcription": str(record.get("ocr_annotation", {}).get("transcription") or record.get("ocr_annotation", {}).get("text") or ""),
            "ocr_instance_id": str(record.get("ocr_annotation", {}).get("instance_id") or ""),
            "source_image": str(record["image"]), "source_name": entry["name"], "source_index": entry["source_index"],
            "source_manifest": str(entry["manifest"]), "char_count": len(record.get("chars", [])),
            "char_labels": [int(item["label"]) for item in record.get("chars", []) if "label" in item],
            "resize_mode": self.resize_mode, "resize_scale": scale, "dataset_version": self.dataset_version,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """@luna-comment Stack tensor fields while retaining per-record metadata lists."""
        if not batch:
            raise ValueError("cannot collate an empty batch")
        tensor_keys = ("lq", "gt", "lq_rgb", "gt_rgb", "input_yuv420", "target_yuv444", "valid_mask", "text_mask", "blend_mask", "valid_mask_rgb", "text_mask_rgb")
        result = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
        for key in batch[0]:
            if key not in result:
                result[key] = [item[key] for item in batch]
        return result


__all__ = ["DATASET_VERSION", "BasicTextLineRemote58Dataset"]
