#!/usr/bin/env python3
"""Measure CRNN input contracts, feature stability, and fixed-width distortion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realesrgan.metrics.crnn.crnn import CRNN


LAYER_NAMES = ("pooling0", "pooling1", "pooling2", "pooling3", "relu6")
IMAGE_RESAMPLING = getattr(Image, "Resampling", Image)


def sha256_file(path: Path) -> str:
    """@guang.yang-comment
    Input: a local file path.
    Output: lowercase SHA-256 digest of the complete file.
    Key Steps: stream the file in 1 MiB chunks and update one hash object.
    Key Params: chunk size limits memory use for model and image manifests.
    State/Dependencies: reads only the requested file.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """@guang.yang-comment
    Input: a scalar, NumPy value, tensor-derived value, mapping, or sequence.
    Output: a JSON-serializable equivalent with non-finite floats represented as null.
    Key Steps: recursively normalize containers and convert numeric library scalars.
    Key Params: NaN and infinity are intentionally preserved as null for artifact validity.
    State/Dependencies: pure conversion with no external state.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    """@guang.yang-comment
    Input: an output path and an in-memory JSON-compatible payload.
    Output: an indented UTF-8 JSON artifact.
    Key Steps: create the parent directory and serialize after non-finite normalization.
    Key Params: ensure_ascii=False keeps Chinese transcripts auditable.
    State/Dependencies: writes only the requested artifact path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_conditions(path: Path) -> list[dict]:
    """@guang.yang-comment
    Input: the experiment YAML condition file.
    Output: ordered condition dictionaries with required fields checked.
    Key Steps: parse YAML, validate unique IDs and geometry/mode fields, and return a detached list.
    Key Params: supported geometry values are fixed/aspect and supported modes are eval/train.
    State/Dependencies: reads one YAML file; no configuration is modified.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", [])
    if not conditions:
        raise ValueError(f"No conditions found in {path}")
    ids = [condition.get("id") for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("Condition IDs must be unique")
    for condition in conditions:
        if condition.get("geometry") not in {"fixed", "aspect"}:
            raise ValueError(f"Unsupported geometry: {condition}")
        if condition.get("mode") not in {"eval", "train"}:
            raise ValueError(f"Unsupported mode: {condition}")
    return [dict(condition) for condition in conditions]


def load_manifest(path: Path, max_samples: int | None, seed: int) -> list[dict]:
    """@guang.yang-comment
    Input: a frozen JSONL manifest, optional sample limit, and seed.
    Output: deterministic records with absolute source paths resolved by the caller's environment.
    Key Steps: parse non-empty lines, validate required pair/text fields, and select a stable subset when requested.
    Key Params: max_samples is used only for smoke runs and never rewrites the source manifest.
    State/Dependencies: reads one manifest file and returns independent dictionaries.
    """
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"sample_id", "pair_id", "image_path_gt", "image_path_lq", "polygon", "transcript"}
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Manifest record {record.get('sample_id')} misses {sorted(missing)}")
    if max_samples is not None and max_samples < len(records):
        rng = random.Random(seed)
        positive = [record for record in records if record.get("transcript")]
        background = [record for record in records if not record.get("transcript")]
        rng.shuffle(positive)
        rng.shuffle(background)
        positive_target = min(len(positive), max(1, int(round(max_samples * 0.87))))
        chosen = positive[:positive_target] + background[: max_samples - positive_target]
        if len(chosen) < max_samples:
            remainder = [record for record in records if record not in chosen]
            rng.shuffle(remainder)
            chosen.extend(remainder[: max_samples - len(chosen)])
        records = sorted(chosen[:max_samples], key=lambda item: item["sample_id"])
    return records


def snapshot_manifest(records: list[dict], output_path: Path) -> None:
    """@guang.yang-comment
    Input: manifest records used by this run and an output path.
    Output: a JSONL snapshot containing exactly the run's sample records.
    Key Steps: create the parent directory and serialize one deterministic record per line.
    Key Params: sort_keys=True makes snapshots comparable across reruns.
    State/Dependencies: writes only the requested snapshot file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def order_quad(points: np.ndarray) -> np.ndarray:
    """@guang.yang-comment
    Input: four polygon points in arbitrary order.
    Output: top-left, top-right, bottom-right, bottom-left points.
    Key Steps: use coordinate sums/differences and an angular fallback for ties.
    Key Params: exactly four finite points are required by the perspective crop.
    State/Dependencies: pure NumPy computation.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    if len({tuple(point) for point in ordered}) != 4:
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        cyclic = points[np.argsort(angles)]
        start = np.argmin(cyclic.sum(axis=1))
        cyclic = np.roll(cyclic, -int(start), axis=0)
        if cyclic[1, 0] < cyclic[3, 0]:
            cyclic = cyclic[[0, 3, 2, 1]]
        ordered = cyclic.astype(np.float32)
    return ordered


def crop_polygon(image: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    """@guang.yang-comment
    Input: a BGR source image and a four-point polygon from the frozen manifest.
    Output: a perspective-rectified BGR text-line crop.
    Key Steps: order the polygon, estimate opposing-edge dimensions, and warp with OpenCV.
    Key Params: linear interpolation preserves the same crop contract for GT and LQ views.
    State/Dependencies: allocates a crop and leaves the source image unchanged.
    """
    quad = order_quad(np.asarray(polygon, dtype=np.float32))
    width = max(int(round(np.linalg.norm(quad[1] - quad[0]))), int(round(np.linalg.norm(quad[2] - quad[3]))))
    height = max(int(round(np.linalg.norm(quad[3] - quad[0]))), int(round(np.linalg.norm(quad[2] - quad[1]))))
    if width < 4 or height < 4:
        raise ValueError(f"Invalid crop geometry: width={width}, height={height}")
    target = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    crop = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    if crop is None or crop.size == 0:
        raise ValueError("Perspective crop returned an empty image")
    return crop


def read_crop(record: dict, view: str) -> np.ndarray:
    """@guang.yang-comment
    Input: one manifest record and either the ``gt`` or ``lq`` view selector.
    Output: the corresponding BGR crop loaded from disk.
    Key Steps: resolve the view path, decode with OpenCV, and apply the shared polygon crop.
    Key Params: the same polygon is used for both paired views to prevent contract-dependent sample selection.
    State/Dependencies: reads one source image; no source image is modified.
    """
    key = "image_path_gt" if view == "gt" else "image_path_lq"
    image = cv2.imread(record[key], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(record[key])
    return crop_polygon(image, record["polygon"])


def preprocess_crop(crop: np.ndarray, condition: dict) -> torch.Tensor:
    """@guang.yang-comment
    Input: a BGR crop and one resolved condition dictionary.
    Output: a CPU float tensor shaped [1,3,32,W] in the condition's numeric range and RGB order.
    Key Steps: convert BGR to RGB, apply PIL bilinear for fixed geometry or tensor bicubic for aspect geometry, then normalize.
    Key Params: height 32, dynamic width clamp [16,1024], and explicit ranges [-1,1]/[0,1]/[0,255].
    State/Dependencies: uses PIL/OpenCV/Torch interpolation; does not mutate the crop.
    """
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    height, source_width = rgb.shape[:2]
    if condition["geometry"] == "fixed":
        width = int(condition["width"])
        resized = np.asarray(
            Image.fromarray(rgb).resize((width, 32), resample=IMAGE_RESAMPLING.BILINEAR),
            dtype=np.uint8,
        )
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0).float()
    else:
        width = max(16, min(int(round(source_width * 32.0 / max(1, height))), 1024))
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = F.interpolate(tensor, size=(32, width), mode="bicubic", align_corners=False)
        tensor = tensor * 255.0
    low, high = condition["range"]
    if low == -1.0 and high == 1.0:
        tensor = tensor / 127.5 - 1.0
    elif low == 0.0 and high == 1.0:
        tensor = tensor / 255.0
    elif low == 0.0 and high == 255.0:
        pass
    else:
        raise ValueError(f"Unsupported input range: {condition['range']}")
    return tensor.contiguous()


def input_statistics(tensor: torch.Tensor) -> dict:
    """@guang.yang-comment
    Input: a finite-or-nonfinite input tensor.
    Output: min/max/mean/std, shape, and finite/non-finite counts.
    Key Steps: flatten detached CPU values and compute statistics on finite values only.
    Key Params: zero finite values produce null scalar statistics rather than fabricated zeros.
    State/Dependencies: does not retain the computation graph.
    """
    values = tensor.detach().float().cpu().reshape(-1).numpy()
    finite = np.isfinite(values)
    valid = values[finite]
    return {
        "input_shape": list(tensor.shape),
        "input_min": float(np.min(valid)) if valid.size else math.nan,
        "input_max": float(np.max(valid)) if valid.size else math.nan,
        "input_mean": float(np.mean(valid)) if valid.size else math.nan,
        "input_std": float(np.std(valid)) if valid.size else math.nan,
        "input_finite_count": int(np.count_nonzero(finite)),
        "input_nonfinite_count": int(values.size - np.count_nonzero(finite)),
    }


def activation_statistics(tensor: torch.Tensor) -> dict:
    """@guang.yang-comment
    Input: one CNN activation tensor captured by a forward hook.
    Output: shape, distribution quantiles, ReLU-zero ratio, and non-finite counts.
    Key Steps: detach, sample at most 50k values for quantiles, and compute robust finite statistics.
    Key Params: P50/P95/P99 and max absolute value are the plan's activation-state diagnostics.
    State/Dependencies: consumes a detached view and releases no model state.
    """
    values = tensor.detach().float().reshape(-1)
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    if finite.numel() > 50000:
        indices = torch.linspace(0, finite.numel() - 1, 50000, device=finite.device).long()
        finite_for_quantile = finite[indices]
    else:
        finite_for_quantile = finite
    if finite_for_quantile.numel():
        abs_values = finite_for_quantile.abs()
        quantiles = torch.quantile(abs_values, torch.tensor([0.50, 0.95, 0.99], device=abs_values.device))
        finite_mean = float(finite_for_quantile.mean().item())
        finite_std = float(finite_for_quantile.std(unbiased=False).item())
        abs_max = float(abs_values.max().item())
        p50, p95, p99 = [float(value.item()) for value in quantiles]
    else:
        finite_mean = finite_std = abs_max = p50 = p95 = p99 = math.nan
    return {
        "shape": list(tensor.shape),
        "mean": finite_mean,
        "std": finite_std,
        "abs_p50": p50,
        "abs_p95": p95,
        "abs_p99": p99,
        "abs_max": abs_max,
        "relu_zero_ratio": float((values == 0).float().mean().item()) if values.numel() else math.nan,
        "nan_count": int(torch.isnan(values).sum().item()),
        "inf_count": int(torch.isinf(values).sum().item()),
    }


def feature_vector(tensor: torch.Tensor) -> torch.Tensor:
    """@guang.yang-comment
    Input: one captured CNN activation shaped [1,C,H,W].
    Output: an L2-normalized feature vector after adaptive pooling to [1,C,1,32].
    Key Steps: adaptive-average-pool height and width, flatten, and normalize per sample.
    Key Params: width 32 is only a contract-comparison proxy and is not the formal CP implementation.
    State/Dependencies: retains the input graph when the caller supplies a graph-connected tensor.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4-D activation, got {tuple(tensor.shape)}")
    pooled = F.adaptive_avg_pool2d(tensor, (1, 32)).flatten(1)
    return F.normalize(pooled, p=2, dim=1, eps=1e-12)


def forward_with_hooks(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    layer_names: tuple[str, ...],
    with_grad: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """@guang.yang-comment
    Input: a frozen CRNN model, input tensor, CNN child names, and graph-retention flag.
    Output: logits and captured activation tensors for the requested layers.
    Key Steps: register temporary hooks, execute one forward pass, and remove all handles in a finally block.
    Key Params: with_grad controls whether input-gradient diagnostics retain autograd history.
    State/Dependencies: temporarily observes model modules; hook handles are always released.
    """
    captures: dict[str, torch.Tensor] = {}
    handles = []
    for name in layer_names:
        module = dict(model.cnn.named_children()).get(name)
        if module is None:
            raise KeyError(f"CRNN CNN layer not found: {name}")

        def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor, key: str = name) -> None:
            """@guang.yang-comment
            Input: one forward-hook callback invocation and its captured layer name.
            Output: no return value; stores the live activation in the enclosing capture map.
            Key Steps: associate the module output with the stable layer key.
            Key Params: default-bound key prevents late-binding across hook registrations.
            State/Dependencies: mutates only the temporary captures dictionary.
            """
            captures[key] = output

        handles.append(module.register_forward_hook(capture))
    try:
        if with_grad:
            logits = model(tensor)
        else:
            with torch.no_grad():
                logits = model(tensor)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captures


def decode_logits(logits: torch.Tensor, alphabet: list[str]) -> dict:
    """@guang.yang-comment
    Input: CRNN logits shaped [T,1,C] and the checkpoint alphabet without the CTC blank.
    Output: greedy CTC text, mean max probability, mean entropy, and predicted length.
    Key Steps: softmax, argmax, collapse repeats/blanks, and aggregate confidence diagnostics.
    Key Params: class zero is the blank index; class i maps to alphabet[i-1].
    State/Dependencies: pure Torch/CPU conversion; no model state is changed.
    """
    probabilities = logits.detach().float().softmax(dim=-1)
    max_probability, indices = probabilities.max(dim=-1)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    sequence = indices[:, 0].tolist()
    text = []
    previous = 0
    for index in sequence:
        if index != 0 and index != previous and index - 1 < len(alphabet):
            text.append(alphabet[index - 1])
        previous = index
    return {
        "prediction": "".join(text),
        "ctc_mean_max_prob": float(max_probability.mean().item()),
        "ctc_mean_entropy": float(entropy.mean().item()),
        "pred_length": len(text),
        "blank_output": not text,
    }


def edit_distance(source: str, target: str) -> int:
    """@guang.yang-comment
    Input: two Unicode strings.
    Output: Levenshtein edit distance as an integer.
    Key Steps: dynamic-program the insertion, deletion, and substitution minimum with two rows.
    Key Params: empty strings use the standard length boundary conditions.
    State/Dependencies: pure Python memory bounded by the shorter string length.
    """
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            current.append(min(
                current[-1] + 1,
                previous[target_index] + 1,
                previous[target_index - 1] + (source_char != target_char),
            ))
        previous = current
    return previous[-1]


def text_metrics(transcript: str, prediction: str) -> dict:
    """@guang.yang-comment
    Input: manifest transcript and greedy CTC prediction.
    Output: exact accuracy, CER, NED, and validity flag for OCR reporting.
    Key Steps: compute edit distance and normalize by transcript length, with empty-text rows excluded from label metrics.
    Key Params: NED is reported as 1 - normalized edit distance, so higher is better.
    State/Dependencies: pure string metric calculation.
    """
    if not transcript:
        return {"text_valid": False, "acc": math.nan, "cer": math.nan, "ned": math.nan}
    distance = edit_distance(transcript, prediction)
    cer = distance / max(1, len(transcript))
    return {
        "text_valid": True,
        "acc": float(transcript == prediction),
        "cer": float(cer),
        "ned": float(1.0 - min(1.0, cer)),
    }


def tensor_distance(first: np.ndarray, second: np.ndarray) -> float:
    """@guang.yang-comment
    Input: two normalized feature vectors with equal shape.
    Output: root-mean-square Euclidean distance.
    Key Steps: subtract, square, average, and take the square root.
    Key Params: normalized vectors make distances comparable across layers and widths.
    State/Dependencies: pure NumPy arithmetic.
    """
    return float(np.sqrt(np.mean(np.square(first.astype(np.float64) - second.astype(np.float64)))))


def choose_negative_indices(records: list[dict]) -> dict[str, str]:
    """@guang.yang-comment
    Input: manifest records containing positive text pairs.
    Output: sample_id to different-transcript negative sample_id mapping.
    Key Steps: prefer the same language and closest length/aspect, then reject identical transcripts.
    Key Params: deterministic sorting prevents negative-pair selection drift between runs.
    State/Dependencies: reads record metadata only; no image access is required.
    """
    positive = [record for record in records if record.get("transcript")]
    mapping = {}
    for record in positive:
        candidates = [item for item in positive if item["sample_id"] != record["sample_id"] and item.get("transcript") != record.get("transcript")]
        candidates.sort(key=lambda item: (
            item.get("language") != record.get("language"),
            abs(int(item.get("character_count", 0)) - int(record.get("character_count", 0))),
            abs(float(item.get("aspect_ratio", 1.0)) - float(record.get("aspect_ratio", 1.0))),
            item["sample_id"],
        ))
        if candidates:
            mapping[record["sample_id"]] = candidates[0]["sample_id"]
    return mapping


def condition_input_label(condition: dict) -> str:
    """@guang.yang-comment
    Input: a resolved condition dictionary.
    Output: stable human-readable range label for CSV/report grouping.
    Key Steps: format numeric ranges or preserve the best-eval marker.
    Key Params: six decimal places avoid noisy floating-point representations.
    State/Dependencies: pure formatting.
    """
    value = condition["range"]
    if isinstance(value, str):
        return value
    return "[%.6g,%.6g]" % (float(value[0]), float(value[1]))


def build_model(checkpoint: Path, alphabet_size: int, device: torch.device, mode: str) -> torch.nn.Module:
    """@guang.yang-comment
    Input: checkpoint path, alphabet size, target device, and eval/train mode.
    Output: a CRNN with strictly loaded frozen parameters on the target device.
    Key Steps: construct the known RGB CRNN, strip optional DataParallel prefixes, load parameters, freeze gradients, and set mode.
    Key Params: hidden size 256, class count alphabet+blank, and input height 32 match the checkpoint architecture.
    State/Dependencies: reads checkpoint tensors and allocates one model on the selected device.
    """
    model = CRNN(3, 256, alphabet_size + 1, 32)
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must be a state dict, got {type(state).__name__}")
    cleaned = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train(mode == "train")
    return model


def resolve_best_range(conditions: list[dict], metrics: list[dict]) -> list[float]:
    """@guang.yang-comment
    Input: configured conditions and measured per-sample metrics from C00-C05.
    Output: the numeric range with the highest mean GT NED, defaulting to [-1,1] if unavailable.
    Key Steps: group valid GT rows by range label, average NED, and choose the highest score with stable tie-breaking.
    Key Params: only eval conditions with text labels are eligible; geometry is intentionally not used to choose the range.
    State/Dependencies: consumes measured rows and does not rerun inference.
    """
    allowed = {condition["id"] for condition in conditions if condition["id"] != "C06-TATSR" and condition["mode"] == "eval"}
    scores: dict[str, list[float]] = defaultdict(list)
    ranges: dict[str, list[float]] = {}
    for row in metrics:
        if row.get("condition") not in allowed or row.get("view") != "gt" or not row.get("text_valid"):
            continue
        if row.get("ned") is None:
            continue
        label = row["range_label"]
        scores[label].append(float(row["ned"]))
        ranges[label] = row["range"]
    if not scores:
        return [-1.0, 1.0]
    best_label = max(scores, key=lambda label: (float(np.mean(scores[label])), label))
    return [float(value) for value in ranges[best_label]]


def run_condition(
    condition: dict,
    records: list[dict],
    checkpoint: Path,
    alphabet: list[str],
    device: torch.device,
    gradient_limit: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """@guang.yang-comment
    Input: one resolved condition, frozen manifest records, checkpoint/alphabet/device, and gradient sample limit.
    Output: per-sample OCR rows, per-layer activation rows, CP positive/negative rows, and input-gradient rows.
    Key Steps: reload model, process GT/LQ views independently, collect repeatability/hooks, rank feature pairs, and measure gradients.
    Key Params: batch size is one, five CNN hook layers are fixed, and gradients use the first 32 positive pairs by default.
    State/Dependencies: allocates and releases one model; reads source images and writes no artifacts directly.
    """
    model = build_model(checkpoint, len(alphabet), device, condition["mode"])
    metric_rows: list[dict] = []
    activation_rows: list[dict] = []
    feature_cache: dict[str, dict[str, list[np.ndarray]]] = {}
    positive_records = [record for record in records if record.get("transcript")]
    for record_index, record in enumerate(records):
        views = [("gt", record["image_path_gt"]), ("lq", record["image_path_lq"])] if record.get("transcript") else [("lq", record["image_path_lq"])]
        for view, _path in views:
            crop = read_crop(record, view)
            cpu_tensor = preprocess_crop(crop, condition)
            tensor = cpu_tensor.to(device)
            logits, captures = forward_with_hooks(model, tensor, LAYER_NAMES, with_grad=False)
            first_logits = logits.detach()
            second_logits, second_captures = forward_with_hooks(model, tensor, LAYER_NAMES, with_grad=False)
            repeat_difference = float(torch.max(torch.abs(first_logits - second_logits.detach())).item())
            decoded = decode_logits(first_logits, alphabet)
            metrics = text_metrics(record.get("transcript", ""), decoded["prediction"])
            input_stats = input_statistics(cpu_tensor)
            metric_rows.append({
                "sample_id": record["sample_id"],
                "pair_id": record["pair_id"],
                "condition": condition["id"],
                "mode": condition["mode"],
                "view": view,
                "language": record.get("language", "none"),
                "length_bucket": record.get("length_bucket", "0"),
                "structure_bucket": record.get("structure_bucket", "unknown"),
                "character_count": record.get("character_count", 0),
                "aspect_ratio": record.get("aspect_ratio", math.nan),
                "range_label": condition_input_label(condition),
                "range": list(condition["range"]),
                "geometry": condition["geometry"],
                "width": condition.get("width", "dynamic"),
                "transcript": record.get("transcript", ""),
                **decoded,
                **metrics,
                "logit_nonfinite_count": int((~torch.isfinite(first_logits)).sum().item()),
                "repeat_max_abs_diff": repeat_difference,
                **input_stats,
            })
            for layer_name in LAYER_NAMES:
                activation = captures[layer_name]
                stats = activation_statistics(activation)
                activation_rows.append({
                    "sample_id": record["sample_id"],
                    "pair_id": record["pair_id"],
                    "condition": condition["id"],
                    "mode": condition["mode"],
                    "view": view,
                    "layer": layer_name,
                    **stats,
                })
            if record.get("transcript"):
                feature_cache.setdefault(record["sample_id"], {})[view] = [
                    feature_vector(captures[layer_name]).detach().cpu().numpy().reshape(-1)
                    for layer_name in LAYER_NAMES
                ]
            del second_captures, captures, second_logits, logits, tensor, cpu_tensor, crop
        if (record_index + 1) % 25 == 0:
            print(f"{condition['id']}: processed {record_index + 1}/{len(records)}", flush=True)
    negative_mapping = choose_negative_indices(positive_records)
    cache_by_id = {record["sample_id"]: record for record in positive_records}
    cp_rows: list[dict] = []
    for record in positive_records:
        sample_id = record["sample_id"]
        negative_id = negative_mapping.get(sample_id)
        if sample_id not in feature_cache or negative_id not in feature_cache:
            continue
        current = feature_cache[sample_id]
        negative = feature_cache[negative_id]
        distances_pos = [tensor_distance(current["gt"][index], current["lq"][index]) for index in range(len(LAYER_NAMES))]
        distances_neg = [tensor_distance(negative["gt"][index], current["lq"][index]) for index in range(len(LAYER_NAMES))]
        d_pos = float(np.mean(distances_pos))
        d_neg = float(np.mean(distances_neg))
        cp_rows.append({
            "sample_id": sample_id,
            "pair_id": record["pair_id"],
            "condition": condition["id"],
            "language": record.get("language", "none"),
            "length_bucket": record.get("length_bucket", "0"),
            "structure_bucket": record.get("structure_bucket", "unknown"),
            "negative_sample_id": negative_id,
            "negative_transcript": cache_by_id[negative_id].get("transcript", ""),
            **{f"d_pos_{layer}": distances_pos[index] for index, layer in enumerate(LAYER_NAMES)},
            **{f"d_neg_{layer}": distances_neg[index] for index, layer in enumerate(LAYER_NAMES)},
            "d_pos": d_pos,
            "d_neg": d_neg,
            "pos_better": float(d_pos < d_neg),
            "margin": float((d_neg - d_pos) / (abs(d_neg) + abs(d_pos) + 1e-8)),
        })
    gradient_rows: list[dict] = []
    if condition["mode"] == "eval" and condition.get("gradient", True):
        for record in positive_records[:gradient_limit]:
            sample_id = record["sample_id"]
            if sample_id not in feature_cache or "gt" not in feature_cache[sample_id]:
                continue
            crop = read_crop(record, "lq")
            cpu_tensor = preprocess_crop(crop, condition)
            tensor = cpu_tensor.to(device).requires_grad_(True)
            _logits, captures = forward_with_hooks(model, tensor, LAYER_NAMES, with_grad=True)
            target_features = [torch.from_numpy(value).to(device=device, dtype=torch.float32).unsqueeze(0) for value in feature_cache[sample_id]["gt"]]
            for index, layer_name in enumerate(LAYER_NAMES):
                current_feature = feature_vector(captures[layer_name])
                loss = F.mse_loss(current_feature, target_features[index])
                gradient = torch.autograd.grad(loss, tensor, retain_graph=True, allow_unused=True)[0]
                if gradient is None:
                    gradient_rows.append({"sample_id": sample_id, "condition": condition["id"], "layer": layer_name, "gradient_l2": math.nan, "gradient_abs_p50": math.nan, "gradient_abs_p95": math.nan, "gradient_abs_max": math.nan, "gradient_zero_ratio": math.nan, "nan_count": 0, "inf_count": 0})
                    continue
                values = gradient.detach().float().reshape(-1)
                finite = values[torch.isfinite(values)]
                abs_values = finite.abs()
                if abs_values.numel():
                    quantiles = torch.quantile(abs_values, torch.tensor([0.50, 0.95], device=abs_values.device))
                    gradient_rows.append({
                        "sample_id": sample_id,
                        "condition": condition["id"],
                        "layer": layer_name,
                        "gradient_l2": float(torch.linalg.vector_norm(values).item()),
                        "gradient_abs_p50": float(quantiles[0].item()),
                        "gradient_abs_p95": float(quantiles[1].item()),
                        "gradient_abs_max": float(abs_values.max().item()),
                        "gradient_zero_ratio": float((values == 0).float().mean().item()),
                        "nan_count": int(torch.isnan(values).sum().item()),
                        "inf_count": int(torch.isinf(values).sum().item()),
                    })
            del target_features, captures, _logits, tensor, cpu_tensor, crop
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows, activation_rows, cp_rows, gradient_rows


def bootstrap_interval(values: list[float], seed: int, iterations: int = 2000) -> dict:
    """@guang.yang-comment
    Input: paired sample differences, seed, and bootstrap iteration count.
    Output: sample count, mean, and percentile 95% confidence interval.
    Key Steps: resample paired differences with replacement and evaluate the mean statistic.
    Key Params: 2,000 iterations and percentile bounds follow the experiment plan.
    State/Dependencies: uses a local NumPy generator and does not alter global randomness.
    """
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        return {"n": 0, "mean": math.nan, "low": math.nan, "high": math.nan}
    if values_array.size == 1:
        value = float(values_array[0])
        return {"n": 1, "mean": value, "low": value, "high": value}
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values_array.size, size=(iterations, values_array.size))
    means = values_array[indices].mean(axis=1)
    return {"n": int(values_array.size), "mean": float(values_array.mean()), "low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}


def paired_metric_deltas(metrics: list[dict], fixed: str, aspect: str, metric: str, group_key: str | None = None, group_value: str | None = None) -> list[float]:
    """@guang.yang-comment
    Input: per-sample metrics, fixed/aspect condition IDs, metric name, and optional group filter.
    Output: sample_id-paired fixed-minus-aspect differences on LQ text rows.
    Key Steps: index rows by condition/sample, intersect sample IDs, and subtract paired values.
    Key Params: view=lq and text_valid are mandatory to avoid mixing GT and no-text diagnostics.
    State/Dependencies: reads metric dictionaries only.
    """
    indexed = {}
    for row in metrics:
        if row.get("view") != "lq" or not row.get("text_valid") or row.get("condition") not in {fixed, aspect}:
            continue
        if group_key is not None and row.get(group_key) != group_value:
            continue
        value = row.get(metric)
        if value is not None:
            indexed[(row["condition"], row["sample_id"])] = float(value)
    samples = {sample for condition, sample in indexed if condition == fixed} & {sample for condition, sample in indexed if condition == aspect}
    return [indexed[(fixed, sample)] - indexed[(aspect, sample)] for sample in sorted(samples)]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """@guang.yang-comment
    Input: CSV output path, row mappings, and optional ordered field list.
    Output: a UTF-8 CSV artifact with a stable header.
    Key Steps: infer missing fields when needed, create parents, and write nulls as empty cells.
    Key Params: newline='' avoids platform-dependent blank lines in the remote CSV files.
    State/Dependencies: writes only the requested CSV path.
    """
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else json_safe(row.get(field)) for field in fields})


def aggregate_condition_summary(metrics: list[dict], cp_rows: list[dict], activations: list[dict], gradients: list[dict], condition: dict) -> dict:
    """@guang.yang-comment
    Input: raw condition metrics, CP rows, activation rows, gradient rows, and condition metadata.
    Output: one condition-level summary row for ranking and reporting.
    Key Steps: average finite OCR/activation/repeat values and compute CP ordering/gradient counts.
    Key Params: valid text rows exclude no-text backgrounds from OCR means; CP uses d_pos<d_neg.
    State/Dependencies: pure aggregation over in-memory experiment results.
    """
    rows = [row for row in metrics if row["condition"] == condition["id"]]
    valid = [row for row in rows if row.get("text_valid")]
    def mean_field(items: list[dict], key: str) -> float:
        """@guang.yang-comment
        Input: condition rows and one numeric field name.
        Output: the finite arithmetic mean, or NaN when no finite values exist.
        Key Steps: filter missing/non-finite values and average the remaining scalars.
        Key Params: no-text and invalid values are excluded from metric means.
        State/Dependencies: pure aggregation over the enclosing condition rows.
        """
        values = [float(item[key]) for item in items if item.get(key) is not None and math.isfinite(float(item[key]))]
        return float(np.mean(values)) if values else math.nan
    condition_cp = [row for row in cp_rows if row["condition"] == condition["id"]]
    condition_activation = [row for row in activations if row["condition"] == condition["id"]]
    condition_gradient = [row for row in gradients if row["condition"] == condition["id"]]
    return {
        "condition": condition["id"],
        "mode": condition["mode"],
        "geometry": condition["geometry"],
        "width": condition.get("width", "dynamic"),
        "range": condition_input_label(condition),
        "sample_rows": len(rows),
        "valid_text_rows": len(valid),
        "acc_mean": mean_field(valid, "acc"),
        "cer_mean": mean_field(valid, "cer"),
        "ned_mean": mean_field(valid, "ned"),
        "blank_rate": mean_field(rows, "blank_output"),
        "pred_length_mean": mean_field(rows, "pred_length"),
        "ctc_max_prob_mean": mean_field(rows, "ctc_mean_max_prob"),
        "ctc_entropy_mean": mean_field(rows, "ctc_mean_entropy"),
        "repeat_max_abs_diff_max": max([float(row["repeat_max_abs_diff"]) for row in rows], default=math.nan),
        "logit_nonfinite_total": int(sum(int(row.get("logit_nonfinite_count", 0)) for row in rows)),
        "cp_pairs": len(condition_cp),
        "cp_pos_better_probability": mean_field(condition_cp, "pos_better"),
        "cp_margin_mean": mean_field(condition_cp, "margin"),
        "activation_rows": len(condition_activation),
        "gradient_rows": len(condition_gradient),
        "gradient_nonfinite_total": int(sum(int(row.get("nan_count", 0)) + int(row.get("inf_count", 0)) for row in condition_gradient)),
    }


def render_visuals(records: list[dict], output_dir: Path, conditions: dict[str, dict], seed: int, count: int) -> list[dict]:
    """@guang.yang-comment
    Input: manifest records, visual output directory, resolved C03/C00/C06 conditions, seed, and target count.
    Output: saved four-panel comparison images plus a private mapping list for audit.
    Key Steps: crop LQ text lines, render aspect/fixed variants at matched display height, shuffle G0/G1/G2 labels, and save PNGs.
    Key Params: count targets the plan's 120 blind-review images; mapping is not printed on the image.
    State/Dependencies: reads source images and writes only visual artifacts and mapping metadata.
    """
    positive = [record for record in records if record.get("transcript")]
    positive = sorted(positive, key=lambda record: record["sample_id"])[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    mapping_rows = []
    for record in positive:
        crop = read_crop(record, "lq")
        variants = {
            "aspect": preprocess_crop(crop, conditions["C03-DYN-NORM"]),
            "fixed256": preprocess_crop(crop, conditions["C00-SRC"]),
            "fixed128": preprocess_crop(crop, conditions["C06-TATSR"]),
        }
        images = [crop]
        names = ["source"]
        labels = ["source"]
        variant_items = list(variants.items())
        rng.shuffle(variant_items)
        for index, (name, tensor) in enumerate(variant_items):
            values = tensor.squeeze(0).permute(1, 2, 0).numpy()
            if name == "aspect":
                low, high = conditions["C03-DYN-NORM"]["range"]
            elif name == "fixed256":
                low, high = conditions["C00-SRC"]["range"]
            else:
                low, high = conditions["C06-TATSR"]["range"]
            if low == -1.0:
                values = (values + 1.0) * 127.5
            elif high == 1.0:
                values = values * 255.0
            values = np.clip(values, 0, 255).astype(np.uint8)
            values = cv2.cvtColor(values, cv2.COLOR_RGB2BGR)
            images.append(values)
            names.append(name)
            labels.append(f"G{index}")
        display_height = 180
        panels = []
        for image, label in zip(images, labels):
            scale = display_height / max(1, image.shape[0])
            panel = cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), display_height), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 24), (0, 0, 0), thickness=-1)
            cv2.putText(panel, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        canvas = cv2.hconcat(panels)
        file_name = f"{record['sample_id']}.png"
        cv2.imwrite(str(output_dir / file_name), canvas)
        mapping_rows.append({"sample_id": record["sample_id"], "panel_names": names, "panel_labels": labels})
    write_json(output_dir / "visual_mapping.json", mapping_rows)
    return mapping_rows


def write_report(
    output_dir: Path,
    run_meta: dict,
    manifest_report: dict,
    summaries: list[dict],
    cp_rows: list[dict],
    intervals: dict,
    visual_count: int,
    c06_range: list[float],
) -> None:
    """@guang.yang-comment
    Input: output directory, provenance, manifest audit, condition summaries, CP rows, bootstrap intervals, visual count, and C06 range.
    Output: an English Markdown report separating source facts, measured results, and inferences.
    Key Steps: format tables, rank OCR/CP conditions, describe fixed-width deltas, and record limitations.
    Key Params: visual_count and interval records are reported exactly as measured.
    State/Dependencies: writes only ``report.md`` and does not alter numeric artifacts.
    """
    def fmt(value: Any) -> str:
        """@guang.yang-comment
        Input: one report scalar or string value.
        Output: compact Markdown-safe display text, with non-finite values rendered as NA.
        Key Steps: handle missing/non-finite values, then format floats or preserve strings.
        Key Params: six significant digits keep tables readable without hiding effect sizes.
        State/Dependencies: pure formatting helper scoped to report generation.
        """
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "NA"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
    best_ocr = max(summaries, key=lambda row: (-float(row["cer_mean"]) if math.isfinite(float(row["cer_mean"])) else -math.inf, float(row["ned_mean"]) if math.isfinite(float(row["ned_mean"])) else -math.inf), default=None)
    best_cp = max(summaries, key=lambda row: float(row["cp_pos_better_probability"]) if math.isfinite(float(row["cp_pos_better_probability"])) else -math.inf, default=None)
    lines = [
        "# CRNN Input Contract Validation Report",
        "",
        "## 1. Source facts",
        "",
        f"- Checkpoint: `{run_meta['checkpoint']}`; SHA-256 `{run_meta['checkpoint_sha256']}`.",
        f"- Alphabet: {run_meta['alphabet_length']} characters; SHA-256 `{run_meta['alphabet_sha256']}`.",
        f"- Runtime: Python `{run_meta['python']}`, PyTorch `{run_meta['torch']}`, CUDA `{run_meta['cuda']}`, GPU `{run_meta['gpu']}`.",
        f"- Code revision: `{run_meta['code_revision']}`. The remote project is not a Git worktree, so no commit hash was available.",
        f"- Manifest rows: {manifest_report.get('total_rows', 'NA')} ({manifest_report.get('selected_text_rows', 'NA')} text and {manifest_report.get('selected_background_rows', 'NA')} no-text proxy rows).",
        "- Structure density is a Canny edge-ratio proxy, not stroke-level annotation.",
        "",
        "## 2. Measured condition results",
        "",
        "| Condition | Mode | Geometry | Range | OCR NED | OCR CER | Blank rate | CP P(pos<neg) | Repeat max |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["condition"], row["mode"], row["geometry"], row["range"], fmt(row["ned_mean"]), fmt(row["cer_mean"]), fmt(row["blank_rate"]), fmt(row["cp_pos_better_probability"]), fmt(row["repeat_max_abs_diff_max"])))
    lines.extend([
        "",
        f"Best OCR condition by measured CER/NED: `{best_ocr['condition'] if best_ocr else 'NA'}`.",
        f"Best CP ordering condition by measured P(D_pos < D_neg): `{best_cp['condition'] if best_cp else 'NA'}`.",
        f"C06 resolved numeric range: `{c06_range}`.",
        "",
        "## 3. Fixed-width paired comparisons",
        "",
        "Deltas are fixed minus aspect-preserving on paired LQ rows. Positive CER delta and negative NED delta indicate fixed-width degradation.",
        "",
        "| Comparison | Group | Metric | N | Mean | Bootstrap 95% CI |",
        "|---|---|---|---:|---:|---|",
    ])
    for item in intervals.get("comparisons", []):
        interval = item["interval"]
        lines.append("| %s - %s | %s | %s | %s | %s | [%s, %s] |" % (item["fixed"], item["aspect"], item["group"], item["metric"], fmt(interval["n"]), fmt(interval["mean"]), fmt(interval["low"]), fmt(interval["high"])))
    lines.extend([
        "",
        "## 4. CP and gradient interpretation",
        "",
        f"- CP rows measured: {len(cp_rows)}.",
        f"- Blind-review comparison images saved: {visual_count}.",
        "- Input gradients are diagnostic proxies. They do not implement or validate the final CRNN-CP loss formula.",
        "- C10/C11 are mode diagnostics only. They are not candidates for a frozen loss network.",
        "",
        "## 5. Conclusion rules applied",
        "",
        "- The effective contract must be selected jointly from OCR, activation finiteness/stability, CP pair ordering, and gradient validity.",
        "- A fixed-width strategy is not accepted as the default when paired confidence intervals show degradation in Chinese long/high-density groups or when CP ordering falls.",
        "- No checkpoint training contract is inferred from the Real-CE `[0,255]` condition alone.",
        "",
        "## 6. Limitations",
        "",
        "- The remote project has no Git metadata; provenance is bound to file hashes instead of a commit.",
        "- The clear/reference view uses the available same-name `documents_52mm` subset paired with `52mm`; this is a data-source fact, not a claim that the two captures are pixel-aligned GT/LQ.",
        "- Human review remains an external judgment step even though the blind-review panels are generated.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """@guang.yang-comment
    Input: command-line arguments supplied by the smoke or full-run launcher.
    Output: validated paths, device, condition selection, seed, and visual switch.
    Key Steps: register required artifacts and optional sample/condition limits.
    Key Params: max-samples enables the mandatory 10-sample smoke run; save-visuals enables blind panels.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--alphabet", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--visual-count", type=int, default=120)
    return parser.parse_args()


def checkpoint_metadata(checkpoint: Path, alphabet: list[str]) -> dict:
    """@guang.yang-comment
    Input: checkpoint path and decoded alphabet.
    Output: checkpoint type, tensor count, key shape checks, and SHA-256 metadata.
    Key Steps: load on CPU, inspect first/last architecture tensors, and validate RGB input/class count.
    Key Params: first convolution must be [64,3,3,3] and final classifier must equal alphabet+blank.
    State/Dependencies: reads checkpoint tensors into CPU memory and releases them before returning.
    """
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state).__name__}")
    keys = list(state)
    first_key = next((key for key in keys if key.endswith("cnn.conv0.weight")), keys[0])
    output_key = next((key for key in keys if key.endswith("rnn.1.embedding.bias")), keys[-1])
    first_shape = list(state[first_key].shape)
    output_shape = list(state[output_key].shape)
    if first_shape != [64, 3, 3, 3]:
        raise ValueError(f"Unexpected first convolution shape: {first_shape}")
    if output_shape != [len(alphabet) + 1]:
        raise ValueError(f"Unexpected output class shape: {output_shape}; alphabet={len(alphabet)}")
    return {
        "type": type(state).__name__,
        "tensor_count": len(state),
        "first_weight_key": first_key,
        "first_weight_shape": first_shape,
        "output_bias_key": output_key,
        "output_bias_shape": output_shape,
        "sha256": sha256_file(checkpoint),
    }


def runtime_metadata(device: torch.device) -> dict:
    """@guang.yang-comment
    Input: the selected Torch device.
    Output: Python/Torch/CUDA versions, GPU count/name, and deterministic backend settings.
    Key Steps: query runtime modules and the selected CUDA device without changing allocation state.
    Key Params: cuDNN deterministic=True and benchmark=False reduce repeated-run drift.
    State/Dependencies: reads the current process runtime and GPU metadata.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else "CPU"
    return {
        "python": __import__("sys").version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu": gpu_name,
        "device": str(device),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def main() -> None:
    """@guang.yang-comment
    Input: command-line experiment paths and switches.
    Output: all plan-specified CSV/JSON/visual/report artifacts under output-dir.
    Key Steps: validate provenance, run smoke/full selected conditions, compute paired intervals, render visuals, and write report.
    Key Params: conditions, sample limit, seed, gradient limit 32, and visual target 120.
    State/Dependencies: reads checkpoint/alphabet/manifest/source images and writes only output-dir artifacts.
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    alphabet = list(args.alphabet.read_text(encoding="utf-8").splitlines()[0])
    all_conditions = load_conditions(Path(__file__).with_name("crnn_contract_conditions.yaml"))
    requested = {item.strip() for item in args.conditions.split(",")} if args.conditions != "all" else {condition["id"] for condition in all_conditions}
    conditions = [condition for condition in all_conditions if condition["id"] in requested]
    if not conditions:
        raise ValueError(f"No requested conditions found: {args.conditions}")
    records = load_manifest(args.manifest, args.max_samples, args.seed)
    snapshot_manifest(records, args.output_dir / "sample_manifest_snapshot.jsonl")
    manifest_report_path = args.manifest.with_suffix(".report.json")
    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8")) if manifest_report_path.is_file() else {"total_rows": len(records)}
    checkpoint_info = checkpoint_metadata(args.checkpoint, alphabet)
    runtime_info = runtime_metadata(device)
    code_files = {
        "diagnostic_script": sha256_file(Path(__file__)),
        "condition_config": sha256_file(Path(__file__).with_name("crnn_contract_conditions.yaml")),
        "crnn_source": sha256_file(Path(__file__).parents[2] / "realesrgan/metrics/crnn/crnn.py"),
    }
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unversioned_worktree"
    run_meta = {
        **runtime_info,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "alphabet": str(args.alphabet),
        "alphabet_length": len(alphabet),
        "alphabet_sha256": sha256_file(args.alphabet),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "code_revision": revision,
        "code_file_sha256": code_files,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "conditions_requested": sorted(requested),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(args.output_dir / "checkpoint_meta.json", checkpoint_info)
    write_json(args.output_dir / "run_meta.json", run_meta)
    log_path = args.output_dir / "logs" / "progress.jsonl"
    log_path.write_text("", encoding="utf-8")
    all_metrics: list[dict] = []
    all_activations: list[dict] = []
    all_cp: list[dict] = []
    all_gradients: list[dict] = []
    measured_conditions = []
    for condition in conditions:
        if condition["id"] == "C06-TATSR" and condition.get("range") == "best_eval":
            continue
        if condition["range"] == "best_eval":
            continue
        started = time.time()
        metric_rows, activation_rows, cp_rows, gradient_rows = run_condition(condition, records, args.checkpoint, alphabet, device, gradient_limit=32)
        all_metrics.extend(metric_rows)
        all_activations.extend(activation_rows)
        all_cp.extend(cp_rows)
        all_gradients.extend(gradient_rows)
        measured_conditions.append(condition)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"condition": condition["id"], "seconds": time.time() - started, "metric_rows": len(metric_rows), "cp_rows": len(cp_rows)}, ensure_ascii=False) + "\n")
    best_range = resolve_best_range(measured_conditions, all_metrics)
    for condition in conditions:
        if condition["id"] == "C06-TATSR" and condition.get("range") == "best_eval":
            condition["range"] = best_range
    remaining = [condition for condition in conditions if condition not in measured_conditions]
    for condition in remaining:
        started = time.time()
        metric_rows, activation_rows, cp_rows, gradient_rows = run_condition(condition, records, args.checkpoint, alphabet, device, gradient_limit=32)
        all_metrics.extend(metric_rows)
        all_activations.extend(activation_rows)
        all_cp.extend(cp_rows)
        all_gradients.extend(gradient_rows)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"condition": condition["id"], "seconds": time.time() - started, "metric_rows": len(metric_rows), "cp_rows": len(cp_rows), "resolved_range": condition["range"]}, ensure_ascii=False) + "\n")
    summaries = [aggregate_condition_summary(all_metrics, all_cp, all_activations, all_gradients, condition) for condition in conditions]
    interval_payload = {"comparisons": []}
    comparison_groups = [("C00-SRC", "C03-DYN-NORM"), ("C06-TATSR", "C03-DYN-NORM")]
    for fixed, aspect in comparison_groups:
        for group_key, group_name in ((None, "all"), ("language", "zh"), ("language", "en"), ("length_bucket", ">32"), ("structure_bucket", "high")):
            for metric in ("cer", "ned"):
                values = paired_metric_deltas(all_metrics, fixed, aspect, metric, group_key, group_name if group_key else None)
                interval_payload["comparisons"].append({"fixed": fixed, "aspect": aspect, "group": group_name, "metric": metric, "interval": bootstrap_interval(values, args.seed + len(interval_payload["comparisons"]))})
    write_csv(args.output_dir / "per_sample_metrics.csv", all_metrics)
    write_csv(args.output_dir / "layer_activation_stats.csv", all_activations)
    write_csv(args.output_dir / "cp_pair_ranking.csv", all_cp)
    write_csv(args.output_dir / "input_gradient_stats.csv", all_gradients)
    write_csv(args.output_dir / "condition_summary.csv", summaries)
    write_json(args.output_dir / "bootstrap_intervals.json", interval_payload)
    visual_count = 0
    if args.save_visuals:
        resolved = {condition["id"]: condition for condition in conditions}
        visual_mapping = render_visuals(records, args.output_dir / "visuals", resolved, args.seed + 99, args.visual_count)
        visual_count = len(visual_mapping)
    run_meta["resolved_c06_range"] = best_range
    run_meta["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_meta["visual_count"] = visual_count
    write_json(args.output_dir / "run_meta.json", run_meta)
    write_report(args.output_dir, run_meta, manifest_report, summaries, all_cp, interval_payload, visual_count, best_range)
    print(json.dumps({"output_dir": str(args.output_dir), "conditions": [row["condition"] for row in summaries], "samples": len(records), "visual_count": visual_count, "c06_range": best_range}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
