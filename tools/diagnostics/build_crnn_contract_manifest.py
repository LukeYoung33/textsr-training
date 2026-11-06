#!/usr/bin/env python3
"""Build a frozen paired text-line manifest for the CRNN contract experiment."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_paths(root: Path) -> list[Path]:
    """@guang.yang-comment
    Input: an image-root directory.
    Output: sorted readable-candidate image paths; no files are modified.
    Key Steps: recurse, filter by supported suffix, and sort deterministically.
    Key Params: IMAGE_SUFFIXES defines the accepted image extensions.
    State/Dependencies: depends only on the local filesystem.
    """
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def annotation_candidates(
    image_path: Path, image_root: Path, annotation_root: Path
) -> list[Path]:
    """@guang.yang-comment
    Input: one image path and the corresponding annotation/image roots.
    Output: canonical TXT annotation candidates in priority order.
    Key Steps: preserve the image-relative path and try the repository's three naming conventions.
    Key Params: the ``_res``, same-stem, and ``res_`` variants cover current RealCE labels.
    State/Dependencies: does not read or write annotations.
    """
    relative = image_path.relative_to(image_root)
    base = (annotation_root / relative).with_suffix("")
    return [
        Path(f"{base}_res.txt"),
        base.with_suffix(".txt"),
        base.parent / f"res_{base.name}.txt",
    ]


def read_txt_annotation(path: Path) -> list[dict]:
    """@guang.yang-comment
    Input: a RealCE comma-separated TXT annotation file.
    Output: raw records containing four-point polygons and text strings.
    Key Steps: decode UTF-8 or GBK, parse eight coordinates, and retain the remaining comma-joined text.
    Key Params: records with fewer than eight coordinates are discarded before canonical validation.
    State/Dependencies: reads one annotation file and returns an in-memory list.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="gbk")
    records = []
    for line in content.splitlines():
        parts = [value.strip() for value in line.lstrip("\ufeff").split(",")]
        if len(parts) < 9:
            continue
        try:
            polygon = np.asarray(
                [float(value) for value in parts[:8]], dtype=np.float32
            ).reshape(4, 2)
        except ValueError:
            continue
        records.append({"polygon": polygon, "text": ",".join(parts[8:]).strip()})
    return records


def canonical_records(records: Iterable[dict], width: int, height: int) -> list[dict]:
    """@guang.yang-comment
    Input: raw polygon/text records and the source image dimensions.
    Output: finite, non-degenerate records with coordinates clipped to image bounds.
    Key Steps: reject empty text, malformed polygons, non-finite values, degenerate boxes, and out-of-domain coordinates.
    Key Params: one-pixel tolerance accepts annotation rounding at image borders.
    State/Dependencies: deterministic validation; the source records are not mutated.
    """
    valid = []
    for record in records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        try:
            polygon = np.asarray(record["polygon"], dtype=np.float32).reshape(4, 2)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(polygon).all():
            continue
        if polygon[:, 0].min() < -1 or polygon[:, 1].min() < -1:
            continue
        if polygon[:, 0].max() > width + 1 or polygon[:, 1].max() > height + 1:
            continue
        clipped = polygon.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
        box_width = float(clipped[:, 0].max() - clipped[:, 0].min())
        box_height = float(clipped[:, 1].max() - clipped[:, 1].min())
        if box_width < 2 or box_height < 2:
            continue
        valid.append({"polygon": clipped, "text": text})
    return valid


def load_annotations(
    image_path: Path, image_root: Path, annotation_root: Path, width: int, height: int
) -> tuple[Path | None, list[dict]]:
    """@guang.yang-comment
    Input: an image path, annotation roots, and image dimensions.
    Output: the first existing annotation path and its canonical records, or ``(None, [])``.
    Key Steps: try canonical filenames and validate records against the image domain.
    Key Params: candidate ordering matches the existing RealCE manifest tool.
    State/Dependencies: reads annotation files but does not alter them.
    """
    for candidate in annotation_candidates(image_path, image_root, annotation_root):
        if candidate.is_file():
            return candidate, canonical_records(
                read_txt_annotation(candidate), width=width, height=height
            )
    return None, []


def order_quad(points: np.ndarray) -> np.ndarray:
    """@guang.yang-comment
    Input: four two-dimensional polygon points.
    Output: points ordered top-left, top-right, bottom-right, bottom-left.
    Key Steps: use coordinate sums and differences, then fall back to angle sorting for ties.
    Key Params: the four-point contract is required by the perspective cropper.
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


def warp_polygon(image: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    """@guang.yang-comment
    Input: a BGR image and one four-point polygon in source-image coordinates.
    Output: a perspective-rectified BGR crop, or ``None`` for invalid geometry.
    Key Steps: order the quadrilateral, estimate target dimensions, and apply a linear perspective warp.
    Key Params: target dimensions use the larger opposing-edge lengths and are clamped to safe minimums.
    State/Dependencies: OpenCV allocates the returned crop; the input image is unchanged.
    """
    quad = order_quad(polygon)
    width = max(
        int(round(np.linalg.norm(quad[1] - quad[0]))),
        int(round(np.linalg.norm(quad[2] - quad[3]))),
    )
    height = max(
        int(round(np.linalg.norm(quad[3] - quad[0]))),
        int(round(np.linalg.norm(quad[2] - quad[1]))),
    )
    if width < 4 or height < 4:
        return None
    target = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR)


def normalise_text(text: str, alphabet: set[str]) -> str:
    """@guang.yang-comment
    Input: an annotation transcript and the checkpoint alphabet.
    Output: a transcript containing only supported characters.
    Key Steps: remove whitespace and unsupported code points while preserving character order.
    Key Params: the supplied alphabet is the single source for recognizer-compatible labels.
    State/Dependencies: pure string processing.
    """
    return "".join(char for char in text.replace("\u3000", " ").strip() if char in alphabet)


def classify_text(text: str) -> str:
    """@guang.yang-comment
    Input: a normalized transcript.
    Output: ``zh``, ``en``, or ``punct`` language bucket.
    Key Steps: detect CJK code points first, then ASCII letters/digits, and classify residual symbols as punctuation.
    Key Params: a mixed CJK/ASCII transcript remains in the Chinese bucket for conservative coverage accounting.
    State/Dependencies: Unicode code-point checks only.
    """
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    if any(char.isascii() and char.isalnum() for char in text):
        return "en"
    return "punct"


def length_bucket(length: int) -> str:
    """@guang.yang-comment
    Input: a non-negative transcript character count.
    Output: the plan-defined length bucket label.
    Key Steps: apply ordered inclusive boundaries.
    Key Params: Chinese and English share the common five bucket labels for cross-language reporting.
    State/Dependencies: pure arithmetic.
    """
    if length <= 4:
        return "1-4"
    if length <= 8:
        return "5-8"
    if length <= 16:
        return "9-16"
    if length <= 32:
        return "17-32"
    return ">32"


def quad_geometry(polygon: np.ndarray) -> tuple[int, int, float]:
    """@guang.yang-comment
    Input: one four-point polygon.
    Output: estimated original crop width, height, and width/height ratio.
    Key Steps: order the quadrilateral and average opposing-edge lengths.
    Key Params: a minimum height of one prevents division by zero for malformed labels.
    State/Dependencies: pure NumPy computation.
    """
    quad = order_quad(polygon)
    width = max(1.0, float((np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2))
    height = max(1.0, float((np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2))
    return int(round(width)), int(round(height)), width / height


def edge_density(crop: np.ndarray | None) -> float:
    """@guang.yang-comment
    Input: a BGR crop or ``None``.
    Output: Canny edge-pixel ratio used as a structure-density proxy.
    Key Steps: grayscale conversion, bounded Canny detection, and mean of the binary edge map.
    Key Params: Canny thresholds 80/160 are a fixed proxy, not a stroke annotation.
    State/Dependencies: OpenCV image conversion; the input is not mutated.
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))


def structure_bucket(value: float) -> str:
    """@guang.yang-comment
    Input: an edge-density proxy in [0, 1].
    Output: low, medium, or high structure-density bucket.
    Key Steps: apply fixed transparent thresholds.
    Key Params: 0.05 and 0.12 are reporting thresholds and are not stroke-level labels.
    State/Dependencies: pure thresholding.
    """
    if value < 0.05:
        return "low"
    if value < 0.12:
        return "medium"
    return "high"


def choose_stratified(candidates: list[dict], target: int, seed: int) -> list[dict]:
    """@guang.yang-comment
    Input: candidate text-line records, target count, and deterministic seed.
    Output: a shuffled, approximately language/length-balanced subset.
    Key Steps: shuffle each language-length bucket, round-robin buckets, then fill any shortfall from remaining records.
    Key Params: language targets are 50% Chinese, 35% English, and 15% punctuation when available.
    State/Dependencies: only local random state is used; input records are not modified.
    """
    rng = random.Random(seed)
    target = min(target, len(candidates))
    category_targets = {"zh": int(round(target * 0.50)), "en": int(round(target * 0.35))}
    category_targets["punct"] = max(0, target - category_targets["zh"] - category_targets["en"])
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate["language"], candidate["length_bucket"])].append(candidate)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict] = []
    used_ids: set[str] = set()
    for language in ("zh", "en", "punct"):
        language_buckets = [
            bucket for (bucket_language, _), bucket in sorted(buckets.items()) if bucket_language == language
        ]
        wanted = category_targets[language]
        while wanted and language_buckets:
            progressed = False
            for bucket in language_buckets:
                if bucket and wanted:
                    candidate = bucket.pop()
                    selected.append(candidate)
                    used_ids.add(candidate["candidate_id"])
                    wanted -= 1
                    progressed = True
            if not progressed:
                break
    if len(selected) < target:
        remaining = [candidate for candidate in candidates if candidate["candidate_id"] not in used_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])
    rng.shuffle(selected)
    return selected[:target]


def rectangle_iou(rectangle: np.ndarray, polygon: np.ndarray) -> float:
    """@guang.yang-comment
    Input: an axis-aligned rectangle and a text polygon.
    Output: IoU of their axis-aligned bounding boxes.
    Key Steps: convert both shapes to min/max bounds and compute intersection over union.
    Key Params: a zero union returns zero to keep background selection deterministic.
    State/Dependencies: pure NumPy arithmetic.
    """
    ax1, ay1, ax2, ay2 = rectangle[:, 0].min(), rectangle[:, 1].min(), rectangle[:, 0].max(), rectangle[:, 1].max()
    bx1, by1, bx2, by2 = polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def build_background_records(
    image_root: Path,
    clear_root: Path,
    annotation_root: Path,
    alphabet: set[str],
    target: int,
    seed: int,
) -> list[dict]:
    """@guang.yang-comment
    Input: the real-camera image root, optional clear-reference root, annotation root, alphabet, target, and seed.
    Output: deterministic no-transcript rectangle records that avoid valid text polygons when possible.
    Key Steps: scan images, generate a grid of rectangles, reject text-overlapping candidates, and retain fallback regions.
    Key Params: rectangle sizes are bounded at 256x96 and IoU must stay below 0.02 for annotated images.
    State/Dependencies: reads source images/annotations; no source files are changed.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    for image_path in image_paths(image_root):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        _, annotations = load_annotations(image_path, image_root, annotation_root, width, height)
        clear_path = clear_root / image_path.relative_to(image_root)
        if not clear_path.is_file():
            clear_path = image_path
        proposals = []
        rect_width = max(64, min(256, width // 4))
        rect_height = max(32, min(96, height // 12))
        for y_fraction in (0.12, 0.35, 0.58, 0.78):
            for x_fraction in (0.08, 0.38, 0.68):
                x = int(max(0, min(width - rect_width, width * x_fraction)))
                y = int(max(0, min(height - rect_height, height * y_fraction)))
                proposals.append((x, y, rect_width, rect_height))
        rng.shuffle(proposals)
        for x, y, rect_w, rect_h in proposals:
            rectangle = np.asarray([[x, y], [x + rect_w, y], [x + rect_w, y + rect_h], [x, y + rect_h]], dtype=np.float32)
            if any(rectangle_iou(rectangle, item["polygon"]) >= 0.02 for item in annotations):
                continue
            records.append({
                "candidate_id": f"bg-{image_path.stem}-{len(records):05d}",
                "pair_id": f"bg-pair-{image_path.stem}-{len(records):05d}",
                "source": "RealCE_train_52mm_no_text_proxy",
                "image_path_gt": str(clear_path),
                "image_path_lq": str(image_path),
                "polygon": rectangle.tolist(),
                "transcript": "",
                "transcript_raw": "",
                "language": "none",
                "character_count": 0,
                "length_bucket": "0",
                "is_lq": True,
                "is_gt": True,
                "mask_path": "",
            })
            if len(records) >= target:
                return records
    return records[:target]


def enrich_records(records: list[dict], seed: int) -> list[dict]:
    """@guang.yang-comment
    Input: selected manifest records with image paths and polygons.
    Output: records enriched with crop geometry, structure proxy, and degradation buckets.
    Key Steps: crop clear/real views, measure Canny density and Laplacian sharpness, then assign quantile buckets.
    Key Params: low/medium/high buckets use deterministic one-third quantiles; structure thresholds are fixed by structure_bucket.
    State/Dependencies: reads image pixels and returns new dictionaries without modifying source data.
    """
    del seed
    enriched = []
    degradation_values = []
    for index, record in enumerate(records):
        polygon = np.asarray(record["polygon"], dtype=np.float32)
        width, height, aspect_ratio = quad_geometry(polygon)
        clear_image = cv2.imread(record["image_path_gt"], cv2.IMREAD_COLOR)
        lq_image = cv2.imread(record["image_path_lq"], cv2.IMREAD_COLOR)
        clear_crop = warp_polygon(clear_image, polygon) if clear_image is not None else None
        lq_crop = warp_polygon(lq_image, polygon) if lq_image is not None else None
        structure = edge_density(clear_crop)
        gray_lq = cv2.cvtColor(lq_crop, cv2.COLOR_BGR2GRAY) if lq_crop is not None else np.zeros((1, 1), dtype=np.uint8)
        laplacian = float(cv2.Laplacian(gray_lq, cv2.CV_64F).var())
        degradation_score = 1.0 / (1.0 + laplacian)
        degradation_values.append(degradation_score)
        item = dict(record)
        item.update({
            "sample_id": f"sample-{index:05d}",
            "polygon": polygon.round(3).tolist(),
            "original_height": height,
            "original_width": width,
            "aspect_ratio": round(aspect_ratio, 6),
            "structure_density_proxy": round(structure, 8),
            "structure_bucket": structure_bucket(structure),
            "degradation_score_proxy": round(degradation_score, 10),
        })
        enriched.append(item)
    if degradation_values:
        first, second = np.quantile(np.asarray(degradation_values), [1 / 3, 2 / 3])
        for item in enriched:
            score = item["degradation_score_proxy"]
            item["degradation_bucket"] = "light" if score <= first else "medium" if score <= second else "heavy"
    return enriched


def parse_args() -> argparse.Namespace:
    """@guang.yang-comment
    Input: command-line arguments from the experiment launcher.
    Output: validated argument namespace for manifest construction.
    Key Steps: register paths, sample targets, and deterministic seed.
    Key Params: 400 text rows and 60 no-text rows are the plan defaults.
    State/Dependencies: argparse only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-root", required=True, type=Path)
    parser.add_argument("--real-root", required=True, type=Path)
    parser.add_argument("--annotation-root", required=True, type=Path)
    parser.add_argument("--alphabet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--text-samples", type=int, default=400)
    parser.add_argument("--no-text-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> None:
    """@guang.yang-comment
    Input: command-line paths to authorized RealCE data and output files.
    Output: a JSONL frozen manifest and a JSON construction audit report.
    Key Steps: load alphabet, create paired text/background records, enrich metadata, and serialize deterministically.
    Key Params: sample targets and seed come from parse_args and are recorded in the report.
    State/Dependencies: reads RealCE images/annotations and writes only the requested manifest/report paths.
    """
    args = parse_args()
    alphabet = set(args.alphabet.read_text(encoding="utf-8").splitlines()[0])
    candidates: list[dict] = []
    for real_path in image_paths(args.real_root):
        clear_path = args.clear_root / real_path.relative_to(args.real_root)
        if not clear_path.is_file():
            continue
        image = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        anno_path, records = load_annotations(real_path, args.real_root, args.annotation_root, image.shape[1], image.shape[0])
        mask_path = ""
        for candidate in (
            args.annotation_root.parent / "label_mask_52mm_renov" / real_path.name,
            args.annotation_root.parent / "label_mask_52mm_renov" / f"{real_path.stem}_mask.png",
        ):
            if candidate.is_file():
                mask_path = str(candidate)
                break
        del anno_path
        for line_index, record in enumerate(records):
            transcript = normalise_text(record["text"], alphabet)
            if not transcript:
                continue
            language = classify_text(transcript)
            polygon = np.asarray(record["polygon"], dtype=np.float32)
            candidates.append({
                "candidate_id": f"txt-{real_path.stem}-{line_index:04d}",
                "pair_id": f"pair-{real_path.stem}-{line_index:04d}",
                "source": "RealCE_train_documents_52mm_to_real_camera",
                "image_path_gt": str(clear_path),
                "image_path_lq": str(real_path),
                "polygon": polygon.tolist(),
                "transcript": transcript,
                "transcript_raw": record["text"],
                "language": language,
                "character_count": len(transcript),
                "length_bucket": length_bucket(len(transcript)),
                "is_lq": True,
                "is_gt": True,
                "mask_path": mask_path,
            })
    selected_text = choose_stratified(candidates, args.text_samples, args.seed)
    backgrounds = build_background_records(
        args.real_root,
        args.clear_root,
        args.annotation_root,
        alphabet,
        args.no_text_samples,
        args.seed + 1,
    )
    records = enrich_records(selected_text + backgrounds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    counts = Counter(record["language"] for record in records)
    report = {
        "seed": args.seed,
        "alphabet": str(args.alphabet),
        "clear_root": str(args.clear_root),
        "real_root": str(args.real_root),
        "annotation_root": str(args.annotation_root),
        "candidate_text_rows": len(candidates),
        "selected_text_rows": len(selected_text),
        "selected_background_rows": len(backgrounds),
        "total_rows": len(records),
        "language_counts": dict(counts),
        "length_counts": dict(Counter(record["length_bucket"] for record in records if record["language"] != "none")),
        "structure_counts": dict(Counter(record["structure_bucket"] for record in records)),
        "degradation_counts": dict(Counter(record["degradation_bucket"] for record in records)),
        "clear_reference_image_count": len({record["image_path_gt"] for record in selected_text}),
        "real_image_count": len({record["image_path_lq"] for record in records}),
        "structure_density_note": "Canny edge ratio is a structure-density proxy, not stroke-level annotation.",
        "manifest": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
