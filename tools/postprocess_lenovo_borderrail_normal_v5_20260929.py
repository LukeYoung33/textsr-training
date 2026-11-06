"""Apply frozen text-line V5 to the normal full-frame inference outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.text_lq_scurve_mild import V5_CONFIG, _luma_bgr, _run_normalized, preprocess_text_lq_bgr


DATA_ROOT = Path("/home/guang.yang/data/lenovo_mojito_textsr_20260928_whiteboard_v4")
MANIFEST = DATA_ROOT / "manifests/train.jsonl"
SOURCE_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_three_routes_20260929")
OUTPUT_ROOT = Path("/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/lenovo_borderrail_ft50k_whiteboard33_v5post_rawdelta_20260929")
ARMS = tuple(f"{arm}_h{height}" for arm in
             ("v5_borderrail", "v6_borderrail", "gtfit_v5_borderrail", "gtfit_v6_borderrail")
             for height in (96, 64))


def _records() -> list[dict]:
    """@luna-comment
    Input: The configured JSONL training manifest with lq/gt/ocr paths.
    Output: Exactly 33 distinct records with paths resolved under DATA_ROOT.
    Key Steps: Parse rows and reject an unexpected count or duplicate identity.
    Key Params: MANIFEST and DATA_ROOT bind the historical evaluation batch.
    State/Dependencies: Reads the manifest; does not modify records on disk.
    """
    rows = []
    for line in MANIFEST.open(encoding="utf-8"):
        row = json.loads(line)
        rows.append({"sample_id": row["sample_id"],
                     **{key: DATA_ROOT / row["files"][key] for key in ("lq", "gt", "ocr")}})
    if len(rows) != 33 or len({row["sample_id"] for row in rows}) != 33:
        raise ValueError("expected 33 distinct whiteboard images")
    return rows


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    """@luna-comment
    Input: OCR polygon or xyxy box and full-image dimensions in pixels.
    Output: Clipped expanded integer xyxy crop, or None for invalid geometry.
    Key Steps: Find extrema, validate extent, pad and clip to the image.
    Key Params: Horizontal 3% and vertical 12% context, at least two pixels.
    State/Dependencies: Pure NumPy geometry; no persistent state.
    """
    raw = item.get("polygon") or item.get("bbox_xyxy")
    if raw is None:
        return None
    points = np.asarray(raw, np.float32)
    if points.shape == (4,):
        x0, y0, x1, y1 = map(float, points)
    elif points.ndim == 2 and points.shape[1] == 2:
        x0, y0 = map(float, points.min(axis=0))
        x1, y1 = map(float, points.max(axis=0))
    else:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    pad_y = max(2, round((y1 - y0) * .12))
    pad_x = max(2, round((x1 - x0) * .03))
    return (max(0, int(np.floor(x0 - pad_x))), max(0, int(np.floor(y0 - pad_y))),
            min(width, int(np.ceil(x1 + pad_x))), min(height, int(np.ceil(y1 + pad_y))))


def _smoothstep(value: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: Array of normalized ramp coordinates.
    Output: Cubic ramp clipped to [0,1] with the same shape.
    Key Steps: Clip values and evaluate the cubic polynomial.
    Key Params: Unit endpoints determine the transition window.
    State/Dependencies: Pure NumPy computation without shared state.
    """
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


PASTE_MODE = "all_boxes_board_frame_rulefix_conservative_polarity_v6"


def _blend_weight(crop: np.ndarray) -> np.ndarray:
    """@luna-comment
    Input: HWC crop used to determine the pixel canvas dimensions.
    Output: HxW float32 rectangular feather weight in [0,1].
    Key Steps: Measure nearest-edge distance and apply a smooth ramp.
    Key Params: V5 border protection followed by a 32-pixel transition.
    State/Dependencies: Reads V5_CONFIG; no filesystem or persistent state.
    """
    height, width = crop.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    border_distance = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy)).astype(np.float32)
    return _smoothstep((border_distance - V5_CONFIG.border_px) / 32.0).astype(np.float32)


def _board_frame_weight(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """@luna-comment
    Input: Full uint8 BGR image, including any green chalkboard boundary.
    Output: Full-size float32 protection weight and board-detection report.
    Key Steps: Detect a large HSV green support, fit its hull, feather its perimeter.
    Key Params: One-eighth scale, color/area gates and small-map distance ramp.
    State/Dependencies: OpenCV/NumPy only; unqualified boards use unit weights.
    """
    height, width = image.shape[:2]
    small = cv2.resize(image, (max(1, width // 8), max(1, height // 8)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    board_color = ((hsv[:, :, 0] >= 55) & (hsv[:, :, 0] <= 105) &
                   (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 190))
    support = cv2.morphologyEx(board_color.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((height, width), np.float32), {"board_frame_protection": "no_board"}
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area_share = cv2.contourArea(largest) / float(small.shape[0] * small.shape[1])
    if area_share < 0.15 or w < small.shape[1] * 0.3 or h < small.shape[0] * 0.2:
        return np.ones((height, width), np.float32), {"board_frame_protection": "no_large_board"}
    mask = np.zeros(small.shape[:2], np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(largest)], -1, 1, thickness=-1)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    distance = inside + outside
    small_weight = _smoothstep((distance - 3.0) / 4.0).astype(np.float32)
    weight = cv2.resize(small_weight, (width, height), interpolation=cv2.INTER_LINEAR)
    return weight, {"board_frame_protection": "continuous_perimeter", "board_bbox_small_xywh": [x, y, w, h],
                    "board_area_share": round(area_share, 4), "protected_pixels": int(np.count_nonzero(weight < 0.5))}


def _process_every_box(crop: np.ndarray, background_sigma: float = 32.0,
                       suppress_background_noise: bool = False,
                       ambiguity_mode: str = "forced",
                       delta_filter: bool = False) -> tuple[np.ndarray, dict]:
    """@luna-comment
    Input: uint8 BGR OCR crop and ambiguity/residual-filter controls.
    Output: Same-size candidate BGR and effective processing-branch report.
    Key Steps: Try V5, force uncertain polarity or use bounded edges, optionally high-pass the delta.
    Key Params: ambiguity_mode selects the fallback; delta_filter enables background_sigma and optional edge support.
    State/Dependencies: Frozen V5_CONFIG and OpenCV/NumPy; source crop is unchanged.
    """
    candidate, telemetry = preprocess_text_lq_bgr(crop, config=V5_CONFIG, return_telemetry=True)
    initial_reason = telemetry.get("fallback_reason")
    mode = "v5"
    if initial_reason == "ambiguous_border_polarity" and ambiguity_mode == "conservative_edge":
        luminance = _luma_bgr(crop)
        low = cv2.GaussianBlur(luminance, (0, 0), 1.2)
        edge_delta = np.clip(0.3 * (luminance - low), -4.0, 4.0)
        candidate = np.clip(np.rint(crop.astype(np.float32) + edge_delta[:, :, None]), 0, 255).astype(np.uint8)
        mode = "conservative_edge"
        telemetry = {"fallback_reason": None}
    elif initial_reason is not None:
        luminance = _luma_bgr(crop)
        threshold, _ = cv2.threshold(luminance.astype(np.uint8), 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        polarity = 1 if np.median(luminance) >= threshold else -1
        normalized = luminance if polarity == 1 else 255.0 - luminance
        candidate, telemetry = _run_normalized(crop, normalized, polarity, V5_CONFIG)
        mode = "v5_forced_polarity"
        telemetry["forced_polarity"] = "dark_text_light_background" if polarity == 1 else "light_text_dark_background"
    if telemetry.get("fallback_reason") is not None:
        luminance = _luma_bgr(crop)
        low = cv2.GaussianBlur(luminance, (0, 0), 1.2)
        edge_delta = np.clip(0.65 * (luminance - low), -12.0, 12.0)
        candidate = np.clip(np.rint(crop.astype(np.float32) + edge_delta[:, :, None]), 0, 255).astype(np.uint8)
        mode = "bounded_edge"
    if not delta_filter:
        return candidate, {"processing_mode": mode, "initial_v5_reason": initial_reason,
                           "forced_v5_reason": telemetry.get("fallback_reason") if mode == "bounded_edge" else None,
                           "delta_filter": "none"}
    delta = candidate.astype(np.float32) - crop.astype(np.float32)
    low_frequency = cv2.GaussianBlur(delta, (0, 0), background_sigma)
    if suppress_background_noise:
        guide = cv2.GaussianBlur(_luma_bgr(crop), (0, 0), 1.3)
        grad_x = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        grad_y = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        edge_support = _smoothstep((np.maximum(np.abs(grad_x), np.abs(grad_y)) - 4.0) / 12.0)
    else:
        edge_support = np.ones(crop.shape[:2], np.float32)
    candidate = np.clip(np.rint(crop.astype(np.float32) +
                                 (delta - low_frequency) * edge_support[:, :, None]), 0, 255).astype(np.uint8)
    return candidate, {"processing_mode": mode, "initial_v5_reason": initial_reason,
                       "forced_v5_reason": telemetry.get("fallback_reason") if mode == "bounded_edge" else None,
                       "delta_filter": "highpass_edge"}


def _stabilize_ppt_delta(image: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, int]:
    """@luna-comment
    Input: Candidate and reference BGR images with matching dimensions.
    Output: Median-stabilized uint8 candidate and number of changed pixels.
    Key Steps: Median-filter each signed residual channel and reconstruct.
    Key Params: Five-pixel median window acts on residuals, not source pixels.
    State/Dependencies: OpenCV/NumPy only; both inputs remain unchanged.
    """
    delta = image.astype(np.float32) - reference.astype(np.float32)
    stable = np.stack([cv2.medianBlur(delta[:, :, channel], 5) for channel in range(3)], axis=2)
    result = np.clip(np.rint(reference.astype(np.float32) + stable), 0, 255).astype(np.uint8)
    return result, int(np.count_nonzero(np.any(result != image, axis=2)))


def _dehalo_horizontal_rules(image: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """@luna-comment
    Input: Candidate and reference full-frame BGR images.
    Output: Corrected BGR, applied-alpha map and per-rule geometric reports.
    Key Steps: Detect long dark horizontal rules, estimate upper background, return bright surroundings.
    Key Params: Width/height gates and distance/luminance ramps bound the edit support.
    State/Dependencies: OpenCV/NumPy only; input images are not mutated.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    dark = (gray < 70).astype(np.uint8)
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (max(160, width // 20), 1)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, 8)
    result = image.astype(np.float32)
    applied = np.zeros((height, width), np.uint8)
    reports = []
    for x, y, box_width, box_height, _ in stats[1:]:
        if box_width < width * 0.4 or box_height > 30 or y < 100 or y + box_height + 20 >= height:
            continue
        x0, x1 = max(0, x - 8), min(width, x + box_width + 8)
        y0, y1 = max(0, y - 20), min(height, y + box_height + 20)
        top = reference[max(0, y - 35):max(0, y - 20), x0:x1]
        if top.size == 0:
            continue
        background = np.median(top, axis=0).astype(np.float32)
        background = cv2.GaussianBlur(background[None, :, :], (0, 0), 15.0)[0]
        background_y = (background[:, 0] * 0.114 + background[:, 1] * 0.587 +
                        background[:, 2] * 0.299)
        current = result[y0:y1, x0:x1]
        current_y = cv2.cvtColor(np.clip(current, 0, 255).astype(np.uint8),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
        rows = np.arange(y0, y1)[:, None]
        distance = np.maximum(np.maximum(y - rows, rows - (y + box_height - 1)), 0)
        fade = 1.0 - _smoothstep((distance - 10.0) / 10.0)
        alpha = _smoothstep((current_y - background_y[None, :] - 2.0) / 8.0) * fade
        result[y0:y1, x0:x1] = current * (1.0 - alpha[:, :, None]) + background[None, :, :] * alpha[:, :, None]
        applied[y0:y1, x0:x1] = np.maximum(applied[y0:y1, x0:x1],
                                            np.rint(alpha * 255.0).astype(np.uint8))
        reports.append({"xywh": [int(x), int(y), int(box_width), int(box_height)],
                        "corrected_pixels": int(np.count_nonzero(alpha > 0.1))})
    return np.clip(np.rint(result), 0, 255).astype(np.uint8), applied, reports


def _preview(before: np.ndarray, after: np.ndarray, lq_path: Path, gt_path: Path,
             output: Path, before_label: str) -> None:
    """@luna-comment
    Input: Before/after BGR, matching LQ/GT file paths, output path and label.
    Output: Four-panel labeled JPEG written to output; no return value.
    Key Steps: Read references, verify shapes, resize and assemble panels.
    Key Params: Maximum panel width 850 pixels and JPEG quality 90.
    State/Dependencies: OpenCV image I/O; raises on unavailable or mismatched images.
    """
    lq = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    panels = [lq, before, after, gt]
    if any(panel is None or panel.shape != before.shape for panel in panels):
        raise ValueError("LQ/normal/postprocess/GT image shapes differ")
    target_width = min(850, before.shape[1])
    target_height = max(1, int(round(before.shape[0] * target_width / before.shape[1])))
    views = [cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_AREA) for panel in panels]
    title = np.full((34, 4 * target_width, 3), 245, np.uint8)
    for index, label in enumerate(("LQ", before_label, "V5 POST", "GT")):
        cv2.putText(title, label, (index * target_width + 12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (20, 20, 20), 2)
    cv2.imwrite(str(output), np.vstack((title, np.hstack(views))), [cv2.IMWRITE_JPEG_QUALITY, 90])


def apply_postprocess(before: np.ndarray, items: list[dict],
                      background_sigma: float = 32.0,
                      suppress_background_noise: bool = False,
                      ambiguity_mode: str = "forced",
                      delta_filter: bool = False) -> tuple[np.ndarray, list[dict], np.ndarray, dict]:
    """@luna-comment
    Input: Full uint8 BGR image, OCR items and ambiguity/residual controls.
    Output: Composited BGR, per-line reports, board weights and board report.
    Key Steps: Process expanded crops, average weighted signed changes, protect board perimeter and round once.
    Key Params: Overlap denominator is at least one; per-crop controls are forwarded unchanged.
    State/Dependencies: Frozen V5_CONFIG, OpenCV/NumPy; no filesystem writes or input mutation.
    """
    height, width = before.shape[:2]
    delta_sum = np.zeros((height, width, 3), np.float32)
    weight_sum = np.zeros((height, width), np.float32)
    line_reports = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("ignore") or not item.get("visible", True):
            continue
        box = _box(item, width, height)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        crop = before[y0:y1, x0:x1]
        candidate, processing = _process_every_box(crop, background_sigma=background_sigma,
                                                   suppress_background_noise=suppress_background_noise,
                                                   ambiguity_mode=ambiguity_mode,
                                                   delta_filter=delta_filter)
        changed = np.any(candidate != crop, axis=2)
        report = {"ocr_index": index, "crop_xyxy": list(box),
                  "raw_changed_pixels": int(changed.sum()), **processing}
        weight = _blend_weight(crop)
        delta_sum[y0:y1, x0:x1] += (candidate.astype(np.float32) - crop.astype(np.float32)) * weight[:, :, None]
        weight_sum[y0:y1, x0:x1] += weight
        report["blended_pixels"] = int(np.count_nonzero(weight > 0))
        line_reports.append(report)
    correction = delta_sum / np.maximum(weight_sum, 1.0)[:, :, None]
    frame_weight, frame_report = _board_frame_weight(before)
    correction *= frame_weight[:, :, None]
    after = np.clip(np.rint(before.astype(np.float32) + correction), 0, 255).astype(np.uint8)
    return after, line_reports, frame_weight, frame_report


def render_one(arm: str, row: dict, output_root: Path, source_root: Path = SOURCE_ROOT,
               route: str = "normal", ocr_from_source: bool = False,
               delta_filter: bool = False) -> dict:
    """@luna-comment
    Input: Arm/sample manifest, model-output root, route and OCR/filter controls.
    Output: Saved native postprocessed PNG, diagnostic previews/masks and JSON report.
    Key Steps: Reuse matching cache; load output/OCR; process lines; apply PPT-only repairs; export.
    Key Params: Raw V5 by default; conservative ambiguity fallback; optional residual filter.
    State/Dependencies: apply_postprocess and preview helpers; writes only selected output directory.
    """
    sample_id = row["sample_id"]
    source = source_root / arm / route / sample_id / "output.png"
    ocr_path = source.parent / "ocr_merged.json" if ocr_from_source else row["ocr"]
    directory = output_root / arm / sample_id
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    paste_mode = PASTE_MODE if delta_filter else PASTE_MODE + "_raw_v5"
    if result_path.is_file() and (directory / "output.png").is_file() and (directory / "preview.jpg").is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if (previous.get("paste_mode") == paste_mode and previous.get("source") == str(source)
                and previous.get("ocr_source") == str(ocr_path)
                and previous.get("preview_label") == route):
            return previous
    before = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if before is None:
        raise ValueError(f"cannot decode model output: {source}")
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    after, line_reports, frame_weight, frame_report = apply_postprocess(
        before, ocr.get("lines", []), background_sigma=4.0,
        suppress_background_noise=sample_id.startswith("lenovo_ppt"),
        ambiguity_mode="conservative_edge", delta_filter=delta_filter)
    if sample_id.startswith("lenovo_ppt"):
        after, stabilized_pixels = _stabilize_ppt_delta(after, before)
        after, rule_mask, rule_reports = _dehalo_horizontal_rules(after, before)
    else:
        stabilized_pixels = 0
        rule_mask = np.zeros(before.shape[:2], np.uint8)
        rule_reports = []
    cv2.imwrite(str(directory / "board_frame_protection.png"),
                np.rint((1.0 - frame_weight) * 255).astype(np.uint8))
    cv2.imwrite(str(directory / "horizontal_rule_dehalo.png"), rule_mask)
    output = directory / "output.png"
    if not cv2.imwrite(str(output), after):
        raise ValueError(f"cannot save output: {output}")
    cv2.imwrite(str(directory / "output_full.jpg"), after, [cv2.IMWRITE_JPEG_QUALITY, 94])
    _preview(before, after, row["lq"], row["gt"], directory / "preview.jpg", route.upper())
    record = {"arm": arm, "sample_id": sample_id, "source": str(source), "ocr_source": str(ocr_path),
              "output": str(output), "preview_label": route,
              "paste_mode": paste_mode,
              "board_frame": frame_report,
              "stabilized_delta_pixels": stabilized_pixels,
              "horizontal_rules": rule_reports,
              "lines": len(line_reports), "successful_lines": sum(report["processing_mode"] != "bounded_edge" for report in line_reports),
              "processed_lines": sum(report["blended_pixels"] > 0 and report["raw_changed_pixels"] > 0
                                     for report in line_reports),
              "changed_pixels": int(np.count_nonzero(np.any(after != before, axis=2))), "line_reports": line_reports}
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    """@luna-comment
    Input: CLI arm/sample/output/source/route/OCR-source and residual-filter options.
    Output: Per-image artifacts and arm summary JSON; prints progress.
    Key Steps: Parse options, bind manifest subset, execute render_one, publish summary.
    Key Params: Two OpenCV threads; historical default roots and eight arms.
    State/Dependencies: Reads configured inputs and writes explicit output tree.
    """
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--sample-id")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--route", default="normal")
    parser.add_argument("--ocr-from-source", action="store_true")
    parser.add_argument("--delta-filter", action="store_true",
                        help="Re-enable the previous high-pass and edge-support filtering for comparison")
    args = parser.parse_args()
    arms = (args.arm,) if args.arm else ARMS
    rows = _records()
    if args.sample_id:
        rows = [row for row in rows if row["sample_id"] == args.sample_id]
        if len(rows) != 1:
            raise ValueError("sample ID must match exactly once")
    for arm in arms:
        summaries = []
        for number, row in enumerate(rows, 1):
            result = render_one(arm, row, args.output_root, args.source_root, args.route,
                                args.ocr_from_source, args.delta_filter)
            summaries.append({key: result[key] for key in ("sample_id", "changed_pixels", "lines", "successful_lines")})
            print(json.dumps({"arm": arm, "image": number, "total": len(rows), **summaries[-1]}, ensure_ascii=False), flush=True)
        (args.output_root / arm / "summary.json").write_text(
            json.dumps({"arm": arm, "images": len(summaries), "rows": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8")


if __name__ == "__main__":
    main()
