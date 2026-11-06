"""PP-OCRv6 tiny detection ONNX inference with DB postprocess.

The model returns a fixed-size probability map.  This script decodes it to
text quadrilaterals (or polygons) in the original image coordinate system.

Example (run in the paddleX environment):
    python deploy/infer_ppocrv6_tiny_det_onnx.py --image input.jpg --save-intermediates
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "deploy" / "ppocrv6_tiny_onnx" / "ppocrv6_tiny_det.onnx"


def preprocess(image: np.ndarray, limit_side_len: int, max_side_limit: int):
    """PP-OCRv6 tiny: BGR -> resize to multiples of 32 -> normalize -> NCHW."""
    src_h, src_w = image.shape[:2]
    ratio = limit_side_len / min(src_h, src_w) if min(src_h, src_w) < limit_side_len else 1.0
    resize_h, resize_w = int(src_h * ratio), int(src_w * ratio)
    if max(resize_h, resize_w) > max_side_limit:
        ratio = max_side_limit / max(resize_h, resize_w)
        resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    resized = cv2.resize(image, (resize_w, resize_h))
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = normalized.transpose(2, 0, 1)[None, ...]
    return resized, tensor, (src_h, src_w)


def mini_box(contour: np.ndarray):
    rect = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(rect)), key=lambda point: point[0])
    i1, i2, i3, i4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        i1, i4 = 0, 1
    else:
        i1, i4 = 1, 0
    if points[3][1] > points[2][1]:
        i2, i3 = 2, 3
    else:
        i2, i3 = 3, 2
    return np.asarray([points[i1], points[i2], points[i3], points[i4]], dtype=np.float32), min(rect[1])


def box_score_fast(probability: np.ndarray, box: np.ndarray) -> float:
    height, width = probability.shape
    box = box.copy()
    xmin, xmax = np.clip([np.floor(box[:, 0].min()), np.ceil(box[:, 0].max())], 0, width - 1).astype(np.int32)
    ymin, ymax = np.clip([np.floor(box[:, 1].min()), np.ceil(box[:, 1].max())], 0, height - 1).astype(np.int32)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] -= xmin
    box[:, 1] -= ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(probability[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def unclip(points: np.ndarray, unclip_ratio: float):
    polygon = Polygon(points)
    if polygon.area <= 0 or polygon.length <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(polygon.area * unclip_ratio / polygon.length)
    return np.asarray(expanded[0], dtype=np.float32) if len(expanded) == 1 else None


def to_original_coords(points: np.ndarray, map_shape, source_shape) -> np.ndarray:
    map_h, map_w = map_shape
    src_h, src_w = source_shape
    points = points.copy()
    points[:, 0] = np.clip(np.round(points[:, 0] / map_w * src_w), 0, src_w)
    points[:, 1] = np.clip(np.round(points[:, 1] / map_h * src_h), 0, src_h)
    return points.astype(np.int32)


def decode_db(probability, source_shape, thresh, box_thresh, unclip_ratio, box_type):
    """Equivalent DB postprocess, plus candidate records for debug rendering."""
    bitmap = probability > thresh
    contours, _ = cv2.findContours((bitmap.astype(np.uint8) * 255), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    records, core_polys, unclip_polys, polys, scores = [], [], [], [], []
    for index, contour in enumerate(contours[:3000]):
        record = {"contour_index": index, "contour": contour}
        if box_type == "quad":
            candidate, short_side = mini_box(contour)
            if short_side < 3:
                record["status"] = "reject_small_contour"
                records.append(record)
                continue
        else:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            candidate = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
            if len(candidate) < 4:
                record["status"] = "reject_too_few_vertices"
                records.append(record)
                continue
        record["candidate"] = candidate
        score = box_score_fast(probability, candidate)
        record["score"] = score
        if score < box_thresh:
            record["status"] = "reject_low_score"
            records.append(record)
            continue
        expanded = unclip(candidate, unclip_ratio)
        if expanded is None:
            record["status"] = "reject_unclip"
            records.append(record)
            continue
        record["unclip"] = expanded
        if box_type == "quad":
            expanded, short_side = mini_box(expanded.reshape(-1, 1, 2))
            if short_side < 5:
                record["status"] = "reject_small_unclip"
                records.append(record)
                continue
        core_original = to_original_coords(candidate, probability.shape, source_shape)
        unclip_original = to_original_coords(record["unclip"], probability.shape, source_shape)
        final_poly = to_original_coords(expanded, probability.shape, source_shape)
        record["core_poly"] = core_original
        record["unclip_poly"] = unclip_original
        record["status"] = "accept"
        record["final_poly"] = final_poly
        records.append(record)
        core_polys.append(core_original)
        unclip_polys.append(unclip_original)
        polys.append(final_poly)
        scores.append(score)
    return bitmap, records, core_polys, unclip_polys, polys, scores


def heatmap(probability):
    return cv2.applyColorMap(np.round(np.clip(probability, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def draw_stage(probability, records, stage):
    canvas = heatmap(probability)
    for record in records:
        accepted = record["status"] == "accept"
        color = (0, 190, 0) if accepted else (0, 0, 220)
        if stage == "contours":
            geometry = record["contour"]
        elif stage == "score":
            if "candidate" not in record:
                continue
            geometry = record["candidate"].reshape(-1, 1, 2).astype(np.int32)
        else:
            if not accepted:
                continue
            geometry = record["unclip"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [geometry], True, color, 2)
        if stage == "score" and "score" in record:
            x, y = record["candidate"].min(axis=0).astype(int)
            cv2.putText(canvas, f"{record['contour_index']}:{record['score']:.2f}", (x, max(20, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def draw_final(image, polys, scores):
    canvas = image.copy()
    for index, (poly, score) in enumerate(zip(polys, scores)):
        cv2.polylines(canvas, [poly.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
        x, y = poly.min(axis=0)
        cv2.putText(canvas, f"{index}:{score:.2f}", (int(x), max(20, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="PP-OCRv6 tiny det ONNX inference and DB decode")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("output") / "ppocrv6_tiny_det_result.json")
    parser.add_argument("--box-type", choices=("quad", "poly"), default="quad")
    parser.add_argument("--thresh", type=float, default=0.20, help="Probability-map binarization threshold")
    parser.add_argument("--box-thresh", type=float, default=0.40, help="Candidate mean-score threshold")
    parser.add_argument("--unclip-ratio", type=float, default=1.40)
    parser.add_argument("--limit-side-len", type=int, default=736)
    parser.add_argument("--max-side-limit", type=int, default=4000)
    parser.add_argument("--save-intermediates", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    resized, tensor, source_shape = preprocess(image, args.limit_side_len, args.max_side_limit)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    probability = session.run(None, {session.get_inputs()[0].name: tensor})[0][0, 0]
    bitmap, records, core_polys, unclip_polys, polys, scores = decode_db(probability, source_shape, args.thresh, args.box_thresh, args.unclip_ratio, args.box_type)

    result = {
        "image_path": str(Path(args.image)),
        "model_path": str(args.model),
        "input_shape_nchw": list(tensor.shape),
        "source_shape_hwc": list(image.shape),
        "postprocess": {"thresh": args.thresh, "box_thresh": args.box_thresh, "unclip_ratio": args.unclip_ratio, "box_type": args.box_type},
        "core_polys": [poly.tolist() for poly in core_polys],
        "unclip_polys": [poly.tolist() for poly in unclip_polys],
        "dt_polys": [poly.tolist() for poly in polys],
        "dt_scores": [round(float(score), 6) for score in scores],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_intermediates:
        debug_dir = args.output.parent / f"{args.output.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        np.save(debug_dir / "input_nchw.npy", tensor)
        np.save(debug_dir / "probability_map.npy", probability)
        cv2.imwrite(str(debug_dir / "00_source.jpg"), image)
        cv2.imwrite(str(debug_dir / "01_resized_bgr.jpg"), resized)
        cv2.imwrite(str(debug_dir / "02_probability_map.png"), heatmap(probability))
        cv2.imwrite(str(debug_dir / f"03_binary_mask_thresh_{args.thresh:.2f}.png"), bitmap.astype(np.uint8) * 255)
        cv2.imwrite(str(debug_dir / "04_contours.png"), draw_stage(probability, records, "contours"))
        cv2.imwrite(str(debug_dir / "05_score_filter.png"), draw_stage(probability, records, "score"))
        cv2.imwrite(str(debug_dir / "06_unclip.png"), draw_stage(probability, records, "unclip"))
        cv2.imwrite(str(debug_dir / "07_final_polys.jpg"), draw_final(image, polys, scores))
        result["debug_dir"] = str(debug_dir.resolve())
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Detected {len(polys)} text regions. JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()


