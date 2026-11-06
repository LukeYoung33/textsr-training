"""Reference BASIC V2 deployment pipeline.

The module intentionally keeps pure geometry/tensor helpers independent from
ONNX Runtime so they can be tested with fake detector/restorer runners.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


DET_HW = (768, 1024)
CANVAS_HW = (96, 480)
CONTENT_H = 72
Y4UV_SHAPE = (1, 12, 48, 240)
SKIP_SHAPE = (1, 3, 96, 480)
OUT_SHAPE = (1, 3, 96, 480)
DEFAULT_FEATHER_WIDTH = 16


@dataclass(frozen=True)
class RestoreGeometry:
    """@shanlan-comment
    Input: restoration content height, tile overlap, resize kernel, and width alignment policy.
    Output: immutable geometry contract passed through the full-image restoration path.
    Key Steps: group preprocessing values so adapters cannot update trace metadata without changing pixels.
    Key Params: legacy defaults remain 72-pixel content, 96 overlap, and bicubic resize.
    State/Dependencies: OpenCV interpolation constants; no mutable state.
    """

    content_h: int = CONTENT_H
    overlap: int = 96
    interpolation: int = cv2.INTER_CUBIC
    force_even_width: bool = False


@dataclass
class YUV420Frame:
    """@shanlan-comment
    Input: planar uint8 YUV420 arrays and color metadata.
    Output: explicit frame object.
    Key Steps: retain full-resolution Y and half-resolution U/V.
    Key Params: matrix, range and chroma siting.
    State/Dependencies: NumPy arrays owned by caller.
    """
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    matrix: str = "bt601"
    range: str = "limited"
    chroma_siting: str = "center"


@dataclass
class Detection:
    """@shanlan-comment
    Input: decoded polygon, score and status.
    Output: serializable detection record.
    Key Steps: retain reason and raw score for trace.
    Key Params: rejection reason codes.
    State/Dependencies: NumPy polygon converted before JSON serialization.
    """
    polygon: np.ndarray
    score: float
    raw_score: float = 0.0
    status: str = "accepted"
    reason: str = ""


@dataclass
class ROIResult:
    """@shanlan-comment
    Input: restored inverse-oriented YUV444 ROI and source geometry.
    Output: immutable ROI contribution record for unordered full-frame composition.
    Key Steps: retain score, alpha polygon, Global-S trace and fallback state.
    Key Params: source bbox and orientation define inverse placement.
    State/Dependencies: NumPy arrays are local scratch owned by this result.
    """
    yuv444: np.ndarray
    polygon: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    alpha: np.ndarray
    global_s: dict[str, Any]
    orientation: str = "horizontal"
    fallback: str = ""


def bgr_to_yuv420(bgr: np.ndarray) -> YUV420Frame:
    """@shanlan-comment
    Input: uint8 BGR HxWx3 with even H/W.
    Output: planar uint8 Y, U, V at full/half resolution.
    Key Steps: convert through OpenCV BT.601 limited-range I420 and split planes.
    Key Params: input dimensions must be even.
    State/Dependencies: OpenCV color conversion; metadata is explicit.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("BGR input must be HxWx3")
    h, w = bgr.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("YUV420 requires even source dimensions")
    packed = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    return YUV420Frame(packed[:h].copy(), packed[h:h + h // 4].reshape(h // 2, w // 2).copy(),
                       packed[h + h // 4:].reshape(h // 2, w // 2).copy())


def yuv420_to_bgr(frame: YUV420Frame) -> np.ndarray:
    """@shanlan-comment
    Input: planar uint8 YUV420Frame with matching dimensions.
    Output: uint8 BGR image at Y resolution.
    Key Steps: pack I420 planes and call OpenCV conversion.
    Key Params: BT.601 limited-range representation.
    State/Dependencies: OpenCV conversion only.
    """
    h, w = frame.y.shape
    packed = np.concatenate([frame.y.reshape(-1), frame.u.reshape(-1), frame.v.reshape(-1)])
    return cv2.cvtColor(packed.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)


def yuv444_to_bgr(yuv444: np.ndarray) -> np.ndarray:
    """@shanlan-comment
    Input: YUV444 channel-first uint8 or normalized float tensor.
    Output: uint8 BGR preview.
    Key Steps: convert channel-first to HWC and use OpenCV YUV conversion.
    Key Params: normalized values are scaled to 0..255.
    State/Dependencies: OpenCV color conversion.
    """
    arr=np.asarray(yuv444)
    if arr.dtype.kind == "f" and arr.max() <= 1.5: arr=np.clip(arr*255,0,255)
    return cv2.cvtColor(np.transpose(np.clip(arr,0,255).astype(np.uint8),(1,2,0)),cv2.COLOR_YUV2BGR)


def read_yuv420(path: str | Path, width: int, height: int, *, matrix: str = "bt601",
                range_name: str = "limited") -> YUV420Frame:
    """@shanlan-comment
    Input: raw planar I420 file, even width/height, and color metadata.
    Output: explicit YUV420Frame with Y full-resolution and U/V half-resolution.
    Key Steps: validate exact byte count and split the I420 payload.
    Key Params: width/height and matrix/range are recorded, never inferred.
    State/Dependencies: local file read only; rejects truncated or trailing bytes.
    """
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("raw YUV420 dimensions must be positive and even")
    payload = Path(path).read_bytes(); expected = width * height * 3 // 2
    if len(payload) != expected:
        raise ValueError(f"raw YUV420 byte count {len(payload)} != expected {expected}")
    arr = np.frombuffer(payload, np.uint8); y_end = width * height; c_size = width * height // 4
    y = arr[:y_end].reshape(height, width).copy()
    u = arr[y_end:y_end + c_size].reshape(height // 2, width // 2).copy()
    v = arr[y_end + c_size:].reshape(height // 2, width // 2).copy()
    return YUV420Frame(y, u, v, matrix, range_name, "center")


def write_yuv420(path: str | Path, frame: YUV420Frame) -> None:
    """@shanlan-comment
    Input: planar YUV420Frame and destination path.
    Output: exact planar I420 byte stream on disk.
    Key Steps: concatenate Y/U/V planes after shape validation.
    Key Params: no color conversion or resampling.
    State/Dependencies: destination parent is created.
    """
    h,w=frame.y.shape
    if frame.u.shape!=(h//2,w//2) or frame.v.shape!=frame.u.shape: raise ValueError("invalid YUV420 plane shapes")
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(np.concatenate([frame.y.ravel(),frame.u.ravel(),frame.v.ravel()]).tobytes())


def run_bgr_adapter(bgr: np.ndarray, detector: Any, restorer: Runner, debug_dir: str | Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """@shanlan-comment
    Input: JPG/BGR test adapter image and pipeline runners.
    Output: cropped-back BGR preview and trace.
    Key Steps: pad odd dimensions, convert to YUV420 once, process authoritative frame.
    Key Params: border replicate and original shape restoration.
    State/Dependencies: process_yuv420 owns all production processing.
    """
    frame, preview, trace = run_bgr_adapter_frame(bgr, detector, restorer, debug_dir=debug_dir)
    return preview, trace


def run_bgr_adapter_frame(
    bgr: np.ndarray,
    detector: Any,
    restorer: Runner,
    debug_dir: str | Path | None = None,
    *,
    restore_geometry: RestoreGeometry | None = None,
) -> tuple[YUV420Frame, np.ndarray, dict[str, Any]]:
    """@shanlan-comment
    Input: uint8 BGR image, including odd dimensions.
    Output: authoritative even YUV420 frame, cropped BGR preview, JSON trace.
    Key Steps: replicate-pad once, convert once, process frame, crop preview.
    Key Params: original dimensions and the effective restoration geometry are retained in trace.
    State/Dependencies: process_yuv420 is the only restoration path.
    """
    h,w=bgr.shape[:2]
    padded=cv2.copyMakeBorder(bgr,0,h%2,0,w%2,cv2.BORDER_REPLICATE) if h%2 or w%2 else bgr
    frame=bgr_to_yuv420(padded)
    out,trace=process_yuv420(
        frame,
        detector,
        restorer,
        debug_dir=debug_dir,
        restore_geometry=restore_geometry,
    )
    trace["source_adapter"]="BGR->YUV420 test adapter"; trace["original_hw"]=[h,w]; trace["padded_hw"]=list(padded.shape[:2])
    return out,yuv420_to_bgr(out)[:h,:w],trace


def detector_normalize_bgr(bgr: np.ndarray) -> np.ndarray:
    """@shanlan-comment
    Input: uint8 BGR image of any size.
    Output: float32 NCHW tensor [1,3,768,1024].
    Key Steps: direct resize, divide by 255, subtract mean, divide std, transpose.
    Key Params: BGR mean [.485,.456,.406], std [.229,.224,.225].
    State/Dependencies: fixed direct resize, no RGB swap or letterbox.
    """
    image = cv2.resize(bgr, (DET_HW[1], DET_HW[0]), interpolation=cv2.INTER_LINEAR)
    x = image.astype(np.float32) / 255.0
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.transpose(x, (2, 0, 1))[None]


def _unclip(poly: np.ndarray, ratio: float) -> np.ndarray:
    """@shanlan-comment
    Input: convex quadrilateral in map coordinates.
    Output: one expanded path from polygon offset.
    Key Steps: compute area/perimeter distance and expand rectangle.
    Key Params: distance=area*ratio/perimeter.
    State/Dependencies: Shapely area/perimeter and Pyclipper offset.
    """
    polygon = Polygon(np.asarray(poly, np.float64))
    area = abs(float(polygon.area)); peri = float(polygon.length)
    if not polygon.is_valid or peri <= 1e-6 or area <= 1e-6:
        return np.empty((0, 2), np.float32)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath([(int(round(x * 1000)), int(round(y * 1000))) for x, y in poly],
                   pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(area * ratio / peri * 1000.0)
    if len(paths) != 1:
        return np.empty((0, 2), np.float32)
    result = np.asarray(paths[0], np.float32) / 1000.0
    return result


def _stable_quad(poly: np.ndarray) -> np.ndarray:
    """@shanlan-comment
    Input: four-point polygon in arbitrary cyclic order.
    Output: clockwise quad starting at the top-left-like minimum x+y point.
    Key Steps: convex hull, centroid-angle sort, deterministic rotation.
    Key Params: exactly four output points.
    State/Dependencies: NumPy/OpenCV geometry.
    """
    q = cv2.boxPoints(cv2.minAreaRect(np.asarray(poly, np.float32)))
    order = np.argsort(q[:, 0]); left, right = q[order[:2]], q[order[2:]]
    left = left[np.argsort(left[:, 1])]; right = right[np.argsort(right[:, 1])]
    return np.asarray([left[0], right[0], right[1], left[1]], np.float32)


def decode_db_probability(probability: np.ndarray, *, thresh: float = .20,
                          box_thresh: float = .40, unclip_ratio: float = 1.40,
                          max_candidates: int = 3000) -> list[Detection]:
    """@shanlan-comment
    Input: probability map [1,1,H,W] or [H,W], finite float values.
    Output: ordered quad detections with score/status/reason fields.
    Key Steps: threshold, RETR_LIST contours, short-side 3 gate, fast masked
      mean, score gate, one-path unclip, short-side 5 gate.
    Key Params: thresh=.20, box_thresh=.40, unclip_ratio=1.40, max=3000;
      no dilation and no NMS.
    State/Dependencies: OpenCV contour/minAreaRect; map coordinates are retained.
    """
    p = np.asarray(probability, dtype=np.float32)
    if p.ndim == 4: p = p[0, 0]
    if p.ndim != 2: raise ValueError("probability must be [1,1,H,W] or [H,W]")
    if not np.isfinite(p).all(): raise ValueError("probability map contains NaN/Inf")
    mask = (p > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Detection] = []
    for contour in contours[:max_candidates]:
        rect = cv2.minAreaRect(contour)
        poly = _stable_quad(cv2.boxPoints(rect))
        if min(rect[1]) < 3:
            out.append(Detection(poly, 0., 0., "rejected", "DET_SMALL_CONTOUR")); continue
        inside = np.zeros_like(mask); cv2.fillPoly(inside, [np.rint(poly).astype(np.int32)], 1)
        score = float(p[inside.astype(bool)].mean()) if inside.any() else 0.
        if score < box_thresh:
            out.append(Detection(poly, score, score, "rejected", "DET_LOW_SCORE")); continue
        expanded = _unclip(poly, unclip_ratio)
        if len(expanded) < 4:
            out.append(Detection(poly, score, score, "rejected", "DET_UNCLIP_FAILED")); continue
        expanded_rect=cv2.boxPoints(cv2.minAreaRect(expanded.astype(np.float32)))
        if min(cv2.minAreaRect(expanded_rect)[1]) < 5:
            out.append(Detection(expanded, score, score, "rejected", "DET_SMALL_UNCLIP")); continue
        out.append(Detection(_stable_quad(expanded_rect), score, score, "accepted", "accepted"))
    return out


def map_polygon_to_source(poly: np.ndarray, source_hw: tuple[int, int], map_hw: tuple[int, int],
                          detector_hw: tuple[int, int] = DET_HW) -> np.ndarray:
    """@shanlan-comment
    Input: detector-map polygon, source H/W, probability-map H/W.
    Output: integer source polygon, rounded and clipped to image bounds.
    Key Steps: map->detector fixed-resize->source independent x/y scales.
    Key Params: direct-resize detector dimensions are 1024x768.
    State/Dependencies: no aspect-preserving padding.
    """
    sh, sw = source_hw; mh, mw = map_hw
    q = np.asarray(poly, np.float32).copy()
    q[:, 0] *= detector_hw[1] / mw / (detector_hw[1] / sw)
    q[:, 1] *= detector_hw[0] / mh / (detector_hw[0] / sh)
    q[:, 0] = np.clip(np.rint(q[:, 0]), 0, sw - 1)
    q[:, 1] = np.clip(np.rint(q[:, 1]), 0, sh - 1)
    return q.astype(np.int32)


def filter_detection(poly: np.ndarray, source_hw: tuple[int, int]) -> tuple[bool, str]:
    """@shanlan-comment
    Input: source-coordinate polygon and source H/W.
    Output: acceptance and one deterministic business reason.
    Key Steps: bounding width/height/area gates.
    Key Params: minimum height 12, width 24, area 288 source pixels.
    State/Dependencies: preserves the first rejection reason.
    """
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    area = abs(cv2.contourArea(poly.astype(np.float32)))
    if h < 12: return False, "small_height"
    if w < 24: return False, "small_width"
    if area < 288: return False, "small_area"
    return True, "accepted"


def crop_even_roi(frame: YUV420Frame, poly: np.ndarray, context_ratio: float = .15) -> tuple[YUV420Frame, tuple[int, int, int, int]]:
    """@shanlan-comment
    Input: planar YUV420 frame and source polygon.
    Output: even-aligned YUV420 ROI and (x,y,w,h) source bbox.
    Key Steps: bound polygon, expand to even coordinates, crop Y and chroma.
    Key Params: coordinates clipped to source; no implicit context padding.
    State/Dependencies: chroma crop uses half-resolution coordinates.
    """
    h, w = frame.y.shape; x, y, rw, rh = cv2.boundingRect(poly.astype(np.int32))
    context = max(int(round(rh * context_ratio)), DEFAULT_FEATHER_WIDTH + 2)
    px, py = context, context
    x -= px; y -= py; rw += 2*px; rh += 2*py
    x = max(0, x & ~1); y = max(0, y & ~1); x2 = min(w, (x + rw + 1) & ~1); y2 = min(h, (y + rh + 1) & ~1)
    if x2 <= x or y2 <= y: raise ValueError("empty ROI")
    return YUV420Frame(frame.y[y:y2, x:x2].copy(), frame.u[y//2:y2//2, x//2:x2//2].copy(),
                       frame.v[y//2:y2//2, x//2:x2//2].copy(), frame.matrix, frame.range, frame.chroma_siting), (x, y, x2-x, y2-y)


def normalize_roi_orientation(roi: YUV420Frame, polygon: np.ndarray) -> tuple[YUV420Frame, np.ndarray, str]:
    """@shanlan-comment
    Input: YUV420 ROI and polygon in ROI coordinates.
    Output: horizontal-oriented ROI, transformed polygon, and orientation tag.
    Key Steps: rotate tall text clockwise; retain inverse tag for composition.
    Key Params: vertical when polygon bbox height exceeds width by 1.2x.
    State/Dependencies: OpenCV plane rotation; chroma rotates independently.
    """
    _, _, w, h = cv2.boundingRect(polygon.astype(np.int32))
    if h <= 1.2 * max(w, 1): return roi, polygon, "horizontal"
    rotated = YUV420Frame(cv2.rotate(roi.y, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(roi.u, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(roi.v, cv2.ROTATE_90_CLOCKWISE), roi.matrix, roi.range, roi.chroma_siting)
    original_h = roi.y.shape[0]
    q = np.stack([original_h - 1 - polygon[:, 1], polygon[:, 0]], 1).astype(np.int32)
    if np.any(q[:, 0] < 0) or np.any(q[:, 0] >= rotated.y.shape[1]) or np.any(q[:, 1] < 0) or np.any(q[:, 1] >= rotated.y.shape[0]):
        raise ValueError("rotated polygon out of bounds")
    return rotated, q, "rotate_cw"


def inverse_orientation_yuv444(yuv: np.ndarray, orientation: str) -> np.ndarray:
    """@shanlan-comment
    Input: restored YUV444 channel-first ROI and orientation tag.
    Output: restored ROI in source bbox orientation.
    Key Steps: apply counter-clockwise rotation for clockwise-normalized input.
    Key Params: horizontal is identity.
    State/Dependencies: OpenCV rotates each channel consistently.
    """
    if orientation == "horizontal": return yuv
    if orientation != "rotate_cw": raise ValueError(f"unknown orientation {orientation}")
    return np.stack([cv2.rotate(ch, cv2.ROTATE_90_COUNTERCLOCKWISE) for ch in yuv], axis=0)


def resize_pad_tiles(
    roi: YUV420Frame,
    *,
    content_h: int = CONTENT_H,
    canvas_hw: tuple[int, int] = CANVAS_HW,
    overlap: int = 96,
    interpolation: int = cv2.INTER_CUBIC,
    force_even_width: bool = False,
) -> list[dict[str, Any]]:
    """@shanlan-comment
    Input: even YUV420 ROI.
    Output: tile records with resized Y/U/V, valid rect, geometry metadata, and edge padding.
    Key Steps: aspect-resize to the selected content height, pad vertically, and overlap-slice long rows.
    Key Params: defaults preserve the legacy 96x480 canvas, height 72, overlap 96, and bicubic kernel.
    State/Dependencies: all tiles retain source scale and valid width metadata.
    """
    if content_h <= 0 or content_h > canvas_hw[0] or content_h % 2:
        raise ValueError("content height must be positive, even, and no larger than the canvas")
    if canvas_hw[0] % 2 or canvas_hw[1] % 2 or overlap < 0 or overlap >= canvas_hw[1]:
        raise ValueError("canvas dimensions must be even and overlap must be within the tile width")
    h, w = roi.y.shape; scale = content_h / float(h); ew = max(1, int(round(w * scale)))
    if force_even_width:
        ew = max(2, ew + (ew % 2))
    yoff=(canvas_hw[0]-content_h)//2; uoff=yoff//2
    y = cv2.resize(roi.y, (ew, content_h), interpolation=interpolation)
    u = cv2.resize(roi.u, (max(1, ew//2), content_h//2), interpolation=interpolation)
    v = cv2.resize(roi.v, (max(1, ew//2), content_h//2), interpolation=interpolation)
    canvas_w = canvas_hw[1]; step = max(1, canvas_w - overlap)
    starts = [0] if ew <= canvas_w else list(range(0, ew - canvas_w + 1, step))
    if ew > canvas_w and starts[-1] != ew - canvas_w: starts.append(ew - canvas_w)
    result = []
    for start in starts:
        valid_w = min(canvas_w, ew - start); yc = np.empty(canvas_hw, np.uint8); yc[:] = y[0,0]; yc[yoff:yoff+content_h,:valid_w] = y[:,start:start+valid_w]
        if valid_w < canvas_w: yc[yoff:yoff+content_h,valid_w:] = yc[yoff:yoff+content_h,valid_w-1:valid_w]
        yc[:yoff] = yc[yoff:yoff+1]; yc[yoff+content_h:] = yc[yoff+content_h-1:yoff+content_h]
        chroma_valid_w = max(1, (valid_w + 1) // 2)
        uc = np.empty((canvas_hw[0] // 2, canvas_w // 2), np.uint8)
        vc = np.empty_like(uc)
        u_slice = u[:, start // 2:start // 2 + chroma_valid_w]
        v_slice = v[:, start // 2:start // 2 + chroma_valid_w]
        uc[:] = u[0,0]; vc[:] = v[0,0]
        uc[uoff:uoff+content_h // 2, :u_slice.shape[1]] = u_slice
        vc[uoff:uoff+content_h // 2, :v_slice.shape[1]] = v_slice
        if u_slice.shape[1] < canvas_w // 2:
            uc[uoff:uoff+content_h // 2, u_slice.shape[1]:] = u_slice[:, -1:]
            vc[uoff:uoff+content_h // 2, v_slice.shape[1]:] = v_slice[:, -1:]
        uc[:uoff] = uc[uoff:uoff+1]; uc[uoff+content_h//2:] = uc[uoff+content_h//2-1:uoff+content_h//2]
        vc[:uoff] = vc[uoff:uoff+1]; vc[uoff+content_h//2:] = vc[uoff+content_h//2-1:uoff+content_h//2]
        result.append({"y": yc, "u": uc, "v": vc, "start": start, "line_start_x": start, "valid_w": valid_w,
                       "scale": scale, "valid_rect": (0, yoff, valid_w, content_h), "padding": (yoff, canvas_hw[0]-yoff-content_h),
                       "overlap": overlap, "interpolation": int(interpolation), "force_even_width": bool(force_even_width)})
    return result


def merge_restore_tiles(tiles: list[dict[str, Any]], outputs: list[np.ndarray]) -> np.ndarray:
    """@shanlan-comment
    Input: tile metadata and matching [1,3,96,480] restoration outputs.
    Output: merged YUV444 line [3,content_height,line_width], excluding vertical padding.
    Key Steps: derive valid rows from tile metadata, apply overlap ramps, and normalize accumulated weights.
    Key Params: valid_rect, line_start_x, valid_w, and per-tile overlap define placement.
    State/Dependencies: raises on shape mismatch or uncovered line pixels.
    """
    if len(tiles) != len(outputs) or not tiles: raise ValueError("tile/output count mismatch")
    valid_rects=[t.get("valid_rect",(0,12,int(t["valid_w"]),72)) for t in tiles]
    valid_heights={int(rect[3]) for rect in valid_rects}; y_offsets={int(rect[1]) for rect in valid_rects}
    if len(valid_heights) != 1 or len(y_offsets) != 1: raise ValueError("tile valid-row geometry mismatch")
    content_h=valid_heights.pop(); yoff=y_offsets.pop()
    line_w=max(t["line_start_x"]+t["valid_w"] for t in tiles); total=np.zeros((3,content_h,line_w),np.float32); weights=np.zeros((content_h,line_w),np.float32)
    for tile,out in zip(tiles,outputs):
        if out.shape != OUT_SHAPE: raise ValueError(f"tile output shape {out.shape}")
        st=tile["line_start_x"]; vw=tile["valid_w"]; blend_width=int(tile.get("overlap",96)); ramp=np.ones(vw,np.float32)
        if st > 0: ramp[:min(blend_width,vw)] = np.linspace(0,1,min(blend_width,vw),endpoint=False)
        if st+vw < line_w: ramp[-min(blend_width,vw):] = np.minimum(ramp[-min(blend_width,vw):],np.linspace(1,0,min(blend_width,vw),endpoint=False))
        crop=out[0,:,yoff:yoff+content_h,:vw]; total[:,:,st:st+vw]+=crop*ramp[None,None,:]; weights[:,st:st+vw]+=ramp[None,:]
    if np.any(weights <= 0): raise ValueError("tile merge has uncovered pixels")
    return total/weights[None]


def build_y4uv_skip(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """@shanlan-comment
    Input: uint8 tile planes Y[96,480], U/V[48,240].
    Output: float32 y4uv [1,12,48,240] and skip [1,3,96,480].
    Key Steps: torch-compatible 2x2 PixelUnshuffle Y, repeat U/V four times,
      bilinear upsample chroma for skip, concatenate [Y1..Y4,U1..U4,V1..V4].
    Key Params: normalized [0,1], align_corners=False, batch fixed at one.
    State/Dependencies: NumPy implementation matches torch PixelUnshuffle order.
    """
    if y.shape != (96,480) or u.shape != (48,240) or v.shape != (48,240): raise ValueError("tile shapes must be 96x480/48x240")
    yy = y.astype(np.float32) / 255.; uu = u.astype(np.float32) / 255.; vv = v.astype(np.float32) / 255.
    y4 = yy.reshape(48,2,240,2).transpose(1,3,0,2).reshape(4,48,240)
    y4uv = np.concatenate([y4, np.repeat(uu[None],4,0), np.repeat(vv[None],4,0)], 0)[None].astype(np.float32)
    uv = cv2.resize(np.stack([uu,vv], -1), (480,96), interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
    skip = np.concatenate([yy[None], uv], 0)[None].astype(np.float32)
    return y4uv, skip


def y4uv_to_bgr(y4uv: np.ndarray) -> np.ndarray:
    """@shanlan-comment
    Input: normalized Y4UV tensor [1,12,48,240].
    Output: BGR preview reconstructed from Y4 and representative U/V channels.
    Key Steps: inverse PixelUnshuffle Y, select repeated chroma representatives, convert YUV420.
    Key Params: channel order [Y1..Y4,U1..U4,V1..V4].
    State/Dependencies: NumPy/OpenCV only; diagnostic path does not affect inference.
    """
    x=np.asarray(y4uv)[0]; y4=x[:4].transpose(1,2,0).reshape(48,240,2,2).transpose(0,2,1,3).reshape(96,480)
    u=x[4]; v=x[8]
    return yuv420_to_bgr(YUV420Frame(np.clip(y4*255,0,255).astype(np.uint8),np.clip(u*255,0,255).astype(np.uint8),np.clip(v*255,0,255).astype(np.uint8)))


def skip_to_bgr(skip: np.ndarray) -> np.ndarray:
    """@shanlan-comment
    Input: normalized skip tensor [1,3,96,480] in [Y,U,V] order.
    Output: BGR preview after YUV444 conversion.
    Key Steps: transpose channel-first and call YUV444 conversion.
    Key Params: fixed deployment skip shape.
    State/Dependencies: OpenCV diagnostic conversion only.
    """
    return yuv444_to_bgr(np.asarray(skip)[0])


class Runner(Protocol):
    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """@shanlan-comment
        Input: model-specific NumPy tensor mapping.
        Output: model output tensor.
        Key Steps: execute one batch-one inference.
        Key Params: implementation validates fixed shapes.
        State/Dependencies: detector or restoration runtime.
        """
        ...


class DetectorONNXRunner:
    """@shanlan-comment
    Input: detector NCHW float32 tensor [1,3,768,1024].
    Output: probability map [1,1,H,W] from the sole detector output.
    Key Steps: discover actual ONNX input name and invoke session.
    Key Params: input shape and finite output are validated.
    State/Dependencies: ONNX Runtime session.
    """
    def __init__(self, path: str | Path):
        """@shanlan-comment
        Input: detector ONNX path.
        Output: initialized ONNX Runtime detector session.
        Key Steps: load CPU session and discover input name.
        Key Params: fixed detector input contract.
        State/Dependencies: onnxruntime.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc: raise RuntimeError("onnxruntime is required for ONNX inference") from exc
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
    def run(self, inputs: np.ndarray) -> np.ndarray:
        """@shanlan-comment
        Input: detector NCHW float32 tensor.
        Output: finite probability map [1,1,H,W].
        Key Steps: feed discovered input and validate output shape.
        Key Params: batch and channel equal one.
        State/Dependencies: ONNX Runtime session.
        """
        out = self.session.run(None, {self.input_name: np.asarray(inputs, np.float32)})[0]
        result = np.asarray(out, np.float32)
        if result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 1:
            raise ValueError(f"detector output must be [1,1,H,W], got {result.shape}")
        return result


class RestoreONNXRunner:
    """@shanlan-comment
    Input: dict with y4uv [1,12,48,240] and skip [1,3,96,480].
    Output: yuv444 [1,3,96,480].
    Key Steps: discover names, validate both tensors, invoke sole output.
    Key Params: fixed batch-one deployment contract.
    State/Dependencies: ONNX Runtime session.
    """
    def __init__(self, path: str | Path):
        """@shanlan-comment
        Input: restoration ONNX path.
        Output: initialized dual-input session.
        Key Steps: load session and validate y4uv/skip names.
        Key Params: fixed deployment tensor shapes.
        State/Dependencies: onnxruntime.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc: raise RuntimeError("onnxruntime is required for ONNX inference") from exc
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.names = {item.name for item in inputs}
        if not {"y4uv", "skip"}.issubset(self.names): raise ValueError(f"restore inputs missing y4uv/skip: {self.names}")
    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """@shanlan-comment
        Input: y4uv and skip tensors with frozen shapes.
        Output: yuv444 [1,3,96,480].
        Key Steps: validate, feed by actual names, validate output.
        Key Params: batch one.
        State/Dependencies: ONNX Runtime session.
        """
        if set(("y4uv", "skip")) - set(inputs): raise ValueError("restore requires y4uv and skip")
        if inputs["y4uv"].shape != Y4UV_SHAPE or inputs["skip"].shape != SKIP_SHAPE: raise ValueError("invalid restore input shape")
        out = np.asarray(self.session.run(None, {"y4uv":inputs["y4uv"], "skip":inputs["skip"]})[0], np.float32)
        if out.shape != OUT_SHAPE: raise ValueError(f"restore output must be {OUT_SHAPE}, got {out.shape}")
        return out




def apply_global_s(y: np.ndarray, valid_mask: np.ndarray | None = None, *, min_pixels: int = 256) -> tuple[np.ndarray, dict[str, Any]]:
    """@shanlan-comment
    Input: uint8 restored Y plane or float image.
    Output: adjusted Y and trace containing applied/bypass reason.
    Key Steps: robust percentiles, dark/bright population test, smooth S LUT.
    Key Params: p01/p99, min_pixels=256, separation 0.08..0.30, target=.65.
    State/Dependencies: identity bypass on insufficient confidence; Y-only.
    """
    arr = np.asarray(y, np.float32); mask = np.ones(arr.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if mask.shape != arr.shape: raise ValueError("valid_mask shape must equal Y shape")
    valid = arr[mask & np.isfinite(arr)]
    trace = {"global_s_applied": False, "bypass_reason": ""}
    if valid.size < min_pixels: trace["bypass_reason"]="insufficient_pixels"; return y.copy(), trace
    hist, edges = np.histogram(valid, bins=256, range=(0,255)); kernel=np.array([.25,.5,.25]); smooth=np.convolve(hist.astype(np.float32),kernel,mode='same')
    p01, p99 = np.percentile(valid, [1,99]); sep = (p99-p01)/255.; midpoint=(p01+p99)/2
    dark_mass=float(smooth[:int(midpoint)].sum()/max(smooth.sum(),1)); bright_mass=float(smooth[int(midpoint):].sum()/max(smooth.sum(),1))
    confidence=float(min(1., sep/.30) * min(1., min(dark_mass,bright_mass)/.03))
    if sep < .08 or sep > .30 or min(dark_mass,bright_mass) < .03 or confidence < .15:
        trace.update(hist_p01=float(p01),hist_p99=float(p99),hist_delta=float(p99-p01),mass_dark=dark_mass,mass_bright=bright_mass,confidence=confidence,bypass_reason="histogram_gate"); return y.copy(), trace
    mu_dark=float(np.average(edges[:-1][:int(midpoint)],weights=smooth[:int(midpoint)])); mu_bright=float(np.average(edges[:-1][int(midpoint):],weights=smooth[int(midpoint):]))
    target_delta=.65*255.; gamma=float(np.clip(1.0+(1.0-(mu_bright-mu_dark)/255.)*1.6,1.,2.6))
    z=np.clip((arr-p01)/max(p99-p01,1e-6),0,1); out=arr.copy(); out[mask]=(np.power(z[mask],gamma)*255.).clip(0,255)
    trace.update(global_s_applied=True,hist_p01=float(p01),hist_p99=float(p99),hist_delta=float(p99-p01),mu_dark=mu_dark,mu_bright=mu_bright,mass_dark=dark_mass,mass_bright=bright_mass,confidence=confidence,gamma=gamma,target_delta=target_delta)
    return out.astype(y.dtype), trace


def polygon_feather_mask(shape: tuple[int,int], polygon: np.ndarray, width: int = DEFAULT_FEATHER_WIDTH) -> np.ndarray:
    """@shanlan-comment
    Input: output shape and source polygon.
    Output: float32 alpha mask with one inside polygon and outward decay.
    Key Steps: fill polygon, distance-transform outside, apply raised-cosine decay.
    Key Params: outward feather width defaults to 16 pixels.
    State/Dependencies: OpenCV rasterization; outside pixels remain untouched.
    """
    hard = np.zeros(shape, np.uint8)
    cv2.fillPoly(hard, [polygon.astype(np.int32)], 1)
    if not hard.any(): return hard.astype(np.float32)
    outside = cv2.distanceTransform(1-hard, cv2.DIST_L2, 3)
    alpha = np.ones(shape, np.float32)
    band = outside < float(max(1, width))
    t = np.clip(outside[band] / float(max(1, width)), 0, 1)
    alpha[band] = 0.5 * (1.0 + np.cos(np.pi * t))
    alpha[outside >= float(max(1, width))] = 0.0
    alpha[hard.astype(bool)] = 1.0
    return alpha


def hard_polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """@shanlan-comment
    Input: image shape and polygon coordinates.
    Output: boolean mask true only inside the polygon.
    Key Steps: rasterize polygon without outward feather.
    Key Params: no dilation or transition band.
    State/Dependencies: OpenCV rasterization; used only by explicit Global-S experiments.
    """
    mask=np.zeros(shape,np.uint8)
    cv2.fillPoly(mask,[np.asarray(polygon,np.int32)],1)
    return mask.astype(bool)


def compose_all_rois(frame: YUV420Frame, results: list[ROIResult]) -> YUV420Frame:
    """@shanlan-comment
    Input: source YUV420Frame and unordered successful ROIResult records.
    Output: one frame with weighted Y/U/V full-resolution composition.
    Key Steps: accumulate prediction contributions with spatial alpha times
      detector quality, normalize once, preserve zero-alpha pixels, then
      downsample chroma exactly once.
    Key Params: spatial alpha alone controls source/prediction union; detector
      score only controls relative prediction weight; order is irrelevant.
    State/Dependencies: source planes remain bit-exact outside all polygons.
    """
    h,w=frame.y.shape; ysum=np.zeros((h,w),np.float32); usum=np.zeros((h,w),np.float32); vsum=np.zeros((h,w),np.float32); weight=np.zeros((h,w),np.float32); union_complement=np.ones((h,w),np.float32)
    for result in results:
        x,y,rw,rh=result.bbox; restored=np.asarray(result.yuv444,np.float32)
        if restored.shape != (3,rh,rw): raise ValueError(f"ROI result shape {restored.shape} != {(3,rh,rw)}")
        restored=np.clip(restored,0,255)
        spatial_alpha=np.clip(np.asarray(result.alpha,np.float32),0,1)
        quality=float(np.clip(result.score,0,1))
        prediction_weight=spatial_alpha*quality
        ysum[y:y+rh,x:x+rw]+=restored[0]*prediction_weight
        usum[y:y+rh,x:x+rw]+=restored[1]*prediction_weight
        vsum[y:y+rh,x:x+rw]+=restored[2]*prediction_weight
        weight[y:y+rh,x:x+rw]+=prediction_weight
        union_complement[y:y+rh,x:x+rw]*=(1-spatial_alpha)
    coverage=1-union_complement; valid=coverage>0; restored_y=np.divide(ysum,np.maximum(weight,1e-6)); restored_u=np.divide(usum,np.maximum(weight,1e-6)); restored_v=np.divide(vsum,np.maximum(weight,1e-6)); outy=frame.y.astype(np.float32); outu=cv2.resize(frame.u,(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32); outv=cv2.resize(frame.v,(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    outy[valid]=outy[valid]*(1-coverage[valid])+restored_y[valid]*coverage[valid]; outu[valid]=outu[valid]*(1-coverage[valid])+restored_u[valid]*coverage[valid]; outv[valid]=outv[valid]*(1-coverage[valid])+restored_v[valid]*coverage[valid]
    new_u=cv2.resize(np.clip(outu,0,255).astype(np.uint8),(w//2,h//2),interpolation=cv2.INTER_AREA); new_v=cv2.resize(np.clip(outv,0,255).astype(np.uint8),(w//2,h//2),interpolation=cv2.INTER_AREA)
    pooled=coverage.reshape(h//2,2,w//2,2).max(axis=(1,3))>0; new_u[~pooled]=frame.u[~pooled]; new_v[~pooled]=frame.v[~pooled]
    return YUV420Frame(np.clip(outy,0,255).astype(np.uint8),new_u,new_v,frame.matrix,frame.range,frame.chroma_siting)


DEBUG_STAGES = ["00_input", "01_detector_input", "02_detector_output", "03_detection_decode", "04_roi_crop", "05_tile_preprocess", "06_restore_output", "07_tile_merge", "08_inverse_resize", "09_global_s", "10_composite"]


def _debug_dirs(root: str | Path) -> dict[str, Path]:
    """@shanlan-comment
    Input: debug root path.
    Output: stage-name to created directory mapping.
    Key Steps: create the fixed 11-stage hierarchy once.
    Key Params: DEBUG_STAGES order is the pipeline contract.
    State/Dependencies: filesystem writes occur only when debug is requested.
    """
    base=Path(root); base.mkdir(parents=True,exist_ok=True); dirs={stage:base/stage for stage in DEBUG_STAGES}
    for path in dirs.values(): path.mkdir(parents=True,exist_ok=True)
    return dirs


def _debug_write(path: Path, value: Any, kind: str, manifest: list[dict[str, Any]], *, roi: int | None = None, tile: int | None = None) -> None:
    """@shanlan-comment
    Input: destination, NumPy/image/JSON value, artifact kind and manifest list.
    Output: non-empty artifact plus relative manifest record.
    Key Steps: route raw tensors/json or checked cv2 BGR/gray writes.
    Key Params: kind is bgr_preview, gray_diagnostic, raw_tensor, or json.
    State/Dependencies: OpenCV, NumPy and filesystem.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    if kind == "raw_tensor": np.save(path.with_suffix(".npy"), np.asarray(value)); actual=path.with_suffix(".npy")
    elif kind == "json": path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8"); actual=path
    else:
        arr=np.asarray(value)
        if kind == "bgr_preview" and (arr.ndim != 3 or arr.shape[2] != 3): raise ValueError("BGR preview must be HxWx3")
        if not cv2.imwrite(str(path),arr): raise IOError(f"failed to write {path}")
        actual=path
    stage = next((ancestor.name for ancestor in [actual.parent, *actual.parents] if ancestor.name in DEBUG_STAGES), None)
    if stage is None: raise ValueError(f"artifact is outside a debug stage: {actual}")
    root = next(ancestor.parent for ancestor in [actual.parent, *actual.parents] if ancestor.name == stage)
    manifest.append({"stage":stage,"path":str(actual.relative_to(root)).replace("\\", "/"),"kind":kind,"roi":roi,"tile":tile})


def accumulate_debug_coverage(frame_shape: tuple[int, int], results: list[ROIResult]) -> tuple[np.ndarray, np.ndarray]:
    """@shanlan-comment
    Input: full-frame H/W and successful ROI results.
    Output: accumulated spatial-alpha*score prediction weight and spatial-union coverage.
    Key Steps: sum prediction weights separately from union=1-product(1-spatial_alpha).
    Key Params: score affects relative prediction weight only; union controls source/prediction blend.
    State/Dependencies: NumPy only; diagnostic calculation mirrors compose_all_rois.
    """
    weight=np.zeros(frame_shape,np.float32); union_complement=np.ones(frame_shape,np.float32)
    for result in results:
        x,y,w,h=result.bbox
        spatial_alpha=np.clip(np.asarray(result.alpha,np.float32),0,1)
        quality=float(np.clip(result.score,0,1))
        weight[y:y+h,x:x+w]+=spatial_alpha*quality
        union_complement[y:y+h,x:x+w]*=(1-spatial_alpha)
    return weight,1-union_complement


def process_yuv420(
    frame: YUV420Frame,
    detector: Any,
    restorer: Runner,
    *,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    enable_global_s: bool = False,
    restore_geometry: RestoreGeometry | None = None,
) -> tuple[YUV420Frame, dict[str, Any]]:
    """@shanlan-comment
    Input: authoritative planar YUV420Frame; detector/restorer runners.
    Output: composited YUV420Frame and JSON-serializable trace.
    Key Steps: derive temporary BGR detector view without re-encoding source,
      decode/map/filter ROIs, restore tiles, composite onto source planes.
    Key Params: fixed detector contract plus an explicit restoration geometry; omitted geometry preserves V1 defaults.
    State/Dependencies: source frame remains the sole output base.
    """
    geometry=restore_geometry or RestoreGeometry()
    if geometry.content_h <= 0 or geometry.content_h > CANVAS_HW[0] or geometry.content_h % 2:
        raise ValueError("invalid restoration content height")
    if geometry.overlap < 0 or geometry.overlap >= CANVAS_HW[1]:
        raise ValueError("invalid restoration overlap")
    if frame.y.dtype != np.uint8 or frame.u.dtype != np.uint8 or frame.v.dtype != np.uint8: raise ValueError("YUV planes must be uint8")
    h,w=frame.y.shape
    if frame.u.shape!=(h//2,w//2) or frame.v.shape!=frame.u.shape: raise ValueError("invalid YUV420 plane geometry")
    start=time.perf_counter(); bgr=yuv420_to_bgr(frame); det_input=detector_normalize_bgr(bgr); dirs=_debug_dirs(debug_dir) if debug_dir is not None else None; manifest=[]
    if dirs is not None:
        _debug_write(dirs["00_input"] / "source_bgr.jpg", bgr, "bgr_preview", manifest)
        _debug_write(dirs["00_input"] / "source_y.npy", frame.y, "raw_tensor", manifest)
        _debug_write(dirs["00_input"] / "source_u.npy", frame.u, "raw_tensor", manifest)
        _debug_write(dirs["00_input"] / "source_v.npy", frame.v, "raw_tensor", manifest)
        _debug_write(dirs["01_detector_input"] / "detector_view_bgr.jpg", bgr, "bgr_preview", manifest)
        _debug_write(dirs["01_detector_input"] / "fixed_1024x768_bgr.jpg", cv2.resize(bgr,(1024,768)), "bgr_preview", manifest)
        _debug_write(dirs["01_detector_input"] / "normalized_nchw.npy", det_input, "raw_tensor", manifest)
    prob=detector.run(det_input) if hasattr(detector,'run') else detector(det_input)
    prob=np.asarray(prob,np.float32); detections=decode_db_probability(prob); results=[]
    if dirs is not None:
        _debug_write(dirs["02_detector_output"] / "probability_map.npy", prob, "raw_tensor", manifest)
        _debug_write(dirs["02_detector_output"] / "probability_map.png", np.clip(prob[0,0]*255,0,255).astype(np.uint8), "gray_diagnostic", manifest)
        _debug_write(dirs["02_detector_output"] / "binary.png", (prob[0,0]>.20).astype(np.uint8)*255, "gray_diagnostic", manifest)
    trace={"source_hw":[h,w],"config":{"enable_global_s":bool(enable_global_s),"restore_content_height":int(geometry.content_h),"restore_overlap":int(geometry.overlap),"restore_resize_interpolation":int(geometry.interpolation),"restore_force_even_width":bool(geometry.force_even_width)},"detector_params":{"thresh":.20,"box_thresh":.40,"unclip_ratio":1.40,"max_candidates":3000},"det_input_shape":list(det_input.shape),"probability_map_shape":list(prob.shape),"probability_min":float(prob.min()),"probability_max":float(prob.max()),"detections":[],"rois":[],"fallbacks":[]}
    for d in detections:
        if d.status != "accepted": trace["detections"].append({"status":"rejected","reason":d.reason,"score":float(d.score),"polygon":np.asarray(d.polygon).tolist()}); continue
        poly=map_polygon_to_source(d.polygon,(h,w),prob.shape[-2:]); ok,reason=filter_detection(poly,(h,w)); d.status="accepted" if ok else "rejected"; d.reason=reason; record={"status":d.status,"reason":reason,"score":float(d.score),"polygon":poly.tolist()}; trace["detections"].append(record)
        if not ok: continue
        roi,bbox=crop_even_roi(frame,poly); x,y,rw,rh=bbox; local=poly-np.array([x,y]); oriented,oriented_poly,orientation=normalize_roi_orientation(roi,local); tiles=resize_pad_tiles(oriented,content_h=geometry.content_h,overlap=geometry.overlap,interpolation=geometry.interpolation,force_even_width=geometry.force_even_width); outputs=[]; fallback=""; roi_id=len(results)
        if dirs is not None:
            _debug_write(dirs["04_roi_crop"] / f"roi_{roi_id:04d}" / "source_crop_bgr.jpg", yuv420_to_bgr(roi), "bgr_preview", manifest, roi=roi_id)
            _debug_write(dirs["04_roi_crop"] / f"roi_{roi_id:04d}" / "crop_bgr.jpg", yuv420_to_bgr(oriented), "bgr_preview", manifest, roi=roi_id)
            _debug_write(dirs["04_roi_crop"] / f"roi_{roi_id:04d}" / "crop_y.npy", oriented.y, "raw_tensor", manifest, roi=roi_id)
        try:
            for tile_id,tile in enumerate(tiles):
                y4uv,skip=build_y4uv_skip(tile["y"],tile["u"],tile["v"]); out=restorer.run({"y4uv":y4uv,"skip":skip}); outputs.append(out)
                if dirs is not None:
                    _debug_write(dirs["05_tile_preprocess"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_y_bgr.jpg", cv2.cvtColor(tile["y"],cv2.COLOR_GRAY2BGR), "bgr_preview", manifest, roi=roi_id,tile=tile_id)
                    tile_frame=YUV420Frame(tile["y"],tile["u"],tile["v"])
                    _debug_write(dirs["05_tile_preprocess"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_yuv420_bgr.jpg", yuv420_to_bgr(tile_frame), "bgr_preview", manifest, roi=roi_id,tile=tile_id)
                    _debug_write(dirs["05_tile_preprocess"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_y4uv_bgr.jpg", y4uv_to_bgr(y4uv), "bgr_preview", manifest, roi=roi_id,tile=tile_id)
                    _debug_write(dirs["05_tile_preprocess"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_skip_bgr.jpg", skip_to_bgr(skip), "bgr_preview", manifest, roi=roi_id,tile=tile_id)
                    _debug_write(dirs["05_tile_preprocess"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_y4uv.npy", y4uv, "raw_tensor", manifest, roi=roi_id,tile=tile_id)
                    _debug_write(dirs["05_tile_preprocess"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_skip.npy", skip, "raw_tensor", manifest, roi=roi_id,tile=tile_id)
                    _debug_write(dirs["06_restore_output"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_yuv444.npy", out, "raw_tensor", manifest, roi=roi_id,tile=tile_id)
                    _debug_write(dirs["06_restore_output"] / f"roi_{roi_id:04d}" / f"tile_{tile_id:04d}_yuv444_bgr.jpg", yuv444_to_bgr(out[0]), "bgr_preview", manifest, roi=roi_id,tile=tile_id)
            merged=merge_restore_tiles(tiles,outputs); restored=np.stack([cv2.resize(ch,(oriented.y.shape[1],oriented.y.shape[0]),interpolation=cv2.INTER_CUBIC) for ch in merged],0); restored=np.clip(restored,0,1)*255 if restored.max()<=1.5 else np.clip(restored,0,255)
            if dirs is not None:
                _debug_write(dirs["07_tile_merge"] / f"roi_{roi_id:04d}" / "merged_yuv444.npy", merged, "raw_tensor", manifest, roi=roi_id)
                _debug_write(dirs["07_tile_merge"] / f"roi_{roi_id:04d}" / "merged_yuv444_bgr.jpg", yuv444_to_bgr(merged), "bgr_preview", manifest, roi=roi_id)
                _debug_write(dirs["08_inverse_resize"] / f"roi_{roi_id:04d}" / "inverse_resized_yuv444.npy", restored, "raw_tensor", manifest, roi=roi_id)
                _debug_write(dirs["08_inverse_resize"] / f"roi_{roi_id:04d}" / "inverse_resized_yuv444_bgr.jpg", yuv444_to_bgr(restored), "bgr_preview", manifest, roi=roi_id)
            hard_mask=hard_polygon_mask(oriented.y.shape,oriented_poly); before=restored.copy()
            if enable_global_s:
                restored[0],gs=apply_global_s(restored[0],hard_mask)
            else:
                gs={"global_s_applied":False,"bypass_reason":"disabled_by_config"}
            after=restored.copy()
            if dirs is not None:
                _debug_write(dirs["09_global_s"] / f"roi_{roi_id:04d}" / "before_y.npy", before[0], "raw_tensor", manifest, roi=roi_id)
                _debug_write(dirs["09_global_s"] / f"roi_{roi_id:04d}" / "after_y.npy", after[0], "raw_tensor", manifest, roi=roi_id)
                _debug_write(dirs["09_global_s"] / f"roi_{roi_id:04d}" / "before_bgr.jpg", yuv444_to_bgr(before), "bgr_preview", manifest, roi=roi_id)
                _debug_write(dirs["09_global_s"] / f"roi_{roi_id:04d}" / "after_bgr.jpg", yuv444_to_bgr(after), "bgr_preview", manifest, roi=roi_id)
                _debug_write(dirs["09_global_s"] / f"roi_{roi_id:04d}" / "global_s.json", gs, "json", manifest, roi=roi_id)
            restored=inverse_orientation_yuv444(restored,orientation); alpha=polygon_feather_mask((rh,rw),local); results.append(ROIResult(restored,poly,bbox,float(d.score),alpha,gs,orientation,""))
            trace["rois"].append({"bbox":list(bbox),"orientation":orientation,"tiles":[{"valid_rect":list(t["valid_rect"]),"line_start_x":int(t["line_start_x"]),"valid_w":int(t["valid_w"]),"overlap":int(t["overlap"]),"interpolation":int(t["interpolation"]),"force_even_width":bool(t["force_even_width"]),"input_y4uv_shape":list(Y4UV_SHAPE),"input_skip_shape":list(SKIP_SHAPE),"output_shape":list(o.shape)} for t,o in zip(tiles,outputs)],"global_s":gs,"fallback":""})
        except Exception as exc:
            fallback=f"RESTORE_FAILED:{type(exc).__name__}"; trace["fallbacks"].append(fallback); trace["rois"].append({"bbox":list(bbox),"orientation":orientation,"fallback":fallback})
    out_frame=compose_all_rois(frame,results); trace.update(accepted_count=len(results),elapsed_ms=(time.perf_counter()-start)*1000,output_hw=list(out_frame.y.shape),source_adapter="YUV420 authoritative")
    if debug_dir is not None:
        d=Path(debug_dir); d.mkdir(parents=True,exist_ok=True)
        overlay=bgr.copy()
        for item in trace["detections"]:
            poly=np.asarray(item.get("polygon",[]),np.int32)
            if len(poly)==4: cv2.polylines(overlay,[poly],True,(0,255,0) if item.get("status")=="accepted" else (0,0,255),2)
        _debug_write(dirs["03_detection_decode"] / "detections_overlay.jpg", overlay, "bgr_preview", manifest)
        _debug_write(dirs["03_detection_decode"] / "detections.json", trace["detections"], "json", manifest)
        _debug_write(dirs["10_composite"] / "source_pre_composite.jpg", bgr, "bgr_preview", manifest)
        _debug_write(dirs["10_composite"] / "final_composited_bgr.jpg", yuv420_to_bgr(out_frame), "bgr_preview", manifest)
        weight,coverage=accumulate_debug_coverage(frame.y.shape,results)
        _debug_write(dirs["10_composite"] / "weight.npy", weight, "raw_tensor", manifest)
        _debug_write(dirs["10_composite"] / "coverage.png", np.clip(coverage*255,0,255).astype(np.uint8), "gray_diagnostic", manifest)
        for item in manifest:
            try: item["path"]=str(Path(item["path"]).relative_to(d))
            except ValueError: pass
        (d/"debug_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        trace["debug_stage_roots"]={stage:stage for stage in DEBUG_STAGES}; trace["debug_manifest"]="debug_manifest.json"
    json.dumps(trace)
    return out_frame, trace


def run_pipeline(bgr: np.ndarray, detector: Any, restorer: Runner, *, debug: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """@shanlan-comment
    Input: uint8 BGR test-adapter image and detector/restorer runners.
    Output: BGR output and JSON-serializable complete run trace.
    Key Steps: YUV adapter, detector normalize/decode/map/filter, ROI tiles,
      dual-input restore, merge and polygon composite.
    Key Params: all V1 constants are module constants or decoder defaults.
    State/Dependencies: fake runners or ONNXRunner; batch remains one.
    """
    original_hw=bgr.shape[:2]; h,w=original_hw; padded=bgr
    if h%2 or w%2:
        padded=cv2.copyMakeBorder(bgr,0,h%2,0,w%2,cv2.BORDER_REPLICATE)
    frame=bgr_to_yuv420(padded); out_frame,trace=process_yuv420(frame,detector,restorer,debug=debug)
    output=yuv420_to_bgr(out_frame)[:h,:w]; trace['source_adapter']='BGR->YUV420 test adapter'; trace['original_hw']=list(original_hw); trace['padded_hw']=list(padded.shape[:2])
    return output, trace


class FakeDetector:
    """Deterministic detector used by fixture smoke/tests."""
    """@shanlan-comment
    Input: fixed probability map fixture.
    Output: same probability map regardless of detector tensor.
    Key Steps: retain deterministic fixture.
    Key Params: map shape is caller supplied.
    State/Dependencies: none.
    """
    def __init__(self, probability: np.ndarray):
        """@shanlan-comment
        Input: deterministic probability fixture.
        Output: initialized fake detector.
        Key Steps: retain fixture.
        Key Params: fixture shape.
        State/Dependencies: none.
        """
        self.probability=probability
    def run(self, x: np.ndarray) -> np.ndarray:
        """@shanlan-comment
        Input: detector tensor.
        Output: configured probability fixture.
        Key Steps: return fixture unchanged.
        Key Params: none.
        State/Dependencies: none.
        """
        return self.probability


class FakeRestorer:
    """Identity-like deployment runner validating the dual input contract."""
    """@shanlan-comment
    Input: dual deployment tensors.
    Output: skip tensor as deterministic YUV444 fixture.
    Key Steps: validate exact contract and return identity-like output.
    Key Params: batch-one fixed shapes.
    State/Dependencies: none.
    """
    def run(self, inputs: dict[str,np.ndarray]) -> np.ndarray:
        """@shanlan-comment
        Input: y4uv/skip deployment tensors.
        Output: identity-like skip output.
        Key Steps: validate fixed shapes and return skip.
        Key Params: batch one.
        State/Dependencies: none.
        """
        if inputs['y4uv'].shape != Y4UV_SHAPE or inputs['skip'].shape != SKIP_SHAPE: raise ValueError('invalid dual input shape')
        return inputs['skip'].copy()


def main(argv: list[str] | None = None) -> int:
    """@shanlan-comment
    Input: CLI arguments for JPG, detector ONNX and restoration ONNX.
    Output: output image and JSON run trace; nonzero on missing required models.
    Key Steps: load BGR, construct ONNX runners, execute pipeline, save artifacts.
    Key Params: --input, --detector, --restorer, --output, --trace.
    State/Dependencies: cv2 and onnxruntime for production execution.
    """
    p=argparse.ArgumentParser(description='BASIC V2 YUV420 deployment pipeline')
    p.add_argument('--input',required=True); p.add_argument('--detector',required=True); p.add_argument('--restorer',required=True); p.add_argument('--output',required=True); p.add_argument('--trace'); p.add_argument('--output-i420'); p.add_argument('--debug-dir')
    p.add_argument('--input-format', choices=('jpg','i420'), default='jpg'); p.add_argument('--width',type=int); p.add_argument('--height',type=int)
    a=p.parse_args(argv)
    detector=DetectorONNXRunner(a.detector); restorer=RestoreONNXRunner(a.restorer)
    if a.input_format == 'i420':
        if a.width is None or a.height is None: p.error('--width and --height are required for --input-format i420')
        frame=read_yuv420(a.input,a.width,a.height); out_frame,trace=process_yuv420(frame,detector,restorer,debug_dir=a.debug_dir); out=yuv420_to_bgr(out_frame)
    else:
        image=cv2.imread(a.input,cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(a.input)
        out_frame,out,trace=run_bgr_adapter_frame(image,detector,restorer,debug_dir=a.debug_dir)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(a.output,out): raise IOError(f"failed to write {a.output}")
    preview=Path(a.output).with_name(Path(a.output).stem+"_preview.jpg")
    if not cv2.imwrite(str(preview),out): raise IOError(f"failed to write {preview}")
    if a.output_i420: write_yuv420(a.output_i420,out_frame)
    if a.debug_dir:
        d=Path(a.debug_dir); d.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(d/'final.jpg'),out): raise IOError(f"failed to write {d/'final.jpg'}")
    if a.trace: Path(a.trace).write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0


if __name__ == '__main__': raise SystemExit(main())
