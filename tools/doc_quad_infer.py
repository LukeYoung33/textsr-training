"""Whole-document quadrilaterals from local YOLO segmentation weights.

The default fits document boundaries or uses an approximate oriented mask envelope.
Both produce a four-corner projective warp; --strict-boundary rejects approximations.
Raw contours remain available for diagnosis; --crop-only is explicitly diagnostic.
Boundary fitting does not prove planarity, metric aspect or semantic orientation.
Ultralytics is imported only by the CLI; geometry requires NumPy and OpenCV.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class BoundaryFitError(ValueError):
    """A rejected four-edge fit carrying JSON-ready diagnostic evidence."""

    def __init__(self, reason: str, evidence: dict[str, Any]):
        """@luna-comment
        Input: Rejection reason and finite JSON-ready fitting evidence.
        Output: ValueError subtype with diagnostic evidence attached.
        Key Steps: Preserve reason and evidence without synthesizing corner values.
        Key Params: Evidence describes failed gates, not a successful fallback.
        State/Dependencies: No shared state or external resources.
        """
        super().__init__(reason)
        self.evidence = evidence


def order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """@luna-comment
    Input: Four finite, distinct pixel-coordinate vertices in arbitrary order.
    Output: Float32 strictly convex quad in clockwise image-coordinate order.
    Key Steps: Sort cyclically; choose edge pair closest to image horizontal axes.
    Key Params: A 1e-4 cross-product tolerance rejects near-collinear corners.
    State/Dependencies: NumPy and OpenCV only; no shared state or image mutation.
    """
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("A quad requires four finite xy vertices.")
    if len(cv2.convexHull(quad)) != 4:
        raise ValueError("Quad corners must form a strictly convex polygon.")
    center = quad.mean(axis=0)
    angle = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angle)]
    edges = np.roll(quad, -1, axis=0) - quad
    turns = edges[:, 0] * np.roll(edges[:, 1], -1)
    turns -= edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(turns <= 1e-4):
        raise ValueError("Quad contains a degenerate corner.")
    horizontal = np.abs(edges[:, 0]) / np.linalg.norm(edges, axis=1)
    pair = 0 if horizontal[0] + horizontal[2] >= horizontal[1] + horizontal[3] else 1
    indices = [index for index in (pair, pair + 2) if edges[index, 0] > 0]
    if not indices:
        indices = [pair, pair + 2]
    start = min(indices, key=lambda index: (float((quad[index, 1] + quad[(index + 1) % 4, 1]) / 2), index))
    return np.roll(quad, -start, axis=0)


def sample_contour(points: np.ndarray, count: int = 512) -> np.ndarray:
    """@luna-comment
    Input: Closed polygon vertices and desired uniform boundary sample count.
    Output: Float32 contour sampled uniformly by perimeter distance.
    Key Steps: Drop zero-length segments; interpolate cumulative arc lengths.
    Key Params: 512 samples limit cost and avoid sparse-vertex weighting bias.
    State/Dependencies: NumPy only; no source image or model state.
    """
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    valid = lengths > 1e-7
    starts, ends, lengths = points[valid], following[valid], lengths[valid]
    if len(lengths) < 3 or count < 12:
        raise ValueError("Insufficient contour segments or samples.")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float64)))
    distances = np.linspace(0, cumulative[-1], count, endpoint=False)
    indices = np.minimum(np.searchsorted(cumulative, distances, side="right") - 1, len(lengths) - 1)
    fraction = ((distances - cumulative[indices]) / lengths[indices])[:, None]
    return (starts[indices] + fraction * (ends[indices] - starts[indices])).astype(np.float32)


def edge_measurements(quad: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """@luna-comment
    Input: Clockwise quad and finite boundary sample positions.
    Output: Per-point distances to four lines, edge projections and edge lengths.
    Key Steps: Compute normalized cross/dot products for each directed edge.
    Key Params: Projection zero/one denotes each line segment's corner endpoints.
    State/Dependencies: NumPy geometry only; caller validates nondegenerate edges.
    """
    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    relative = points[:, None, :] - quad[None, :, :]
    distances = np.abs(edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]) / lengths
    projections = np.sum(relative * edges[None, :, :], axis=2) / lengths ** 2
    return distances, projections, lengths


def fit_boundary_lines(seed: np.ndarray, samples: np.ndarray, diagonal: float) -> np.ndarray:
    """@luna-comment
    Input: Four-corner hull seed, uniform raw-contour samples and visible diagonal.
    Output: Convex intersections of four robustly fitted supporting lines.
    Key Steps: Assign nearby samples; trim corners; fit Huber lines; intersect twice.
    Key Params: Four-percent diagonal fit band and 6..94 percent edge interiors.
    State/Dependencies: OpenCV fitLine and NumPy solves; no rectangle substitution.
    """
    quad = order_quad(seed)
    for _ in range(2):
        distances, projections, _ = edge_measurements(quad, samples)
        assignment = distances.argmin(axis=1)
        normals, offsets = [], []
        edges = np.roll(quad, -1, axis=0) - quad
        for index in range(4):
            selected = ((assignment == index) & (distances[:, index] <= max(3.0, diagonal * 0.04))
                        & (projections[:, index] >= 0.06) & (projections[:, index] <= 0.94))
            if int(selected.sum()) < 6:
                raise ValueError("insufficient_samples_for_four_boundary_lines")
            fitted = cv2.fitLine(samples[selected], cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
            direction = fitted[:2].astype(np.float64)
            if np.dot(direction, edges[index]) < 0:
                direction = -direction
            normal = np.array([-direction[1], direction[0]])
            normals.append(normal)
            offsets.append(float(np.dot(normal, fitted[2:])))
        normals, offsets = np.asarray(normals), np.asarray(offsets)
        vertices = [np.linalg.solve(normals[[(index - 1) % 4, index]],
                                    offsets[[(index - 1) % 4, index]]) for index in range(4)]
        quad = order_quad(vertices)
    return quad


def boundary_support(quad: np.ndarray, samples: np.ndarray, diagonal: float) -> list[dict[str, Any]]:
    """@luna-comment
    Input: Candidate quad, uniformly sampled raw boundary and visible diagonal.
    Output: Four dictionaries describing observed straight-line support.
    Key Steps: Assign samples; measure supported fraction/span and residual quantiles.
    Key Params: Tolerance=max(1.5px,min(1.5 percent diagonal,3.5 percent edge)).
    State/Dependencies: NumPy only; metrics cannot establish physical planarity.
    """
    distances, projections, lengths = edge_measurements(quad, samples)
    assignment = distances.argmin(axis=1)
    evidence = []
    for index in range(4):
        selected = assignment == index
        residuals, positions = distances[selected, index], projections[selected, index]
        tolerance = max(1.5, min(0.015 * diagonal, 0.035 * float(lengths[index])))
        supported = (residuals <= tolerance) & (positions >= -0.02) & (positions <= 1.02)
        support_positions = np.clip(positions[supported], 0.0, 1.0)
        evidence.append({
            "edge_index": index, "sample_count": int(selected.sum()),
            "support_fraction": float(supported.mean()) if len(residuals) else 0.0,
            "support_span": float(np.ptp(support_positions)) if len(support_positions) else 0.0,
            "residual_q70_pixels": float(np.quantile(residuals, 0.70)) if len(residuals) else None,
            "residual_q90_pixels": float(np.quantile(residuals, 0.90)) if len(residuals) else None,
            "tolerance_pixels": tolerance,
        })
    return evidence


def enclosing_quad(candidate: np.ndarray, points: np.ndarray, margin: float) -> np.ndarray:
    """@luna-comment
    Input: Robust four-edge fit, finite contour samples and nonnegative margin.
    Output: Quad conservatively expanded toward the outer supported boundary quantile.
    Key Steps: Assign nearby edge samples; exclude remote spurs; intersect support lines.
    Key Params: Outer tenth percentile within the support band; explicit margin afterward.
    State/Dependencies: NumPy geometry; excluded mask area is separately measured/gated.
    """
    quad = order_quad(candidate).astype(np.float64)
    edges = np.roll(quad, -1, axis=0) - quad
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    distances, projections, lengths = edge_measurements(quad, points)
    assignment = distances.argmin(axis=1)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    offsets = np.sum(quad * normals, axis=1)
    for index in range(4):
        tolerance = max(1.5, min(0.015 * diagonal, 0.035 * float(lengths[index])))
        selected = ((assignment == index) & (distances[:, index] <= tolerance)
                    & (projections[:, index] >= -0.02) & (projections[:, index] <= 1.02))
        if not selected.any():
            raise ValueError("no_observed_support_for_boundary_offset")
        offsets[index] = min(offsets[index], float(np.quantile(points[selected] @ normals[index], 0.10))) - margin
    vertices = []
    for index in range(4):
        previous = (index - 1) % 4
        vertices.append(
            np.linalg.solve(normals[[previous, index]], offsets[[previous, index]])
        )
    return order_quad(vertices)


def mask_coverage(polygon: np.ndarray, quad: np.ndarray, image_shape: Sequence[int]) -> dict[str, float]:
    """@luna-comment
    Input: Raw visible polygon, fitted quad and original image dimensions.
    Output: Approximate retained/excluded predicted-mask area and raster scale.
    Key Steps: Rasterize both polygons at bounded resolution and count intersection.
    Key Params: At most 1024 pixels on the longest side; no mask is publicly emitted.
    State/Dependencies: Temporary OpenCV/NumPy rasters; excludes do not imply non-text.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    scale = min(1.0, 1024.0 / max(height, width))
    shape = (max(2, int(math.ceil(height * scale))), max(2, int(math.ceil(width * scale))))
    region = np.zeros(shape, np.uint8)
    enclosure = np.zeros(shape, np.uint8)
    cv2.fillPoly(region, [np.rint(polygon * scale).astype(np.int32)], 1)
    cv2.fillPoly(enclosure, [np.rint(quad * scale).astype(np.int32)], 1)
    area = int(np.count_nonzero(region))
    retained = int(np.count_nonzero(region & enclosure))
    coverage = retained / area if area else 0.0
    return {"mask_coverage": coverage, "excluded_mask_area_fraction": 1.0 - coverage,
            "mask_coverage_raster_scale": scale}


def polygon_to_quad(
    polygon: Sequence[Sequence[float]],
    image_shape: Sequence[int],
    *,
    crop_only: bool = False,
    strict_boundary: bool = False,
    allow_perspective: bool | None = None,
    margin: float = 0.0,
    min_area: float = 16.0,
    min_coverage: float = 0.97,
    max_enclosure_ratio: float = 1.35,
    min_edge_support: float = 0.70,
    min_edge_span: float = 0.65,
) -> dict[str, Any]:
    """@luna-comment
    Input: Visible-mask xy polygon in original-image pixels and image H,W shape.
    Output: Supported boundary quad, approximate rotated envelope, or diagnostic crop.
    Key Steps: Fit four supported lines; otherwise enclose the hull unless strict mode.
    Key Params: Strict mode rejects failed fits; approximation retains visible mask extent.
    State/Dependencies: OpenCV/NumPy; legacy allow_perspective is a no-op compatibility flag.
    """
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if min(height, width) < 2:
        raise ValueError("Image dimensions must be at least two pixels.")
    thresholds = [margin, min_area, min_coverage, max_enclosure_ratio, min_edge_support, min_edge_span]
    if not np.isfinite(thresholds).all() or margin < 0 or min_area <= 0:
        raise ValueError("Geometry thresholds must be finite and margin nonnegative.")
    if not 0 < min_coverage <= 1 or max_enclosure_ratio < 1 or not 0 < min_edge_support <= 1 or not 0 < min_edge_span <= 1:
        raise ValueError("Invalid coverage, enclosure, or edge-support threshold.")
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Visible polygon must contain at least three xy points.")
    if not np.isfinite(points).all():
        raise ValueError("Visible polygon contains nonfinite coordinates.")
    raw_points = points.copy()
    clipped = np.clip(points, [0, 0], [width - 1, height - 1]).astype(np.float32)
    clipping_applied = not np.array_equal(clipped, points)
    points = clipped
    hull = cv2.convexHull(points).reshape(-1, 2)
    hull_area = float(cv2.contourArea(hull))
    polygon_area = float(cv2.contourArea(points))
    if len(hull) < 3 or hull_area < min_area or polygon_area < min_area:
        raise ValueError("Visible polygon area is degenerate or below min_area.")

    lower = np.maximum(np.floor(points.min(axis=0) - margin), [0, 0])
    upper = np.minimum(np.ceil(points.max(axis=0) + margin), [width - 1, height - 1])
    frame_contact = bool(
        np.any(points <= [0.5, 0.5])
        or np.any(points >= [width - 1.5, height - 1.5])
    )
    evidence: dict[str, Any] = {
        "source": "predicted_visible_mask_boundary",
        "method": "uniform_contour_hull_seed_huber_lines_bounded_support_intersections",
        "four_edge_fit_passed": False,
        "physical_planarity_verified": False,
        "true_aspect_ratio_known": False,
        "thresholds": {
            "min_area_pixels": min_area, "min_mask_coverage": min_coverage,
            "max_enclosure_area_ratio": max_enclosure_ratio,
            "min_edge_support_fraction": min_edge_support, "min_edge_support_span": min_edge_span,
            "edge_tolerance_rule": "max(1.5px,min(0.015*diagonal,0.035*edge_length))",
            "raw_support_margin_pixels": margin,
            "supported_boundary_outward_quantile": 0.10,
            "mask_coverage_method": "raster_intersection_longest_side_at_most_1024",
        },
    }
    if crop_only:
        selected = order_quad([lower, [upper[0], lower[1]], upper, [lower[0], upper[1]]])
        evidence["method"] = "axis_aligned_diagnostic_crop_only"
    else:
        selected = None
    best = None
    best_error = math.inf
    perimeter = cv2.arcLength(hull, True)
    diagonal = max(float(np.linalg.norm(upper - lower)), 1.0)
    samples = sample_contour(points)
    attempts = []
    seen = set()
    for epsilon in (() if crop_only else (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)):
        approximation = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
        if len(approximation) != 4:
            continue
        key = tuple(order_quad(approximation).ravel())
        if key in seen:
            continue
        seen.add(key)
        try:
            fitted = fit_boundary_lines(approximation, samples, diagonal)
            candidate = enclosing_quad(fitted, samples, margin)
        except (ValueError, np.linalg.LinAlgError) as error:
            attempts.append({"reason": str(error), "seed_epsilon_fraction": epsilon})
            continue
        # Snap numerical noise only; substantive out-of-frame intersections are rejected.
        if np.any(candidate < -0.001) or np.any(candidate > [width - 0.999, height - 0.999]):
            attempts.append({"reason": "boundary_intersections_outside_observed_image"})
            continue
        candidate = np.clip(candidate, [0, 0], [width - 1, height - 1]).astype(np.float32)
        candidate_area = float(cv2.contourArea(candidate))
        intersection, _ = cv2.intersectConvexConvex(hull, candidate)
        coverage = min(1.0, max(0.0, float(intersection) / hull_area))
        ratio = candidate_area / hull_area
        per_edge = boundary_support(candidate, samples, diagonal)
        candidate_evidence = {
            "seed_epsilon_fraction": epsilon, "hull_coverage": coverage,
            "enclosure_area_ratio": ratio, "edges": per_edge,
            **mask_coverage(points, candidate, image_shape),
        }
        edges = np.roll(candidate, -1, axis=0) - candidate
        relative = samples[:, None, :] - candidate[None, :, :]
        signed = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
        candidate_evidence["excluded_contour_sample_fraction"] = float(np.any(signed < -1e-3, axis=1).mean())
        reason = None
        if candidate_area < min_area or candidate_evidence["mask_coverage"] < min_coverage or ratio > max_enclosure_ratio:
            reason = "boundary_area_or_coverage_gate_failed"
        elif any(edge["sample_count"] < 6 or edge["support_fraction"] < min_edge_support
                 or edge["support_span"] < min_edge_span for edge in per_edge):
            reason = "insufficient_straight_support_on_all_four_edges"
        if reason:
            attempts.append({"reason": reason, **candidate_evidence})
            continue
        boundary_error = sum(edge["residual_q70_pixels"] for edge in per_edge) / diagonal
        if boundary_error < best_error:
            best = candidate
            best_error = boundary_error
            evidence.update(four_edge_fit_passed=True, **candidate_evidence)
    geometry_type = "diagnostic_crop" if crop_only else "boundary_quad"
    selection_reason = "explicit_crop_only_diagnostic" if crop_only else "four_observed_boundary_lines_supported"
    if not crop_only:
        if best is None:
            evidence["attempts"] = attempts
            reason = attempts[-1]["reason"] if attempts else "no_four_edge_hull_seed"
            if strict_boundary:
                raise BoundaryFitError(reason, evidence)
            rectangle = cv2.minAreaRect(hull)
            center, size, angle = rectangle
            selected = order_quad(cv2.boxPoints((center, (size[0] + 2 * margin, size[1] + 2 * margin), angle)))
            geometry_type = "approximate_quad"
            selection_reason = "boundary_fit_failed_using_oriented_mask_envelope"
            evidence.update(
                method="mask_convex_hull_min_area_rotated_rectangle",
                boundary_fit_failure_reason=reason,
                approximation_is_physical_boundary=False,
                **mask_coverage(points, selected, image_shape),
            )
        else:
            selected = best
    return {
        "quad": selected.astype(float).tolist(),
        "geometry_type": geometry_type,
        "counted_as_boundary_success": geometry_type == "boundary_quad",
        "counted_as_requested_quad": not crop_only,
        "selection_reason": selection_reason,
        "raw_contour_xy": raw_points.astype(float).tolist(),
        "coordinate_space": "original_image_pixel_centers",
        "corner_order": "clockwise_with_edge_pair_closest_to_image_axes",
        "semantic_upright_known": False,
        "geometry_evidence": evidence,
        "frame_contact": frame_contact,
        "visible_mask_coordinates_clipped": clipping_applied,
        "hidden_geometry_recovered": False,
        "occlusion_status": "unknown",
        "text_completeness_verified": False,
        "rectification_guaranteed": False,
        "extent_semantics": {"diagnostic_crop": "diagnostic_visible_envelope", "boundary_quad": "fitted_observed_document_boundary", "approximate_quad": "oriented_visible_mask_envelope"}[geometry_type],
    }


def warp_quad(
    image: np.ndarray,
    quad: Sequence[Sequence[float]],
    *,
    max_side: int = 2048,
    max_pixels: int = 4_000_000,
    allow_outside: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """@luna-comment
    Input: BGR image and convex quad, optionally extending beyond observed pixels.
    Output: Bounded image and JSON-ready transform/size/heuristic metadata.
    Key Steps: Use longest opposing edge lengths plus one; cap size; warp image.
    Key Params: Output allocation caps; outside extent bounded by one image diagonal.
    State/Dependencies: OpenCV interpolation; image is not modified; ratio is heuristic.
    """
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("Expected a nonempty three-channel BGR image.")
    if max_side < 2 or max_pixels < 4:
        raise ValueError("Output limits must allow at least a 2-by-2 image.")
    ordered = order_quad(quad)
    height, width = image.shape[:2]
    outside = bool(np.any(ordered < 0) or np.any(ordered > [width - 1, height - 1]))
    if outside and not allow_outside:
        raise ValueError("Warp quad must stay inside observed image coordinates.")
    diagonal = math.hypot(width, height)
    if np.any(ordered < -diagonal) or np.any(ordered > np.array([width - 1, height - 1]) + diagonal):
        raise ValueError("Warp quad exceeds the bounded outside-image extent.")
    lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    target_width = max(2, int(math.ceil(float(max(lengths[0], lengths[2])) - 1e-4)) + 1)
    target_height = max(2, int(math.ceil(float(max(lengths[1], lengths[3])) - 1e-4)) + 1)
    scale = min(
        1.0,
        max_side / max(target_width, target_height),
        math.sqrt(max_pixels / (target_width * target_height)),
    )
    output_width = max(2, int(math.floor(target_width * scale)))
    output_height = max(2, int(math.floor(target_height * scale)))
    if output_width * output_height > max_pixels:
        if output_width >= output_height:
            output_width = max(2, max_pixels // output_height)
        else:
            output_height = max(2, max_pixels // output_width)
    destination = np.array(
        [[0, 0], [output_width - 1, 0],
         [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
        raise ValueError("Warp transform is singular or nonfinite.")
    warped = cv2.warpPerspective(
        image, transform, (output_width, output_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return warped, {
        "output_width": output_width,
        "output_height": output_height,
        "image_to_output": transform.tolist(),
        "transform_type": "homography",
        "orientation_policy": "minimum_image_axis_rotation_not_semantic_upright",
        "semantic_upright_known": False,
        "size_rule": "maximum_opposing_pixel_edge_lengths_plus_one",
        "metric_aspect_ratio_recovered": False,
        "output_downscaled": scale < 1.0,
        "hidden_pixels_reconstructed": False,
        "padding_applied": outside,
        "unknown_region_padding": "constant_white_not_reconstructed" if outside else "none",
        "border_mode": "constant_white",
        "outside_extent_limit_pixels": diagonal,
    }


def read_image(path: Path) -> np.ndarray:
    """@luna-comment
    Input: Existing local image path, including non-ASCII characters.
    Output: Decoded BGR image; raises for an unreadable or unsupported image.
    Key Steps: Read bytes through pathlib and decode with OpenCV.
    Key Params: IMREAD_COLOR provides a consistent three-channel input.
    State/Dependencies: Reads one file; no external downloads or shared state.
    """
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """@luna-comment
    Input: Output path and BGR image supported by its filename extension.
    Output: Writes encoded image bytes, with non-ASCII filesystem paths supported.
    Key Steps: Encode with OpenCV, then write bytes using pathlib.
    Key Params: The suffix selects the codec; output directories must exist.
    State/Dependencies: Writes only the supplied artifact path.
    """
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def collect_images(source: Path, *, recursive: bool = False) -> list[Path]:
    """@luna-comment
    Input: Image file or directory and optional recursive enumeration flag.
    Output: Deterministically sorted image paths; rejects empty or missing sources.
    Key Steps: Check source type, filter known extensions and sort paths.
    Key Params: Recursive is opt-in to avoid traversing unintended datasets.
    State/Dependencies: Filesystem metadata only; no image decoding or model loading.
    """
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {source}")
        return [source]
    if not source.is_dir():
        raise ValueError(f"Source does not exist: {source}")
    iterator = source.rglob("*") if recursive else source.iterdir()
    paths = sorted(
        (item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: str(item).casefold(),
    )
    if not paths:
        raise ValueError(f"No supported images found: {source}")
    return paths


def process_result(
    result: Any,
    image: np.ndarray,
    *,
    confidence: float = 0.25,
    crop_only: bool = False,
    strict_boundary: bool = False,
    allow_perspective: bool | None = None,
    margin: float = 0.0,
    min_area: float = 16.0,
    max_side: int = 2048,
    max_pixels: int = 4_000_000,
) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
    """@luna-comment
    Input: Public YOLO-like result, original BGR image and geometry/output options.
    Output: JSON-ready per-image record, overlay and one warped image per instance.
    Key Steps: Draw raw contours; fit boundaries or oriented mask envelopes; warp instances.
    Key Params: Strict boundary mode rejects approximations; crop_only is diagnostic.
    State/Dependencies: Public result fields only; no torch or Ultralytics import.
    """
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be in [0,1].")
    if tuple(getattr(result, "orig_shape", image.shape[:2])) != tuple(image.shape[:2]):
        raise ValueError("Prediction original-image shape does not match the supplied image.")
    overlay = image.copy()
    payload: dict[str, Any] = {
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "instances": [],
        "rejected": [],
        "boundary_quad_count": 0,
        "approximate_quad_count": 0,
        "warp_count": 0,
        "diagnostic_crop_count": 0,
        "overlay_legend": {"raw_contour": "magenta", "boundary_quad": "green", "approximate_quad": "cyan", "diagnostic_crop": "orange"},
        "limitations": [
            "Predicted visible support does not establish full text coverage.",
            "Occluded and out-of-frame geometry or text is not recovered.",
            "Boundary gates do not prove planarity or metric aspect ratio.",
            "Upstream detector NMS can already have removed overlapping instances.",
        ],
    }
    crops: list[np.ndarray] = []
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or len(boxes) == 0:
        return payload, overlay, crops
    polygons = [] if masks is None else masks.xy
    if masks is not None and len(polygons) != len(boxes):
        raise ValueError("Segmentation polygon and detection counts do not match.")
    for index in range(len(boxes)):
        score = float(boxes.conf[index])
        if not math.isfinite(score) or score < confidence:
            payload["rejected"].append({"prediction_index": index, "reason": "low_confidence"})
            continue
        if masks is None:
            payload["rejected"].append({"prediction_index": index, "reason": "missing_mask"})
            continue
        raw = np.asarray(polygons[index], dtype=np.float32)
        raw_valid = raw.ndim == 2 and raw.shape[1] == 2
        raw_json = [[float(value) if np.isfinite(value) else None for value in point] for point in raw] if raw_valid else []
        if raw_valid and np.isfinite(raw).all() and len(raw) >= 2:
            cv2.polylines(overlay, [np.rint(raw).astype(np.int32)], True, (210, 70, 210), 1, cv2.LINE_AA)
        try:
            instance = polygon_to_quad(
                polygons[index], image.shape, crop_only=crop_only,
                strict_boundary=strict_boundary,
                margin=margin, min_area=min_area,
            )
            crop, warp_metadata = warp_quad(
                image, instance["quad"], max_side=max_side, max_pixels=max_pixels,
                allow_outside=instance["geometry_type"] == "approximate_quad",
            )
        except ValueError as error:
            payload["rejected"].append({
                "prediction_index": index, "confidence": score, "class_id": int(boxes.cls[index]),
                "reason": str(error), "raw_contour_xy": raw_json,
                "geometry_evidence": getattr(error, "evidence", {}),
            })
            continue
        instance.update(
            instance_id=len(crops), prediction_index=index, confidence=score,
            class_id=int(boxes.cls[index]), warp=warp_metadata,
        )
        payload["instances"].append(instance)
        crops.append(crop)
        is_boundary = instance["geometry_type"] == "boundary_quad"
        kind = instance["geometry_type"]
        payload[kind + "_count"] += 1
        payload["warp_count"] += int(kind != "diagnostic_crop")
        color = (40, 220, 40) if is_boundary else ((255, 220, 0) if kind == "approximate_quad" else (0, 180, 255))
        vertices = np.rint(instance["quad"]).astype(np.int32)
        cv2.polylines(overlay, [vertices], True, color, 2, cv2.LINE_AA)
        anchor = tuple(vertices[0] + [0, 14])
        cv2.putText(
            overlay, f"{instance['instance_id']} {score:.2f}", anchor,
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return payload, overlay, crops


def write_result(
    output_directory: Path,
    payload: dict[str, Any],
    overlay: np.ndarray,
    crops: Sequence[np.ndarray],
) -> Path:
    """@luna-comment
    Input: Per-image directory, result record, overlay and aligned instance crops.
    Output: result.json, overlay.png and instance_NNN.png; returns JSON path.
    Key Steps: Create directory; write images; bind relative names; write strict JSON.
    Key Params: Array counts must agree; empty detections still write JSON/overlay.
    State/Dependencies: Filesystem artifacts only; existing same-name outputs replaced.
    """
    if len(crops) != len(payload["instances"]):
        raise ValueError("Crop count must equal retained instance count.")
    output_directory.mkdir(parents=True, exist_ok=True)
    write_image(output_directory / "overlay.png", overlay)
    for index, (instance, crop) in enumerate(zip(payload["instances"], crops)):
        filename = f"instance_{index:03d}.png"
        write_image(output_directory / filename, crop)
        instance["image_file"] = filename
    payload["overlay_file"] = "overlay.png"
    result_path = output_directory / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """@luna-comment
    Input: Optional CLI argument list; otherwise reads process arguments.
    Output: Validated namespace, or argparse help/error exit.
    Key Steps: Define weights, boundary/envelope warp controls and strict opt-in mode.
    Key Params: Crop-only is diagnostic; strict-boundary forbids approximate envelopes.
    State/Dependencies: argparse only; help does not import torch or load weights.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Existing local YOLO segmentation .pt weights.")
    parser.add_argument("--source", type=Path, required=True, help="Image or directory; non-ASCII paths supported.")
    parser.add_argument("--output", type=Path, required=True, help="Output root containing one directory per input.")
    parser.add_argument("--recursive", action="store_true", help="Include nested source directories.")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.8, help="Upstream NMS threshold; overlap recall needs validation.")
    parser.add_argument("--max-det", type=int, default=100)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--crop-only", action="store_true", help="Diagnostic axis-aligned crop only; NOT a successful boundary fit.")
    mode.add_argument("--strict-boundary", action="store_true", help="Reject failed four-edge fits instead of returning approximate oriented quads.")
    mode.add_argument("--allow-perspective", action="store_true", help="Deprecated compatibility flag: boundary fitting and homography are already the default.")
    parser.add_argument("--margin", type=float, default=0.0, help="Outward padding in original-image pixels.")
    parser.add_argument("--min-area", type=float, default=16.0)
    parser.add_argument("--max-side", type=int, default=2048, help="Maximum crop/warp side in pixels.")
    parser.add_argument("--max-pixels", type=int, default=4_000_000, help="Maximum crop/warp pixel count.")
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf and --iou must be in [0,1].")
    if args.imgsz < 32 or args.max_det < 1 or args.max_side < 2 or args.max_pixels < 4:
        parser.error("Invalid image size, detection count, or output limits.")
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("--margin must be finite and nonnegative.")
    if not math.isfinite(args.min_area) or args.min_area <= 0:
        parser.error("--min-area must be finite and positive.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """@luna-comment
    Input: CLI arguments selecting existing local weights and image inputs.
    Output: Artifacts plus run.json and a machine-readable final stdout summary.
    Key Steps: Validate paths; lazy-load YOLO; infer each image; write results/errors.
    Key Params: Default boundary or approximate envelope homography; optional strict mode.
    State/Dependencies: Public Ultralytics API; no training, downloads or extra NMS.
    """
    args = parse_args(argv)
    if not args.weights.is_file() or args.weights.suffix.lower() != ".pt":
        raise ValueError("--weights must reference an existing local .pt file.")
    images = collect_images(args.source, recursive=args.recursive)
    from ultralytics import YOLO

    model = YOLO(str(args.weights), task="segment")
    if model.task != "segment":
        raise ValueError("Weights must contain a segmentation model.")
    args.output.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {
        "weights": str(args.weights.resolve()),
        "source": str(args.source.resolve()),
        "mode": "crop_only_diagnostic" if args.crop_only else ("strict_boundary" if args.strict_boundary else "boundary_or_approximate_quad"),
        "confidence_threshold": args.conf,
        "upstream_nms_iou": args.iou,
        "configuration": {
            "imgsz": args.imgsz, "device": args.device, "max_det": args.max_det,
            "margin": args.margin, "min_area": args.min_area,
            "max_side": args.max_side, "max_pixels": args.max_pixels,
        },
        "results": [],
        "errors": [],
    }
    source_root = args.source if args.source.is_dir() else args.source.parent
    for source_path in images:
        try:
            image = read_image(source_path)
            results = model.predict(
                source=image, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                max_det=args.max_det, device=args.device, retina_masks=True,
                verbose=False, save=False,
            )
            if len(results) != 1:
                raise ValueError("Expected exactly one prediction result for one image.")
            payload, overlay, crops = process_result(
                results[0], image, confidence=args.conf,
                crop_only=args.crop_only, margin=args.margin,
                strict_boundary=args.strict_boundary,
                min_area=args.min_area, max_side=args.max_side, max_pixels=args.max_pixels,
            )
            payload["source"] = str(source_path.resolve())
            relative = source_path.relative_to(source_root)
            result_directory = args.output / relative.parent / relative.name
            result_path = write_result(result_directory, payload, overlay, crops)
            run["results"].append({
                "source": str(source_path.resolve()),
                "result_json": str(result_path.resolve()),
                "instance_count": len(crops),
                "boundary_quad_count": payload["boundary_quad_count"],
                "approximate_quad_count": payload["approximate_quad_count"],
                "warp_count": payload["warp_count"],
                "diagnostic_crop_count": payload["diagnostic_crop_count"],
            })
        except (OSError, ValueError, cv2.error) as error:
            run["errors"].append({"source": str(source_path), "error": str(error)})
    run["status"] = "completed" if not run["errors"] else "failed"
    run_path = args.output / "run.json"
    run_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(
        {"status": run["status"], "run_json": str(run_path.resolve()), "results": run["results"],
         "error_count": len(run["errors"])}, ensure_ascii=True,
    ))
    return 0 if not run["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
