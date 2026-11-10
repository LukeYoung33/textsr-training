"""Isolated Astra document-box candidate; all estimation uses the 640-square raster."""
from __future__ import annotations

import time

import cv2
import numpy as np

CONFIG = dict(canvas=640, pitch=8, window=65, gamma=1.8, maximum_alpha=0.75,
              foreground_quantiles=[0.10, 0.90], midpoint_fraction=[0.2, 0.8],
              noise_multiple=6.0, haar_mad_denominator=0.67448975,
              resize="BGR_INTER_LINEAR_then_OpenCV_Y", exterior="identity",
              interpolation="four_LUT_outputs_at_native_Y_half_pixel_coordinates")


def curve(values, low, middle, high, alpha, gamma=1.8):
    """@luna-comment
    Input: Float Y samples, ordered scalar rails, alpha in [0,.75], gamma>=1.
    Output: Float mapped Y with identity exterior; no mutation or quantization.
    Key Steps: Evaluate two clipped power branches, select interior, mix identity.
    Key Params: Gamma1.8; degenerate rails have the exact identity limit.
    State/Dependencies: NumPy only; parameters are fixed for this node.
    """
    values = np.asarray(values, np.float64)
    if high <= low or not low < middle < high:
        return values.copy()
    left = low + (middle-low)*np.clip((values-low)/(middle-low), 0, 1)**gamma
    right = high - (high-middle)*np.clip((high-values)/(high-middle), 0, 1)**gamma
    powered = np.where(values <= middle, left, right)
    inside = (values > low) & (values < high)
    return values + alpha*np.where(inside, powered-values, 0)


def otsu_anchors(histogram):
    """@luna-comment
    Input: Nonnegative 256-bin counts from one clipped 640-domain window.
    Output: L/T/H, upper-class fraction and a degenerate Boolean.
    Key Steps: Otsu partition, class Q10/Q90, population-weighted midpoint.
    Key Params: T fraction .2+.6*p_high avoids empty-valley threshold bias.
    State/Dependencies: NumPy only; empty/single-class distributions are identity.
    """
    hist = np.asarray(histogram, np.float64)
    count = np.cumsum(hist)
    total = count[-1]
    if not total or np.count_nonzero(hist) < 2:
        value = float(np.argmax(hist))
        return value, value, value, 0., True
    moment = np.cumsum(hist*np.arange(256))
    denominator = count*(total-count)
    variance = np.divide((total*moment-count*moment[-1])**2, denominator,
                         out=np.zeros(256), where=denominator > 0)
    threshold = int(np.argmax(variance))
    lower = hist[:threshold+1]
    upper = hist[threshold+1:]
    low = float(np.searchsorted(np.cumsum(lower), np.ceil(lower.sum()*.1)))
    high = float(threshold+1+np.searchsorted(np.cumsum(upper), np.ceil(upper.sum()*.9)))
    fraction = float(upper.sum()/total)
    middle = low+(.2+.6*fraction)*(high-low)
    return low, middle, high, fraction, high <= low


def estimate(image):
    """@luna-comment
    Input: Native uint8 BGR full-document raster; geometry is unchanged.
    Output: Immutable-use grid LUTs, rails, sigma/alpha fields, canvas and timings.
    Key Steps: Resize640, local histograms, Haar absolute MAD, node curve construction.
    Key Params: Pitch8/window65; alpha=.75*d²/(d²+(6sigma)²+1).
    State/Dependencies: OpenCV/NumPy; native statistical pixel count is exactly zero.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected native uint8 BGR")
    start = time.perf_counter()
    small = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    signed = canvas.astype(np.int16)
    # Twice the normalized Haar coefficient: integer bins retain half-code values.
    haar2 = np.abs(signed[:-1, :-1]-signed[:-1, 1:]-signed[1:, :-1]+signed[1:, 1:])
    haar2 = np.pad(haar2, ((0, 1), (0, 1)), mode="edge")
    nodes = np.r_[np.arange(0, 640, 8), 639].astype(np.int32)
    shape = (len(nodes), len(nodes))
    anchors = np.zeros((*shape, 3), np.float32)
    sigma = np.zeros(shape, np.float32)
    alpha = np.zeros(shape, np.float32)
    fraction = np.zeros(shape, np.float32)
    tables = np.empty((*shape, 256), np.float32)
    degenerate = 0
    values = np.arange(256, dtype=np.float64)
    for iy, cy in enumerate(nodes):
        y0, y1 = max(0, cy-32), min(640, cy+33)
        for ix, cx in enumerate(nodes):
            x0, x1 = max(0, cx-32), min(640, cx+33)
            patch = canvas[y0:y1, x0:x1]
            hist = np.bincount(patch.ravel(), minlength=256)
            low, mid, high, population, flat = otsu_anchors(hist)
            noise_hist = np.bincount(haar2[y0:y1, x0:x1].ravel(), minlength=511)
            half_median = np.searchsorted(np.cumsum(noise_hist), (patch.size+1)//2)/2
            noise = float(half_median/CONFIG['haar_mad_denominator'])
            span = high-low
            strength = .75*span*span/(span*span+(6*noise)**2+1)
            anchors[iy, ix] = [low, mid, high]
            sigma[iy, ix], alpha[iy, ix], fraction[iy, ix] = noise, strength, population
            tables[iy, ix] = curve(values, low, mid, high, strength)
            degenerate += int(flat)
    if np.min(np.diff(tables, axis=2)) < -1e-5:
        raise AssertionError("Nonmonotone node LUT")
    return dict(nodes=nodes, tables=tables, anchors=anchors, sigma=sigma, alpha=alpha,
                high_fraction=fraction, canvas=canvas, config=CONFIG.copy(),
                native_shape=list(image.shape[:2]), native_statistical_pixels=0,
                analysis_shape=[640, 640], degenerate_nodes=degenerate,
                prepare_ms=1000*(time.perf_counter()-start))


def axis_map(length, nodes):
    """@luna-comment
    Input: Positive native axis length and increasing analysis node centers.
    Output: Left/right node indices and float interpolation weights per native pixel.
    Key Steps: Direct-resize half-pixel inverse geometry, boundary clamp, bracket search.
    Key Params: Analysis extent640; last node639 has a seven-pixel final interval.
    State/Dependencies: NumPy only; no image-content statistics.
    """
    position = np.clip((np.arange(length, dtype=np.float64)+.5)*640/length-.5,
                       nodes[0], nodes[-1])
    right = np.clip(np.searchsorted(nodes, position, side='right'), 1, len(nodes)-1)
    left = right-1
    weight = ((position-nodes[left])/(nodes[right]-nodes[left])).astype(np.float32)
    return left, right, weight


def render(image, context, strength=1., chunk=128):
    """@luna-comment
    Input: Matching native BGR/context, fixed strength in [0,1], positive row chunk.
    Output: New native BGR and path/gamut/timing telemetry, source unchanged.
    Key Steps: Gather four LUT values at original Y, interpolate, common gamut clamp.
    Key Params: LOW strength.5/STANDARD1; ties-even integer delta; no native estimation.
    State/Dependencies: Context is read-only; bounded row arrays, OpenCV Y conversion.
    """
    if not 0 <= strength <= 1 or chunk <= 0 or list(image.shape[:2]) != context['native_shape']:
        raise ValueError("Invalid strength/chunk/geometry")
    start = time.perf_counter()
    height, width = image.shape[:2]
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    xl, xr, wx = axis_map(width, context['nodes'])
    yl, yr, wy = axis_map(height, context['nodes'])
    output = np.empty_like(image)
    tables = context['tables']
    limited = 0
    for top in range(0, height, chunk):
        bottom = min(top+chunk, height)
        original_y = y[top:bottom]
        upper_left = tables[yl[top:bottom, None], xl[None, :], original_y]
        upper_right = tables[yl[top:bottom, None], xr[None, :], original_y]
        lower_left = tables[yr[top:bottom, None], xl[None, :], original_y]
        lower_right = tables[yr[top:bottom, None], xr[None, :], original_y]
        upper = upper_left+(upper_right-upper_left)*wx[None, :]
        lower = lower_left+(lower_right-lower_left)*wx[None, :]
        mapped = upper+(lower-upper)*wy[top:bottom, None]
        requested = np.rint((mapped-original_y)*strength).astype(np.int16)
        source = image[top:bottom].astype(np.int16)
        bounded = np.clip(requested, -source.min(axis=2), 255-source.max(axis=2))
        limited += int(np.count_nonzero(bounded != requested))
        output[top:bottom] = (source+bounded[:, :, None]).astype(np.uint8)
    return output, dict(render_ms=1000*(time.perf_counter()-start),
                        gamut_limited_pixels=limited, strength=strength, chunk_rows=chunk,
                        native_statistical_pixels=0, detector_calls=0, ocr_calls=0,
                        warp_calls=0, model_calls=0)
