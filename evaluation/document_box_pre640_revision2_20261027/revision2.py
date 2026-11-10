"""640-only continuous-role rails and explicit two-sided overshoot compression."""
from __future__ import annotations

import time

import cv2
import numpy as np

from tools.adaptive_scurve_robust_v2 import cluster_mode
from evaluation.document_box_pre640_astra_20261027 import pre640 as frozen_renderer

CONFIG = dict(version='pre640_revision2_soft_exterior', canvas=640, pitch=8, window=65,
              low_core=.10, high_core=.90, cluster_sigma=4., role_transition=.25,
              tail_tau=2., contrast_support=24., low_slope=.40, standard_slope=.10,
              interior='identity', resize='BGR_INTER_LINEAR_then_OpenCV_Y',
              native_renderer='unchanged_pre640_half_pixel_four_LUT_common_gamut')


def smoothstep(value):
    """@luna-comment
    Input: Scalar or array role evidence.
    Output: Continuous0..1 background-role weight.
    Key Steps: Clamp then cubic smoothstep.
    Key Params: Endpoints0/1; no binary polarity decision.
    State/Dependencies: NumPy only, no external state.
    """
    value = np.clip(value, 0., 1.)
    return value*value*(3-2*value)


def rails(histogram):
    """@luna-comment
    Input: One256-bin640-window histogram.
    Output: Ordered L/H and class-core/cluster/population diagnostics.
    Key Steps: Otsu partition, conservative cores, continuous background-only cluster mix.
    Key Params: Q10/Q90; role transition.25; originalV2cluster_sigma4.
    State/Dependencies: Existing signed cluster_mode; uniform distribution exact limit.
    """
    histogram = np.asarray(histogram, np.float64)
    count = np.cumsum(histogram)
    total = count[-1]
    if not total or np.count_nonzero(histogram) < 2:
        value = float(np.argmax(histogram))
        return dict(low=value, high=value, low_core=value, high_core=value,
                    low_cluster=value, high_cluster=value, high_fraction=0.,
                    dark_role=0., light_role=0., degenerate=True)
    moment = np.cumsum(histogram*np.arange(256))
    denominator = count*(total-count)
    variance = np.divide((total*moment-count*moment[-1])**2, denominator,
                         out=np.zeros(256), where=denominator > 0)
    threshold = int(np.argmax(variance))
    lower, upper = histogram[:threshold+1], histogram[threshold+1:]
    low_core = float(np.searchsorted(np.cumsum(lower), np.ceil(lower.sum()*.10)))
    high_core = float(threshold+1+np.searchsorted(np.cumsum(upper), np.ceil(upper.sum()*.90)))
    low_cluster = class_cluster(histogram, 0, threshold)
    high_cluster = class_cluster(histogram, threshold+1, 255)
    population = float(upper.sum()/total)
    dark_role = float(smoothstep((.5-population)/.25))
    light_role = float(smoothstep((population-.5)/.25))
    low = low_core+dark_role*(low_cluster-low_core)
    high = high_core+light_role*(high_cluster-high_core)
    if not low < high:
        raise ValueError('Unordered class rails')
    return dict(low=low, high=high, low_core=low_core, high_core=high_core,
                low_cluster=low_cluster, high_cluster=high_cluster,
                high_fraction=population, dark_role=dark_role, light_role=light_role,
                degenerate=False)


def class_cluster(histogram, start, stop):
    """@luna-comment
    Input: Histogram and inclusive class bounds in original Y units.
    Output: Original-unit maximal-mass cluster center, without endpoint peak inflation.
    Key Steps: Zero-pad class, invoke unchangedV2clustering, undo bin offset.
    Key Params: Sixteen zero bins on each side give four Gaussian sigma4 widths.
    State/Dependencies: Existing cluster_mode is unchanged; only new class wrapper differs.
    """
    padded = np.pad(np.asarray(histogram[start:stop+1], np.float64), (16, 16))
    return float(cluster_mode(padded, 0, len(padded)-1)-16+start)


def curve(values, low, high, slope, tau=2., support_scale=24.):
    """@luna-comment
    Input: Float Y, local L/H, exterior slope in[0,1], positive tau/support.
    Output: Monotone float mapping, identity interior and smooth compressed exterior.
    Key Steps: Signed distance beyond rails, exponential residual compression, span support.
    Key Params: STANDARDs.1/LOWs.4,tau2,support24; zero-span exact identity.
    State/Dependencies: NumPy only; no fitted native data or output-based switch.
    """
    values = np.asarray(values, np.float64)
    if not 0 <= slope <= 1 or tau <= 0 or support_scale <= 0 or high < low:
        raise ValueError('Invalid tail parameters')
    span = high-low
    weight = span*span/(span*span+support_scale*support_scale)
    below, above = np.maximum(low-values, 0), np.maximum(values-high, 0)
    lower_tail = slope*below+(1-slope)*tau*(-np.expm1(-below/tau))
    upper_tail = slope*above+(1-slope)*tau*(-np.expm1(-above/tau))
    compressed = np.where(values < low, low-lower_tail,
                          np.where(values > high, high+upper_tail, values))
    return values+weight*(compressed-values)


def estimate(image):
    """@luna-comment
    Input: Native uint8 BGR full-document raster.
    Output:640-only rails, role telemetry and two node LUT fields.
    Key Steps: Direct640resize, local histograms/roles, fixed LOW/STANDARD soft tails.
    Key Params: Nodes0,8,..632,639; clipped65window; no middleS/no noisealpha.
    State/Dependencies: Existing cluster function, OpenCV/NumPy; no native statistics.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 BGR')
    started = time.perf_counter()
    canvas = cv2.cvtColor(cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR),
                          cv2.COLOR_BGR2YCrCb)[:, :, 0]
    nodes = np.r_[np.arange(0, 640, 8), 639].astype(np.int32)
    shape = (len(nodes), len(nodes))
    fields = {key: np.empty(shape, np.float32) for key in
              ['low', 'high', 'low_core', 'high_core', 'low_cluster', 'high_cluster',
               'high_fraction', 'dark_role', 'light_role']}
    tables = {arm: np.empty((*shape, 256), np.float32) for arm in ['LOW', 'STANDARD']}
    values = np.arange(256)
    degenerate = 0
    for iy, cy in enumerate(nodes):
        for ix, cx in enumerate(nodes):
            patch = canvas[max(0, cy-32):min(640, cy+33), max(0, cx-32):min(640, cx+33)]
            model = rails(np.bincount(patch.ravel(), minlength=256))
            for key in fields:
                fields[key][iy, ix] = model[key]
            for arm, slope in [('LOW', .4), ('STANDARD', .1)]:
                tables[arm][iy, ix] = curve(values, model['low'], model['high'], slope)
            degenerate += int(model['degenerate'])
    if any(np.min(np.diff(table, axis=2)) < -1e-5 for table in tables.values()):
        raise AssertionError('Nonmonotone LUT')
    return dict(nodes=nodes, tables=tables, fields=fields, canvas=canvas,
                native_shape=list(image.shape[:2]), analysis_shape=[640, 640],
                native_statistical_pixels=0, degenerate_nodes=degenerate,
                config=CONFIG.copy(), prepare_ms=1000*(time.perf_counter()-started))


def render(image, context, arm='STANDARD', chunk=128):
    """@luna-comment
    Input: Matching native BGR, frozen context, LOW/STANDARD and chunk size.
    Output: Exact existing-renderer BGR and explicit revised-branch telemetry.
    Key Steps: Select declared LUT field, delegate unchanged half-pixel native mapping.
    Key Params: Runtime strength1; LOW differs by tail slope, not post-hoc identity mix.
    State/Dependencies: frozen_renderer.render unchanged; no new native estimation.
    """
    if arm not in context['tables']:
        raise ValueError('Unknown arm')
    packed = dict(nodes=context['nodes'], tables=context['tables'][arm],
                  native_shape=context['native_shape'])
    result, telemetry = frozen_renderer.render(image, packed, strength=1., chunk=chunk)
    telemetry.update(version=CONFIG['version'], arm=arm,
                     exterior_slope=.4 if arm == 'LOW' else .1,
                     tau=2., support_scale=24., interior='identity')
    return result, telemetry
