"""Experimental detector-canvas S-curves; frozen V1 remains the deployment baseline."""
import time

import cv2
import numpy as np

from evaluation.adaptive_prepost_frozen_v1_20261020.lowres_curve import (
    strip_histograms, prepare_mapping, apply_tables,
)


def regularize_field(values, weights, sigma, floor=32.):
    """@luna-comment
    Input: Nonempty float candidates, positive weights, Gaussian sigma and value floor.
    Output: Positive float32 continuous background nodes, including zero reliable seeds.
    Key Steps: Normalized convolution then two bounded Cauchy residual reweightings.
    Key Params: Residual scale8 Y; background floor32, raw current rail floor0.
    State/Dependencies: OpenCV replicated-end filtering; no native pixels or mutable globals.
    """
    values = np.asarray(values, np.float32)[None, :]
    weights = np.asarray(weights, np.float32)[None, :]
    active = weights
    for _ in range(3):
        numerator = cv2.GaussianBlur(values*active, (0, 0), sigmaX=sigma,
                                     borderType=cv2.BORDER_REPLICATE)
        denominator = cv2.GaussianBlur(active, (0, 0), sigmaX=sigma,
                                       borderType=cv2.BORDER_REPLICATE)
        result = numerator/np.maximum(denominator, 1e-6)
        active = weights/(1+((values-result)/8)**2)
    return np.maximum(result[0], floor).astype(np.float32)


def cluster_mode(histogram, start, stop):
    """@luna-comment
    Input: Nonnegative normalized histogram and inclusive valid class bounds.
    Output: Float cluster center chosen by integrated local mass, not tallest bin.
    Key Steps: Smooth class, separate significant peaks at deep valleys; choose total mass.
    Key Params: Gaussian sigma4; peak floor5%, valley ratio0.5; empty class midpoint.
    State/Dependencies: Small NumPy vectors only, no external state.
    """
    part = np.asarray(histogram[start:stop+1], np.float64)
    if not part.sum():
        return (start+stop)/2
    smooth = cv2.GaussianBlur(part[None, :], (0, 0), sigmaX=4,
                              borderType=cv2.BORDER_REPLICATE)[0]
    peaks = np.flatnonzero((smooth >= np.r_[-1, smooth[:-1]]) &
                           (smooth >= np.r_[smooth[1:], -1]) & (smooth >= smooth.max()*.05))
    retained = [int(peaks[0])]
    valleys = []
    for peak in peaks[1:]:
        previous = retained[-1]
        valley = previous+int(np.argmin(smooth[previous:peak+1]))
        if smooth[valley] < .5*min(smooth[previous], smooth[peak]):
            valleys.append(valley)
            retained.append(int(peak))
        elif smooth[peak] > smooth[previous]:
            retained[-1] = int(peak)
    boundaries = [0, *valleys, len(part)]
    masses = [part[a:b].sum() for a, b in zip(boundaries[:-1], boundaries[1:])]
    winner = int(np.argmax(masses))
    left, right = boundaries[winner:winner+2]
    weights = part[left:right]
    return float(np.dot(np.arange(left, right)+start, weights)/max(float(weights.sum()), 1))


def role_samples(normalized, centers, x0, modes):
    """@luna-comment
    Input: Normalized detector row, global node centers, row x origin, low/high modes.
    Output: Two sparse row-local YX coordinate arrays, at most4 per node per role.
    Key Steps: Search each4-column strip for closest original samples to each mode.
    Key Params: Rank by absolute mode distance; no output or confidence gate.
    State/Dependencies: Source-role coordinates remain frame-owned and reusable by post.
    """
    nodes, height, width = len(centers), *normalized.shape
    padded = np.pad(normalized, ((0, 0), (0, nodes*4-width)), constant_values=np.inf)
    stripes = padded.reshape(height, nodes, 4).transpose(1, 0, 2).reshape(nodes, height*4)
    samples = []
    for mode in modes:
        distance = np.abs(stripes-mode)
        count = min(4, height*4)
        nearest = np.argpartition(distance, count-1, axis=1)[:, :count]
        yy = nearest//4
        xx = nearest % 4+4*np.arange(nodes)[:, None]
        selected_distance = np.take_along_axis(distance, nearest, axis=1)
        valid = (xx < width) & (selected_distance <= 16)
        samples.append(np.stack([yy[valid], xx[valid]], axis=1).astype(np.int16))
    return samples


def estimate_line(canvas, bbox, collect_roles=False, foreground_core=True):
    """@luna-comment
    Input:1984x1472 uint8 Y canvas, half-open detector box; optional sparse roles.
    Output: Regularized field, ordered normalized rails, roles and diagnostic telemetry.
    Key Steps: Prefix quantiles, edge agreement, field fit, background cluster and foreground core.
    Key Params: Pitch4; height-scaled window/sigma; foreground10%/90% core; no seed branch.
    State/Dependencies: Frozen strip histograms; all statistical pixels are detector pixels.
    """
    x0, y0, x1, y1 = bbox
    row = canvas[y0:y1, x0:x1]
    hist = strip_histograms(row, 4)
    indices = np.arange(len(hist))
    radius = max(2, int(round(row.shape[0]/32)))
    starts, ends = np.maximum(0, indices-radius), np.minimum(len(hist), indices+radius)
    prefix = np.vstack([np.zeros((1, 256), np.int32), np.cumsum(hist, axis=0)])
    cumulative = np.cumsum(prefix[ends]-prefix[starts], axis=1)
    quantiles = np.array([np.argmax(cumulative >= np.ceil(cumulative[:, -1]*q)[:, None], axis=1)
                          for q in (.70, .85, .90)])
    edge = strip_histograms(np.concatenate([row[:2], row[-2:]]), 4)
    eprefix = np.vstack([np.zeros((1, 256), np.int32), np.cumsum(edge, axis=0)])
    ecdf = np.cumsum(eprefix[ends]-eprefix[starts], axis=1)
    edge_q85 = np.argmax(ecdf >= np.ceil(ecdf[:, -1]*.85)[:, None], axis=1)
    edge_levels = []
    for band in [row[:2], row[-2:]]:
        bh = strip_histograms(band, 4)
        bp = np.vstack([np.zeros((1, 256), np.int32), np.cumsum(bh, axis=0)])
        bcdf = np.cumsum(bp[ends]-bp[starts], axis=1)
        edge_levels.append(np.argmax(bcdf >= np.ceil(bcdf[:, -1]*.85)[:, None], axis=1))
    agreement = 1/(1+((edge_levels[0]-edge_levels[1])/16)**2)
    candidates = agreement*edge_q85+(1-agreement)*quantiles[1]
    spread = quantiles[2]-quantiles[0]
    disagreement = edge_q85-quantiles[1]
    weights = .05+.95/((1+(spread/12)**2)*(1+(disagreement/16)**2))
    sigma = max(2., row.shape[0]*.35/4)
    paper = regularize_field(candidates, weights, sigma)
    centers = np.minimum(indices*4+1.5, row.shape[1]-1).astype(np.float32)+x0
    profile = np.interp(np.arange(x0, x1), centers, paper).astype(np.float32)
    normalized = row.astype(np.float32)*128/profile[None, :]
    bins = np.clip(np.rint(normalized).astype(np.int32), 0, 1023)
    histogram = np.bincount(bins.ravel(), minlength=1024)
    count = np.cumsum(histogram, dtype=np.float64)
    moment = np.cumsum(histogram*np.arange(1024))
    denominator = count*(count[-1]-count)
    variance = np.divide((count[-1]*moment-count*moment[-1])**2, denominator,
                         out=np.zeros(1024), where=denominator > 0)
    threshold = int(np.argmax(variance))
    background_is_low = threshold >= 128
    low = (cluster_mode(histogram, 0, threshold) if background_is_low else
           float(np.argmax(histogram[:threshold+1])))
    high = (float(threshold+1+np.argmax(histogram[threshold+1:])) if background_is_low else
            cluster_mode(histogram, threshold+1, 1023))
    if foreground_core:
        if background_is_low:
            foreground = histogram[threshold+1:]
            high = float(threshold+1+np.searchsorted(np.cumsum(foreground),
                         max(1, int(np.ceil(foreground.sum()*.90)))))
        else:
            foreground = histogram[:threshold+1]
            low = float(np.searchsorted(np.cumsum(foreground),
                        max(1, int(np.ceil(foreground.sum()*.10)))))
    mid = threshold+.5
    fallback = not 0 <= low < mid < high
    if fallback:
        low, mid, high = 0., 64., 128.
    roles = role_samples(normalized, centers, x0, [low, high]) if collect_roles else []
    return dict(canvas_bbox=list(bbox), centers=centers, paper=paper, profile=profile,
                normalized_anchors=[low, mid, high], roles=roles, candidates=candidates,
                weights=weights, sigma_nodes=sigma, window_nodes=2*radius,
                numeric_fallback=fallback, all_columns=True, row_height=row.shape[0],
                background_is_low=background_is_low)


def soft_curve(source, target, gamma):
    """@luna-comment
    Input: Nx3 ordered source/target rails, finite gamma>=1.
    Output: Monotone in-gamut Nx256 Q8.8 LUT retaining S center and saturating exterior values.
    Key Steps: Clipped power branches join at the middle and saturate at both rails.
    Key Params: Zero exterior slope; gamma1.8 pre. Frozen post remains independent.
    State/Dependencies: Low-resolution arrays only; monotonicity asserted, no forced repair.
    """
    low, mid, high = [source[:, i, None] for i in range(3)]
    rl, rt, rh = [target[:, i, None] for i in range(3)]
    value = np.arange(256, dtype=np.float32)[None, :]
    left = rl+(rt-rl)*np.clip((value-low)/(mid-low), 0, 1)**gamma
    right = rh-(rh-rt)*np.clip((high-value)/(high-mid), 0, 1)**gamma
    lut = np.where(value <= mid, left, right)
    tables = np.rint(np.clip(lut, 0, 255)*256).astype(np.uint16)
    if np.any(np.diff(tables.astype(np.int32), axis=1) < 0):
        raise ValueError('Nonmonotone soft curve')
    return tables


def build_tables(model, gamma=1.8, reference=None):
    """@luna-comment
    Input: Estimated model and optional same-node reference packed rails; gamma>=1.
    Output: Frozen-renderer-compatible LUTs and rails at every node.
    Key Steps: Scale normalized anchors by B; common gamut fit; soft S/transfer evaluation.
    Key Params: No validity gates; ordered positive spans required for numerical safety.
    State/Dependencies: soft_curve; reference belongs to same frame geometry.
    """
    if 'direct_anchors' in model:
        anchors = model['direct_anchors']
    else:
        anchors = model['paper'][:, None]/128*np.asarray(model['normalized_anchors'], np.float32)
        anchors *= np.minimum(1., 255/np.maximum(anchors[:, 2], 1))[:, None]
    target = anchors if reference is None else np.asarray(reference['anchors'], np.float32)
    if not np.isfinite(gamma) or gamma < 1 or np.any(np.diff(anchors, axis=1) <= 0):
        raise ValueError('Invalid gamma or anchors')
    return dict(tables=soft_curve(anchors, target, gamma), anchors=anchors,
                centers=model['centers'], valid=np.ones(len(anchors), bool), gamma=gamma)


def prepare_context(image, boxes, reference_id='', canvas_y=None, foreground_core=True):
    """@luna-comment
    Input: Native uint8 BGR, ordered integer boxes, ID and optional1984x1472 canvas Y.
    Output: Frame-owned robust-pre context with tables and timings; no post role cache.
    Key Steps: Canvas once, estimate field and foreground core per line, build soft pre LUTs.
    Key Params: Gamma1.8; core10%/90% foreground quantiles; fixed detector geometry, no output gates.
    State/Dependencies: Context must be kept until post; does not mutate source/boxes.
    """
    height, width = image.shape[:2]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 BGR')
    start = time.perf_counter()
    if canvas_y is None:
        canvas_y = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0], (1984, 1472))
    if canvas_y.shape != (1472, 1984):
        raise ValueError('Wrong canvas geometry')
    lines, estimate_seconds, lut_seconds = [], 0., 0.
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = map(int, box)
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError('Invalid native box')
        cb = [int(np.floor(x0*1984/width)), int(np.floor(y0*1472/height)),
              int(np.ceil(x1*1984/width)), int(np.ceil(y1*1472/height))]
        tick = time.perf_counter()
        model = estimate_line(canvas_y, cb, collect_roles=False, foreground_core=foreground_core)
        estimate_seconds += time.perf_counter()-tick
        tick = time.perf_counter()
        packed = build_tables(model)
        lut_seconds += time.perf_counter()-tick
        lines.append(dict(index=index, bbox=list(map(int, box)), model=model, packed=packed))
    return dict(version='adaptive_scurve_robust_pre_v2_2', reference_id=reference_id,
                foreground_core=foreground_core,
                source_shape=[height, width], canvas_shape=[1472, 1984], lines=lines,
                timings=dict(prepare_ms=(time.perf_counter()-start)*1000,
                             estimate_ms=estimate_seconds*1000, lut_ms=lut_seconds*1000))


def apply_preprocess(image, context):
    """@luna-comment
    Input: Original native BGR and matching V2 frame context.
    Output: New BGR, all columns processed with original first ownership.
    Key Steps: Original Y once, unchanged frozen mapping/rendering per row, owned-pixel paste.
    Key Params: No statistics/powers at native resolution; no chroma or border protection.
    State/Dependencies: Frozen prepare_mapping/apply_tables, no source mutation.
    """
    if list(image.shape[:2]) != list(context['source_shape']):
        raise ValueError('Source/context shape mismatch')
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    output = image.copy()
    owned = np.zeros(y.shape, bool)
    for line in context['lines']:
        x0, y0, x1, y1 = line['bbox']
        mapping = prepare_mapping(line['packed'], line['bbox'], y.shape, context['canvas_shape'])
        processed, _ = apply_tables(image[y0:y1, x0:x1], y[y0:y1, x0:x1], line['packed'], mapping)
        free = ~owned[y0:y1, x0:x1]
        output[y0:y1, x0:x1][free] = processed[free]
        owned[y0:y1, x0:x1] = True
    return output


def make_post_context(context):
    """@luna-comment
    Input: Robust preprocessing context with geometry and original source anchors.
    Output: JSON reference in the frozen postprocessor schema; no rejected role samples.
    Key Steps: Copy geometry/centers/anchors and explicitly identify reference estimator.
    Key Params: Schema versionV1 selects unchanged frozen post; reference remains current pre.
    State/Dependencies: No image/statistics work; caller can release pre LUTs afterward.
    """
    return dict(version='adaptive_dualmode_scurve_v1', reference_id=context['reference_id'],
                reference_estimator=context['version'],
                source_shape=list(context['source_shape']), canvas_shape=list(context['canvas_shape']),
                geometry='aligned full frame; independent axis scaling only',
                lines=[dict(index=l['index'], bbox=l['bbox'], model=dict(
                    canvas_bbox=l['model']['canvas_bbox'], normalized_anchors=l['model']['normalized_anchors']),
                    packed=dict(centers=np.asarray(l['packed']['centers']).tolist(),
                                anchors=np.asarray(l['packed']['anchors']).tolist()))
                       for l in context['lines']])


def postprocess_bgr(prediction, context, gamma=1.2):
    """@luna-comment
    Input: Aligned uint8 BGR prediction and runtime robust or serialized frozen-schema context.
    Output: Unchanged frozenV1postprocessed BGR and telemetry using current pre reference.
    Key Steps: Adapt runtime reference schema if necessary; delegate to frozen postprocessor.
    Key Params: Default gamma1.2; rejected soft post tails/spatial role fitting removed.
    State/Dependencies: Frozen package unchanged; no production integration or chroma repair.
    """
    from evaluation.adaptive_prepost_frozen_v1_20261020.postprocess import postprocess_bgr as frozen_post
    if context['version'] != 'adaptive_dualmode_scurve_v1':
        context = make_post_context(context)
    return frozen_post(prediction, context, gamma)
