"""Versioned full-frame M2 completion and adjacent-background harmonic repair."""
from pathlib import Path
import sys
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.preview_mask_guided_vosr_strokes import weighted_field

VERSION = 'm2-direct-strong-adjacent-dirichlet-rgb-v1-20261104'


def ink_field(image, mask, valid, background):
    """@luna-comment Input:original RGB/M2/valid/local background. Output:ink/support/anchors and skips. Key Steps:same-component contrast-tail original donors;local normalized RGB interpolation. Key Params:sigma6,quantiles .60/.98,min4;both polarities. State/Dependencies:original pixels only,numpy/scipy."""
    labels, count = ndi.label((mask > .5) & valid, np.ones((3, 3)))
    field = np.zeros_like(image); support = np.zeros(mask.shape, bool); anchors = support.copy()
    if not count:
        return field, support, anchors, 0
    nearest = ndi.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    owners = labels[tuple(nearest)]; skipped = 0
    for number, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        sl = np.s_[max(0, box[0].start-28):min(mask.shape[0], box[0].stop+28), max(0, box[1].start-28):min(mask.shape[1], box[1].stop+28)]
        amplitude = np.linalg.norm(image[sl]-background[sl], axis=2)
        possible = (labels[sl] == number) & (mask[sl] > .7) & (amplitude > .08)
        if possible.sum() < 6:
            skipped += 1; continue
        lo, hi = np.quantile(amplitude[possible], [.60, .98])
        donor = possible & (amplitude >= lo) & (amplitude <= hi)
        if donor.sum() < 4:
            skipped += 1; continue
        local, _ = weighted_field(image[sl], donor.astype(np.float32), 6)
        domain = owners[sl] == number
        field[sl][domain] = local[domain]; support[sl] |= domain; anchors[sl] |= donor
    return field, support, anchors, skipped


def nearest_background(image, background_domain, unknown, ink):
    """@luna-comment Input:original RGB/background/unknown/ink. Output:Dirichlet solution and unresolved/anchor counts. Key Steps:identify closed counters in bounded boxes;solve separate four-connected unknown regions from adjacent original donors. Key Params:fixed known background,text-neighbor exclusion. State/Dependencies:scipy sparse;no far constant fill."""
    unknown = unknown.copy(); unresolved = np.zeros(unknown.shape, bool)
    labels, _ = ndi.label(background_domain); boxes = ndi.find_objects(labels); anchor_count = 0
    for number in np.unique(labels[unknown]):
        if not number:
            continue
        sl = boxes[number-1]; region = labels[sl] == number
        if (region & ~unknown[sl]).any():
            continue
        distance = ndi.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        reliable = region & (distance >= max(1.25, .8*distance.max())) & (np.linalg.norm(image[sl]-ink[sl], axis=2) > .12)
        if reliable.any():
            unknown[sl][reliable] = False; anchor_count += int(reliable.sum())
        else:
            unresolved[sl] |= region; unknown[sl][region] = False
    result = image.copy(); components, _ = ndi.label(unknown)
    for number, box in enumerate(ndi.find_objects(components), 1):
        if box is None:
            continue
        sl = np.s_[max(0, box[0].start-1):min(image.shape[0], box[0].stop+1), max(0, box[1].start-1):min(image.shape[1], box[1].stop+1)]
        active = components[sl] == number; bg = background_domain[sl]; original = image[sl]
        ys, xs = np.where(active); n = len(xs); indices = np.full(active.shape, -1, np.int32); indices[ys, xs] = np.arange(n)
        rows = []; cols = []; values = []; degree = np.zeros(n); rhs = np.zeros((n, 3))
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            yy = ys+dy; xx = xs+dx; inside = (yy >= 0) & (yy < bg.shape[0]) & (xx >= 0) & (xx < bg.shape[1])
            ids = np.flatnonzero(inside); yy = yy[inside]; xx = xx[inside]; allowed = bg[yy, xx]
            ids = ids[allowed]; yy = yy[allowed]; xx = xx[allowed]; degree[ids] += 1
            neighbors = indices[yy, xx]; free = neighbors >= 0
            rows.extend(ids[free]); cols.extend(neighbors[free]); values.extend([-1.]*int(free.sum()))
            rhs[ids[~free]] += original[yy[~free], xx[~free]]
        assert (degree > 0).all()
        rows.extend(np.arange(n)); cols.extend(np.arange(n)); values.extend(degree)
        matrix = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        solution = spsolve(matrix, rhs); assert np.isfinite(solution).all()
        result[sl][active] = solution.astype(np.float32)
    assert np.array_equal(result[background_domain & ~unknown], image[background_domain & ~unknown])
    return result, unknown, unresolved, anchor_count


def repair(raw, m2, validity):
    """@luna-comment Input:uint8 VOSR RGB/M2/valid. Output:strong repaired RGB and factual audit. Key Steps:original ink endpoints,missing fill,false localization,adjacent harmonic background,one soft edge. Key Params:8px detector band,single-pixel unknown expansion,no ROI. State/Dependencies:immutable inputs;unsupported donors preserved/reported."""
    assert raw.dtype == m2.dtype == validity.dtype == np.uint8 and raw.shape[:2] == m2.shape == validity.shape
    image = raw.astype(np.float32)/255; mask = m2.astype(np.float32)/255; valid = validity >= 254
    shape = np.clip((mask-.2)/.6, 0, 1); distance = ndi.distance_transform_edt(mask <= .2)
    if not ((mask > .5) & valid).any() or not ((mask < .05) & valid).any():
        return raw.copy(), dict(changed_pixels=0, unsupported_image=True, unresolved_background_pixels=0)
    background, _ = weighted_field(image, ((mask < .05) & valid).astype(np.float32), 8)
    ink, support, anchors, skips = ink_field(image, mask, valid, background)
    delta = ink-background; den = (delta*delta).sum(2); alpha = ((image-background)*delta).sum(2)/(den+1e-7)
    healthy = anchors | (alpha >= shape-.06)
    missing = valid & support & (mask > .2) & (den > .05**2) & ~healthy & (shape-alpha > .08)
    filled = image.copy(); filled[missing] = (background+shape[..., None]*delta)[missing]
    # A wider background field only detects contamination; never supplies final RGB.
    donors = (distance >= 8) & valid
    detector_bg, _ = weighted_field(image, donors.astype(np.float32), 12) if donors.any() else (background, None)
    detect_ink, detect_support, _, detect_skips = ink_field(image, mask, valid, detector_bg)
    delta = detect_ink-detector_bg; den = (delta*delta).sum(2); residual = filled-detector_bg
    alpha = (residual*delta).sum(2)/(den+1e-7)
    alignment = np.abs((residual*delta).sum(2))/(np.linalg.norm(residual, axis=2)*np.sqrt(den)+1e-7)
    band = valid & detect_support & (distance <= 8) & (shape < .99) & (den > .05**2)
    false = band & (((alpha-shape > .015) & (alignment > .75)) | ((shape == 0) & (alpha < -.03) & (alignment > .8) & (distance <= 4)))
    taper = np.clip((8-distance)/3, 0, 1); taper = taper*taper*(3-2*taper)
    old_target = detector_bg+shape[..., None]*delta
    old_candidate = np.rint(np.clip(filled+taper[..., None]*(old_target-filled), 0, 1)*255).astype(np.uint8)
    filled_bytes = np.rint(np.clip(filled, 0, 1)*255).astype(np.uint8)
    false &= np.any(old_candidate != filled_bytes, axis=2)
    bg = (shape == 0) & valid
    unknown = ndi.binary_dilation(false, iterations=1) & bg
    solved, unknown, unresolved, counter_anchors = nearest_background(image, bg, unknown, detect_ink)
    nearest = ndi.distance_transform_edt(~bg, return_distances=False, return_indices=True)
    adjacent = solved[tuple(nearest)]
    edge = false & (shape > 0) & (shape < .99) & detect_support & valid & ~anchors
    result = filled.copy(); result[unknown] = solved[unknown]
    result[edge] = (adjacent+shape[..., None]*(detect_ink-adjacent))[edge]
    editable = missing | unknown | edge
    value = np.rint(np.clip(result, 0, 1)*255).astype(np.uint8); value[~editable] = raw[~editable]
    assert np.array_equal(value[~valid], raw[~valid]) and np.array_equal(value[bg & ~unknown], raw[bg & ~unknown])
    assert np.array_equal(value[anchors], raw[anchors])
    changed = np.any(value != raw, axis=2)
    return value, dict(changed_pixels=int(changed.sum()), filled_pixels=int(missing.sum()), cleaned_background_pixels=int(unknown.sum()), soft_edge_pixels=int(edge.sum()), ink_donor_pixels=int(anchors.sum()), skipped_components=skips, detector_skipped_components=detect_skips, unresolved_background_pixels=int(unresolved.sum()), counter_anchor_pixels=counter_anchors, invalid_changed_pixels=0, known_background_changed_pixels=0, healthy_anchor_changed_pixels=0)


def invariant_smoke():
    """@luna-comment Input:none. Output:black/white synthetic invariant evidence. Key Steps:damage interiors/add outer shadow on sloped background;repair and check fixed pixels/locality. Key Params:two polarities,exact uint8 invariants. State/Dependencies:CPU only."""
    reports = []
    for reverse in [False, True]:
        yy, xx = np.mgrid[:80, :100]; background = np.repeat((.2+.001*xx)[..., None], 3, 2)
        mask = np.zeros((80, 100), np.uint8); mask[15:65, 30:40] = 255; mask[15:25, 30:65] = 255
        image = background.copy(); image[mask > 0] = .86; image[35:42, 30:40] = .45; image[30:55, 40:43] += .14
        if reverse:
            image = 1-image
        raw = np.rint(image*255).astype(np.uint8); validity = np.full(mask.shape, 255, np.uint8); validity[:4] = 0
        output, audit = repair(raw, mask, validity)
        assert audit['filled_pixels'] > 0 and audit['cleaned_background_pixels'] > 0
        assert np.array_equal(output[:4], raw[:4]) and np.array_equal(output[:, :15], raw[:, :15])
        reports.append(audit)
    return dict(passed=True, polarities=2, reports=reports)
