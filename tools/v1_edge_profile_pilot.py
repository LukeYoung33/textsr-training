"""Experimental short-segment edge profiles anchored to the frozen V1 operator."""
from dataclasses import dataclass
import time

import cv2
import numpy as np

from tools.light_text_postprocess import support_alpha


@dataclass(frozen=True)
class ProfileConfig:
    """@luna-comment Input: U/H/S factor switches. Output: immutable low-strength pilot setting. Key Steps: keep all original V1 gains/caps. Key Params: shared fixed HALO_MU/SLOPE_GAMMA below. State/Dependencies: C intentionally absent until stroke-interior coordination is solved."""
    remove_usm: bool = False
    halo: bool = False
    slope: bool = False


HALO_MU = .25
SLOPE_GAMMA = 1.1


VARIANTS = {
    'V1_zero': ProfileConfig(),
    'U': ProfileConfig(remove_usm=True),
    'UH': ProfileConfig(remove_usm=True, halo=True),
    'US': ProfileConfig(remove_usm=True, slope=True),
    'UHS': ProfileConfig(remove_usm=True, halo=True, slope=True),
}


def v1_float_stages(image, boxes):
    """@luna-comment Input: rectified BGR8 and OCR boxes. Output: original Y, pre-USM adjusted, frozen V1 target, support alpha and threshold. Key Steps: reproduce frozen V1 standard equations exactly. Key Params: gain .75, sharp .65, contrast/final cap24, 16px support feather. State/Dependencies: frozen V1 reference remains read-only; focused test asserts exact float stages/output."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected BGR8')
    h, w = image.shape[:2]
    y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    small = cv2.resize(y, (max(1, (w+7)//8), max(1, (h+7)//8)), interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT_101),
                            (w, h), interpolation=cv2.INTER_LINEAR)
    fine = y-cv2.GaussianBlur(y, (0, 0), .8, borderType=cv2.BORDER_REFLECT_101)
    samples = fine[::8, ::8]
    sigma = float(np.median(np.abs(samples-np.median(samples)))*1.4826)
    threshold = float(np.clip(1.5*sigma, 2., 6.))
    residual = y-background
    paper = np.clip((background-80.)/96., 0, 1)
    paper = paper*paper*(3-2*paper)
    dark = -np.maximum(-residual-threshold, 0)*paper
    bright = np.maximum(residual-threshold, 0)*(1-paper)
    contrast = .75*(dark+bright)
    contrast = contrast/(1+np.abs(contrast)/24.)
    adjusted = y+contrast
    detail = adjusted-cv2.GaussianBlur(adjusted, (0, 0), .8, borderType=cv2.BORDER_REFLECT_101)
    detail = np.sign(detail)*np.maximum(np.abs(detail)-threshold*.5, 0)
    low = cv2.erode(adjusted, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REFLECT_101)
    high = cv2.dilate(adjusted, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REFLECT_101)
    target = np.clip(adjusted+.65*detail, low, high)
    alpha = support_alpha((h, w), boxes)
    return {'y': y, 'adjusted': adjusted, 'target': target, 'alpha': alpha,
            'background': background, 'threshold': threshold, 'noise_sigma': sigma}


def area_profile(nx, ny, t0, scale, xy, low, high, gamma=1.):
    """@luna-comment Input: segment normal/offset/width, Nx2 pixel centers, shared low/high platforms. Output: continuous logistic profile integrated by 4x4 Gauss quadrature over each unit pixel. Key Steps: evaluate fixed model at quadrature nodes with tensor weights. Key Params: gamma changes slope only, t0 remains fixed. State/Dependencies: NumPy; approximation is measured against native final, not a geometry guarantee."""
    nodes = np.array([-.4305681557970263, -.1699905217924281, .1699905217924281, .4305681557970263], np.float32)
    weights = np.array([.173927422568727, .326072577431273, .326072577431273, .173927422568727], np.float32)
    offsets = np.array([(a, b) for a in nodes for b in nodes], np.float32)
    quadrature = np.outer(weights, weights).ravel()
    projected = xy[:, 0]*nx+xy[:, 1]*ny-t0
    shifts = offsets[:, 0]*nx+offsets[:, 1]*ny
    z = np.clip(gamma*(projected[:, None]+shifts[None, :])/scale, -18, 18)
    value = 1./(1.+np.exp(-z))
    return (low+(high-low)*(value@quadrature)).astype(np.float32)


def normal_remap(y, nx, ny, offset):
    """@luna-comment Input: float Y, per-pixel unit normal and signed offset. Output: bilinear normal sample. Key Steps: reflect-border remap on full warp. Key Params: pixel offset. State/Dependencies: OpenCV float maps; used only in read-only edge evidence."""
    h, w = y.shape
    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    return cv2.remap(y, xx+offset*nx, yy+offset*ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def segment_candidates(stages):
    """@luna-comment Input: V1 adjusted and support alpha. Output: edge mask, normal-side samples and counters. Key Steps: Sobel/5x5 coherence, two-sided platform separation and distant-side stability; reject thin/double edges before expensive fit. Key Params: 12px local windows, side distances2/4/5/7, separation30. State/Dependencies: OpenCV; heuristic eligibility only, not halo ground truth."""
    y = stages['adjusted']
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)/8
    mag = cv2.magnitude(gx, gy)
    nx, ny = gx/(mag+1e-6), gy/(mag+1e-6)
    a = cv2.boxFilter(gx*gx, -1, (5, 5))
    b = cv2.boxFilter(gy*gy, -1, (5, 5))
    c = cv2.boxFilter(gx*gy, -1, (5, 5))
    coherence = np.sqrt((a-b)**2+4*c*c)/(a+b+1e-6)
    sample = {d: normal_remap(y, nx, ny, d) for d in (-7, -5, -4, -2, 2, 4, 5, 7)}
    separated = sample[7]-sample[-7]
    stable = (np.abs(sample[-7]-sample[-5])<10)&(np.abs(sample[7]-sample[5])<10)
    edge = (mag>12)&(coherence>.78)&stable&(separated>30)&(stages['alpha']>.01)
    reverse = ((sample[-4]>sample[-2]+2)|(sample[4]<sample[2]-2))
    reverse_field = cv2.GaussianBlur((reverse&edge).astype(np.float32), (0, 0), 2.)
    return {'edge': edge, 'reverse': reverse, 'reverse_field': reverse_field, 'gx': gx, 'gy': gy, 'mag': mag,
            'minus': sample[-7], 'plus': sample[7],
            'candidate_edge_pixels': int(edge.sum()), 'reverse_edge_pixels': int(np.count_nonzero(edge&reverse))}


def fit_short_segment(stages, evidence, left, top, size=12):
    """@luna-comment Input: overlapping 12px edge window and frozen V1/evidence arrays. Output: shared platform/normal/t0/s fit or None plus reject reason. Key Steps: require one coherent edge, stable segment plateaus and phase, grid-fit continuous area model to actual adjusted samples. Key Params: >=7 edge pixels, fit median normalized error<=.12. State/Dependencies: no writes; rejected windows retain V1."""
    h, w = stages['y'].shape
    right, bottom = min(left+size, w), min(top+size, h)
    edge = evidence['edge'][top:bottom, left:right]
    yi, xi = np.nonzero(edge)
    if len(xi)<7:
        return None, 'few_edge_pixels'
    normals = np.stack((evidence['gx'][top:bottom, left:right][edge], evidence['gy'][top:bottom, left:right][edge]), axis=1)
    units = normals/(np.linalg.norm(normals, axis=1, keepdims=True)+1e-6)
    mean_normal = np.mean(units, axis=0)
    if np.linalg.norm(mean_normal)<.88:
        return None, 'mixed_normals'
    n = mean_normal/np.linalg.norm(mean_normal)
    low_samples = evidence['minus'][top:bottom, left:right][edge]
    high_samples = evidence['plus'][top:bottom, left:right][edge]
    low = float(np.median(low_samples))
    high = float(np.median(high_samples))
    if (high-low<30 or np.median(np.abs(low_samples-low))>8
            or np.median(np.abs(high_samples-high))>8):
        return None, 'unstable_platforms'
    xy = np.stack((left+xi, top+yi), axis=1).astype(np.float32)
    observed = stages['adjusted'][top:bottom, left:right][edge]
    normalized = (observed-low)/(high-low)
    transition = (normalized>.08)&(normalized<.92)
    if int(transition.sum())<6:
        return None, 'few_transition_pixels'
    xy, normalized, normals = xy[transition], normalized[transition], normals[transition]
    if len(xy)>16:
        select = np.linspace(0, len(xy)-1, 16).astype(np.int32)
        xy, normalized, normals = xy[select], normalized[select], normals[select]
    projected = xy@n
    derivative = (normals@n)/(high-low)
    derivative = np.maximum(derivative, .05)
    seed = float(np.median(projected-(normalized-.5)/derivative))
    phase_mad = float(np.median(np.abs((projected-(normalized-.5)/derivative)-seed)))
    if phase_mad>.65:
        return None, 'unstable_phase'
    best = None
    for scale in (.32, .45, .60, .80, 1.05, 1.40):
        for phase in (-.30, -.15, 0., .15, .30):
            t0 = seed+phase
            predicted = area_profile(n[0], n[1], t0, scale, xy, low, high)
            error = float(np.median(np.abs((predicted-low)/(high-low)-normalized)))
            if best is None or error<best[0]:
                best = (error, t0, scale)
    if best[0]>.12:
        return None, 'poor_area_fit'
    yy, xx = np.mgrid[top:bottom, left:right]
    all_xy = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
    model = area_profile(n[0], n[1], best[1], best[2], all_xy, low, high).reshape(xx.shape)
    band = np.abs(n[0]*xx+n[1]*yy-best[1])<3.5
    residual = np.abs((stages['adjusted'][top:bottom, left:right]-model)/(high-low))
    band_residual = residual[band]
    residual_p90 = float(np.percentile(band_residual, 90)) if band_residual.size else 0.
    unexplained = band&(residual>.20)&(evidence['reverse_field'][top:bottom, left:right]<.05)
    unexplained_fraction = float(np.count_nonzero(unexplained)/max(1, np.count_nonzero(band)))
    if unexplained_fraction>.18:
        return None, 'unexplained_second_structure'
    reverse = evidence['reverse'][top:bottom, left:right][edge]
    return {'nx': float(n[0]), 'ny': float(n[1]), 'low': low, 'high': high,
            't0': best[1], 'scale': best[2], 'fit_error': best[0],
            'phase_mad': phase_mad, 'reverse_fraction': float(np.mean(reverse)),
            'normal_band_residual_p90': residual_p90,
            'unexplained_residual_fraction': unexplained_fraction,
            'left': left, 'top': top, 'right': right, 'bottom': bottom}, None


def profile_corrections(stages, evidence):
    """@luna-comment Input: frozen V1 stages and short-edge evidence. Output: smooth U/H/S float correction maps and telemetry. Key Steps: fit every eligible overlapping window, evaluate fixed continuous area profiles, blend weighted corrections with V1 fallback; H requires reverse lobe and overshoot. Key Params: stride6, mu .25, gamma1.1. State/Dependencies: no arbitrary segment budget or write; sparse coverage is reported."""
    y = stages['adjusted']
    h, w = y.shape
    edge_sum = cv2.boxFilter(evidence['edge'].astype(np.uint8), cv2.CV_16S, (12, 12), normalize=False)
    centers = np.argwhere(edge_sum[6::6, 6::6]>=7)
    reverse_field = evidence['reverse_field']
    sums = [np.zeros((h, w), np.float32) for _ in range(4)]
    weights = sums[0]
    accepted = halo_segments = 0
    errors = []
    band_errors = []
    rejected = {}
    diagnostic_mask = evidence.get('diagnostic_mask')
    diagnostic_candidates = diagnostic_accepted = 0
    for gy, gx in centers:
        top, left = int(gy*6), int(gx*6)
        in_diagnostic = (diagnostic_mask is not None and
                         bool(diagnostic_mask[min(top+6, h-1), min(left+6, w-1)]))
        diagnostic_candidates += int(in_diagnostic)
        segment, reason = fit_short_segment(stages, evidence, left, top)
        if segment is None:
            rejected[reason] = rejected.get(reason, 0)+1
            continue
        accepted += 1
        diagnostic_accepted += int(in_diagnostic)
        errors.append(segment['fit_error'])
        band_errors.append(segment['normal_band_residual_p90'])
        yy, xx = np.mgrid[top:segment['bottom'], left:segment['right']]
        xy = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
        base = area_profile(segment['nx'], segment['ny'], segment['t0'], segment['scale'], xy,
                            segment['low'], segment['high']).reshape(xx.shape)
        steep = area_profile(segment['nx'], segment['ny'], segment['t0'], segment['scale'], xy,
                             segment['low'], segment['high'], SLOPE_GAMMA).reshape(xx.shape)
        normal_distance = segment['nx']*xx+segment['ny']*yy-segment['t0']
        radial = np.exp(-.5*(normal_distance/3.5)**2)
        taper = np.sin(np.pi*(xx-left+.5)/12.)**2*np.sin(np.pi*(yy-top+.5)/12.)**2
        weight = (radial*taper).astype(np.float32)
        sl = np.s_[top:segment['bottom'], left:segment['right']]
        local_y = y[sl]
        weights[sl] += weight
        sums[1][sl] += weight*(local_y-stages['target'][sl])
        sums[3][sl] += weight*(steep-base)
        if segment['reverse_fraction']>.5:
            over = np.maximum(segment['low']-local_y, local_y-segment['high'])-2.
            exterior_gate = np.clip(over/4., 0, 1)
            exterior_gate = exterior_gate*exterior_gate*(3-2*exterior_gate)
            interior_gate = np.clip(reverse_field[sl]*5., 0, 1)*np.clip(np.abs(base-local_y)/4., 0, 1)
            halo_gate = np.maximum(exterior_gate, interior_gate)
            correction = HALO_MU*(base-local_y)*halo_gate
            if np.any(np.abs(correction)>.1):
                halo_segments += 1
                sums[2][sl] += weight*correction
    active = weights>1e-6
    gain = np.zeros_like(weights)
    gain[active] = np.minimum(weights[active], 1.)/weights[active]
    corrections = {'U': sums[1]*gain, 'H': sums[2]*gain, 'S': sums[3]*gain}
    telemetry = {'candidate_windows': int(len(centers)), 'accepted_segments': accepted,
                 'halo_segments': halo_segments, 'candidate_edge_pixels': evidence['candidate_edge_pixels'],
                 'reverse_edge_pixels': evidence['reverse_edge_pixels'],
                 'rejected_windows_by_reason': rejected,
                 'fit_error_median': float(np.median(errors)) if errors else None,
                 'normal_band_residual_p90_median': float(np.median(band_errors)) if band_errors else None,
                 'reliable_blend_pixels': int(np.count_nonzero(active)),
                 'U_float_hits': int(np.count_nonzero(np.abs(corrections['U'])>.5)),
                 'H_float_hits': int(np.count_nonzero(np.abs(corrections['H'])>.5)),
                 'S_float_hits': int(np.count_nonzero(np.abs(corrections['S'])>.5))}
    if diagnostic_mask is not None:
        telemetry['diagnostic_roi'] = {
            'candidate_windows': diagnostic_candidates,
            'accepted_segments': diagnostic_accepted,
            'U_float_hits': int(np.count_nonzero((np.abs(corrections['U'])>.5)&diagnostic_mask)),
            'H_float_hits': int(np.count_nonzero((np.abs(corrections['H'])>.5)&diagnostic_mask)),
            'S_float_hits': int(np.count_nonzero((np.abs(corrections['S'])>.5)&diagnostic_mask))}
    return corrections, telemetry


def finish_v1_output(image, stages, target):
    """@luna-comment Input: BGR8 source, exact V1 stage Y/alpha and candidate float target. Output: BGR8, integer map and clipping telemetry after frozen V1 contract. Key Steps: shared scalar correction for all channels. Key Params: cap24. State/Dependencies: no writes; zero-switch tested exact against frozen enhance_bgr."""
    raw = target-stages['y']
    delta = np.clip(raw, -24., 24.)*stages['alpha']
    lower = -image.min(2).astype(np.float32)
    upper = 255-image.max(2).astype(np.float32)
    capped = np.minimum(np.maximum(delta, lower), upper)
    integer = np.rint(capped).astype(np.int16)
    limits = {'total_delta_cap_hits': int(np.count_nonzero(np.abs(raw)>24.)),
              'gamut_cap_hits': int(np.count_nonzero(np.abs(capped-delta)>.01)),
              'rounded_nonzero_pixels': int(np.count_nonzero(integer))}
    return (image.astype(np.int16)+integer[:, :, None]).astype(np.uint8), integer, limits


def enhance_bgr_v1_profile(image, boxes, variant='U', prepared=None):
    """@luna-comment Input: BGR8 warp, OCR boxes and V1/U/UH/US/UHS variant. Output: BGR8 and branch telemetry. Key Steps: exact V1 float stages, conservative short-segment U/H/S corrections, single frozen V1 output contract. Key Params: low mu .25/gamma1.1; no C or stronger USM. State/Dependencies: prepared stage maps can be reused across variants of the same source warp."""
    started = time.perf_counter()
    if variant not in VARIANTS:
        raise ValueError('Unknown profile variant')
    if prepared is None:
        stages = v1_float_stages(image, boxes)
        corrections, detail = ({}, {}) if variant == 'V1_zero' else profile_corrections(stages, segment_candidates(stages))
    else:
        stages, corrections, detail = prepared
    if variant == 'V1_zero':
        target = stages['target']
    else:
        target = stages['target']+corrections['U']
        if VARIANTS[variant].halo:
            target = target+corrections['H']
        if VARIANTS[variant].slope:
            target = target+corrections['S']
    output, integer, limits = finish_v1_output(image, stages, target)
    report = {'variant': variant, 'seconds': time.perf_counter()-started, **detail, **limits,
              'halo_mu': HALO_MU, 'slope_gamma': SLOPE_GAMMA,
              'changed_pixels': int(np.count_nonzero(integer)),
              'max_abs_delta': int(np.abs(integer).max()),
              'support_pixels': int(np.count_nonzero(stages['alpha'])),
              'outside_support_unchanged': bool(np.array_equal(output[stages['alpha']==0], image[stages['alpha']==0])),
              'common_channel_delta': True}
    return output, report
