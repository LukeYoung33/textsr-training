"""Read-only RCA measurements for the two user-reported V2 camera failures."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_light_post_v2_ablation import output_for
from tools.light_text_postprocess_v2 import (PRESETS, robust_threshold,
    trusted_plateau_envelope, dehalo_luma, polarity_contrast,
    directional_antialias, weak_sharpen)
from tools.run_camera_light_post_v2 import MANIFEST, OUT, documents_for, mapped_path, read_bgr


REGIONS = {'006': (1600, 2020, 1850, 2140), '099': (1280, 2620, 2570, 2780)}


def summarize(values):
    """@luna-comment Input: numeric array. Output: robust distribution summary. Key Steps: flatten and select quantiles. Key Params: p05/p50/p95. State/Dependencies: no writes."""
    a = np.asarray(values).ravel()
    return {'n': int(a.size), 'min': float(a.min()) if a.size else None,
            'p05': float(np.percentile(a, 5)) if a.size else None,
            'median': float(np.median(a)) if a.size else None,
            'p95': float(np.percentile(a, 95)) if a.size else None,
            'max': float(a.max()) if a.size else None}


def crop_mask(shape, matrix, box):
    """@luna-comment Input: warp size, native-to-warp H and native crop. Output: rectified polygon mask. Key Steps: project four crop corners. Key Params: half-open crop. State/Dependencies: OpenCV only."""
    x0, y0, x1, y1 = box
    points = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32)
    warped = cv2.perspectiveTransform(points, matrix.astype(np.float64))[0]
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(warped).astype(np.int32), 1)
    return mask.astype(bool), warped.tolist()


def load_case(identity):
    """@luna-comment Input: 006/099 manifest ID. Output: exact baseline, bound geometry, four native variants. Key Steps: resolve existing final and existing receipts. Key Params: read only source/output. State/Dependencies: authorized Y paths."""
    row = next(v for v in json.loads(MANIFEST.read_text(encoding='utf-8'))['rows'] if v['id'] == identity)
    source = mapped_path(row['images']['vosr_v5_block'])
    receipt = json.loads((source.parent.parent/'receipts'/(identity+'.json')).read_text(encoding='utf-8'))
    variants = {'BL': read_bgr(source)}
    for name, key in [('V1', 'light_v1_standard'), ('V2low', 'light_v2_low'), ('V2std', 'light_v2_standard')]:
        variants[name] = read_bgr(mapped_path(row['images'][key]))
    return variants, documents_for(identity, receipt)


def native_measure(identity, variants, box):
    """@luna-comment Input: four actual native outputs and user-text crop. Output: same-pixel brightness and stroke measurements. Key Steps: source-derived dark/bright stroke masks, pair differences and extrema. Key Params: all masks diagnostic, not ground truth. State/Dependencies: no writes."""
    x0, y0, x1, y1 = box
    y = {key: cv2.cvtColor(value[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.int16) for key, value in variants.items()}
    gray = y['BL'].astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), 9.)
    dark = gray < background-10
    bright = gray > background+10
    report = {'crop_xyxy': box, 'source_dark_pixels': int(dark.sum()), 'source_bright_pixels': int(bright.sum())}
    for name in ('V1', 'V2low', 'V2std'):
        diff = y[name]-y['BL']
        report[name+'_minus_BL'] = {'dark': summarize(diff[dark]), 'bright': summarize(diff[bright]), 'all': summarize(diff)}
    gap = y['V2std']-y['V1']
    report['V2std_minus_V1'] = {'dark': summarize(gap[dark]), 'bright': summarize(gap[bright]),
                                 'dark_ge_20': int(np.count_nonzero(gap[dark] >= 20)), 'all': summarize(gap)}
    if dark.any():
        positions = np.argwhere(dark & (gap >= 15))
        selected = positions[np.argsort(-gap[positions[:, 0], positions[:, 1]])[:10]] if len(positions) else np.empty((0, 2), int)
        report['dark_gap_examples'] = [{'xy': [int(x0+x), int(y0+yy)], **{key: int(value[yy, x]) for key, value in y.items()}}
                                      for yy, x in selected]
    return report


def blue_edge_bands(variants, contrast_off, box):
    """@luna-comment Input: actual 006 outputs and contrast-off ablation. Output: signed blue-background changes by distance to white letter cores. Key Steps: source-only color/brightness masks and Euclidean distance bands. Key Params: diagnostic proxy, not a labeled halo ground truth. State/Dependencies: no writes."""
    x0, y0, x1, y1 = box
    crops = {key: value[y0:y1, x0:x1] for key, value in variants.items()}
    crops['no_contrast'] = contrast_off[y0:y1, x0:x1]
    luminance = {key: cv2.cvtColor(value, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.int16) for key, value in crops.items()}
    source = crops['BL'].astype(np.int16)
    blue = (source[:, :, 0] > source[:, :, 2]+20) & (luminance['BL'] < 100)
    white_core = luminance['BL'] > 200
    distance = cv2.distanceTransform((~white_core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    bands = {}
    for low, high in [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 999)]:
        pixels = blue & (distance >= low) & (distance < high)
        bands[f'{low}-{high}'] = {'n': int(pixels.sum())}
        for key in ('V1', 'V2low', 'V2std', 'no_contrast'):
            diff = luminance[key][pixels]-luminance['BL'][pixels]
            bands[f'{low}-{high}'][key+'_minus_BL'] = summarize(diff)
            bands[f'{low}-{high}'][key+'_darkened_ge_3'] = int(np.count_nonzero(diff <= -3))
    return bands


def rectified_measure(identity, baseline, documents, box):
    """@luna-comment Input: exact final/geometry and failure ROI. Output: internal frozen V2 gate/sharpen evidence. Key Steps: reproduce read-only stage maps and collect projected-crop distributions. Key Params: standard preset. State/Dependencies: no core modification."""
    cfg = PRESETS['standard']
    parts = []
    for item, ocr, _ in documents:
        h, w = ocr['warp_hw']
        matrix = np.asarray(item['warp']['image_to_output'], np.float64)
        image = cv2.warpPerspective(baseline, matrix, (w, h))
        mask, corners = crop_mask((h, w), matrix, box)
        if not mask.any():
            continue
        y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        threshold, _ = robust_threshold(y)
        envelope = trusted_plateau_envelope(y, threshold)
        target, halo_hits = dehalo_luma(y, cfg.halo, threshold, envelope)
        small = cv2.resize(target, (max(1, (w+7)//8), max(1, (h+7)//8)), interpolation=cv2.INTER_AREA)
        background = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5), (w, h))
        residual = target-background
        dark_energy = cv2.boxFilter(np.maximum(-residual-threshold, 0), -1, (31, 31))
        bright_energy = cv2.boxFilter(np.maximum(residual-threshold, 0), -1, (31, 31))
        dominance = (dark_energy-bright_energy)/(dark_energy+bright_energy+threshold)
        confidence = np.clip((np.abs(dominance)-.25)/.55, 0, 1)
        confidence *= np.clip((dark_energy+bright_energy-threshold)/(3*threshold+1e-6), 0, 1)
        contrasted, contrast_hits = polarity_contrast(target, background, threshold, cfg.contrast, cfg.max_delta)
        antialiased, aa_hits = directional_antialias(contrasted, cfg.aa)
        sharpened, sharp_hits = weak_sharpen(antialiased, threshold, cfg.sharp, envelope)
        low, high, trusted = envelope
        detail = antialiased-cv2.GaussianBlur(antialiased, (0, 0), .8)
        detail = np.sign(detail)*np.maximum(np.abs(detail)-threshold*.5, 0)
        intended = cfg.sharp*detail
        sharp_change = sharpened-antialiased
        clip_alone = np.clip(antialiased+intended, low, high)-(antialiased+intended)
        stroke = mask & (residual < -10)
        bright_stroke = mask & (residual > 10)
        active = mask & trusted
        clip_over_gain = active & (np.abs(sharp_change)>np.abs(intended)+.5)
        part = {'instance_id': item['instance_id'], 'projected_corners': corners,
                'threshold': threshold, 'mask_pixels': int(mask.sum()),
                'global_hits': {'halo': halo_hits, 'contrast': contrast_hits, 'aa': aa_hits, 'sharp': sharp_hits},
                'ROI_counts': {'dark_stroke': int(stroke.sum()), 'bright_stroke': int(bright_stroke.sum()),
                               'trusted': int(active.sum()), 'sharp_changed_gt_0p5': int(np.count_nonzero(mask & (np.abs(sharp_change)>.5))),
                               'sharp_clip_over_gain': int(clip_over_gain.sum()),
                               'halo_change_gt_0p5': int(np.count_nonzero(mask & (np.abs(target-y)>.5))),
                               'aa_change_gt_0p5': int(np.count_nonzero(mask & (np.abs(antialiased-contrasted)>.5)))},
                'dark_stroke': {'dark_energy': summarize(dark_energy[stroke]), 'bright_energy': summarize(bright_energy[stroke]),
                                'dominance': summarize(dominance[stroke]), 'confidence': summarize(confidence[stroke]),
                                'contrast_float_delta': summarize((contrasted-target)[stroke]),
                                'contrast_suppressed_count': int(np.count_nonzero(stroke & (confidence==0)))},
                'bright_stroke': {'contrast_float_delta': summarize((contrasted-target)[bright_stroke]),
                                  'confidence': summarize(confidence[bright_stroke])},
                'sharp': {'change': summarize(sharp_change[mask]), 'intended_gain_delta': summarize(intended[mask]),
                          'clip_adjustment': summarize(clip_alone[mask]),
                          'clip_over_gain_change': summarize(sharp_change[clip_over_gain])}}
        parts.append(part)
    return parts


def run(identity):
    """@luna-comment Input: 006 or 099. Output: Y-only JSON RCA. Key Steps: bind actual native outputs, ablate final V2, inspect rectified internal gate. Key Params: no batch/full result changes. State/Dependencies: dedicated evaluation file."""
    cv2.setNumThreads(2)
    variants, documents = load_case(identity)
    box = REGIONS[identity]
    report = {'id': identity, 'native': native_measure(identity, variants, box)}
    if identity == '006':
        x0, y0, x1, y1 = box
        full, _ = output_for(variants['BL'], documents, PRESETS['standard'])
        report['reproduces_actual_V2std'] = bool(np.array_equal(full, variants['V2std']))
        report['native_ablation_ROI'] = {}
        no_contrast = None
        for name, switch in [('dehalo', 'dehalo'), ('contrast', 'polarity'), ('AA', 'directional_aa'), ('sharpen', 'sharpening')]:
            ablated, _ = output_for(variants['BL'], documents, replace(PRESETS['standard'], **{switch: False}))
            d = ablated[y0:y1, x0:x1, 0].astype(np.int16)-full[y0:y1, x0:x1, 0].astype(np.int16)
            report['native_ablation_ROI'][name] = {'changed': int(np.count_nonzero(d)), 'difference': summarize(d),
                                                     'abs_difference': summarize(np.abs(d)),
                                                     'coordinates_max_abs': [[int(x0+x), int(y0+y)] for y, x in np.argwhere(np.abs(d)==np.abs(d).max())[:10]]}
            if name == 'contrast':
                no_contrast = ablated
                continue
            del ablated
        report['blue_edge_bands'] = blue_edge_bands(variants, no_contrast, box)
        del no_contrast
        del full
    report['rectified'] = rectified_measure(identity, variants['BL'], documents, box)
    destination = OUT/'evaluation'/(identity+'_targeted_failure_rca.json')
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    run(sys.argv[1])
