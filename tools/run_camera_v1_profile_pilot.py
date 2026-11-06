"""Exact-source, three-image native pilot for conservative V1 edge profiles."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.doc_warp_composition import inverse_residual
from tools.run_camera_light_post_v2 import MANIFEST, OUT, digest, documents_for, mapped_path, read_bgr, write_png
from tools.v1_edge_profile_pilot import (HALO_MU, SLOPE_GAMMA, VARIANTS,
    enhance_bgr_v1_profile, profile_corrections, segment_candidates, v1_float_stages)


PILOT = OUT.parent/'v1_profile_pilot'
IDS = ('006', '099', '017')
PRODUCE = ('U', 'UH', 'US', 'UHS')
ROIS = {'006': (1600, 2020, 1850, 2140),
        '099': (1280, 2620, 2570, 2780),
        '017': (1450, 2370, 1940, 2480)}


def finalize_native(baseline, canvas, owned):
    """@luna-comment Input: immutable native BGR8, mapped common residual and ownership mask. Output: native BGR8 plus cap/support telemetry. Key Steps: original V1 single gamut-limited integer scalar delta. Key Params: no extra source-frame sharpening. State/Dependencies: valid-mask/ownership maps from frozen inverse_residual."""
    lower = -baseline.min(2).astype(np.float32)
    upper = 255-baseline.max(2).astype(np.float32)
    clipped = np.minimum(np.maximum(canvas, lower), upper)
    integer = np.rint(clipped).astype(np.int16)
    final = (baseline.astype(np.int16)+integer[:, :, None]).astype(np.uint8)
    outside = bool(np.array_equal(final[~owned], baseline[~owned]))
    common = bool(np.array_equal(final[:, :, 0].astype(np.int16)-baseline[:, :, 0].astype(np.int16), integer)
                  and np.array_equal(final[:, :, 1].astype(np.int16)-baseline[:, :, 1].astype(np.int16), integer)
                  and np.array_equal(final[:, :, 2].astype(np.int16)-baseline[:, :, 2].astype(np.int16), integer))
    if not (outside and common):
        raise AssertionError('Native support/common-delta contract')
    return final, {'changed_pixels_from_BL': int(np.count_nonzero(integer)),
                   'max_abs_delta_from_BL': int(np.abs(integer).max()),
                   'owned_pixels': int(np.count_nonzero(owned)),
                   'outside_owned_unchanged': outside, 'common_channel_delta': common,
                   'native_gamut_cap_hits': int(np.count_nonzero(np.abs(clipped-canvas)>.01))}


def pair_metrics(first, second, roi):
    """@luna-comment Input: two same-size native BGR8 results and source ROI. Output: final integer pair-difference counts/full and ROI. Key Steps: compare common channel 0 directly after full pasteback/cap. Key Params: pairs U-V1, UH-U, US-U, UHS-US. State/Dependencies: no writes."""
    a = first[:, :, 0].astype(np.int16)
    b = second[:, :, 0].astype(np.int16)
    x0, y0, x1, y1 = roi
    d = b-a
    patch = d[y0:y1, x0:x1]
    return {'full_changed_pixels': int(np.count_nonzero(d)), 'full_max_abs': int(np.abs(d).max()),
            'roi_changed_pixels': int(np.count_nonzero(patch)), 'roi_max_abs': int(np.abs(patch).max()),
            'roi_mean_abs': float(np.abs(patch).mean())}


def pilot_one(identity, row, core_hashes):
    """@luna-comment Input: exact selected manifest row/core SHA. Output: four new PNG/receipt paths and per-factor native evidence. Key Steps: verify source/frozen V1, share one short-segment analysis per warp, produce V1_zero plus four variants, inverse-map identical geometry, assert zero equals historical V1 before writes. Key Params: one image at a time. State/Dependencies: exclusive writes only approved Y pilot directory."""
    source = mapped_path(row['images']['vosr_v5_block'])
    source_receipt_path = source.parent.parent/'receipts'/(identity+'.json')
    source_receipt = json.loads(source_receipt_path.read_text(encoding='utf-8'))
    if digest(source) != source_receipt['final_sha256']:
        raise AssertionError('Viewer source hash mismatch '+identity)
    baseline = read_bgr(source)
    if baseline.shape[:2] != (row['height'], row['width']):
        raise AssertionError('Viewer native shape mismatch '+identity)
    historical = mapped_path(row['images']['light_v1_standard'])
    historical_receipt_path = historical.with_suffix('.json')
    historical_receipt = json.loads(historical_receipt_path.read_text(encoding='utf-8'))
    if (historical_receipt['source_sha256'] != source_receipt['final_sha256']
            or historical_receipt['core_sha256'] != core_hashes['frozen_v1']
            or digest(historical) != historical_receipt['final_sha256']):
        raise AssertionError('Frozen V1 receipt mismatch '+identity)
    frozen_v1 = read_bgr(historical)
    documents = documents_for(identity, source_receipt)
    keys = ('V1_zero',)+PRODUCE
    canvas = {key: np.zeros(baseline.shape[:2], np.float32) for key in keys}
    owned = {key: np.zeros(baseline.shape[:2], bool) for key in keys}
    stages_report = {key: [] for key in keys}
    for item, ocr, valid in documents:
        h, w = ocr['warp_hw']
        matrix = np.asarray(item['warp']['image_to_output'], np.float64)
        warp = cv2.warpPerspective(baseline, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        float_stages = v1_float_stages(warp, ocr['boxes'])
        evidence = segment_candidates(float_stages)
        x0, y0, x1, y1 = ROIS[identity]
        source_corners = np.asarray([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], np.float32)
        warp_corners = cv2.perspectiveTransform(source_corners, matrix).reshape(4, 2)
        roi_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(roi_mask, np.rint(warp_corners).astype(np.int32), 1)
        evidence['diagnostic_mask'] = roi_mask.astype(bool)
        corrections, detail = profile_corrections(float_stages, evidence)
        prepared = (float_stages, corrections, detail)
        for key in keys:
            enhanced, telemetry = enhance_bgr_v1_profile(warp, ocr['boxes'], key, prepared=prepared)
            difference = enhanced.astype(np.int16)-warp.astype(np.int16)
            if not (np.array_equal(difference[:, :, 0], difference[:, :, 1])
                    and np.array_equal(difference[:, :, 1], difference[:, :, 2])):
                raise AssertionError('Warp common-channel delta mismatch')
            mapped, _, take, geometry = inverse_residual(baseline, warp, enhanced, matrix, item['quad'], valid, owned[key])
            canvas[key] += mapped[:, :, 0]
            owned[key] |= take
            stages_report[key].append({'instance_id': item['instance_id'], 'ocr_sha256': ocr['hash'],
                                       'telemetry': telemetry, 'geometry': geometry})
            del enhanced, difference, mapped
        del warp, float_stages, evidence, corrections, prepared, roi_mask
    finals = {}
    native_info = {}
    for key in keys:
        finals[key], native_info[key] = finalize_native(baseline, canvas[key], owned[key])
        del canvas[key], owned[key]
    if not np.array_equal(finals['V1_zero'], frozen_v1):
        raise AssertionError('V1_zero native output differs from exact frozen V1 '+identity)
    pairs = {'U_minus_V1': pair_metrics(frozen_v1, finals['U'], ROIS[identity]),
             'UH_minus_U': pair_metrics(finals['U'], finals['UH'], ROIS[identity]),
             'US_minus_U': pair_metrics(finals['U'], finals['US'], ROIS[identity]),
             'UHS_minus_US': pair_metrics(finals['US'], finals['UHS'], ROIS[identity])}
    if any(digest(ROOT/path) != value for path, value in core_hashes['paths'].items()):
        raise AssertionError('Source code changed during pilot generation')
    outputs = {}
    for key in PRODUCE:
        target = PILOT/key/(identity+'.png')
        receipt_path = target.with_suffix('.json')
        if target.exists() or receipt_path.exists():
            raise FileExistsError('Pilot output already exists '+str(target))
        write_png(target, finals[key])
        receipt = {'id': identity, 'variant': key, 'status': 'experimental_visual_pending',
                   'source_url': row['images']['vosr_v5_block'],
                   'source_sha256': source_receipt['final_sha256'],
                   'source_receipt_sha256': digest(source_receipt_path),
                   'document_record_sha256': source_receipt['document_record_sha256'],
                   'frozen_v1_png_sha256': historical_receipt['final_sha256'],
                   'frozen_v1_receipt_sha256': digest(historical_receipt_path),
                   'core_sha256': {k: v for k, v in core_hashes.items() if k!='paths'},
                   'parameters': {'halo_mu': HALO_MU, 'slope_gamma': SLOPE_GAMMA, 'contrast': .75,
                                  'sharp': .65, 'final_max_delta': 24, 'C': 'deferred'},
                   'output_sha256': digest(target), 'native_hw': list(finals[key].shape[:2]),
                   'integration': 'existing_final_to_original_ocr_warp_inverse_common_residual',
                   'V1_zero_pixel_equal': True, 'native': native_info[key],
                   'stage_records': stages_report[key], 'final_integer_factor_pairs': pairs}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        outputs[key] = {'png': target.as_uri(), 'receipt': str(receipt_path),
                        'sha256': receipt['output_sha256'], 'native': native_info[key]}
    return {'id': identity, 'source_sha256': source_receipt['final_sha256'],
            'source_receipt_sha256': digest(source_receipt_path),
            'historical_v1_sha256': historical_receipt['final_sha256'],
            'V1_zero_pixel_equal': True, 'roi_xyxy': ROIS[identity],
            'output': outputs, 'final_integer_factor_pairs': pairs,
            'stages': stages_report['U']}


def main():
    """@luna-comment Input: frozen three-row subset of production manifest. Output: approved-Y pilot PNGs/receipts/summary only. Key Steps: hash-bind cores, process IDs, verify source code unchanged at end. Key Params: low-strength U/H/S only; no main-gallery edits. State/Dependencies: approved Y directory and existing receipt-bound OCR geometry."""
    cv2.setNumThreads(2)
    PILOT.mkdir(parents=True, exist_ok=True)
    sources = {'frozen_v1': 'tools/light_text_postprocess.py',
               'profile': 'tools/v1_edge_profile_pilot.py',
               'runner': 'tools/run_camera_v1_profile_pilot.py'}
    core = {key: digest(ROOT/value) for key, value in sources.items()}
    core['paths'] = {value: core[key] for key, value in sources.items()}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    rows = []
    for identity in IDS:
        row = next(value for value in manifest['rows'] if value['id']==identity)
        result = pilot_one(identity, row, core)
        rows.append(result)
        print(json.dumps({'done': identity, 'pairs': result['final_integer_factor_pairs']}), flush=True)
    if any(digest(ROOT/path) != value for path, value in core['paths'].items()):
        raise AssertionError('Source code changed during pilot run')
    summary = {'status': 'experimental_visual_pending', 'ids': list(IDS),
               'variants': list(PRODUCE), 'C': 'deferred',
               'core_sha256': {key: value for key, value in core.items() if key!='paths'},
               'rows': rows}
    destination = PILOT/'pilot_measurements.json'
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
