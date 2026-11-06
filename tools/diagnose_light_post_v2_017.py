"""Measure source-coordinate edge profiles and switch ablations for rejected camera017 V2."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_light_post_v2_ablation import output_for
from tools.light_text_postprocess_v2 import PRESETS, edge_field, normal_sample
from tools.run_camera_light_post_v2 import MANIFEST, OUT, documents_for, mapped_path, read_bgr

ROI = (1450, 2370, 1940, 2480)


def ring_values(crop, nx, ny, mask):
    """@luna-comment Input: native BGR crop and baseline-derived edge mask/normals. Output: nonnegative bright-side peak above +6 platform per fixed edge. Key Steps: compare normal samples at +1..+4 to +6. Key Params: source-derived local overshoot proxy, not a GT halo annotation. State/Dependencies: OpenCV bilinear samples."""
    y = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    shoulder = np.stack([normal_sample(y, nx, ny, distance)[mask] for distance in range(1, 5)])
    platform = normal_sample(y, nx, ny, 6)[mask]
    return np.maximum(shoulder.max(axis=0)-platform, 0)


def edge_metrics(crop, nx, ny, mask):
    """@luna-comment Input: native BGR ROI and baseline-derived text-edge normals/mask. Output: mean normal profile, side contrast and outer bright-ring statistics. Key Steps: sample identical pixels at -6..+6 along fixed normals. Key Params: exploratory source-derived mask, not GT halo labels. State/Dependencies: OpenCV bilinear remap."""
    y = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    values = np.stack([normal_sample(y, nx, ny, distance)[mask] for distance in range(-6, 7)])
    bright_platform = values[12]
    ring = np.maximum(np.max(values[7:11], axis=0)-bright_platform, 0)
    return {'count': int(mask.sum()), 'profile_minus6_to_plus6': np.round(values.mean(axis=1), 3).tolist(),
            'side_contrast_plus6_minus_minus6': float((values[12]-values[0]).mean()),
            'positive_outer_ring_mean': float(ring.mean()), 'positive_outer_ring_p95': float(np.percentile(ring, 95))}


def main():
    """@luna-comment Input: exact published-source camera017 and its three completed trial files. Output: approved-Y rejected-result diagnostic JSON. Key Steps: select baseline text edges, compare four native profiles, run module-off final-image ablations with the same geometry, verify zero-strength warp identity. Key Params: fixed 490x110 native ROI; no candidate tuning. State/Dependencies: read-only source/results and Y evaluation JSON write."""
    row = next(item for item in json.loads(MANIFEST.read_text(encoding='utf-8'))['rows'] if item['id'] == '017')
    baseline_path = mapped_path(row['images']['vosr_v5_block'])
    receipt = json.loads((baseline_path.parent.parent/'receipts/017.json').read_text(encoding='utf-8'))
    baseline = read_bgr(baseline_path)
    documents = documents_for('017', receipt)
    x0, y0, x1, y1 = ROI
    baseline_crop = baseline[y0:y1, x0:x1]
    base_y = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    nx, ny, _, mag = edge_field(base_y)
    minus3 = normal_sample(base_y, nx, ny, -3)
    plus3 = normal_sample(base_y, nx, ny, 3)
    mask = (mag > 12) & (minus3 < 100) & (plus3 > 130) & ((plus3-minus3) > 50)
    mask[:8] = False
    mask[-8:] = False
    mask[:, :8] = False
    mask[:, -8:] = False
    if int(mask.sum()) < 100:
        raise AssertionError('Insufficient source-derived text edge support')
    data = {'id': '017', 'roi_xyxy': ROI, 'edge_selection': 'Baseline Sobel normal, dark -3 (<100), bright +3 (>130), separation >50; source-derived, not GT', 'profiles': {}, 'ring_relative_to_baseline': {}, 'ablations': {}}
    full_images = {}
    paths = {'baseline': baseline_path, **{key: OUT/key/'017.png' for key in ('light_v1_standard', 'light_v2_low', 'light_v2_standard')}}
    for key, path in paths.items():
        crop = read_bgr(path)[y0:y1, x0:x1]
        full_images[key] = crop
        data['profiles'][key] = edge_metrics(crop, nx, ny, mask)
    base_ring = ring_values(baseline_crop, nx, ny, mask)
    coordinates = np.argwhere(mask)
    for key, crop in full_images.items():
        if key == 'baseline':
            continue
        diff = ring_values(crop, nx, ny, mask)-base_ring
        top = int(np.argmax(diff))
        data['ring_relative_to_baseline'][key] = {'mean_delta': float(diff.mean()), 'p95_delta': float(np.percentile(diff, 95)),
                                                   'pixels_increase_ge_2': int(np.count_nonzero(diff >= 2)),
                                                   'pixels_increase_ge_5': int(np.count_nonzero(diff >= 5)),
                                                   'max_increase': float(diff[top]),
                                                   'max_increase_source_xy': [int(coordinates[top, 1]+x0), int(coordinates[top, 0]+y0)]}
    standard_crop = full_images['light_v2_standard']
    for name, flag in [('contrast', 'polarity'), ('halo', 'dehalo'), ('aa', 'directional_aa'), ('sharp', 'sharpening')]:
        output, reports = output_for(baseline, documents, replace(PRESETS['standard'], **{flag: False}))
        crop = output[y0:y1, x0:x1]
        diff = crop[:, :, 0].astype(np.int16)-standard_crop[:, :, 0].astype(np.int16)
        data['ablations'][name] = {'final_roi_pixels_different': int(np.count_nonzero(diff)),
                                   'max_abs_roi_difference': int(np.abs(diff).max()),
                                   'mean_signed_roi_difference': float(diff.mean()),
                                   'profile': edge_metrics(crop, nx, ny, mask), 'stage_reports': reports}
        del output
    zero, reports = output_for(baseline, documents, PRESETS['off'])
    data['zero_strength_native_identity'] = bool(np.array_equal(zero, baseline))
    data['zero_strength_stage_reports'] = reports
    target = OUT/'evaluation/017_rejected_quality_diagnosis.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
