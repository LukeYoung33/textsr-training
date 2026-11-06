"""Measure final native-frame pixel impact of each V2 module on fixed camera samples."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.doc_warp_composition import inverse_residual
from tools.light_text_postprocess_v2 import PRESETS, enhance_bgr_v2
from tools.run_camera_light_post_v2 import BASE, MANIFEST, OUT, documents_for, mapped_path, read_bgr


def output_for(baseline, documents, config):
    """@luna-comment Input: exact baseline, bound OCR geometry, one V2 switch config. Output: native BGR8 and per-instance hit telemetry. Key Steps: warp existing final, enhance, map only scalar residual through fixed ownership, common gamut-limited integer delta. Key Params: standard gains except one disabled switch. State/Dependencies: no image writes or model inference."""
    canvas = np.zeros(baseline.shape[:2], np.float32)
    owned = np.zeros(baseline.shape[:2], bool)
    reports = []
    for item, ocr, valid in documents:
        h, w = ocr['warp_hw']
        warped = cv2.warpPerspective(baseline, np.asarray(item['warp']['image_to_output'], np.float64), (w, h))
        enhanced, report = enhance_bgr_v2(warped, config=config, boxes=ocr['boxes'])
        mapped, _, take, _ = inverse_residual(baseline, warped, enhanced, item['warp']['image_to_output'], item['quad'], valid, owned)
        canvas += mapped[:, :, 0]
        owned |= take
        reports.append(report)
    delta = np.rint(np.minimum(np.maximum(canvas, -baseline.min(2).astype(np.float32)), 255-baseline.max(2).astype(np.float32))).astype(np.int16)
    return (baseline.astype(np.int16)+delta[:, :, None]).astype(np.uint8), reports


def evaluate(identity):
    """@luna-comment Input: completed camera image ID. Output: approved-Y JSON with final BGR ablation pixel counts. Key Steps: verify source, evaluate standard and each module off, compare native scalar differences. Key Params: one image at a time. State/Dependencies: writes only dedicated Y evaluation folder."""
    row = next(v for v in json.loads(MANIFEST.read_text(encoding='utf-8'))['rows'] if v['id'] == identity)
    baseline_path = mapped_path(row['images']['vosr_v5_block'])
    receipt = json.loads((baseline_path.parent.parent/'receipts'/(identity+'.json')).read_text(encoding='utf-8'))
    baseline = read_bgr(baseline_path)
    documents = documents_for(identity, receipt)
    standard = PRESETS['standard']
    full, reports = output_for(baseline, documents, standard)
    results = {'id': identity, 'baseline_changed_pixels': int(np.count_nonzero(full[:, :, 0] != baseline[:, :, 0])), 'full_stage_hits': reports, 'ablations': {}}
    for name, flag in [('halo', 'dehalo'), ('contrast', 'polarity'), ('aa', 'directional_aa'), ('sharp', 'sharpening'), ('adaptive_feather', 'adaptive_feather'), ('noise', 'noise')]:
        disabled, _ = output_for(baseline, documents, replace(standard, **{flag: False}))
        diff = disabled[:, :, 0].astype(np.int16)-full[:, :, 0].astype(np.int16)
        results['ablations'][name] = {'final_pixels_different': int(np.count_nonzero(diff)), 'max_abs_difference': int(np.abs(diff).max()), 'mean_abs_difference': float(np.abs(diff).mean())}
    target = OUT/'evaluation'/(identity+'_ablation.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('id')
    print(evaluate(parser.parse_args().id))
