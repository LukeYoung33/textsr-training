"""P16: explicit polarity normalization of the complete frozen V13b pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_contour_v6 import read_image
from tools.postprocess_ppt_dehalo_20260924 import prepare, write_png, dilate, smoothstep
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
from tools.postprocess_ppt_clean_background_v13 import clean_background_feather


def prepare_scoped(image, roi):
    """@luna-comment Input: normalized BGR and explicit board polygon mask.
    Output: frozen preparation tuple scoped before core and background sampling.
    Key Steps: original median seed and normalized Gaussian background formulas.
    Key Params: identical median31/sigma2/22/48/exclusion8. State: no global edits.
    """
    source = image.astype(np.float32)
    y = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    h, w = y.shape
    small = cv2.resize(y, (max(1, w//4), max(1, h//4)), interpolation=cv2.INTER_AREA)
    initial = cv2.medianBlur(np.clip(np.rint(small), 0, 255).astype(np.uint8), 31).astype(np.float32)
    initial = cv2.GaussianBlur(initial, (0, 0), 2)
    initial = cv2.resize(initial, (w, h), interpolation=cv2.INTER_LINEAR)
    core = roi & ((initial-y) > np.maximum(24, .18*initial))
    weight = (roi & ~dilate(core, 8)).astype(np.float32)
    den = cv2.GaussianBlur(weight, (0, 0), 22)
    num = cv2.GaussianBlur(source*weight[..., None], (0, 0), 22)
    background = num/np.maximum(den[..., None], 1e-6)
    large_den = cv2.GaussianBlur(weight, (0, 0), 48)
    large_num = cv2.GaussianBlur(source*weight[..., None], (0, 0), 48)
    large = large_num/np.maximum(large_den[..., None], 1e-6)
    blend = smoothstep((den-.02)/.10)[..., None]
    background = blend*background+(1-blend)*large
    support = smoothstep((large_den-.02)/.06)
    distance = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return source, y, roi, core, background, support, distance


def process_image(source, polarity, roi=None, support_fallback=False):
    """@luna-comment Input: uint8 BGR and explicit text polarity.
    Output: result, geometry/debug buffers, stage timing and protection receipt.
    Key Steps: normalize BEFORE preparation; frozen V4/P15b; inverse complement.
    Key Params: frozen constants; opt-in support fallback. No input mutation.
    """
    if polarity not in ('dark-on-light', 'light-on-dark'):
        raise ValueError('Explicit supported polarity required')
    inverse = polarity == 'light-on-dark'
    normalized = 255-source if inverse else source
    start = time.perf_counter()
    prepared = prepare(normalized, robust_seed=True) if roi is None else prepare_scoped(normalized, roi)
    prep_seconds = time.perf_counter()-start
    v4, maps, _ = normal_correct(prepared)
    v4_seconds = time.perf_counter()-start-prep_seconds
    base = prepared[0]+maps['correction_weight'][..., None]*(prepared[4]-prepared[0])
    processed, debug, report = clean_background_feather(base, prepared, exterior_cap=True,
                                                       support_fallback=support_fallback)
    result = 255-processed if inverse else processed
    changed = np.any(result != source, axis=2)
    assert not changed[prepared[3]].any()
    assert not changed[~prepared[2]].any()
    report.update(polarity=polarity, prepare_seconds=prep_seconds, v4_seconds=v4_seconds,
                  total_seconds=time.perf_counter()-start,
                  original_core_changed=int(changed[prepared[3]].sum()),
                  outside_roi_changed=int(changed[~prepared[2]].sum()),
                  roi_pixels=int(prepared[2].sum()))
    debug.update(normalized_input=normalized, roi=prepared[2], core=prepared[3],
                 v4_original_domain=255-v4 if inverse else v4,
                 target_original_domain=255-debug['target'] if inverse else debug['target'])
    return result, debug, report


def main():
    """@luna-comment Input: explicit sample path/polarity and new output directory.
    Output: full result, readable debug PNGs, hash and roundtrip receipt.
    Key Steps: execute pipeline once; save normalized maps and source-domain colors.
    Key Params: sample only. State/Dependencies: local files; no inference/batch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--polarity', choices=['dark-on-light', 'light-on-dark'], default='dark-on-light')
    parser.add_argument('--board-polygon', help='Explicit full-resolution x,y;x,y;... vertices')
    parser.add_argument('--support-fallback', action='store_true', help='Retain input at unsupported background positions')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    source = read_image(args.input)
    roi = None
    if args.board_polygon:
        vertices = np.array([[int(v) for v in point.split(',')] for point in args.board_polygon.split(';')], np.int32)
        roi = np.zeros(source.shape[:2], np.uint8)
        cv2.fillPoly(roi, [vertices], 1)
        roi = roi.astype(bool)
    result, debug, report = process_image(source, args.polarity, roi, args.support_fallback)
    report['board_polygon'] = args.board_polygon
    write_png(args.output/'result.png', result)
    write_png(args.output/'source.png', source)
    for name, value in debug.items():
        if value.dtype == bool:
            encoded = value.astype(np.uint8)*255
        elif value.ndim == 3:
            encoded = np.clip(np.rint(value), 0, 255).astype(np.uint8)
        else:
            scale = 255 if ('weight' in name or name.startswith('support') or name == 'fallback_confidence') else 1
            encoded = np.clip(np.rint(value*scale), 0, 255).astype(np.uint8)
        write_png(args.output/(name+'.png'), encoded)
    overlay = source.copy()
    overlay[debug['core']] = (0, 255, 0)
    contours, _ = cv2.findContours(debug['roi'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
    write_png(args.output/'roi_core_overlay.png', overlay)
    if args.support_fallback:
        low = np.clip(np.rint(source.astype(np.float32)+.5*(result.astype(np.float32)-source)), 0, 255).astype(np.uint8)
        write_png(args.output/'low_strength.png', low)
        preview = np.concatenate([cv2.resize(im, (1040, 780), interpolation=cv2.INTER_AREA)
                                  for im in (source, result)], axis=1)
        write_png(args.output/'comparison_preview.png', preview)
    assert np.array_equal(read_image(args.output/'result.png'), result)
    for path in args.output.glob('*.png'):
        if path.name == 'comparison_preview.png':
            assert read_image(path).shape == (780, 2080, 3)
            continue
        assert read_image(path).shape[:2] == source.shape[:2]
    assert digest == hashlib.sha256(args.input.read_bytes()).hexdigest()
    report.update(input=str(args.input), source_sha256=digest, shape=list(source.shape),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  debug_domain='normalized except source, result, v4_original_domain, target_original_domain, roi_core_overlay',
                  decoded_pngs=len(list(args.output.glob('*.png'))))
    (args.output/'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
