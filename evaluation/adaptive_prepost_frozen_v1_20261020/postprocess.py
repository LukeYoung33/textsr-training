"""Adaptive tonal clarity and reference brightness repair; aligned-frame prototype."""
from pathlib import Path

import cv2
import numpy as np

from .lowres_curve import (
    estimate_line, prepare_mapping, apply_tables,
)


def build_transfer_tables(current, reference, gamma=1.2):
    """@luna-comment
    Input: Current low-resolution model, frozen reference packed anchors, gamma>=1.
    Output: Monotone Q8.8 current-to-reference LUTs for every detector-column node.
    Key Steps: Recover current local rails; map both S branches into reference rails/T.
    Key Params: Gamma1 is brightness-only;1.2 mild tonal clarity; no identity/confidence gate.
    State/Dependencies: All power operations occur here on low-resolution tables only.
    """
    if not np.isfinite(gamma) or gamma < 1:
        raise ValueError('gamma must be finite and at least1')
    if not np.array_equal(current['centers'], reference['centers']):
        raise ValueError('Current/reference canvas nodes must match')
    anchors = current['paper'][:, None]/128*np.asarray(current['normalized_anchors'])[None, :]
    anchors *= np.minimum(1., 255/np.maximum(anchors[:, 2], 1))[:, None]
    target = np.asarray(reference['anchors'], dtype=np.float64)
    if not (np.all(np.diff(anchors, axis=1) > 0) and np.all(np.diff(target, axis=1) > 0)):
        raise ValueError('Unordered anchors')
    low, mid, high = [anchors[:, i, None] for i in range(3)]
    rl, rt, rh = [target[:, i, None] for i in range(3)]
    value = np.arange(256)[None, :]
    left = rl+(rt-rl)*np.clip((value-low)/(mid-low), 0, 1)**gamma
    right = rh-(rh-rt)*np.clip((high-value)/(high-mid), 0, 1)**gamma
    lut = np.where(value <= mid, left, right)
    tables = np.rint(lut*256).astype(np.uint16)
    assert np.all(np.diff(tables.astype(np.int32), axis=1) >= 0)
    return dict(tables=tables, valid=np.ones(len(tables), bool), centers=current['centers'],
                anchors=anchors, target_anchors=target, gamma=gamma)


def postprocess_bgr(prediction, context, gamma=1.2):
    """@luna-comment
    Input: Aligned model-output uint8 BGR and original frozen preprocessing context.
    Output: Brightness-repaired/tonally sharpened BGR and per-line telemetry.
    Key Steps: Low-resolution current estimates; reuse reference rails; lookup-only first-owner render.
    Key Params: Gamma1/1.2 ablation; scaled boxes; no resizing of output, masks or border protection.
    State/Dependencies: Requires same physical frame; rotation/warp needs an explicit external transform.
    """
    if context['version'] != 'adaptive_dualmode_scurve_v1':
        raise ValueError('Unsupported preprocessing context')
    if prediction.dtype != np.uint8 or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError('Expected uint8 BGR')
    height, width = prediction.shape[:2]
    sh, sw = context['source_shape']
    y = cv2.cvtColor(prediction, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    canvas = cv2.resize(y, (1984, 1472))
    output = prediction.copy()
    owned = np.zeros(y.shape, bool)
    reports = []
    for line in context['lines']:
        current = estimate_line(canvas, line['model']['canvas_bbox'], lower_estimator='mode')
        packed = build_transfer_tables(current, line['packed'], gamma)
        a, b, c, d = line['bbox']
        box = [max(0, int(np.floor(a*width/sw))), max(0, int(np.floor(b*height/sh))),
               min(width, int(np.ceil(c*width/sw))), min(height, int(np.ceil(d*height/sh)))]
        x0, y0, x1, y1 = box
        mapping = prepare_mapping(packed, box, y.shape, context['canvas_shape'])
        processed, _ = apply_tables(prediction[y0:y1, x0:x1], y[y0:y1, x0:x1], packed, mapping)
        free = ~owned[y0:y1, x0:x1]
        output[y0:y1, x0:x1][free] = processed[free]
        owned[y0:y1, x0:x1] = True
        reports.append(dict(index=line['index'], bbox=box, all_columns=bool(mapping['full'].all()),
                            current_anchors=current['normalized_anchors'],
                            reference_anchors=line['model']['normalized_anchors'], gamma=gamma,
                            target_reused=True, current_reestimated=True))
    return output, reports


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--context', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gamma', type=float, default=1.2)
    args = parser.parse_args()
    prediction = cv2.imread(str(args.prediction))
    if prediction is None:
        raise ValueError('Cannot decode model prediction')
    context = json.loads(args.context.read_text(encoding='utf-8'))
    output, reports = postprocess_bgr(prediction, context, args.gamma)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise ValueError('Cannot save postprocessed output')
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        reference_id=context['reference_id'], gamma=args.gamma, geometry=context['geometry'],
        current_reestimated=True, target_reused=True, native_statistical_pixels=0,
        lines=reports), indent=2), encoding='utf-8')
