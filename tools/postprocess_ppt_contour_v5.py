"""Sample-only, bounded exterior low-frequency luminance repair over V4."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare, smoothstep, write_png


def repair_contour(previous, prepared):
    """@luna-comment Input: uint8 V4 and robust background buffers. Output: V5/debug.
    Key Steps: estimate masked low-frequency residual; subtract bounded signed Y.
    Key Params: sigma2, deadzone1.5, max3 levels, exterior ramps10..16/28..40.
    State/Dependencies: immutable inputs; equal RGB correction preserves float chroma.
    """
    src, original_y, roi, core, bg, support, distance = prepared
    image = previous.astype(np.float32)
    y = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    by = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    weight = (roi & ~core & (distance > 8)).astype(np.float32)
    weight *= smoothstep((original_y - by + 12) / 8)
    denominator = cv2.GaussianBlur(weight, (0, 0), 2)
    residual = cv2.GaussianBlur((y - by) * weight, (0, 0), 2) / np.maximum(denominator, 1e-6)
    envelope = smoothstep((distance - 10) / 6) * (1 - smoothstep((distance - 28) / 12))
    envelope *= roi * (~core) * support * smoothstep((denominator - .1) / .4)
    envelope *= smoothstep((original_y - by + 12) / 8)
    delta = np.sign(residual) * np.clip(np.abs(residual) - 1.5, 0, 3) * envelope
    output = np.clip(np.rint(image - delta[..., None]), 0, 255).astype(np.uint8)
    protected = core | ~roi | (distance <= 10) | (distance >= 40)
    assert np.array_equal(output[protected], previous[protected])
    return output, delta, residual


def main():
    """@luna-comment Input: source/output CLI paths. Output: two V5 samples and A/B.
    Key Steps: load original/V4, prepare, repair, decode-check, save diagnostic report.
    Key Params: two fixed sample IDs; no batch support or inference.
    State/Dependencies: read-only originals/V4; exclusive new output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    reports = []
    for ident in ('044920599', '052927799'):
        path = next(args.input.glob(f'*{ident}_preprocessInput.png'))
        src = cv2.imdecode(np.fromfile(str(path), np.uint8), 1)
        name = path.stem + '_dehalo.png'
        previous = cv2.imdecode(np.fromfile(str(Path('results/ppt_dehalo_20260924/sample_v4_normal') / name), np.uint8), 1)
        prepared = prepare(src, robust_seed=True)
        result, delta, residual = repair_contour(previous, prepared)
        write_png(args.output / name, result)
        assert np.array_equal(cv2.imdecode(np.fromfile(str(args.output / name), np.uint8), 1), result)
        write_png(args.output / 'debug' / (path.stem + '_signed_delta.png'), np.clip(np.rint(128 + delta * 30), 0, 255).astype(np.uint8))
        report = dict(image=name, max_change=int(np.abs(result.astype(np.int16)-previous).max()),
                      changed_pixels=int(np.any(result != previous, axis=2).sum()), protected_changes=0,
                      inner_radius=10, outer_radius=40, maximum_luma_delta=3, sigma=2)
        crops = dict(practice=(950, 740, 2150, 990), function=(1300, 510, 2530, 750),
                     symbols=(1300, 1130, 2800, 1320), color=(600, 1690, 2700, 1890))
        for label, (x0,y0,x1,y1) in crops.items():
            panels = []
            for title, image in [('V4',previous),('V5 exterior luminance',result)]:
                panel = cv2.copyMakeBorder(image[y0:y1,x0:x1],26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
                cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                panels.append(panel)
            write_png(args.output/'comparisons'/(path.stem+'_'+label+'.png'),np.concatenate(panels))
        reports.append(report)
        (args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
