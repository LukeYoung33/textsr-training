"""Two-sample P8 experiment: smooth the final correction, never the image."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare, smoothstep, write_png
from tools.postprocess_ppt_contour_v5 import repair_contour


def smooth_correction(base, delta, prepared):
    """@luna-comment Input: V4, float D5, fixed buffers. Output: V6/D6/protection.
    Key Steps: Gaussian final field then clearance feather; preserve protected pixels.
    Key Params: sigma3, clearance1..6, d10..40, original residual floor-12.
    State/Dependencies: immutable arrays; OpenCV; no image smoothing.
    """
    _, y, roi, core, bg, _, distance = prepared
    by = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    protected = core | ~roi | (distance <= 10) | (distance >= 40) | (y-by <= -12)
    clearance = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    correction = cv2.GaussianBlur(delta, (0, 0), 3) * smoothstep((clearance-1)/5)
    floating = base.astype(np.float32) - correction[..., None]
    clipped = int(np.count_nonzero((floating < 0) | (floating > 255)))
    result = np.clip(np.rint(floating), 0, 255).astype(np.uint8)
    assert np.array_equal(result[protected], base[protected])
    assert np.max(np.abs(correction)) <= min(3, np.max(np.abs(delta))) + 1e-5
    return result, correction, protected, clipped


def read_image(path):
    """@luna-comment Input: Unicode PNG path. Output: decoded BGR; fail on corruption.
    Key Steps: byte load and decode. Key Params: color mode1.
    State/Dependencies: read-only filesystem and OpenCV.
    """
    result = cv2.imdecode(np.fromfile(str(path), np.uint8), 1)
    if result is None:
        raise ValueError(str(path))
    return result


def main():
    """@luna-comment Input: two-sample source/output paths. Output: images/debug/receipt.
    Key Steps: reconstruct D5, run P8, assert contracts, compare and measure.
    Key Params: fixed sample IDs and matched crops; no batch switch.
    State/Dependencies: read-only originals/V4/V5; exclusive new result directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    reports = []
    crops = dict(practice=(950,740,2150,990), title=(950,450,2750,760),
                 color=(600,1690,2700,1890), angles=(1400,1300,2700,1540))
    for ident in ('044920599', '052927799'):
        path = next(args.input.glob(f'*{ident}_preprocessInput.png'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name = path.stem + '_dehalo.png'
        root = Path('results/ppt_dehalo_20260924')
        base = read_image(root/'sample_v4_normal'/name)
        old = read_image(root/'sample_v5_contour_final'/name)
        prepared = prepare(read_image(path), robust_seed=True)
        rebuilt, d5, _ = repair_contour(base, prepared)
        assert np.array_equal(rebuilt, old)
        result, d6, protected, clipped = smooth_correction(base, d5, prepared)
        write_png(args.output/name, result)
        assert np.array_equal(read_image(args.output/name), result)
        report = dict(image=name, source_correction='unquantized_D5', sigma=3,
                      clearance_zero=1, clearance_full=6, max_d5=float(np.abs(d5).max()),
                      max_d6=float(np.abs(d6).max()), clipping_count=clipped,
                      protected_changed=0, changed_pixels=int(np.any(old != result,axis=2).sum()), crops={})
        fields = []
        for label, field in [('D5',d5),('D6',d6)]:
            debug = np.clip(np.rint(128+30*field),0,255).astype(np.uint8)
            write_png(args.output/'debug'/(path.stem+'_'+label+'.png'),debug)
            fields.append(debug)
        for label,(x0,y0,x1,y1) in crops.items():
            panels = []
            for title,im in [('V4',base),('V5',old),('V6 field blend',result)]:
                panel = cv2.copyMakeBorder(im[y0:y1,x0:x1],26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
                cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                panels.append(panel)
            write_png(args.output/'comparisons'/(path.stem+'_'+label+'.png'),np.concatenate(panels))
            write_png(args.output/'debug'/(path.stem+'_'+label+'_D5_D6.png'),np.concatenate([f[y0:y1,x0:x1] for f in fields]))
            metrics=[]
            for field in (d5,d6):
                patch=field[y0:y1,x0:x1]
                metrics.append(dict(laplacian=float(np.abs(cv2.Laplacian(patch,cv2.CV_32F)).mean()),
                                    gradient=float(np.hypot(*np.gradient(patch)).mean())))
            report['crops'][label]=metrics
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        reports.append(report)
        (args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
