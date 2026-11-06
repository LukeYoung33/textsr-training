"""Two-sample normal-profile dehalo experiment; no batch inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare, smoothstep, write_png


def normal_correct(prepared, strength=1., negative=True):
    """@luna-comment Input: robust background buffers. Output: BGR, maps, receipt.
    Key Steps: luminance overshoot gate; inward-normal bright peak confirms trough;
    outward background anchors validate correction; continuous exterior fade.
    Key Params: inner10/outer28; inward2,4,6,8; outward8,12,16; trough max20.
    State/Dependencies: OpenCV sampling, immutable source; dark cores untouched.
    """
    src, y, roi, core, bg, confidence, distance = prepared
    residual = y - cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    band = roi & ~core & (distance < 28) & (confidence > 0)
    yy, xx = np.nonzero(band)
    smoothed = cv2.GaussianBlur(distance, (0, 0), 1)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)[yy, xx]
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)[yy, xx]
    norm = np.hypot(gx, gy)
    nx, ny = gx / np.maximum(norm, 1e-6), gy / np.maximum(norm, 1e-6)
    dist = distance[yy, xx]
    prior = np.full(len(xx), -1000., np.float32)
    anchors = []
    # OpenCV remap dimensions must stay below 32767, so use compact row blocks.
    for start in range(0, len(xx), 20000):
        end = min(start + 20000, len(xx))
        sub = slice(start, end)
        for step in (2, 4, 6, 8):
            mx = (xx[sub] - step * nx[sub]).astype(np.float32)[None]
            my = (yy[sub] - step * ny[sub]).astype(np.float32)[None]
            value = cv2.remap(residual, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1000)[0]
            dd = cv2.remap(distance, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1000)[0]
            prior[sub] = np.maximum(prior[sub], np.where((dd >= 1) & (dd <= dist[sub] - 1), value, -1000))
        local_anchors = []
        for step in (8, 12, 16):
            mx = (xx[sub] + step * nx[sub]).astype(np.float32)[None]
            my = (yy[sub] + step * ny[sub]).astype(np.float32)[None]
            value = cv2.remap(residual, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1000)[0]
            dd = cv2.remap(distance, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
            inside = cv2.remap(roi.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)[0]
            local_anchors.append(smoothstep((dd - 6) / 4) * (1 - smoothstep((np.abs(value) - 4) / 4)) * (inside > 0))
        anchors.append(np.stack(local_anchors, axis=1))
    anchor = np.concatenate(anchors) if anchors else np.empty((0, 3))
    anchor_weight = np.clip(anchor.sum(axis=1) / 2, 0, 1)
    rr = residual[yy, xx]
    taper = 1 - smoothstep((dist - 10) / 18)
    bright = smoothstep((rr - 2) / 4)
    dark = smoothstep((-rr - 3) / 3) * (1 - smoothstep((-rr - 15) / 5))
    dark *= smoothstep((prior - 3) / 3) * smoothstep((dist - 2) / 2)
    dark *= (1 - smoothstep((dist - 12) / 4)) * smoothstep((norm - .25) / .5) * anchor_weight * .65 * negative
    alpha = strength * confidence[yy, xx] * taper * (bright + dark)
    output = src.copy()
    output[yy, xx] += alpha[:, None] * (bg[yy, xx] - src[yy, xx])
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    changed = np.any(output != src, axis=2)
    assert not changed[core].any() and not changed[~band].any()
    weight = np.zeros_like(y)
    weight[yy, xx] = alpha
    ring = np.zeros_like(y)
    ring[yy, xx] = dark
    samples = np.flatnonzero(dark > .1)[::max(1, int((dark > .1).sum()) // 16)][:16]
    report = dict(core_pixels=int(core.sum()), core_changed=int(changed[core].sum()),
                  outside_changed=int(changed[~band].sum()), changed_pixels=int(changed.sum()),
                  positive_candidates=int((bright > 0).sum()), normal_trough_candidates=int((dark > 0).sum()),
                  outer_radius=28, inner_radius=10, strength=strength, negative=negative,
                  seed_kernel=31, seed_sigma=2, normal_valid_fraction=float((norm > .25).mean()),
                  profiles=[dict(x=int(xx[i]), y=int(yy[i]), distance=float(dist[i]),
                                 prior_peak=float(prior[i]), original_residual=float(rr[i]),
                                 corrected_residual=float(rr[i] * (1 - alpha[i])),
                                 normal_x=float(nx[i]), normal_y=float(ny[i]),
                                 anchor_confidence=float(anchor_weight[i])) for i in samples],
                  background_range_pixels_in_core=int((core & (y > 155) & (y < 185)).sum()))
    return output, dict(core=core.astype(np.float32), correction_weight=weight, normal_trough_weight=ring), report


def main():
    """@luna-comment Input: sample source/output paths. Output: two images/debug/crops.
    Key Steps: robust prepare, normal correction, compare original/v3/v4; hash checks.
    Key Params: only screenshot image IDs 044920599 and 052927799; original resolution.
    State/Dependencies: local files only, never invokes model or launches batch.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for ident in ('044920599', '052927799'):
        path = next(args.input.glob(f'*{ident}_preprocessInput.png'))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        src = cv2.imdecode(np.fromfile(str(path), np.uint8), 1)
        prepared = prepare(src, robust_seed=True)
        output, debug, report = normal_correct(prepared)
        low, _, _ = normal_correct(prepared, negative=False)
        dest = args.output / (path.stem + '_dehalo.png')
        write_png(dest, output)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        decoded = cv2.imdecode(np.fromfile(str(dest), np.uint8), 1)
        assert np.array_equal(decoded, output)
        for name, values in debug.items():
            write_png(args.output / 'debug' / path.stem / (name + '.png'), np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))
        write_png(args.output / 'debug' / path.stem / 'background.png', np.clip(prepared[4], 0, 255).astype(np.uint8))
        oldpath = Path('results/ppt_dehalo_20260924/sample_v3_outward') / dest.name
        old = cv2.imdecode(np.fromfile(str(oldpath), np.uint8), 1)
        crops = dict(practice=(950, 740, 2150, 990), function=(1300, 510, 2530, 750),
                     symbols=(1300, 1130, 2800, 1320), degrees=(1750, 1260, 2950, 1480),
                     color=(600, 1690, 2700, 1890), small=(700, 1530, 2400, 1690))
        for label, (x0, y0, x1, y1) in crops.items():
            panels = []
            for title, im in [('original', src), ('previous_v3', old), ('v4_positive_only', low), ('v4_normal_dehalo', output)]:
                panel = cv2.copyMakeBorder(im[y0:y1, x0:x1], 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                cv2.putText(panel, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                panels.append(panel)
            write_png(args.output / 'comparisons' / (path.stem + '_' + label + '.png'), np.concatenate(panels))
        report.update(source=str(path), sha256=sha, output=str(dest), shape=list(src.shape))
        reports.append(report)
        (args.output / 'receipt.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
