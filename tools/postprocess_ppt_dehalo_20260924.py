"""Offline traditional exterior halo suppression; frozen task plan dated 20260924."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def write_png(path, image):
    """@luna-comment Input: path and uint8 image. Output: lossless PNG file.
    Key Steps: encode and write Unicode-safe. Key Params: PNG compression 3.
    State/Dependencies: creates parent directories; OpenCV/NumPy filesystem I/O.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(str(path))
    data.tofile(str(path))


def dilate(mask, radius):
    """@luna-comment Input: bool mask, radius pixels. Output: expanded bool mask.
    Key Steps: elliptical dilation. Key Params: odd kernel diameter 2r+1.
    State/Dependencies: OpenCV, no shared mutable state.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def smoothstep(value):
    """@luna-comment Input: float array. Output: continuous ramp in [0,1].
    Key Steps: clamp and cubic map. Key Params: endpoints 0 and 1.
    State/Dependencies: NumPy only; no side effects.
    """
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def prepare(image, robust_seed=False):
    """@luna-comment Input: uint8 BGR image. Output: background and fixed masks.
    Key Steps: dominant slide ROI, dark seeds, masked background smoothing.
    Key Params: optional median31 seed; core max(24,.18B); exclusion8; sigma22/48.
    State/Dependencies: OpenCV and SciPy; full-resolution float buffers local.
    """
    source = image.astype(np.float32)
    y = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    h, w = y.shape
    small = cv2.resize(y, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    light = cv2.morphologyEx((small > 85).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(light)
    if n <= 1:
        raise ValueError('No bright slide region')
    region = binary_fill_holes(labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    region = cv2.erode(region, np.ones((9, 9), np.uint8))
    roi = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if robust_seed:
        initial = cv2.medianBlur(np.clip(np.rint(small), 0, 255).astype(np.uint8), 31).astype(np.float32)
        initial = cv2.GaussianBlur(initial, (0, 0), 2)
    else:
        initial = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        initial = cv2.GaussianBlur(initial, (0, 0), 5)
    initial = cv2.resize(initial, (w, h), interpolation=cv2.INTER_LINEAR)
    core = roi & ((initial - y) > np.maximum(24, .18 * initial))
    weight = (roi & ~dilate(core, 8)).astype(np.float32)
    denominator = cv2.GaussianBlur(weight, (0, 0), 22)
    numerator = cv2.GaussianBlur(source * weight[..., None], (0, 0), 22)
    background = numerator / np.maximum(denominator[..., None], 1e-6)
    den_large = cv2.GaussianBlur(weight, (0, 0), 48)
    num_large = cv2.GaussianBlur(source * weight[..., None], (0, 0), 48)
    large = num_large / np.maximum(den_large[..., None], 1e-6)
    blend = smoothstep((denominator - .02) / .10)[..., None]
    background = blend * background + (1 - blend) * large
    support = smoothstep((den_large - .02) / .06)
    distance = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return source, y, roi, core, background, support, distance


def correct(prepared, strength=1., ring_strength=.5, outer_radius=12.):
    """@luna-comment Input: prepare buffers and strengths. Output: uint8 result/debug/report.
    Key Steps: exterior positive RGB cap, halo-evidenced shallow trough correction.
    Key Params: +2 cap; outer fade10..outer_radius; ring deficit3..18, d3..10.
    State/Dependencies: input buffers immutable; correction masks computed from input.
    """
    source, y, roi, core, background, support, distance = prepared
    by = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    if outer_radius <= 10:
        raise ValueError('outer_radius must exceed protected distance 10')
    exterior = roi & ~core & (support > 0) & (distance <= outer_radius)
    taper = 1 - smoothstep((distance - 10) / (outer_radius - 10))
    excess = np.maximum(source - background - 2, 0)
    alpha = np.clip(excess / 3, 0, 1) * (exterior * taper * support)[..., None] * strength
    output = source - alpha * excess
    halo = roi & (support > 0) & (distance >= 1) & (distance <= 6) & (y - by > 4)
    deficit = by - y
    trough = dilate(halo, 3) & exterior & (distance >= 3) & (distance <= 10)
    trough &= (deficit >= 3) & (deficit <= 18) & ~dilate(core, 2)
    ring_taper = smoothstep(distance - 3) * (1 - smoothstep((distance - 8) / 2))
    ring_taper *= smoothstep((deficit - 3) / 3) * (1 - smoothstep((deficit - 15) / 3))
    ring_alpha = ring_strength * trough * ring_taper * support
    output += ring_alpha[..., None] * (background - output)
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    changed = np.any(output != source, axis=2)
    after_y = cv2.cvtColor(output.astype(np.float32), cv2.COLOR_BGR2GRAY)
    report = dict(strength=strength, ring_strength=ring_strength, outer_radius=outer_radius,
                  roi_fraction=float(roi.mean()), core_pixels=int(core.sum()),
                  bright_corrected=int((np.any(alpha > 0, axis=2) & changed).sum()),
                  ring_corrected=int(((ring_alpha > 0) & changed).sum()),
                  max_correction=float(np.max(np.abs(output.astype(np.float32) - source))),
                  core_changed=int((changed & core).sum()),
                  outside_mask_changed=int((changed & ~exterior).sum()),
                  positive_before=float(np.maximum(y - by - 2, 0)[halo].mean()) if halo.any() else 0.,
                  positive_after=float(np.maximum(after_y - by - 2, 0)[halo].mean()) if halo.any() else 0.)
    assert report['core_changed'] == report['outside_mask_changed'] == 0
    return output, dict(roi=roi, core=core, halo=halo, ringing=ring_alpha > 0, changed=changed), report


def main():
    """@luna-comment Input: CLI source/output/sample options. Output: PNGs and receipt.
    Key Steps: enumerate suffix, process sequentially, preserve hashes, decode checks.
    Key Params: sample emits low/default/P1 comparisons; batch uses P1=1/P2=.5.
    State/Dependencies: reads source only; writes new output directory, no inference.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--outer-radius', type=float, default=12.)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    paths = sorted(args.input.glob('*_preprocessInput.png'))
    if args.sample:
        paths = [p for p in paths if '044920599' in p.name or '052927799' in p.name]
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(str(path))
        prepared = prepare(image)
        output, debug, report = correct(prepared, outer_radius=args.outer_radius)
        dest = args.output / (path.stem + '_dehalo.png')
        if dest.exists():
            raise FileExistsError(str(dest))
        write_png(dest, output)
        decoded = cv2.imdecode(np.fromfile(str(dest), np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == image.shape and np.array_equal(decoded, output)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        dbg = args.output / 'debug' / path.stem
        for name, mask in debug.items():
            write_png(dbg / (name + '.png'), mask.astype(np.uint8) * 255)
        if args.sample:
            write_png(dbg / 'background.png', np.clip(np.rint(prepared[4]), 0, 255).astype(np.uint8))
        overview = image.copy()
        overview[~debug['roi']] //= 3
        write_png(dbg / 'roi_overview.png', cv2.resize(overview, (1040, 780), interpolation=cv2.INTER_AREA))
        # Fixed full-resolution crops cover degrees, fine text, colored math, and title.
        crops = {'degrees': (1750, 1260, 2950, 1480), 'small': (700, 1530, 2400, 1690),
                 'title': (600, 560, 2200, 760), 'color': (600, 1690, 2700, 1890),
                 'practice': (950, 740, 2150, 990), 'function': (1300, 510, 2530, 750)}
        variants = [('before', image), ('after', output)]
        if args.sample:
            low, _, low_report = correct(prepared, .5, 0)
            p1, _, p1_report = correct(prepared, 1, 0)
            variants = [('before', image), ('low', low), ('bright_only', p1), ('after', output)]
            report['ab'] = [low_report, p1_report]
            if args.outer_radius != 12:
                previous, _, _ = correct(prepared)
                protected = prepared[6] <= 10
                assert np.array_equal(previous[protected], output[protected])
                report['inner_10px_equal_previous'] = True
                variants = [('original', image), ('previous_12px', previous), ('outward_24px', output)]
        for label, (x0, y0, x1, y1) in crops.items():
            panels = []
            for name, variant in variants:
                crop = variant[y0:y1, x0:x1].copy()
                panel = cv2.copyMakeBorder(crop, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                cv2.putText(panel, name, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                panels.append(panel)
            write_png(args.output / 'comparisons' / (path.stem + '_' + label + '.png'), np.concatenate(panels, axis=0))
        report.update(source=str(path), output=str(dest), sha256=digest, shape=list(image.shape), crops=crops)
        reports.append(report)
        (args.output / 'receipt.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k not in ('ab', 'crops')}, ensure_ascii=True), flush=True)
        del prepared, variants, image, output
    print(f'Completed {len(reports)} images', flush=True)


if __name__ == '__main__':
    main()
