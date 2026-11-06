"""Replay cached V10 candidates with source-aware overlap compositing."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_cached_text_line_verified_normal_scurve_v10 as v10
base = v10.base


def alpha_for(valid, raw, radius=16):
    """@luna-comment Pad support before distance measurement; protect foreground plus six pixels."""
    d = cv2.distanceTransform(np.pad(valid.astype('uint8'), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    a = np.clip((d-1)/radius, 0, 1)
    a = .5-.5*np.cos(np.pi*a)
    core = cv2.dilate(raw.astype('uint8'), np.ones((13,13), 'uint8')).astype(bool) & valid
    a[core] = 1
    return a * valid


def compose(source, numerator, denominator, coverage):
    """@luna-comment Normalize candidate overlaps, then retain source feather independently."""
    candidate = numerator / np.maximum(denominator[None], 1e-12)
    return source*(1-coverage[None])+candidate*coverage[None]


def run(cache_root, output_dir, image_id):
    """@luna-comment Reuse predictions and unchanged V10; save baseline, fixed frame, and final crops."""
    cache_root, output_dir = Path(cache_root), Path(output_dir)
    manifest = base.verify_cache(cache_root)
    frame = next(f for f in manifest['frames'] if f['image_id'] == image_id)
    source = np.load(cache_root/frame['directory']/'frame_yuv444.npy')
    output_dir.mkdir(parents=True, exist_ok=False)
    numerator = np.zeros_like(source)
    denominator = np.zeros(source.shape[1:], np.float32)
    coverage = np.zeros_like(denominator)
    counts = np.zeros(source.shape[1:], np.uint16)
    baseline = source.copy()
    taken = np.zeros(source.shape[1:], bool)
    records = []
    for line in sorted(frame['lines'], key=lambda x:x['line_index']):
        ld = cache_root/line['directory']
        p = np.load(ld/'prediction_yuv444.npy')
        valid = np.load(ld/'valid_support.npy').astype(bool)
        l,t,r,b = json.loads((ld/'geometry.json').read_text())['crop_xyxy']
        q,e,report = v10.fill_v10(p,source[:,t:b,l:r],valid)
        a = alpha_for(valid,e['raw'])
        numerator[:,t:b,l:r] += q*a[None]
        denominator[t:b,l:r] += a
        coverage[t:b,l:r] = np.maximum(coverage[t:b,l:r],a)
        counts[t:b,l:r] += valid.astype(np.uint16)
        write = valid & ~taken[t:b,l:r]
        baseline[:,t:b,l:r][:,write] = q[:,write]
        taken[t:b,l:r] |= valid
        dd = output_dir/f"line_{line['line_index']:04d}"
        base._rgb_png(dd/'reconstructed_line.png',q)
        base._png(dd/'paste_alpha.png',np.rint(a*255).astype('uint8'))
        records.append(dict(line_index=line['line_index'],crop=[l,t,r,b],report=report))
    result = compose(source,numerator,denominator,coverage)
    base._rgb_png(output_dir/'before.png',baseline)
    base._rgb_png(output_dir/'after.png',result)
    base._png(output_dir/'overlap.png',np.minimum(counts*80,255).astype('uint8'))
    for rec in records:
        l,t,r,b = rec['crop']
        dd = output_dir/f"line_{rec['line_index']:04d}"
        base._rgb_png(dd/'final_pasted_crop.png',result[:,t:b,l:r])
        base._rgb_png(dd/'before_pasted_crop.png',baseline[:,t:b,l:r])
    report = dict(image_id=image_id,lines=records,uncovered_exact=bool(np.array_equal(result[:,coverage==0],source[:,coverage==0])),overlap_pixels=int((counts>1).sum()))
    (output_dir/'receipt.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--image-id',required=True)
    run(**vars(p.parse_args()))
