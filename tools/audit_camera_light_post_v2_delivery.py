"""Independently decode and audit all three V5 camera postprocess columns."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_camera_light_post_v2 import KEYS, OUT, digest, mapped_path


def audit():
    """@luna-comment Input: published 280-row 16-choice manifest and Y image/receipt files. Output: independent Y delivery audit. Key Steps: hash every output, decode dimensions, compare common integer RGB delta/caps and source binding, summarize low-vs-standard strength. Key Params: 280x3 native PNGs. State/Dependencies: read-only old/new finals except approved audit JSON."""
    manifest = json.loads((ROOT/'output/camera_2x2_comparison_20261022/manifest.json').read_text(encoding='utf-8'))
    if len(manifest['rows']) != 280 or len(manifest['models']) != 16:
        raise AssertionError('Expected final 280x16 manifest')
    records = []
    for row in manifest['rows']:
        source_path = mapped_path(row['images']['vosr_v5_block'])
        source_hash = digest(source_path)
        with Image.open(source_path) as image:
            source = np.asarray(image.convert('RGB'))
        if source.shape[:2] != (row['height'], row['width']):
            raise AssertionError('Source dimensions '+row['id'])
        strength = {}
        for key in KEYS:
            path = mapped_path(row['images'][key])
            receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if (path != OUT/key/(row['id']+'.png') or digest(path) != receipt['final_sha256']
                    or source_hash != receipt['source_sha256']):
                raise AssertionError('File provenance '+row['id']+' '+key)
            with Image.open(path) as image:
                actual = np.asarray(image.convert('RGB'))
            if actual.shape != source.shape:
                raise AssertionError('Output dimensions '+row['id']+' '+key)
            delta = actual.astype(np.int16)-source.astype(np.int16)
            if not np.array_equal(delta[:, :, 0], delta[:, :, 1]) or not np.array_equal(delta[:, :, 1], delta[:, :, 2]):
                raise AssertionError('Common-channel delta '+row['id']+' '+key)
            scalar = np.abs(delta[:, :, 0])
            cap = {'light_v1_standard': 24, 'light_v2_low': 12, 'light_v2_standard': 20}[key]
            if scalar.max() > cap or int(np.count_nonzero(scalar)) != receipt['changed_pixels']:
                raise AssertionError('Delta bound/count '+row['id']+' '+key)
            counts = np.bincount(scalar.ravel(), minlength=cap+1)
            p95 = int(np.searchsorted(np.cumsum(counts), .95*scalar.size))
            strength[key] = {'changed_pixels': receipt['changed_pixels'], 'p95_abs_delta': p95, 'mean_abs_delta': float(scalar.mean()), 'max_abs_delta': int(scalar.max())}
            del actual, delta, scalar
        records.append({'id': row['id'], 'baseline_sha256': source_hash, 'variants': strength,
                        'low_mean_le_standard': strength['light_v2_low']['mean_abs_delta'] <= strength['light_v2_standard']['mean_abs_delta']})
        print('audited '+row['id'], flush=True)
    target = OUT/'evaluation/DELIVERY_AUDIT.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'rows': len(records), 'variants': len(KEYS),
                                  'low_mean_le_standard_count': sum(v['low_mean_le_standard'] for v in records),
                                  'records': records}, indent=2), encoding='utf-8')
    return target


if __name__ == '__main__':
    print(audit())
