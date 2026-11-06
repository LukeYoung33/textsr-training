"""Build annotation-union sidecars with the bank's exact transform and crop metadata."""
import json
from pathlib import Path

import cv2
import numpy as np


def polygons(document):
    """@luna-comment Input:published OCR JSON. Output:finite source polygons/boxes.
    Key Steps:unwrap existing annotation schemas; retain partial/rotated/vertical lines
    independently of CP eligibility. Key Params:no detector or stroke thresholds.
    State/Dependencies:NumPy; malformed geometry fails, explicit empty lists permitted.
    """
    nested = document.get('ocr', document.get('res', document))
    if 'annotations' in nested or 'annotations' in document:
        entries = nested.get('annotations', document.get('annotations', []))
    elif 'rec_polys' in nested or 'dt_polys' in nested:
        entries = [{'polygon': p} for p in nested.get('rec_polys', nested.get('dt_polys', []))]
    elif 'lines' in document:
        entries = [{'bbox': e['bbox_xyxy']} for e in document['lines'] if 'bbox_xyxy' in e]
    else:
        raise ValueError('Missing annotation list schema')
    result = []
    for item in entries:
        poly = next((item[k] for k in ('polygon', 'poly', 'points') if k in item), None)
        if poly is None:
            box = item.get('bbox')
            if box is None or len(box) != 4:
                raise ValueError('Missing finite line geometry')
            x0, y0, x1, y1 = box
            poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        poly = np.asarray(poly, np.float32).reshape(-1, 2)
        if len(poly) < 3 or not np.isfinite(poly).all():
            raise ValueError('Invalid polygon')
        result.append(poly)
    return result


def build(job):
    """@luna-comment Input:bank shard/source row. Output:bit-packed region masks/coverage receipt.
    Key Steps:raster union before identical recorded homography; crop with actual xy,
    intersect valid support, pack bits. Key Params:no CP filters/padding supervision.
    State/Dependencies:immutable eight-channel shard/annotations; owned sidecars only.
    """
    from tools.prepare_v3pre_training import sha, save
    bank, shard, row = job
    bank = Path(bank)
    root = bank / 'line_regions'
    target = root / (shard['array'] + '.bits.npy')
    receipt = target.with_suffix('.receipt.json')
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        assert previous['source_shard_sha256'] == shard['array_sha256'] and sha(target) == previous['sha256']
        return previous
    assert sha(row['files']['ocr']) == row['output_sha256']['ocr']
    records = json.loads((bank / shard['metadata']).read_text())
    array = np.load(bank / shard['array'], mmap_mode='r')
    h, w = cv2.imread(row['files']['valid_mask'], 0).shape
    original = np.zeros((h, w), np.uint8)
    for poly in polygons(json.loads(Path(row['files']['ocr']).read_text())):
        cv2.fillPoly(original, [np.rint(poly).astype(np.int32)], 1)
    matrices = {}
    masks, coverage = [], []
    for index, record in enumerate(records):
        key = json.dumps(record['matrix'])
        if key not in matrices:
            matrix = np.asarray(record['matrix'], np.float64)
            matrices[key] = original if np.array_equal(matrix, np.eye(3)) else cv2.warpPerspective(original, matrix, (w, h), flags=cv2.INTER_NEAREST)
        x, y = record['xy']
        hh, ww = record['real_hw']
        valid = array[index, ..., 7] > 127
        mask = np.zeros((256, 256), np.uint8)
        mask[:hh, :ww] = matrices[key][y:y+hh, x:x+ww]
        mask &= valid
        masks.append(np.packbits(mask.reshape(-1)))
        coverage.append(float(mask.sum() / max(valid.sum(), 1)))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.save(stream, np.asarray(masks, np.uint8).reshape(-1, 8192), allow_pickle=False)
    temp.replace(target)
    r = dict(passed=True, count=len(records), path=str(target.relative_to(bank)), sha256=sha(target),
             source_ocr_sha256=row['output_sha256']['ocr'], source_shard_sha256=shard['array_sha256'],
             source_metadata_sha256=shard['metadata_sha256'], coverage=coverage)
    save(receipt, r)
    return r
