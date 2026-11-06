"""Isolated native stroke-thinning experiment using existing S2.2 GT and masks."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time

import cv2
import numpy as np

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE = DATA / 'mojito280_reused_v3_vosr_s22_20261029'
MASKS = DATA / 'mojito280_guard_finetune_20261030/source_view/images'
OUT = DATA / 'mojito280_s22_thin_20261104'


def digest(path):
    """@luna-comment Input:file path. Output:SHA256. Key Steps:stream bytes. Key Params:1MiB chunks. State/Dependencies:read-only filesystem."""
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    """@luna-comment Input:path and JSON value. Output:atomic receipt. Key Steps:temporary write/replace. Key Params:UTF8. State/Dependencies:owned destination only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def thin(image, mask, valid, radius, bright_override=None):
    """@luna-comment Input:uint8 BGR/native masks,radius0.5/1px. Output:thinned RGB and support. Key Steps:background polarity;8-direction subpixel gray extrema;copy winning RGB. Key Params:mask>.05 plus1px support,valid>=254. State/Dependencies:OpenCV,source immutable;fine strokes may vanish."""
    assert image.dtype == np.uint8 and image.shape[:2] == mask.shape == valid.shape
    assert radius in (0.5, 1.0)
    src = image.astype(np.float32)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    m = mask.astype(np.float32) / 255
    bgw = (1-m) * (valid >= 254)
    den = cv2.GaussianBlur(bgw, (0, 0), 8)
    bg = cv2.GaussianBlur(gray * bgw, (0, 0), 8) / np.maximum(den, 1e-6)
    bright = cv2.GaussianBlur((gray-bg)*m, (0, 0), 4) >= 0
    if bright_override is not None:
        bright = bright_override
    support = (cv2.dilate((m > .05).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (valid >= 254)
    best, result = gray.copy(), src.copy()
    for angle in np.arange(8)*math.pi/4:
        matrix = np.array([[1, 0, radius*math.cos(angle)], [0, 1, radius*math.sin(angle)]], np.float32)
        candidate = cv2.warpAffine(src, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        candidate_valid = cv2.warpAffine(valid, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        level = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        choose = support & (candidate_valid >= 254) & np.where(bright, level < best, level > best)
        best[choose] = level[choose]
        result[choose] = candidate[choose]
    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    assert np.array_equal(result[~support], image[~support])
    return result, support


def line_support(image, mask, valid, annotations):
    """@luna-comment Input:BGR,region/valid masks,cached native polygons. Output:derived stroke support,polarity map,audit. Key Steps:line outer-ring background and10/90 tails;assign contrast-ranked overlapping lines;derive foreground residual. Key Params:ring5px,mincontrast5,foreground residual2 or5percent. State/Dependencies:cached OCR only;regionmask is not a stroke mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    support = np.zeros(mask.shape, np.uint8)
    bright = np.zeros(mask.shape, bool)
    priority = np.zeros(mask.shape, np.float32)
    audit = []
    for annotation in annotations:
        if annotation.get('ignore', False) or not annotation.get('visible', True):
            continue
        points = np.asarray(annotation['points'], np.float32)
        x0,y0 = np.maximum(np.floor(points.min(0)).astype(int)-7, 0)
        x1,y1 = np.minimum(np.ceil(points.max(0)).astype(int)+8, [image.shape[1], image.shape[0]])
        if x1<=x0 or y1<=y0:
            continue
        area = np.zeros((y1-y0,x1-x0), np.uint8)
        cv2.fillPoly(area, [np.rint(points-[x0,y0]).astype(np.int32)], 1)
        g = gray[y0:y1,x0:x1]
        good = valid[y0:y1,x0:x1]>=254
        core = (area>0)&good
        if core.sum()<16:
            continue
        ring = (cv2.dilate(area,np.ones((11,11),np.uint8))>0)&(area==0)&good
        background = float(np.median(g[ring])) if ring.sum()>=16 else float(np.median(g[core]))
        low, high = np.percentile(g[core],[10,90])
        is_bright = high-background > background-low
        contrast = float(max(high-background,background-low))
        audit.append(dict(bright=bool(is_bright),background=background,contrast=contrast))
        if contrast<5:
            continue
        choose = core & (mask[y0:y1,x0:x1]>12) & (contrast>priority[y0:y1,x0:x1])
        residual = g-background if is_bright else background-g
        support[y0:y1,x0:x1][choose] = (residual[choose]>max(2,contrast*.05)).astype(np.uint8)*255
        bright[y0:y1,x0:x1][choose] = is_bright
        priority[y0:y1,x0:x1][choose] = contrast
    return support, bright, audit


def synthetic_check():
    """@luna-comment Input:none. Output:measured synthetic width receipt. Key Steps:dark/bright stripes,half-height interpolation. Key Params:10px stroke,0.5/1px radius. State/Dependencies:thin and NumPy;raises on sign/radius/support mismatch."""
    checks = []
    for white in (False, True):
        im = np.full((96, 96, 3), 0 if white else 240, np.uint8)
        im[:, 43:53] = 240 if white else 0
        mask = np.zeros((96, 96), np.uint8)
        mask[:, 43:53] = 255
        valid = np.full_like(mask, 255)
        for radius in (.5, 1.):
            result, support = thin(im, mask, valid, radius)
            profile = result[48, :, 0].astype(float)
            profile = profile/240 if white else 1-profile/240
            crossings = []
            for x in range(95):
                if (profile[x] < .5 <= profile[x+1]) or (profile[x] >= .5 > profile[x+1]):
                    crossings.append(x + (.5-profile[x])/(profile[x+1]-profile[x]))
            width = crossings[-1]-crossings[0]
            assert abs(width-(10-2*radius)) < .03, (white, radius, width)
            assert np.array_equal(result[~support], im[~support])
            checks.append(dict(white=white, radius=radius, width=width, expected=10-2*radius))
    return dict(passed=True, checks=checks)


def process(record):
    """@luna-comment Input:frozen source record. Output:two PNG/hash checks and row. Key Steps:verify original/oriented masks,thin independently,atomic receipt. Key Params:existing IDs,1/2px total reductions. State/Dependencies:owned output;no OCR/GPU/source edits."""
    cv2.setNumThreads(1)
    ident = record['id']
    parent = json.loads((MASKS/ident/'receipt.json').read_text())['row']
    paths = {k: Path(parent['files'][k]) for k in ('gt', 'text_mask', 'valid_mask', 'lq', 'ocr')}
    assert paths['gt'] == SOURCE/'results/steep_s'/f'{ident}.png'
    for k, path in paths.items():
        assert digest(path) == parent['output_sha256'][k], (ident, k)
    assert digest(paths['gt']) == record['arms']['steep_s']['sha256']
    assert record['arms']['steep_s']['midpoint_derivative'] == 2.2
    receipt = OUT/'receipts'/f'{ident}.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        assert previous['source_sha256'] == parent['output_sha256']
        if all(digest(OUT/v['path']) == v['sha256'] for v in previous['outputs'].values()):
            return previous
    image = cv2.imread(str(paths['gt']))
    mask = cv2.imread(str(paths['text_mask']), 0)
    valid = cv2.imread(str(paths['valid_mask']), 0)
    assert list(image.shape[:2]) == record['shape_hw']
    ocr = json.loads(paths['ocr'].read_text())
    stroke_mask, bright, polarity_audit = line_support(image, mask, valid, ocr['ocr']['annotations'])
    outputs = {}
    for label, radius in [('minus1', .5), ('minus2', 1.)]:
        result, support = thin(image, stroke_mask, valid, radius, bright)
        delta = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_BGR2GRAY)-cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        wrong_direction = int((support & np.where(bright, delta>1, delta < -1)).sum())
        assert wrong_direction == 0, (ident, label, wrong_direction)
        destination = OUT/label/f'{ident}.png'
        destination.parent.mkdir(exist_ok=True, parents=True)
        temporary = destination.with_name(ident+'.tmp.png')
        assert cv2.imwrite(str(temporary), result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        os.replace(temporary, destination)
        outputs[label] = dict(path=str(destination.relative_to(OUT)), sha256=digest(destination), radius=radius,
            changed_pixels=int(np.any(result != image, axis=2).sum()), outside_support_unchanged=True, wrong_direction_pixels=wrong_direction)
    result = dict(id=ident, name=parent['sample_id'], width=image.shape[1], height=image.shape[0],
                  source={k:str(v) for k,v in paths.items()}, source_sha256=parent['output_sha256'], outputs=outputs, polarity=polarity_audit)
    save(receipt, result)
    return result


def main():
    """@luna-comment Input:CLI workers/optional selected IDs. Output:280 pairs plus status/results. Key Steps:exclusive lock,synthetic gate,hash-bound sources,process pool,completion receipt. Key Params:4CPU workers default. State/Dependencies:Remote58 data;resume only owned receipts."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(OUT/'SYNTHETIC_CHECK.json', synthetic_check())
    records = json.loads((SOURCE/'EXPORT_MANIFEST.json').read_text())['records']
    assert len(records) == 280 and len({r['id'] for r in records}) == 280
    if args.ids:
        records = [r for r in records if r['id'] in args.ids]
        assert len(records) == len(args.ids)
    records.sort(key=lambda r:r['id'])
    save(OUT/'BINDING.json', dict(source=str(SOURCE), manifest_sha256=digest(SOURCE/'EXPORT_MANIFEST.json'),
         code_sha256=digest(__file__), masks=str(MASKS), count=len(records)))
    rows, start = [], time.time()
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process, record) for record in records]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            elapsed = time.time()-start
            state = dict(completed=len(rows), total=len(records), seconds=elapsed,
                         remaining_seconds=elapsed/len(rows)*(len(records)-len(rows)))
            save(OUT/'STATUS.json', state)
            print(json.dumps(state), flush=True)
    save(OUT/'RESULTS.json', dict(passed=True, rows=sorted(rows, key=lambda r:r['id']), seconds=time.time()-start))
    if len(rows) == 280:
        save(OUT/'COMPLETE.json', dict(passed=True, count=280, outputs=560, seconds=time.time()-start))


if __name__ == '__main__':
    main()
