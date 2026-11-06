"""Deterministic full-context frozen preprocessing and native paired shard bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.adaptive_prepost_frozen_v1_20261020.preprocess import prepare_context, apply_preprocess

PROFILE = 'adaptive_preprocessed_augmented_v1_20261020'
CATEGORIES = ['photometry'] * 3 + ['rotation'] * 2 + ['geometry'] * 2 + ['neighbors'] * 2 + ['direction']
HW = (96, 480)
PREPROCESS_MODE = os.environ.get('OFFLINE_PREPROCESS_MODE', 'lq')


def digest(path):
    """@luna-comment
    Input: readable path. Output: SHA256 hex; no mutation.
    Key Steps: stream bounded chunks. Key Params: 4MiB. State/Dependencies: hashlib/filesystem.
    """
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    """@luna-comment
    Input: destination and JSON value. Output: atomically replaced UTF8 JSON.
    Key Steps: write temporary sibling then replace. Key Params: stable keys.
    State/Dependencies: destination filesystem; only build-owned files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def stable_seed(value):
    """@luna-comment
    Input: identity string. Output: deterministic uint32 seed.
    Key Steps: hash UTF8 bytes. Key Params: first four digest bytes.
    State/Dependencies: hashlib; independent of Python hash randomization.
    """
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'little')


def read_annotations(document):
    """@luna-comment
    Input: published OCR document. Output: ordered usable annotation dictionaries.
    Key Steps: unwrap OCR; retain finite boxes and visible nonignored entries.
    Key Params: original order controls first ownership. State/Dependencies: numpy.
    """
    nested = document.get('ocr', document.get('res', document))
    entries = nested.get('annotations', document.get('annotations', []))
    if not entries:
        entries = [{'bbox': line['bbox_xyxy']} for line in document.get('lines', []) if 'bbox_xyxy' in line]
    if not entries:
        polys = nested.get('rec_polys', nested.get('dt_polys', []))
        entries = [{'bbox': [*np.min(p, axis=0), *np.max(p, axis=0)]} for p in polys]
    return [a for a in entries if not a.get('ignore') and a.get('visible') is not False
            and len(a.get('bbox', [])) == 4 and np.isfinite(a['bbox']).all()
            and a['bbox'][2] > a['bbox'][0] and a['bbox'][3] > a['bbox'][1]]


def clipped_boxes(annotations, shape):
    """@luna-comment
    Input: ordered source annotations and image HW. Output: clipped integer half-open boxes.
    Key Steps: floor/ceil and reject empty support. Key Params: no margin enlargement.
    State/Dependencies: immutable original annotations.
    """
    height, width = shape[:2]
    result = []
    for a in annotations:
        x0, y0, x1, y1 = a['bbox']
        box = [max(0, math.floor(x0)), max(0, math.floor(y0)), min(width, math.ceil(x1)), min(height, math.ceil(y1))]
        if box[2] > box[0] and box[3] > box[1]:
            result.append(box)
    return result


def axis_origins(start, end, extent, full):
    """@luna-comment
    Input: native interval, patch extent and full dimension. Output: even covering origins.
    Key Steps: center short spans; cover long ones; clamp to image.
    Key Params: even phase. State/Dependencies: deterministic arithmetic.
    """
    first, last = math.floor(start / 2) * 2, math.ceil((end - extent) / 2) * 2
    points = [round((start + end - extent) / 4) * 2] if last <= first else list(range(first, last, extent)) + [last]
    maximum = max(0, (full - extent) // 2 * 2)
    return sorted({max(0, min(p, maximum)) for p in points})


def source_windows(boxes, shape, text, valid):
    """@luna-comment
    Input: OCR boxes, source shape and authoritative masks. Output: unique native even windows.
    Key Steps: cover context-expanded boxes; retain trusted text; never resize.
    Key Params: historical .03/.12 margins. State/Dependencies: source masks.
    """
    height, width = shape[:2]
    support = (text > 127) & (valid > 127)
    points = set()
    for x0, y0, x1, y1 in boxes:
        dx, dy = max(2, round((x1 - x0) * .03)), max(2, round((y1 - y0) * .12))
        for y in axis_origins(max(0, y0 - dy), min(height, y1 + dy), 96, height):
            for x in axis_origins(max(0, x0 - dx), min(width, x1 + dx), 480, width):
                if support[y:y + 96, x:x + 480].any():
                    points.add((x, y))
    return sorted(points, key=lambda p: (p[1], p[0]))


def transform_boxes(boxes, matrix, shape):
    """@luna-comment
    Input: source boxes, source-to-target homography and target shape. Output: ordered target boxes.
    Key Steps: transform all corners, bound and clip. Key Params: integer half-open API.
    State/Dependencies: OpenCV; transformed boxes retain original order.
    """
    annotations = []
    for x0, y0, x1, y1 in boxes:
        corners = np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
        out = cv2.perspectiveTransform(corners, matrix)[0]
        annotations.append({'bbox': [*out.min(0), *out.max(0)]})
    return clipped_boxes(annotations, shape)


def row_information(boxes, xy, shape=HW):
    """@luna-comment
    Input: transformed boxes and final rectangle. Output: visible row count and CP eligibility.
    Key Steps: intersect boxes, merge coincident row bands, require a fully contained row for CP.
    Key Params: 50% vertical overlap groups row bands. State/Dependencies: OCR proxy, not character inference.
    """
    x, y = xy
    h, w = shape
    visible = [b for b in boxes if b[0] < x + w and b[2] > x and b[1] < y + h and b[3] > y]
    groups = []
    for b in sorted(visible, key=lambda v: (v[1] + v[3]) / 2):
        low, high = max(y, b[1]), min(y + h, b[3])
        if groups and min(groups[-1][1], high) - max(groups[-1][0], low) > .5 * min(high - low, groups[-1][1] - groups[-1][0]):
            groups[-1] = (min(low, groups[-1][0]), max(high, groups[-1][1]))
        else:
            groups.append((low, high))
    eligible = len(groups) == 1 and any(b[1] >= y and b[3] <= y + h for b in visible)
    return len(groups), eligible


def crop_sample(arrays, xy, boxes, category, recipe, matrix, full_support=True):
    """@luna-comment
    Input: paired BGR/masks/support, even rectangle, boxes and recipe provenance.
    Output: uint8 HWC8 sample plus metadata, or rejection reason.
    Key Steps: reject unsupported geometry/too many rows/no trusted text; copy masks and pixels.
    Key Params: at most three visible OCR row bands; 480x96. State/Dependencies: immutable arrays.
    """
    x, y = map(int, xy)
    h, w = HW
    height, width = arrays['lq'].shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return None, 'outside_source'
    if full_support and arrays['support'][y:y + h, x:x + w].min() < 255:
        return None, 'geometric_support'
    valid = arrays['valid_mask'][y:y + h, x:x + w]
    text = arrays['text_mask'][y:y + h, x:x + w]
    if not ((valid > 127) & (text > 127)).any():
        return None, 'no_trusted_text'
    rows, cp = row_information(boxes, (x, y))
    if rows > 3:
        return None, 'more_than_three_rows'
    channels = [arrays[k][y:y + h, x:x + w] for k in ('lq', 'gt')]
    channels += [np.where((text > 127) & (valid > 127), 255, 0).astype('uint8')[..., None], valid[..., None]]
    metadata = dict(category=category, recipe=recipe, xy=[x, y], visible_ocr_rows=rows,
                    cp_eligible=bool(cp and category not in ('direction', 'neighbors')),
                    orientation='OBLIQUE' if category in ('rotation', 'geometry') else 'HORIZONTAL',
                    matrix=matrix.tolist(), geometric_support=1.)
    return (np.concatenate(channels, axis=2), metadata), None


def photometry(image, recipe):
    """@luna-comment
    Input: native BGR uint8 and deterministic lighting recipe. Output: transformed BGR uint8.
    Key Steps: exposure in odds domain, contrast/gamma or smooth spatial shadow; bound range.
    Key Params: EV [-1.2,1.2], contrast [.65,1.8], gamma [.6,1.6], shadow [.45,.85].
    State/Dependencies: numpy; identical field/parameters on both paired images.
    """
    value = image.astype(np.float32) / 255
    kind = recipe['kind']
    if kind == 'exposure':
        gain = 2 ** recipe['ev']
        value = value * gain / (1 + (gain - 1) * value)
    elif kind == 'contrast':
        value = .5 + (value - .5) * recipe['contrast']
    elif kind == 'gamma':
        value = value ** recipe['gamma']
    else:
        h, w = value.shape[:2]
        xx, yy = np.meshgrid(np.linspace(0, 1, w, dtype=np.float32), np.linspace(0, 1, h, dtype=np.float32))
        axis = xx * np.cos(recipe['angle']) + yy * np.sin(recipe['angle'])
        field = 1 - (1 - recipe['minimum']) * np.exp(-.5 * ((axis - recipe['center']) / recipe['width']) ** 2)
        value *= field[..., None]
    return np.clip(np.rint(value * 255), 0, 255).astype(np.uint8)


def recipe_for(identity, slot, shape):
    """@luna-comment
    Input: source identity, variant slot and shape. Output: primary category, recipe and homography.
    Key Steps: deterministic category cycle and parameter strata; compose native transforms.
    Key Params: three augmentation slots per parent; all continuous intervals covered across parents.
    State/Dependencies: seeded numpy RNG; no global RNG mutation.
    """
    seed = stable_seed(identity + f':{slot}')
    rng = np.random.default_rng(seed)
    category = CATEGORIES[(stable_seed(identity) + slot * 3) % len(CATEGORIES)]
    u = (seed % 1000 + rng.random()) / 1000
    h, w = shape[:2]
    matrix = np.eye(3, dtype=np.float64)
    recipe = dict(seed=seed, stratum=seed % 1000)
    if category == 'photometry':
        kind = ('exposure', 'contrast', 'gamma', 'shadow')[seed % 4]
        recipe.update(kind=kind, ev=-1.2 + 2.4 * u, contrast=.65 + 1.15 * u,
                      gamma=.6 + u, minimum=.45 + .4 * u, width=.15 + .30 * rng.random(),
                      center=float(rng.uniform(-.2, 1.2)), angle=float(rng.uniform(-math.pi, math.pi)))
    elif category == 'rotation':
        angle = (-1 if seed % 2 else 1) * (2 + 13 * u)
        matrix[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        recipe.update(angle=angle)
    elif category == 'geometry':
        if seed % 2:
            shear = -.18 + .36 * u
            sy = float(rng.uniform(-.06, .06))
            matrix = np.array([[1, shear, -shear * h / 2], [sy, 1, -sy * w / 2], [0, 0, 1]], np.float64)
            recipe.update(kind='affine', shear=shear, shear_y=sy)
        else:
            amount = .01 + .04 * u
            src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            dst = src + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.float32([w, h]) * amount
            matrix = cv2.getPerspectiveTransform(src, dst)
            recipe.update(kind='perspective', normalized_jitter=amount)
    elif category == 'direction':
        matrix = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64)
        recipe.update(kind='upside_down', angle=180)
    else:
        recipe.update(kind=('upper', 'lower', 'both')[seed % 3])
    return category, recipe, matrix


def neighbor_windows(boxes, shape, kind):
    """@luna-comment
    Input: source boxes, image shape and neighbor side. Output: centered candidate rectangles.
    Key Steps: keep short principal rows central; vary horizontal origins; require actual partial neighbors.
    Key Params: main row center error <=1px, <=3 total OCR bands. State/Dependencies: authoritative OCR.
    """
    h, w = shape[:2]
    candidates = []
    for main in boxes:
        if main[3] - main[1] > 90:
            continue
        y = int(round(((main[1] + main[3]) / 2 - 48) / 2)) * 2
        if y < 0 or y + 96 > h:
            continue
        for x in axis_origins(main[0], main[2], 480, w):
            visible = [b for b in boxes if b[0] < x + 480 and b[2] > x and b[1] < y + 96 and b[3] > y]
            upper = any(b[1] < y < b[3] and b[3] < (main[1] + main[3]) / 2 for b in visible)
            lower = any(b[1] < y + 96 < b[3] and b[1] > (main[1] + main[3]) / 2 for b in visible)
            matched = {'upper': upper and not lower, 'lower': lower and not upper, 'both': upper and lower}[kind]
            if matched and row_information(boxes, (x, y))[0] <= 3:
                candidates.append((x, y))
    return sorted(set(candidates))


def prepare_variant(original, boxes, category, recipe, matrix, reference):
    """@luna-comment
    Input: full-source pair/masks and one common recipe. Output: augmented pair/masks.
    Key Steps: common photometry/geometry, conservative support, selected frozen preprocessing.
    Key Params: mode lq/both/none; gamma1.8; each selected image gets its own context.
    State/Dependencies: frozen package and process-local PREPROCESS_MODE; inputs unchanged.
    """
    arrays = {k: v.copy() for k, v in original.items()}
    if category == 'photometry':
        for key in ('lq', 'gt'):
            arrays[key] = photometry(arrays[key], recipe)
    height, width = arrays['lq'].shape[:2]
    if not np.array_equal(matrix, np.eye(3)):
        for key, value in arrays.items():
            if key == 'support':
                support = cv2.warpPerspective(value.astype(np.float32) / 255, matrix, (width, height),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                arrays[key] = np.where(support >= 1., 255, 0).astype(np.uint8)
                continue
            interpolation = cv2.INTER_NEAREST if key.endswith('mask') else cv2.INTER_LINEAR
            arrays[key] = cv2.warpPerspective(value, matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
        boxes = transform_boxes(boxes, matrix, arrays['lq'].shape)
    if PREPROCESS_MODE not in ('lq', 'both', 'none'):
        raise ValueError('Unknown offline preprocessing mode')
    keys = ('lq', 'gt') if PREPROCESS_MODE == 'both' else (() if PREPROCESS_MODE == 'none' else ('lq',))
    for key in keys:
        context = prepare_context(arrays[key], boxes, reference_id=reference + ':' + key)
        arrays[key] = apply_preprocess(arrays[key], context)
    return arrays, boxes


def vertical_samples(original, annotations, reference):
    """@luna-comment
    Input: original pair/masks and glyph boxes. Output: tight synthetic upright-column +/-90 samples.
    Key Steps: select small trusted glyphs; estimate valid nontext ring backgrounds, compose one shared tight column,
      preprocess in full-size frame context; rotate the complete column into native horizontal patches.
    Key Params: 2px guard, glyph width<=86, height<=86; no glyph resizing or fake semantic mask.
    State/Dependencies: authoritative glyph boxes and source masks; synthetic-layout label mandatory.
    """
    height, width = original['gt'].shape[:2]
    chosen = []
    for annotation in annotations:
        if annotation.get('font_runs'):
            runs = annotation['font_runs']
            for run in runs:
                if not str(run.get('char', '')).strip() or len(run.get('bbox', [])) != 4:
                    continue
                x0, y0, x1, y1 = run['bbox']
                box = [max(0, math.floor(x0) - 2), max(0, math.floor(y0) - 2), min(width, math.ceil(x1) + 2), min(height, math.ceil(y1) + 2)]
                x0, y0, x1, y1 = box
                if not (8 <= x1 - x0 <= 86 and 8 <= y1 - y0 <= 86):
                    continue
                valid = original['valid_mask'][y0:y1, x0:x1]
                text = original['text_mask'][y0:y1, x0:x1]
                if valid.min() < 255 or (text > 127).sum() < 8:
                    continue
                chosen.append(box)
                if len(chosen) == 5:
                    break
        if len(chosen) == 5:
            break
    if len(chosen) < 3:
        return []
    while sum(b[3] - b[1] + 1 for b in chosen) + 4 > 470:
        chosen.pop()
    glyph_w = max(b[2] - b[0] for b in chosen)
    col_h = sum(b[3] - b[1] + 1 for b in chosen) + 4
    col_w = glyph_w + 4
    xbase = max(0, (width // 2 - col_w // 2) // 2 * 2)
    ybase = max(0, (height // 2 - col_h // 2) // 2 * 2)
    if xbase + col_w > width or ybase + col_h > height:
        return []
    arrays = {}
    backgrounds = {}
    background_support = (cv2.dilate((original['text_mask'] > 127).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0) & (original['valid_mask'] == 255)
    local_backgrounds = {key: [] for key in ('lq', 'gt')}
    for x0, y0, x1, y1 in chosen:
        xa, ya, xb, yb = max(0, x0 - 48), max(0, y0 - 48), min(width, x1 + 48), min(height, y1 + 48)
        support = background_support[ya:yb, xa:xb]
        if support.sum() < 32:
            return []
        for key in ('lq', 'gt'):
            local_backgrounds[key].append(np.median(original[key][ya:yb, xa:xb][support], axis=0).astype(np.float32))
    for key in ('lq', 'gt'):
        backgrounds[key] = np.median(local_backgrounds[key], axis=0).astype(np.float32)
        arrays[key] = np.broadcast_to(backgrounds[key].astype(np.uint8), (height, width, 3)).copy()
    arrays['text_mask'] = np.zeros((height, width), np.uint8)
    arrays['valid_mask'] = np.full((height, width), 255, np.uint8)
    arrays['support'] = np.full((height, width), 255, np.uint8)
    cursor = ybase + 2
    for glyph_index, (x0, y0, x1, y1) in enumerate(chosen):
        gh, gw = y1 - y0, x1 - x0
        left = xbase + 2 + (glyph_w - gw) // 2
        fade = np.ones((gh, gw), np.float32)
        fade[[0, -1], :] = 0
        fade[:, [0, -1]] = 0
        for key in ('lq', 'gt'):
            glyph = original[key][y0:y1, x0:x1].astype(np.float32)
            bg = local_backgrounds[key][glyph_index]
            alpha = fade
            if key == 'lq':
                yy, xx = np.indices((gh, gw), dtype=np.float32)
                edge_distance = np.minimum(np.minimum(xx, gw - 1 - xx), np.minimum(yy, gh - 1 - yy))
                ramp = np.clip(edge_distance / min(6., gw / 4, gh / 4), 0, 1)
                alpha = ramp * ramp * (3 - 2 * ramp)
            arrays[key][cursor:cursor + gh, left:left + gw] = np.clip(np.rint(backgrounds[key] + (glyph - bg) * alpha[..., None]), 0, 255).astype(np.uint8)
        arrays['text_mask'][cursor:cursor + gh, left:left + gw] = original['text_mask'][y0:y1, x0:x1]
        cursor += gh + 1
    boxes = [[xbase, ybase, xbase + col_w, ybase + col_h]]
    arrays, boxes = prepare_variant(arrays, boxes, 'vertical', {}, np.eye(3), reference)
    result = []
    for turns in (1, 3):
        rotated = {k: np.ascontiguousarray(np.rot90(v, turns)) for k, v in arrays.items()}
        rotated['lq'] = rotated['lq']
        cx = ybase + col_h / 2 if turns == 1 else height - 1 - (ybase + col_h / 2)
        cy = width - 1 - (xbase + col_w / 2) if turns == 1 else xbase + col_w / 2
        xy = (round((cx - 240) / 2) * 2, round((cy - 48) / 2) * 2)
        rect = [[xy[0] + 1, xy[1] + 1, xy[0] + 479, xy[1] + 95]]
        recipe = dict(kind='synthetic_tight_vertical_rotated', angle=turns * 90, source_glyph_boxes=chosen,
                      synthetic_background=True, background_estimator='valid_nontext_ring48_exclusion12',
                      lq_edge_feather=6, glyph_resizing=False)
        item, rejection = crop_sample(rotated, xy, rect, 'direction', recipe, np.eye(3))
        if item:
            item[1]['orientation'] = 'VERTICAL'
            result.append(item)
    return result


def process_parent(job):
    """@luna-comment
    Input: one build-owned parent job. Output: atomic shard/metadata receipt or explicit rejection.
    Key Steps: verify source, prepare base/variants, crop actual samples, save mmap array and metadata.
    Key Params: three augmentation slots; max_base bounds pilots; receipt hashes paired masks explicitly.
    State/Dependencies: read-only sources, independent per-parent outputs; no model training.
    """
    cv2.setNumThreads(1)
    source, output, split, row, limit, build_id = job
    source, output = Path(source), Path(output)
    name = f"{split}/{row['dataset']}/{row['sample_id']}"
    receipt_path = output / 'shards' / (name + '.receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['build_id'] != build_id:
            raise ValueError('Existing output belongs to another build')
        for key in ('array', 'metadata'):
            if digest(output / receipt[key]) != receipt[key + '_sha256']:
                raise ValueError('Existing receipt hash mismatch')
        return receipt
    started = time.perf_counter()
    arrays, source_hashes = {}, {}
    for key in ('lq', 'gt', 'valid_mask', 'text_mask'):
        path = source / row['files'][key]
        source_hashes[key] = digest(path)
        if source_hashes[key] != row['output_sha256'][key]:
            raise ValueError(f"Source changed: {name}/{key}")
        arrays[key] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arrays[key] is None or arrays[key].dtype != np.uint8:
            raise ValueError('Expected published uint8 source')
    if len({a.shape[:2] for a in arrays.values()}) != 1:
        raise ValueError('Pair/mask dimensions differ')
    ocr_path = source / row['files']['ocr']
    source_hashes['ocr'] = digest(ocr_path)
    if source_hashes['ocr'] != row['output_sha256']['ocr']:
        raise ValueError('OCR source changed')
    annotations = read_annotations(json.loads(ocr_path.read_text(encoding='utf-8')))
    boxes = clipped_boxes(annotations, arrays['gt'].shape)
    extra = source / 'OCR_ppocrv6_medium' / split / (row['sample_id'] + '.json')
    if extra.exists():
        supplementary = read_annotations(json.loads(extra.read_text(encoding='utf-8')))
        for box in clipped_boxes(supplementary, arrays['gt'].shape):
            if not any(abs((box[1] + box[3]) / 2 - (b[1] + b[3]) / 2) < .25 * min(box[3] - box[1], b[3] - b[1])
                       and min(box[2], b[2]) > max(box[0], b[0]) for b in boxes):
                boxes.append(box)
        source_hashes['supplemental_ocr'] = digest(extra)
    arrays['support'] = np.full(arrays['gt'].shape[:2], 255, np.uint8)
    windows = source_windows(boxes, arrays['gt'].shape, arrays['text_mask'], arrays['valid_mask'])
    if limit:
        windows = windows[::max(1, len(windows) // limit)][:limit]
    samples, metadata, reasons = [], [], Counter()
    base, base_boxes = prepare_variant(arrays, boxes, 'baseline', {}, np.eye(3), name)
    for xy in windows:
        item, rejection = crop_sample(base, xy, base_boxes, 'baseline', {}, np.eye(3))
        if item:
            samples.append(item[0]); metadata.append(item[1])
        else:
            reasons['baseline:' + rejection] += 1
    if split == 'train':
        for slot in range(3):
            category, recipe, matrix = recipe_for(name, slot, arrays['gt'].shape)
            transformed, target_boxes = prepare_variant(arrays, boxes, category, recipe, matrix, name + f':{slot}')
            if category == 'neighbors':
                candidates = neighbor_windows(boxes, arrays['gt'].shape, recipe['kind'])
                if not candidates:
                    reasons['neighbors:no_matching_partial_rows'] += 1
                    continue
                candidates = candidates[::max(1, len(candidates) // max(1, len(windows)))][:len(windows)]
            else:
                centers = np.float32([[[x + 240, y + 48] for x, y in windows]])
                if not windows:
                    continue
                mapped = cv2.perspectiveTransform(centers, matrix)[0]
                candidates = sorted({(round((float(x) - 240) / 2) * 2, round((float(y) - 48) / 2) * 2) for x, y in mapped})
            for xy in candidates:
                item, rejection = crop_sample(transformed, xy, target_boxes, category, recipe, matrix)
                if item:
                    samples.append(item[0]); metadata.append(item[1])
                else:
                    reasons[category + ':' + rejection] += 1
        for item in vertical_samples(arrays, annotations, name + ':vertical'):
            samples.append(item[0]); metadata.append(item[1])
    relative_array, relative_meta = 'shards/' + name + '.npy', 'shards/' + name + '.json'
    path = output / relative_array
    path.parent.mkdir(parents=True, exist_ok=True)
    if samples:
        buffer = np.stack(samples)
        with path.with_suffix('.npy.tmp').open('wb') as stream:
            np.save(stream, buffer, allow_pickle=False)
        os.replace(path.with_suffix('.npy.tmp'), path)
    else:
        buffer = np.empty((0, *HW, 8), np.uint8)
        with path.with_suffix('.npy.tmp').open('wb') as stream:
            np.save(stream, buffer, allow_pickle=False)
        os.replace(path.with_suffix('.npy.tmp'), path)
    write_json(output / relative_meta, metadata)
    receipt = dict(build_id=build_id, sample_id=row['sample_id'], source_identity=row['source_identity'],
                   dataset=row['dataset'], split=split, array=relative_array, metadata=relative_meta,
                   array_sha256=digest(path), metadata_sha256=digest(output / relative_meta),
                   source_hashes=source_hashes, count=len(metadata),
                   mask_sha256=hashlib.sha256(np.ascontiguousarray(buffer[..., 6:]).tobytes()).hexdigest(),
                   categories=dict(Counter(x['category'] for x in metadata)), rejections=dict(reasons),
                   seconds=time.perf_counter() - started, bytes=path.stat().st_size)
    write_json(receipt_path, receipt)
    return receipt


def build(args):
    """@luna-comment
    Input: approved source/output roots and deterministic worker/pilot options.
    Output: complete manifested shard bank with compact sampling index and receipt.
    Key Steps: verify splits/frozen hashes, process resumable parents, publish only after all jobs finish.
    Key Params: mode binds build identity; pilot limits/workers bound resources;50% baseline default view.
    State/Dependencies: source masks/OCR, frozen package, process pool and build-owned output.
    """
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_rows = {}
    for split in ('train', 'val'):
        source_rows[split] = [json.loads(s) for s in (source / 'manifests' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if {x['source_identity'] for x in source_rows['train']} & {x['source_identity'] for x in source_rows['val']}:
        raise ValueError('Train/val capture identity overlap')
    frozen = ROOT / 'evaluation/adaptive_prepost_frozen_v1_20261020'
    files = json.loads((frozen / 'manifest.json').read_text())['files']
    for relative, expected in files.items():
        if digest(frozen / relative) != expected:
            raise ValueError('Frozen package integrity failure: ' + relative)
    signature = dict(profile=PROFILE, builder=digest(__file__), frozen_manifest=digest(frozen / 'manifest.json'),
                     source_manifests={s: digest(source / 'manifests' / f'{s}.jsonl') for s in source_rows},
                     pilot_limit=args.limit, max_base=args.max_base, variants=3, preprocess_mode=PREPROCESS_MODE)
    build_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    write_json(output / 'build_signature.json', dict(signature, build_id=build_id))
    jobs = []
    for split, rows in source_rows.items():
        counts = Counter()
        for row in rows:
            if args.limit and counts[row['dataset']] >= args.limit:
                continue
            counts[row['dataset']] += 1
            jobs.append((str(source), str(output), split, row, args.max_base, build_id))
    started = time.perf_counter()
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_parent, job): job[3]['sample_id'] for job in jobs}
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            if len(receipts) % 20 == 0 or len(receipts) == len(jobs):
                progress = dict(completed=len(receipts), total=len(jobs), samples=sum(r['count'] for r in receipts),
                                seconds=time.perf_counter() - started, bytes=sum(r['bytes'] for r in receipts))
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    result = dict(build_id=build_id, profile=PROFILE, root=str(output), splits={}, source_counts={s: len(v) for s, v in source_rows.items()})
    for split in ('train', 'val'):
        shards = sorted([r for r in receipts if r['split'] == split], key=lambda r: (r['dataset'], r['sample_id']))
        index, baseline, augmented = [], [], []
        categories = Counter()
        for shard_index, shard in enumerate(shards):
            meta = json.loads((output / shard['metadata']).read_text())
            for sample_index, record in enumerate(meta):
                item = (shard_index, sample_index)
                index.append(item)
                (baseline if record['category'] == 'baseline' else augmented).append(item)
                categories[record['category']] += 1
        # Equal baseline/augmented sampling mass; all unique assets remain indexed.
        sampling = index
        if split == 'train' and baseline and augmented:
            target = max(len(baseline), len(augmented))
            sampling = [baseline[i % len(baseline)] for i in range(target)] + [augmented[i % len(augmented)] for i in range(target)]
        np.save(output / f'{split}.index.npy', np.asarray(sampling, dtype=np.uint32).reshape(-1, 2))
        np.save(output / f'{split}.unique.npy', np.asarray(index, dtype=np.uint32).reshape(-1, 2))
        manifest = dict(build_id=build_id, split=split, shape_hw=list(HW), dtype='uint8',
                        channels=['B', 'G', 'R', 'GT_B', 'GT_G', 'GT_R', 'text_mask', 'valid_mask'],
                        index=f'{split}.index.npy', unique_index=f'{split}.unique.npy', shards=shards,
                        preprocessing=('none' if PREPROCESS_MODE == 'none' else 'adaptive_dualmode_scurve_v1:gamma1.8'),
                        preprocess_mode=PREPROCESS_MODE, paired_only=True,
                        runtime_preprocessing=False, runtime_geometry=False, categories=dict(categories))
        write_json(output / f'{split}.json', manifest)
        result['splits'][split] = dict(parents=len(shards), nonempty_parents=sum(r['count'] > 0 for r in shards),
                                      unique_samples=len(index), sampling_samples=len(sampling), categories=dict(categories),
                                      bytes=sum(r['bytes'] for r in shards), rejections=dict(sum((Counter(r['rejections']) for r in shards), Counter())))
    result['seconds'] = time.perf_counter() - started
    result['complete'] = True
    write_json(output / 'COMPLETE.json', result)
    print(json.dumps(result), flush=True)


def main():
    """@luna-comment
    Input: CLI source/output/workers/preprocess-mode and pilot limits. Output: offline dataset build.
    Key Steps: bind lq/both/none mode for parent/worker processes, then delegate. Key Params:workers default8.
    State/Dependencies: build; never launches training.
    """
    global PREPROCESS_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--preprocess-mode', choices=('lq', 'both', 'none'), default=PREPROCESS_MODE)
    parser.add_argument('--limit', type=int, default=0, help='parents per family per split; zero means all')
    parser.add_argument('--max-base', type=int, default=0, help='pilot-only per-parent base cap')
    args = parser.parse_args()
    PREPROCESS_MODE = args.preprocess_mode
    os.environ['OFFLINE_PREPROCESS_MODE'] = PREPROCESS_MODE
    if args.workers < 1 or args.limit < 0 or args.max_base < 0:
        parser.error('Invalid worker/limit options')
    build(args)


if __name__ == '__main__':
    main()
