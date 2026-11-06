"""Freeze native OCR-region patch indices and clone the historical 0921 recipe."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml
from PIL import Image


def digest(path):
    """@luna-comment Input: file. Output: SHA256. Key Steps: stream chunks.
    Key Params: 1MiB buffer. State/Dependencies: read-only disk access.
    """
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def annotations(document):
    """@luna-comment Input: published OCR JSON. Output: source-coordinate boxes.
    Key Steps: prefer explicit annotations, otherwise convert recognition polygons.
    Key Params: no coordinate transform/resize inferred. State/Dependencies: numpy.
    """
    nested = document.get('ocr', document.get('res', document))
    entries = nested.get('annotations', document.get('annotations', []))
    if entries:
        result = []
        for entry in entries:
            if entry.get('ignore') or entry.get('visible') is False:
                continue
            if len(entry.get('bbox', [])) == 4:
                result.append(entry['bbox'])
                continue
            points = entry.get('points', entry.get('polygon'))
            if points is not None and len(points) >= 3:
                pts = np.asarray(points).reshape(-1, 2)
                result.append([*pts.min(0).tolist(), *pts.max(0).tolist()])
            elif len(entry.get('bbox', [])) == 4:
                result.append(entry['bbox'])
        return result
    if document.get('lines'):
        return [line['bbox_xyxy'] for line in document['lines'] if len(line.get('bbox_xyxy', [])) == 4]
    polys = nested.get('rec_polys', nested.get('dt_polys', document.get('rec_polys', [])))
    return [[*np.asarray(p).reshape(-1, 2).min(0).tolist(), *np.asarray(p).reshape(-1, 2).max(0).tolist()] for p in polys]


def axis_origins(start, end, canvas, full):
    """@luna-comment Input: clipped region interval, patch extent, full-image extent.
    Output: even nonnegative origins covering region at source resolution.
    Key Steps: center small regions; evenly cover larger regions, clamp to source.
    Key Params: canvas stride upper bound; even origins preserve YUV phase.
    State/Dependencies: deterministic arithmetic, no RNG.
    """
    first = int(math.floor(start/2))*2
    last = int(math.ceil((end-canvas)/2))*2
    if last <= first:
        centered = int(round((start+end-canvas)/4))*2
        positions = [max(last, min(first, centered))]
    else:
        positions = list(range(first, last, canvas)) + [last]
    maximum = int(math.ceil(max(0, full-canvas)/2))*2
    return sorted({max(0, min(int(math.floor(p/2))*2, maximum)) for p in positions})


def index_image(args):
    """@luna-comment Input: root/split/published manifest record. Output: validated crop record.
    Key Steps: verify pair and mask dimensions; cover OCR context boxes; reject empty
    trusted-text patches; deduplicate origins and record original provenance.
    Key Params: historical context ratios .12/.03, fixed 96x480 canvas.
    State/Dependencies: read-only source assets, supplemental OCR used only if available.
    """
    root, split, record = args
    files = {k: str(root / record['files'][k]) for k in ('lq', 'gt', 'text_mask', 'valid_mask', 'ocr')}
    if any('text_lines' in p for p in files.values()):
        raise ValueError('text_lines assets are forbidden')
    shapes = []
    for key in ('lq', 'gt', 'text_mask', 'valid_mask'):
        with Image.open(files[key]) as image:
            shapes.append(image.size)
    if len(set(shapes)) != 1:
        raise ValueError(f"Pair/mask shape mismatch: {record['sample_id']}")
    width, height = shapes[0]
    text = cv2.imread(files['text_mask'], cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(files['valid_mask'], cv2.IMREAD_GRAYSCALE)
    support = (text > 127) & (valid > 127)
    doc = json.loads(Path(files['ocr']).read_text())
    boxes = annotations(doc)
    extra = root / 'OCR_ppocrv6_medium' / split / (record['sample_id']+'.json')
    if extra.exists():
        # Published supplemental detections cover previously unannotated text.
        boxes += annotations(json.loads(extra.read_text()))
    origins = set()
    for box in boxes:
        x0, y0, x1, y1 = map(float, box)
        if not np.isfinite(box).all() or x1 <= x0 or y1 <= y0:
            continue
        padx, pady = max(2, round((x1-x0)*.03)), max(2, round((y1-y0)*.12))
        x0, x1 = max(0, x0-padx), min(width, x1+padx)
        y0, y1 = max(0, y0-pady), min(height, y1+pady)
        if x1 <= x0 or y1 <= y0:
            continue
        for y in axis_origins(y0, y1, 96, height):
            for x in axis_origins(x0, x1, 480, width):
                if support[y:y+96, x:x+480].any():
                    origins.add((x, y))
    return dict(sample_id=record['sample_id'], source_identity=record.get('source_identity', record['sample_id']),
                dataset=record.get('dataset', 'lenovo0920'), files=files, source_hw=[height, width],
                source_hashes=record.get('output_sha256', {}),
                supplemental_ocr=str(extra) if extra.exists() else None,
                patches=[{'xy': [x, y]} for x, y in sorted(origins, key=lambda p: (p[1], p[0]))],
                ocr_boxes=len(boxes))


def prepare(args):
    """@luna-comment Input: output directory and repository historical YAMLs.
    Output: four configs, four patch manifests, source/config provenance report.
    Key Steps: freeze source splits, verify isolation, index native crops, clone recipe.
    Key Params: user-approved V3 Haar225K and paired probabilities 0/1/0/.75.
    State/Dependencies: bounded thread pool for IO; no source data mutation.
    """
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    baseline_path = Path(args.baseline) if args.baseline else repo/'options/train_bucket_3/lenovo_v5_p17_effect_iteration_20260920/train_original_r1_bglf_uv_250k_20260920.yml'
    base = yaml.safe_load(baseline_path.read_text())
    v3_path = Path(args.v3_config) if args.v3_config else repo/'options/train_bucket_3/train_basic_text_line_y4uv_v3_cp_haar_20260820_96x480.yml'
    v3 = yaml.safe_load(v3_path.read_text())
    checkpoint = '/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/basic_text_line_y4uv_v3_cp_haar_20260820_96x480/models/net_g_225000.pth'
    report = {'baseline_sha256': digest(baseline_path), 'checkpoint': checkpoint, 'checkpoint_sha256': digest(checkpoint), 'sources': {}, 'arms': {}}
    for tag, root in [('old', Path('/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop')),
                      ('new', Path('/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261015/full_images'))]:
        identities = {}
        for split in ('train', 'val'):
            source = root/'manifests'/f'{split}.jsonl'
            records = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
            identities[split] = {r.get('source_identity', r['sample_id']) for r in records}
            with ThreadPoolExecutor(max_workers=8) as pool:
                indexed = list(pool.map(index_image, [(root, split, r) for r in records]))
            manifest = out/f'{tag}_{split}.jsonl'
            manifest.write_text(''.join(json.dumps(r, ensure_ascii=True)+'\n' for r in indexed if r['patches']))
            report['sources'][f'{tag}_{split}'] = dict(root=str(root), manifest=str(source), sha256=digest(source),
                images=len(records), indexed_images=sum(bool(r['patches']) for r in indexed),
                patches=sum(len(r['patches']) for r in indexed), dataset_counts=dict(Counter(r['dataset'] for r in indexed)),
                empty_images=[r['sample_id'] for r in indexed if not r['patches']], index_sha256=digest(manifest))
            print(tag, split, report['sources'][f'{tag}_{split}']['patches'], flush=True)
        overlap = identities['train'] & identities['val']
        report['sources'][tag+'_source_overlap'] = sorted(overlap)
        if overlap:
            raise ValueError(f'{tag} source leakage: {len(overlap)}')
    config_dir = out/'configs'
    config_dir.mkdir(exist_ok=True)
    for arm, tag, probability in [('01_0921data_degraded', 'old', 0), ('02_displayfit_paired', 'new', 1),
                                   ('03_displayfit_degraded', 'new', 0), ('04_displayfit_mixed75', 'new', .75)]:
        opt = copy.deepcopy(base)
        opt['name'] = arm
        opt['model_type'] = opt['train']['model_type'] = 'NativeRegion0921Model'
        opt['network_g'] = copy.deepcopy(v3['network_g'])
        opt['paired_probability'] = probability
        opt['path']['pretrain_network_g'] = checkpoint
        opt['path']['strict_load_g'] = True
        opt['path']['resume_state'] = None
        opt['path']['experiments_root'] = str(out/'runs')
        for split in ('train', 'val'):
            settings = opt['datasets'][split]
            for key in ('sources', 'resize_mode', 'ocr_line_crop', 'text_line_crop', 'text_line_gap',
                        'text_line_vertical_context', 'text_line_horizontal_context', 'text_line_boundary_jitter'):
                settings.pop(key, None)
            settings.update(name=f'{arm}_{split}', type='NativeTextRegionDataset', manifest=str(out/f'{tag}_{split}.jsonl'), cache_images=2)
        opt['dist_params']['port'] = 31600 + int(arm[:2])
        # Training loss/optimizer/schedule are exactly cloned, except registry type.
        expected = copy.deepcopy(base['train']); expected['model_type'] = 'NativeRegion0921Model'
        assert opt['train'] == expected
        path = config_dir/f'{arm}.yml'
        path.write_text(yaml.safe_dump(opt, sort_keys=False))
        report['arms'][arm] = {'config': str(path), 'sha256': digest(path), 'paired_probability': probability}
    (out/'preparation.json').write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(json.dumps(report, indent=2), flush=True)


def main():
    """@luna-comment Input: --output. Output: prepared experiment directory.
    Key Steps: parse then prepare. Key Params: explicit isolated output required.
    State/Dependencies: CLI and prepare IO.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--v3-config')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
