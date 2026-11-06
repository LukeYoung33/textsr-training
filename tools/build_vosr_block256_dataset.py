"""Materialize real paired full-image256 grids without OCR-selected network crops."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_offline_augmented_dataset as common


def grid_samples(arrays, boxes, category, recipe, matrix):
    """@luna-comment
    Input: full augmented pair/masks/support and OCR boxes. Output:256-square samples/metadata.
    Key Steps: enumerate fixed source grid, retain any valid area including background,
      mask replicated edge padding and attach contained width>=height OCR candidates in unrotated categories.
    Key Params:stride256; no OCR selection or row cap; augmented geometric support required.
    State/Dependencies:numpy; original arrays unchanged; OCR uprightness assumed, not recognized or rechecked here.
    """
    height, width = arrays['gt'].shape[:2]
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            hh, ww = min(256, height-y), min(256, width-x)
            support = arrays['support'][y:y+hh, x:x+ww]
            if support.min() < 255:
                continue
            valid = arrays['valid_mask'][y:y+hh, x:x+ww]
            if not (valid > 127).any():
                continue
            pixels = np.zeros((256, 256, 8), np.uint8)
            for key, start in (('lq', 0), ('gt', 3)):
                pixels[..., start:start+3] = np.pad(arrays[key][y:y+hh, x:x+ww],
                    ((0, 256-hh), (0, 256-ww), (0, 0)), mode='edge')
            pixels[:hh, :ww, 7] = valid
            pixels[:hh, :ww, 6] = np.where((arrays['text_mask'][y:y+hh, x:x+ww] > 127) & (valid > 127), 255, 0)
            lines = []
            if category in ('baseline', 'photometry', 'neighbors'):
                for bx0, by0, bx1, by1 in boxes:
                    if x <= bx0 < bx1 <= x+ww and y <= by0 < by1 <= y+hh and bx1-bx0 >= by1-by0:
                        if arrays['valid_mask'][by0:by1, bx0:bx1].min() == 255:
                            a,b,c,d = bx0-x, by0-y, bx1-x, by1-y
                            lines.append([a,b,c,b,c,d,a,d,''])
            rows, _ = common.row_information(boxes, (x,y), shape=(hh,ww))
            yield pixels, dict(category=category, recipe=recipe, matrix=matrix.tolist(), xy=[x,y],
                               orientation='HORIZONTAL' if category in ('baseline','photometry','neighbors') else 'OBLIQUE',
                               cp_eligible=bool(lines), cp_lines=lines, visible_ocr_rows=rows,
                               geometric_support=1., real_hw=[hh,ww], background_only=not pixels[...,6].any())


def process_parent(job):
    """@luna-comment
    Input: source/output roots, split,row and build ID. Output: hash-bound paired shard receipt.
    Key Steps: verify source hashes/geometry, apply common full-frame recipes, grid all valid blocks,
      save native pixels and annotation metadata atomically.
    Key Params:three train variants; validation baseline only. State/Dependencies:own output, immutable sources.
    """
    source, output, split, row, build_id = job
    source, output = Path(source), Path(output)
    cv2.setNumThreads(1)
    name = f"{split}/{row['dataset']}/{row['sample_id']}"
    receipt_path = output / 'shards' / (name+'.receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['build_id'] != build_id:
            raise ValueError('Foreign block-bank receipt')
        for key in ('array','metadata'):
            if common.digest(output/receipt[key]) != receipt[key+'_sha256']:
                raise ValueError('Corrupt resumed block shard')
        return receipt
    arrays, hashes = {}, {}
    for key in ('lq','gt','text_mask','valid_mask','ocr'):
        path = source / row['files'][key]
        hashes[key] = common.digest(path)
        if hashes[key] != row['output_sha256'][key]:
            raise ValueError('Source hash mismatch: '+name+'/'+key)
        if key != 'ocr':
            arrays[key] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arrays[key] is None or arrays[key].dtype != np.uint8:
                raise ValueError('Expected native uint8 source')
    if len({a.shape[:2] for a in arrays.values()}) != 1:
        raise ValueError('Block source geometry mismatch')
    annotations = common.read_annotations(json.loads((source/row['files']['ocr']).read_text()))
    boxes = common.clipped_boxes(annotations, arrays['gt'].shape)
    extra = source / 'OCR_ppocrv6_medium' / split / (row['sample_id']+'.json')
    if extra.exists():
        supplementary = common.clipped_boxes(common.read_annotations(json.loads(extra.read_text())), arrays['gt'].shape)
        for box in supplementary:
            if not any(abs((box[1]+box[3])-(b[1]+b[3])) < .5*min(box[3]-box[1],b[3]-b[1]) and min(box[2],b[2])>max(box[0],b[0]) for b in boxes):
                boxes.append(box)
        hashes['supplemental_ocr'] = common.digest(extra)
    arrays['support'] = np.full(arrays['gt'].shape[:2],255,np.uint8)
    recipes = [('baseline',{},np.eye(3))]
    if split == 'train':
        recipes += [common.recipe_for(name, slot, arrays['gt'].shape) for slot in range(3)]
    samples, records = [], []
    for category, recipe, matrix in recipes:
        transformed, transformed_boxes = common.prepare_variant(arrays,boxes,category,recipe,matrix,name+':'+category)
        for pixels, metadata in grid_samples(transformed,transformed_boxes,category,recipe,matrix):
            samples.append(pixels); records.append(metadata)
    relative = 'shards/'+name+'.npy'
    path = output/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    buffer = np.stack(samples) if samples else np.empty((0,256,256,8),np.uint8)
    temp = path.with_suffix('.npy.tmp')
    with temp.open('wb') as stream:
        np.save(stream,buffer,allow_pickle=False)
    temp.replace(path)
    metadata_path = 'shards/'+name+'.json'
    common.write_json(output/metadata_path,records)
    receipt = dict(build_id=build_id,sample_id=row['sample_id'],source_identity=row['source_identity'],dataset=row['dataset'],
                   split=split,array=relative,metadata=metadata_path,count=len(records),bytes=path.stat().st_size,
                   array_sha256=common.digest(path),metadata_sha256=common.digest(output/metadata_path),source_hashes=hashes,
                   background_samples=sum(r['background_only'] for r in records),
                   cp_lines=sum(len(r['cp_lines']) for r in records),categories=dict(Counter(r['category'] for r in records)))
    common.write_json(receipt_path,receipt)
    return receipt


def main():
    """@luna-comment
    Input: source/output/workers/mode CLI. Output: complete train/val block bank and balanced indices.
    Key Steps: bind builder/source/frozen hashes, validate capture isolation, build bounded workers,
      index all samples by category and publish only after every parent completes.
    Key Params:256 grid,16workers default, same50/15/10/10/10/5 draw mixture; no online transforms.
    State/Dependencies:common immutable frozen preprocessing, source-view manifests, own outputs.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--preprocess-mode',choices=('none','both'),default='none')
    args=parser.parse_args()
    common.PREPROCESS_MODE=args.preprocess_mode
    os.environ['OFFLINE_PREPROCESS_MODE']=args.preprocess_mode
    source,output=Path(args.source),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    rows={s:[json.loads(line) for line in (source/'manifests'/(s+'.jsonl')).read_text().splitlines() if line.strip()] for s in ('train','val')}
    if {r['source_identity'] for r in rows['train']} & {r['source_identity'] for r in rows['val']}:
        raise ValueError('Capture overlap')
    frozen=ROOT/'evaluation/adaptive_prepost_frozen_v1_20261020'
    for filename,sha in json.loads((frozen/'manifest.json').read_text())['files'].items():
        if common.digest(frozen/filename)!=sha: raise ValueError('Frozen source mismatch')
    signature=dict(builder=common.digest(__file__),common_builder=common.digest(common.__file__),
                   mode=args.preprocess_mode,source_manifests={s:common.digest(source/'manifests'/(s+'.jsonl')) for s in rows},
                   frozen_manifest=common.digest(frozen/'manifest.json'),shape_hw=[256,256],grid_stride=256)
    build_id=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    common.write_json(output/'build_signature.json',dict(signature,build_id=build_id))
    jobs=[(str(source),str(output),s,r,build_id) for s,items in rows.items() for r in items]
    receipts=[];started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(process_parent,job) for job in jobs]
        for future in as_completed(futures):
            receipts.append(future.result())
            if len(receipts)%20==0 or len(receipts)==len(jobs):
                progress=dict(completed=len(receipts),total=len(jobs),samples=sum(r['count'] for r in receipts),seconds=time.time()-started)
                common.write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    complete=dict(complete=True,build_id=build_id,shape_hw=[256,256],splits={},seconds=time.time()-started)
    for split in ('train','val'):
        shards=sorted([r for r in receipts if r['split']==split],key=lambda r:(r['dataset'],r['sample_id']))
        refs=[];groups={};categories=Counter()
        for si,shard in enumerate(shards):
            for ri,record in enumerate(json.loads((output/shard['metadata']).read_text())):
                refs.append((si,ri));groups.setdefault(record['category'],[]).append((si,ri));categories[record['category']]+=1
        np.save(output/(split+'.index.npy'),np.asarray(refs,np.uint32).reshape(-1,2))
        manifest=dict(build_id=build_id,shape_hw=[256,256],paired_only=True,runtime_preprocessing=False,runtime_geometry=False,
                      preprocess_mode=args.preprocess_mode,index=split+'.index.npy',unique_index=split+'.index.npy',shards=shards,
                      categories=dict(categories),sampling_fractions=None)
        common.write_json(output/(split+'.json'),manifest)
        if split=='train':
            from tools.verify_offline_augmented_dataset import category_balanced_manifest
            category_balanced_manifest(output,manifest,groups)
        complete['splits'][split]=dict(parents=len(shards),nonempty_parents=sum(r['count']>0 for r in shards),samples=len(refs),
                                      categories=dict(categories),background_samples=sum(r['background_samples'] for r in shards),
                                      cp_lines=sum(r['cp_lines'] for r in shards),bytes=sum(r['bytes'] for r in shards))
    common.write_json(output/'COMPLETE.json',complete)
    print(json.dumps(complete),flush=True)


if __name__=='__main__':
    main()
