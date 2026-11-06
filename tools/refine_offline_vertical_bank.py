"""Prepublication vertical-background refinement without recomputing unaffected variants."""
from __future__ import annotations

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
from tools.build_offline_augmented_dataset import digest, write_json, read_annotations, vertical_samples


def refine_parent(job):
    """@luna-comment
    Input: completed build-owned shard, authoritative parent and new deterministic build ID.
    Output: updated receipt; only column-derived sample bytes may change.
    Key Steps: reproduce improved columns, replace appended rows, hash array/metadata, update receipt.
    Key Params: nonvertical rows are bitwise preserved; resume requires matching receipt ID.
    State/Dependencies: current standalone builder and own unpublished bank, source files read-only.
    """
    bank, source, shard, row, build_id = job
    bank, source = Path(bank), Path(source)
    receipt_path = (bank / shard['array']).with_suffix('.receipt.json')
    receipt = json.loads(receipt_path.read_text())
    if receipt['build_id'] == build_id:
        return receipt
    cv2.setNumThreads(1)
    records = json.loads((bank / shard['metadata']).read_text())
    positions = [i for i, r in enumerate(records) if r['recipe'].get('kind') == 'synthetic_tight_vertical_rotated']
    if positions:
        prefix = positions[0]
        assert positions == list(range(prefix, len(records)))
        arrays = {k: cv2.imread(str(source / row['files'][k]), cv2.IMREAD_UNCHANGED)
                  for k in ('lq', 'gt', 'text_mask', 'valid_mask')}
        arrays['support'] = np.full(arrays['gt'].shape[:2], 255, np.uint8)
        annotations = read_annotations(json.loads((source / row['files']['ocr']).read_text(encoding='utf-8')))
        name = f"train/{row['dataset']}/{row['sample_id']}:vertical"
        samples = vertical_samples(arrays, annotations, name)
        path = bank / shard['array']
        if len(samples) == len(positions):
            output = np.lib.format.open_memmap(path, mode='r+')
            for i, (pixels, meta) in enumerate(samples):
                output[prefix + i] = pixels
            output.flush()
            del output
        else:
            original = np.load(path, mmap_mode='r')
            temporary = path.with_suffix('.npy.refining')
            output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(prefix + len(samples), 96, 480, 8))
            output[:prefix] = original[:prefix]
            for i, (pixels, meta) in enumerate(samples):
                output[prefix + i] = pixels
            output.flush()
            del output, original
            os.replace(temporary, path)
        records = records[:prefix] + [item[1] for item in samples]
        write_json(bank / shard['metadata'], records)
        receipt.update(array_sha256=digest(path), metadata_sha256=digest(bank / shard['metadata']),
                       bytes=path.stat().st_size, count=len(records),
                       categories=dict(Counter(r['category'] for r in records)))
    receipt['build_id'] = build_id
    write_json(receipt_path, receipt)
    return receipt


def run(args):
    """@luna-comment
    Input: unpublished structurally complete bank and read-only source root.
    Output: final bank directly reproducible by the updated standalone builder.
    Key Steps: withdraw ready marker, refine parents, republish indexes/receipts under new code hash.
    Key Params: bounded workers; source split/counts and unaffected pixels stay unchanged.
    State/Dependencies: local build-owned marker/manifests; no model training/source mutation.
    """
    bank, source = Path(args.bank), Path(args.source)
    historical = bank / 'COMPLETE.before_vertical_refinement.json'
    if not historical.exists():
        os.replace(bank / 'COMPLETE.json', historical)
    old_complete = json.loads(historical.read_text())
    old_signature = json.loads((bank / 'build_signature.json').read_text())
    signature = {k: v for k, v in old_signature.items() if k != 'build_id'}
    signature['builder'] = digest(ROOT / 'tools/build_offline_augmented_dataset.py')
    build_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    write_json(bank / 'vertical_refinement.json', dict(old_build_id=old_complete['build_id'], new_build_id=build_id,
                                                     tool_sha256=digest(__file__), status='in_progress'))
    started = time.perf_counter()
    source_rows = {json.loads(line)['sample_id']: json.loads(line) for line in (source / 'manifests/train.jsonl').read_text().splitlines() if line.strip()}
    manifests = {s: json.loads((bank / f'{s}.json').read_text()) for s in ('train', 'val')}
    jobs = [(str(bank), str(source), shard, source_rows.get(shard['sample_id']), build_id)
            for s in ('train', 'val') for shard in manifests[s]['shards']]
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(refine_parent, job) for job in jobs]
        for future in as_completed(futures):
            receipts.append(future.result())
            if len(receipts) % 200 == 0 or len(receipts) == len(jobs):
                progress = dict(completed=len(receipts), total=len(jobs), seconds=time.perf_counter() - started)
                write_json(bank / 'refinement_progress.json', progress)
                print(json.dumps(progress), flush=True)
    complete = dict(old_complete, build_id=build_id)
    for split in ('train', 'val'):
        shards = sorted([r for r in receipts if r['split'] == split], key=lambda r: (r['dataset'], r['sample_id']))
        unique, baseline, augmented, categories = [], [], [], Counter()
        for si, shard in enumerate(shards):
            records = json.loads((bank / shard['metadata']).read_text())
            for ri, record in enumerate(records):
                item = (si, ri)
                unique.append(item)
                (baseline if record['category'] == 'baseline' else augmented).append(item)
                categories[record['category']] += 1
        sampling = unique
        if split == 'train' and baseline and augmented:
            target = max(len(baseline), len(augmented))
            sampling = [baseline[i % len(baseline)] for i in range(target)] + [augmented[i % len(augmented)] for i in range(target)]
        np.save(bank / f'{split}.index.npy', np.asarray(sampling, np.uint32).reshape(-1, 2))
        np.save(bank / f'{split}.unique.npy', np.asarray(unique, np.uint32).reshape(-1, 2))
        manifest = manifests[split]
        manifest.update(build_id=build_id, shards=shards, categories=dict(categories))
        write_json(bank / f'{split}.json', manifest)
        complete['splits'][split].update(parents=len(shards), nonempty_parents=sum(r['count'] > 0 for r in shards),
                                         unique_samples=len(unique), sampling_samples=len(sampling), categories=dict(categories),
                                         bytes=sum(r['bytes'] for r in shards))
    write_json(bank / 'build_signature.json', dict(signature, build_id=build_id))
    complete['seconds'] = old_complete['seconds'] + time.perf_counter() - started
    write_json(bank / 'vertical_refinement.json', dict(old_build_id=old_complete['build_id'], new_build_id=build_id,
                                                     tool_sha256=digest(__file__), status='completed', seconds=time.perf_counter() - started))
    write_json(bank / 'COMPLETE.json', complete)
    print(json.dumps(complete), flush=True)


def main():
    """@luna-comment
    Input: CLI bank/source/workers. Output: final prepublication refinement.
    Key Steps: parse explicit roots and delegate. Key Params:8workers default.
    State/Dependencies: own derived dataset only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
