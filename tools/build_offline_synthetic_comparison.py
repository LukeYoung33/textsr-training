"""Generate matching offline synthetic-input shards; never launch training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    """@luna-comment
    Input: approved roots/config, deterministic shard rank and finalize switch.
    Output: verified synthetic shards or a complete matched manifest.
    Key Steps: inject process-local degradation into the exact original builder;
    compare every target/mask/metadata byte; publish after all receipts exist.
    Key Params: rank/world partition parents; seed derives full variant reference.
    State/Dependencies: read-only real bank/source, offline CUDA, frozen package.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'source', 'base', 'output', 'config'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    from tools.train_offline_comparison import bootstrap
    bootstrap(args.repo_root)
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools import build_offline_augmented_dataset as builder
    from realesrgan.data.basic_text_line_gtdeg_dataset import sample_standard_realesrgan_kernels
    from realesrgan.models.realesrnet_model import RealESRNetModel
    from basicsr.utils import DiffJPEG
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    source, base, output = map(Path, (args.source, args.base, args.output))
    output.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load(Path(args.config).read_text())
    base_manifest = json.loads((base / 'train_balanced.json').read_text())
    signature = dict(base_build_id=base_manifest['build_id'], builder_sha256=builder.digest(__file__),
                     augmentation_builder_sha256=builder.digest(builder.__file__),
                     degradation_source_sha256=builder.digest(Path(args.repo_root) / 'realesrgan/models/realesrnet_model.py'),
                     config_sha256=builder.digest(args.config), seed_policy='sha256(full_parent_variant_reference):first4',
                     operator='standard_two_stage_full_parent_before_frozen_preprocess', pilot_limit=args.limit)
    build_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    builder.write_json(output / f'signature.rank{args.rank}.json', dict(signature, build_id=build_id))
    rows = [json.loads(line) for line in (source / 'manifests/train.jsonl').read_text().splitlines() if line.strip()]
    by_id = {(row['dataset'], row['sample_id']): row for row in rows}
    shards = base_manifest['shards'][:args.limit or None]
    if args.finalize:
        receipts = []
        for shard in shards:
            receipt = json.loads((output / Path(shard['array']).with_suffix('.receipt.json')).read_text())
            if receipt['build_id'] != build_id or not receipt.get('matched_target_mask_metadata'):
                raise ValueError('Incomplete or foreign synthetic receipt')
            receipts.append(receipt)
        if args.limit:
            raise ValueError('Pilot cannot publish a formal complete dataset')
        manifest = copy.deepcopy(base_manifest)
        manifest.update(build_id=build_id, shards=receipts, input_origin='offline_synthetic')
        for key in ('index', 'unique_index'):
            shutil.copyfile(base / manifest[key], output / manifest[key])
        builder.write_json(output / 'train_balanced.json', manifest)
        builder.write_json(output / 'build_signature.json', dict(signature, build_id=build_id))
        builder.write_json(output / 'COMPLETE.json', dict(complete=True, build_id=build_id,
                           parents=len(receipts), samples=sum(r['count'] for r in receipts),
                           matched_target_mask_metadata=True, validation_root=str(base)))
        print(json.dumps(dict(complete=True, parents=len(receipts))), flush=True)
        return
    engine = object.__new__(RealESRNetModel)
    engine.opt = dict(opt, high_order_degradation=True, gt_usm=False, scale=1)
    engine.device = torch.device('cuda')
    engine.is_train, engine.finetune_deg, engine.color, engine.queue_size = True, -1., 'rgb', 0
    engine.jpeger = DiffJPEG(differentiable=False).cuda()
    original_prepare = builder.prepare_variant

    def synthetic_prepare(original, boxes, category, recipe, matrix, reference):
        """@luna-comment
        Input: full parent arrays and common geometry/photometry recipe.
        Output: identical augmented target/masks plus preprocessed synthetic LQ.
        Key Steps: reuse common augmentation; seed standard kernels/noise;
        run legacy full-frame degradation then frozen input preprocessing.
        Key Params: negative finetune_deg forces synthetic branch; no queue/color pack.
        State/Dependencies: process-local CUDA engine and exact original builder.
        """
        arrays, transformed_boxes = original_prepare(original, boxes, category, recipe, matrix, reference)
        seed = builder.stable_seed(reference + ':offline_standard_degradation_v1')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        kernels = sample_standard_realesrgan_kernels(opt['datasets']['train'])
        gt = torch.from_numpy(np.ascontiguousarray(arrays['gt'][..., ::-1].transpose(2, 0, 1))).float().unsqueeze(0).div_(255)
        engine.feed_data(dict(gt=gt, **{k: v.unsqueeze(0) for k, v in zip(('kernel1', 'kernel2', 'sinc_kernel'), kernels)}))
        lq = engine.lq[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()[..., ::-1].copy()
        context = builder.prepare_context(lq, transformed_boxes, reference_id=reference)
        arrays['lq'] = builder.apply_preprocess(lq, context)
        return arrays, transformed_boxes

    builder.prepare_variant = synthetic_prepare
    started = time.time()
    try:
        for ordinal, shard in enumerate(shards):
            if ordinal % args.world != args.rank:
                continue
            row = by_id[(shard['dataset'], shard['sample_id'])]
            receipt_path = output / Path(shard['array']).with_suffix('.receipt.json')
            existing = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            if existing and existing.get('matched_target_mask_metadata') and existing['build_id'] == build_id:
                continue
            receipt = builder.process_parent((str(source), str(output), 'train', row, 0, build_id))
            actual = np.load(output / receipt['array'], mmap_mode='r')
            expected = np.load(base / shard['array'], mmap_mode='r')
            if actual.shape != expected.shape or not np.array_equal(actual[..., 3:], expected[..., 3:]):
                raise ValueError('Synthetic target/mask bytes differ: ' + shard['sample_id'])
            if builder.digest(base / shard['metadata']) != receipt['metadata_sha256']:
                raise ValueError('Synthetic sample recipes/order differ: ' + shard['sample_id'])
            if len(actual) and np.array_equal(actual[..., :3], expected[..., :3]):
                raise ValueError('Synthetic input did not change')
            receipt['matched_target_mask_metadata'] = True
            receipt['input_origin'] = 'offline_synthetic'
            builder.write_json(receipt_path, receipt)
            print(json.dumps(dict(rank=args.rank, ordinal=ordinal, total=len(shards), samples=len(actual), seconds=time.time()-started)), flush=True)
    finally:
        builder.prepare_variant = original_prepare


if __name__ == '__main__':
    main()
