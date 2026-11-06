"""Verify actual offline bank pixels, model input routing and loader throughput."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.train_offline_augmented_regions import bootstrap
from tools.build_offline_augmented_dataset import digest, process_parent


def category_balanced_manifest(bank, manifest, groups):
    """@luna-comment
    Input: bank manifest and all unique per-category shard references.
    Output: optional training view with exact50/15/10/10/10/5 percent category mass.
    Key Steps: repeat compact references to meet quotas while retaining every unique asset at least once.
    Key Params: integer units baseline10/photo3/rotation2/geometry2/neighbors2/direction1.
    State/Dependencies: writes only derived index/view, leaves physical assets/build identity unchanged.
    """
    import math
    units = dict(baseline=10, photometry=3, rotation=2, geometry=2, neighbors=2, direction=1)
    if any(not groups.get(category) for category in units):
        raise ValueError('Cannot balance missing augmentation category')
    unit_count = max(math.ceil(len(groups[category]) / unit) for category, unit in units.items())
    draws = []
    for category, unit in units.items():
        group = np.asarray(groups[category], np.uint32)
        draws.append(group[np.arange(unit_count * unit) % len(group)])
    index = np.concatenate(draws)
    np.save(bank / 'train_balanced.index.npy', index)
    derived = dict(manifest, index='train_balanced.index.npy', sampling_fractions={k: v / 20 for k, v in units.items()},
                   sampling_samples=len(index), index_sha256=digest(bank / 'train_balanced.index.npy'))
    path = bank / 'train_balanced.json'
    path.write_text(json.dumps(derived, indent=2), encoding='utf-8')
    return dict(manifest=str(path), sampling_samples=len(index), sampling_fractions=derived['sampling_fractions'],
                all_unique_assets_included=True, index_sha256=derived['index_sha256'])


def gallery(bank, dataset, output):
    """@luna-comment
    Input: complete bank, actual reader and build-owned gallery directory.
    Output: paired PNGs and self-contained local HTML with parameter/source labels.
    Key Steps: choose actual category/parameter representatives; save native pixels and mode labels.
    Key Params: examples are actual training assets, not model predictions.
    State/Dependencies: OpenCV file writes, escaped HTML; source/bank pixels unchanged.
    """
    import cv2
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    chosen = {}
    clipping = []
    for shard_id, shard in enumerate(dataset.shards):
        meta = json.loads((bank / shard['metadata']).read_text())
        for item_id, record in enumerate(meta):
            category, recipe = record['category'], record['recipe']
            if category == 'baseline':
                key = 'baseline_' + shard['dataset']
            elif category == 'photometry':
                key = 'light_' + recipe['kind'] + ('_low' if recipe['stratum'] < 500 else '_high')
            elif category == 'rotation':
                key = 'rotation_' + str(int(recipe['angle'] // 5) * 5)
            elif category == 'direction':
                key = 'direction_' + recipe['kind'] + '_' + str(recipe['angle'])
            else:
                key = category + '_' + recipe['kind']
            if key not in chosen:
                chosen[key] = (shard_id, item_id)
    cards = []
    for ordinal, (label, (shard_id, item_id)) in enumerate(sorted(chosen.items())):
        array, metadata = dataset._load(shard_id)
        sample, record = array[item_id], metadata[item_id]
        filename = f'{ordinal:02d}_{label}.png'
        pair = np.concatenate([sample[..., :3], sample[..., 3:6]], axis=1)
        assert cv2.imwrite(str(output / filename), pair)
        recipe = json.dumps(record['recipe'], sort_keys=True)
        source = dataset.shards[shard_id]
        cards.append(f'<article><h2>{html.escape(label)}</h2><p>{html.escape(source["sample_id"])} / {html.escape(source["dataset"])}</p>'
                     f'<img src="{html.escape(filename)}" width="960" height="96"><details><summary>Parameters</summary><pre>{html.escape(recipe)}</pre></details></article>')
        clipping.append(dict(label=label, gt_zero_fraction=float((sample[..., 3:6] == 0).all(2).mean()),
                             gt_white_fraction=float((sample[..., 3:6] == 255).all(2).mean())))
    page = '<!doctype html><meta charset="utf-8"><title>Offline training dataset</title><style>body{font:15px sans-serif;background:#111820;color:#e4edf5;margin:28px}article{background:#1c2835;padding:16px;margin:18px 0;max-width:1000px}img{max-width:100%;height:auto}h2{font-size:18px}pre{white-space:pre-wrap}</style>'
    mode = dataset.manifest.get('preprocess_mode', 'lq')
    descriptions = {'none': 'Neither LQ nor GT is preprocessed.', 'both': 'Both LQ and GT are independently preprocessed.',
                    'lq': 'Only LQ is preprocessed; GT is the augmented target.'}
    page += '<h1>Actual offline training assets</h1><p>Left: augmented LQ. Right: augmented paired GT. ' + descriptions[mode] + ' Native 480x96 each. No model output.</p>'
    page += '<p>All-source full-context geometry; at most three OCR row bands. Vertical layouts are explicitly synthetic. Parameter examples are not an image-quality acceptance claim.</p>'
    page += ''.join(cards)
    (output / 'index.html').write_text(page, encoding='utf-8')
    return dict(examples=len(cards), index=str(output / 'index.html'), clipping=clipping)


def replay_sources(bank, dataset, source):
    """@luna-comment
    Input: completed bank, its reader and authoritative source root.
    Output: byte-exact four-family parent replay and resume evidence.
    Key Steps: rebuild one parent per family in an isolated verification root, compare
      array/metadata hashes; exercise receipt-based resume without overwriting bank.
    Key Params: max_base=0; replay directory binds bank name/build identity to avoid mode collisions.
    State/Dependencies: current builder/frozen code and immutable bank.
    """
    root = Path(source)
    records = {r['sample_id']: r for r in (json.loads(s) for s in (root / 'manifests/train.jsonl').read_text().splitlines() if s.strip())}
    selected = {}
    for shard in dataset.shards:
        if shard['count'] and shard['dataset'] not in selected:
            selected[shard['dataset']] = shard
    destination = bank.parent / ('verification_replay_' + bank.name + '_' + dataset.manifest['build_id'][:12])
    evidence = {}
    for family, shard in selected.items():
        job = (str(root), str(destination), 'train', records[shard['sample_id']], 0, dataset.manifest['build_id'])
        replay = process_parent(job)
        assert replay['array_sha256'] == shard['array_sha256']
        assert replay['metadata_sha256'] == shard['metadata_sha256']
        resumed = process_parent(job)
        assert replay == resumed
        evidence[family] = dict(sample_id=shard['sample_id'], count=shard['count'], byte_exact=True, resume_exact=True)
    return evidence


def measure(dataset, workers, batch_size, batches):
    """@luna-comment
    Input: dataset, worker/batch counts and measured batch budget.
    Output: warm loader median/p95/mean batch wait and startup time.
    Key Steps: shuffled equal-sized draws with fixed seed, warmup then time identical batches.
    Key Params: four warmup batches, replacement index supports small pilots.
    State/Dependencies: DataLoader processes are shut down by iterator destruction.
    """
    generator = torch.Generator().manual_seed(20261020)
    count = (batches + 4) * batch_size
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=count, generator=generator)
    started = time.perf_counter()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                        num_workers=workers, collate_fn=dataset.collate_fn,
                                        pin_memory=False, persistent_workers=False)
    iterator = iter(loader)
    elapsed, shapes = [], None
    for index in range(batches + 4):
        now = time.perf_counter()
        batch = next(iterator)
        seconds = time.perf_counter() - now
        shapes = list(batch['lq_rgb'].shape)
        if index >= 4:
            elapsed.append(seconds)
    del iterator, loader
    return dict(mean_batch_ms=float(np.mean(elapsed) * 1000), median_batch_ms=float(np.median(elapsed) * 1000),
                p95_batch_ms=float(np.quantile(elapsed, .95) * 1000), wall_seconds=time.perf_counter() - started,
                batch_shape=shapes, workers=workers, batch_size=batch_size, batches=batches)


def verify(args):
    """@luna-comment
    Input: bank and existing training repository, optional baseline manifest/config.
    Output: integrity/reader/model-routing evidence, benchmark and isolated paired YAML.
    Key Steps: bootstrap, sample shards, replay exact bytes, feed real batch without optimization,
      benchmark identical loader settings and write a config preserving losses/budget.
    Key Params: paired_probability=1; runtime preprocessing/homography disabled.
    State/Dependencies: historical model methods, CPU only; no training/checkpoint mutation.
    """
    torch.set_num_threads(1)
    bootstrap(args.repo_root)
    from realesrgan.data.offline_augmented_region_dataset import OfflineAugmentedRegionDataset
    from realesrgan.data.native_text_region_dataset import NativeTextRegionDataset
    from realesrgan.models.native_region_0921_model import NativeRegion0921Model
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    bank = Path(args.bank)
    complete = json.loads((bank / 'COMPLETE.json').read_text())
    evidence = dict(build_id=complete['build_id'], checks={}, source='actual offline bank')
    import cv2
    import platform
    evidence['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
                                   opencv=cv2.__version__, torch=torch.__version__, torch_threads=torch.get_num_threads())
    checks = evidence['checks']
    dataset = OfflineAugmentedRegionDataset(dict(manifest=str(bank / 'train.json'), root=str(bank), unique_index=True))
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(128, len(dataset)), dtype=int))
    categories, sides, angles, lighting, vertical = Counter(), Counter(), [], Counter(), 0
    sample_batch = []
    for index in indices:
        sample = dataset[int(index)]
        shard_id, item_id = map(int, dataset.index[index])
        array, metadata = dataset._load(shard_id)
        raw, meta = array[item_id], metadata[item_id]
        np.testing.assert_allclose(sample['lq_rgb'].numpy(), raw[..., [2, 1, 0]].transpose(2, 0, 1) / 255, atol=4e-8)
        assert raw.shape == (96, 480, 8) and meta['geometric_support'] == 1
        assert meta['visible_ocr_rows'] <= 3
        assert torch.isfinite(sample['lq_rgb']).all() and torch.isfinite(sample['gt_rgb']).all()
        assert bool((sample['text_mask'] <= sample['valid_mask']).all())
        if meta['category'] in ('direction', 'neighbors'):
            assert not sample['cp_eligible']
        if len(sample_batch) < 4:
            sample_batch.append(sample)
    checks['sample_pixels_shapes_masks_eligibility'] = True
    groups = {}
    for shard_id, shard in enumerate(dataset.shards):
        assert (bank / shard['array']).stat().st_size == shard['bytes']
        assert digest(bank / shard['metadata']) == shard['metadata_sha256']
        records = json.loads((bank / shard['metadata']).read_text())
        assert len(records) == shard['count']
        for sample_id, meta in enumerate(records):
            assert meta['geometric_support'] == 1 and meta['visible_ocr_rows'] <= 3
            groups.setdefault(meta['category'], []).append((shard_id, sample_id))
            categories[meta['category']] += 1
            recipe = meta['recipe']
            if meta['category'] == 'neighbors':
                sides[recipe['kind']] += 1
            if meta['category'] == 'rotation':
                angles.append(recipe['angle'])
            if meta['category'] == 'photometry':
                lighting[recipe['kind']] += 1
            vertical += recipe.get('kind') == 'synthetic_tight_vertical_rotated'
    evidence['coverage'] = dict(categories=dict(categories), neighbor_sides=dict(sides), lighting_kinds=dict(lighting),
                                rotation_range=[min(angles), max(angles)] if angles else [], vertical_rotated=vertical)
    checks['all_train_metadata_sizes_hashes_and_geometry'] = True
    assert sum(categories.values()) == complete['splits']['train']['unique_samples']
    checks['full_manifest_count_matches'] = True
    evidence['balanced_training_view'] = category_balanced_manifest(bank, dataset.manifest, groups)
    checks['balanced_all_categories_all_unique_assets'] = True
    batch = dataset.collate_fn(sample_batch)
    model = object.__new__(NativeRegion0921Model)
    model.opt = dict(paired_probability=1., lq_preprocess=dict(enabled=False))
    model.device = torch.device('cpu')
    model.is_train = True
    model.feed_data(batch)
    torch.testing.assert_close(model.lq, rgb_to_y4uv(batch['lq_rgb']), rtol=0, atol=0)
    assert model.last_paired_selection.all() and 'kernel1' not in batch
    checks['actual_model_feed_exact_paired_no_degradation'] = True
    evidence['model_lq_shape'] = list(model.lq.shape)
    train_ids = {s['source_identity'] for s in dataset.shards}
    val_manifest = json.loads((bank / 'val.json').read_text())
    assert not train_ids & {s['source_identity'] for s in val_manifest['shards']}
    checks['train_val_identity_isolation'] = True
    if args.source:
        source = Path(args.source)
        excluded = {json.loads(s)['sample_id'] for s in (source / 'manifests/review_excluded.jsonl').read_text().splitlines() if s.strip()}
        assert not excluded & {s['sample_id'] for s in dataset.shards + val_manifest['shards']}
        checks['review_excluded_absent'] = True
        evidence['parent_replay'] = replay_sources(bank, dataset, args.source)
        checks['four_family_byte_exact_replay_and_resume'] = len(evidence['parent_replay']) == 4
    if args.gallery:
        evidence['gallery'] = gallery(bank, dataset, args.gallery)
    if args.baseline:
        baseline = NativeTextRegionDataset(dict(manifest=args.baseline, phase='train', cache_images=2,
                                                canvas_hw=[96, 480], train_homography_prob=.35,
                                                train_homography_jitter=.015))
        # Alternate order to reduce one-way warm-cache bias.
        paired = OfflineAugmentedRegionDataset(dict(manifest=str(bank / 'train_balanced.json'), root=str(bank)))
        evidence['baseline_loader'] = measure(baseline, args.workers, args.batch_size, args.batches)
        evidence['offline_loader'] = measure(paired, args.workers, args.batch_size, args.batches)
        evidence['offline_loader_repeat'] = measure(paired, args.workers, args.batch_size, args.batches)
        evidence['baseline_loader_repeat'] = measure(baseline, args.workers, args.batch_size, args.batches)
        old = np.mean([evidence[k]['mean_batch_ms'] for k in ('baseline_loader', 'baseline_loader_repeat')])
        new = np.mean([evidence[k]['mean_batch_ms'] for k in ('offline_loader', 'offline_loader_repeat')])
        evidence['offline_to_baseline_wait_ratio'] = float(new / old)
        checks['loader_no_material_regression'] = bool(new <= old * 1.1)
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text())
        from basicsr.utils.registry import ARCH_REGISTRY
        network_options = dict(config['network_g'])
        generator = ARCH_REGISTRY.get(network_options.pop('type'))(**network_options)
        prediction = generator(model.lq)
        assert prediction.shape == model.gt.shape and torch.isfinite(prediction).all()
        smoke_loss = ((prediction - model.gt).abs() * model.valid_mask).mean()
        smoke_loss.backward()
        gradients = [p.grad for p in generator.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        checks['cpu_native_generator_forward_backward_finite'] = True
        evidence['cpu_smoke_loss'] = float(smoke_loss.detach())
        config.update(name='adaptive_offline_paired_v1_20261020', paired_probability=1., high_order_degradation=False)
        config['lq_preprocess'] = dict(enabled=False)
        config['path']['experiments_root'] = str(bank.parent / 'training_runs')
        for split in ('train', 'val'):
            options = config['datasets'][split]
            options.update(name='adaptive_offline_' + split, type='OfflineAugmentedRegionDataset', root=str(bank), manifest=str(bank / f'{split}.json'),
                           train_homography_prob=0, train_homography_jitter=0, cache_shards=8)
            if split == 'train':
                options['manifest'] = str(bank / 'train_balanced.json')
        (bank / 'train_paired.yml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        evidence['config'] = str(bank / 'train_paired.yml')
        checks['training_budget_preserved'] = config['train']['total_iter'] == 250000
    evidence['passed'] = all(checks.values())
    (bank / 'verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2), flush=True)
    if not evidence['passed']:
        raise SystemExit(1)


def main():
    """@luna-comment
    Input: CLI bank/repo/baseline/config and loader limits. Output: actual verification run.
    Key Steps: parse and delegate. Key Params: defaults4workers/4batch/20batches.
    State/Dependencies: CPU verification; no model optimization.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--repo-root')
    parser.add_argument('--baseline')
    parser.add_argument('--config')
    parser.add_argument('--source')
    parser.add_argument('--gallery')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=20)
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
