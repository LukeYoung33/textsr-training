"""Compute only missing camera comparison columns using accepted geometry/V2.2 caches."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/home/guang.yang/workspace/Real-ESRGAN')
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OLD = DATA/'adaptive_preprocessed_augmented_v1_20261020/training_comparison_20261020'
DOC = OLD/'evaluation/camera_docwarp_robust_20261021'
PRE = DATA/'camera_robust_pre_native_v22_20261021'
OUT = DATA/'camera_compare_2x2_20261022'
HISTORY = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments')
INV = HISTORY.parent/'evaluation/mojito_camera_v3_newdeg_20261016/inventory.json'
ARMS = ('original_v3_pre', 'original_v5_pre', 'original_v3_raw', 'original_v5_raw', 'vosr_v3_pre', 'vosr_v5_pre')


def model_paths(arm):
    """@luna-comment Input:arm. Output:config/final checkpoint. Key Steps:explicit training-family mapping. Key Params:250K only. State/Dependencies:read-only training roots."""
    import yaml
    if arm == 'original_v3_raw':
        train = HISTORY/'v3_native_four_20261015'
        config = train/'configs/02_displayfit_paired.yml'
        checkpoint = train/'runs/02_displayfit_paired/models/net_g_250000.pth'
    elif arm == 'original_v5_raw':
        train = HISTORY/'displayfit_4arm_20261014'
        config = train/'configs/train_raw_h96_250k_20261014.yml'
        checkpoint = train/yaml.safe_load(config.read_text())['name']/'models/net_g_250000.pth'
    else:
        train = OLD if arm.startswith('original') else DATA/'adaptive_vosr_preprocessed_augmented_v1_20261021/training_comparison_20261021'
        number = '02_v3' if '_v3_' in arm else '01_v5'
        config = train/'configs'/(number+('_real.yml' if arm.startswith('original') else '_preprocessed.yml'))
        checkpoint = train/'experiments'/yaml.safe_load(config.read_text())['name']/'models/net_g_250000.pth'
    return config, checkpoint


def run(args):
    """@luna-comment
    Input:arm and disjoint inventory partition. Output:lossless finals and provenance receipts.
    Key Steps:strict checkpoint load; hash-bound cached warp/OCR/V2.2; matched geometry; residual unwarp.
    Key Params:raw historical V5 uses height96; other arms native96; post gamma1.2 for pre arms only.
    State/Dependencies:read-only approved caches/CUDA; isolated resumable results; no detection rerun.
    """
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_offline_comparison import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_offline_camera_matrix import sha, save_json, image_write
    from tools.auto_hx_v3_native import restore
    from tools.run_mojito_v5_height96_20261016 import restore_height96
    from tools.doc_warp_composition import inverse_residual
    from evaluation.adaptive_prepost_frozen_v1_20261020.postprocess import postprocess_bgr
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    config, checkpoint = model_paths(args.arm)
    binding = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), config_sha256=sha(config), evaluator_sha256=sha(__file__), preprocessing='V2_2' if args.arm.endswith('_pre') else 'none')
    if args.arm == 'original_v5_raw' and binding['checkpoint_sha256'] != 'a62e473a694fa6ed045622b9ca7f1377c6ab7e32c4bf048dfcca3c7b8f7210a8':
        raise ValueError('Historical raw V5 checkpoint mismatch')
    options = yaml.safe_load(config.read_text())
    model = build_network(options['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['params'], strict=True)
    rows = json.loads(INV.read_text())['images']
    assert len(rows) == 280
    destination = OUT/'results'/args.arm
    for row in rows[:args.limit or None][args.rank::args.shards]:
        started = time.perf_counter()
        identity = row['id']
        receipt = destination/'receipts'/(identity+'.json')
        if receipt.exists():
            cached = json.loads(receipt.read_text())
            if cached['binding'] != binding or sha(OUT/cached['final']) != cached['final_sha256']:
                raise ValueError('Resume binding mismatch')
            continue
        source_file = OLD/'evaluation/camera_final_20261021/source'/(identity+Path(row['name']).suffix.lower())
        assert sha(source_file) == row['sha256']
        source = cv2.imread(str(source_file))
        assert source is not None
        record_file = DOC/'documents'/source_file.name/'result.json'
        record = json.loads(record_file.read_text())
        assert [record['image_height'], record['image_width']] == list(source.shape[:2])
        prepared = json.loads((PRE/'receipts'/(identity+'.json')).read_text()) if args.arm.endswith('_pre') else None
        if prepared:
            assert prepared['source_sha256'] == row['sha256'] and prepared['corrected_document_record_sha256'] == sha(record_file)
        canvas, owned, stages = source.astype(np.float32), np.zeros(source.shape[:2], bool), []
        for item in sorted(record['instances'], key=lambda v:(-v['confidence'], v['instance_id'])):
            ocr_file = DOC/'ocr'/identity/(f"{item['instance_id']:03d}.json")
            ocr = json.loads(ocr_file.read_text())
            assert ocr['source_sha256'] == row['sha256'] and sha(ocr['warp_file']) == ocr['warp_sha256']
            assert sha(DOC/ocr['valid_file']) == ocr['valid_sha256']
            assert np.allclose(ocr['transform'], item['warp']['image_to_output'])
            original = cv2.imread(ocr['warp_file'])
            inputs, context, cached_pre = original, None, None
            if prepared:
                cached_pre = next(v for v in prepared['instances'] if v['instance_id'] == item['instance_id'])
                assert cached_pre['ocr_sha256'] == sha(ocr_file) and cached_pre['warp_sha256'] == ocr['warp_sha256']
                for key in ('pre', 'context'):
                    assert sha(PRE/cached_pre[key]) == cached_pre[key+'_sha256']
                inputs = cv2.imread(str(PRE/cached_pre['pre']))
                context = json.loads((PRE/cached_pre['context']).read_text())
                assert context['reference_estimator'] == 'adaptive_scurve_robust_pre_v2_2'
            method = restore_height96 if args.arm == 'original_v5_raw' else restore
            prediction, details = method(cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB), ocr['boxes'], model, 'cuda', blend_source=cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            post = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
            if context:
                post, _ = postprocess_bgr(post, context, gamma=1.2)
            valid = cv2.imread(str(DOC/ocr['valid_file']), 0)
            delta, alpha, take, geometry = inverse_residual(source, original, post, item['warp']['image_to_output'], item['quad'], valid, owned)
            canvas += delta
            owned |= take
            stages.append(dict(instance_id=item['instance_id'], ocr_sha256=sha(ocr_file), warp_sha256=ocr['warp_sha256'], restoration=details, cached_pre=cached_pre, geometry=geometry))
        final = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
        assert np.array_equal(final[~owned], source[~owned])
        final_file = destination/'final'/(identity+'.png')
        image_write(final_file, final)
        save_json(receipt, dict(id=identity, arm=args.arm, binding=binding, source_sha256=row['sha256'], source_hw=list(source.shape[:2]), final=str(final_file.relative_to(OUT)), final_sha256=sha(final_file), document_record_sha256=sha(record_file), stages=stages, outside_owned_unchanged=True, seconds=time.perf_counter()-started))
        print(json.dumps(dict(arm=args.arm, id=identity, seconds=round(time.perf_counter()-started, 2))), flush=True)
    save_json(destination/f'COMPLETE.{args.rank}.json', dict(binding=binding, limit=args.limit, shards=args.shards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    run(parser.parse_args())
