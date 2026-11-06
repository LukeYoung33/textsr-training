"""Evaluate the current V3 using user-supplied basic full-frame preprocessing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016')
OUT = BASE.with_name('mojito_camera_v3_basic_base_20261016')
PREPROCESSED = BASE.with_name('mojito_fullframe8_225k_20261016')/'basic'
TRAIN = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/v3_native_four_20261015')
CHECKPOINT = TRAIN/'runs/03_displayfit_degraded/models/net_g_250000.pth'
EXPECTED_SHA = '115fdaf386447c03d61681710149620555e178b8921b5f0bd2892f81e25a6937'


def worker(rank, shards, limit):
    """@luna-comment
    Input: shard index/count and verified completed V3 camera assets.
    Output: native V3 PNGs, previews and atomic per-image/checkpoint receipts.
    Key Steps: strict V3 load, directly decode and hash supplied basic RGB, shared native restore,
    preserve provided preprocessing pixels outside boxes, record supplied-input/source evidence.
    Key Params: final250K, canvas96x480/stride64x448/batch8/16px full-box feather.
    State/Dependencies: read-only basic images/original assets, matching training YAML, isolated CUDA process.
    """
    import hashlib
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore, sha, save_json
    assert sha(CHECKPOINT) == EXPECTED_SHA
    settings = yaml.safe_load((TRAIN/'configs/03_displayfit_degraded.yml').read_text())
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    net = build_network(settings['network_g']).cuda().eval()
    net.load_state_dict(torch.load(CHECKPOINT, map_location='cpu')['params'], strict=True)
    upstream = json.loads((BASE/'summary.json').read_text())
    assert upstream['status'] == 'passed' and upstream['images'] == 280
    for name in ('final', 'previews', 'receipts'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    records = []
    for row in upstream['results'][:limit][rank::shards]:
        started = time.perf_counter()
        source = Path(row['source'])
        assert sha(source) == row['source_sha256']
        bgr = cv2.imread(str(source))
        preprocessed_path = PREPROCESSED/(source.stem+'.png')
        preprocessed_sha = sha(preprocessed_path)
        processed = cv2.imread(str(preprocessed_path))
        assert bgr is not None and processed is not None and bgr.shape == processed.shape
        model_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        input_sha = hashlib.sha256(model_rgb.tobytes()).hexdigest()
        assert list(model_rgb.shape[:2]) == [row['height'], row['width']]
        result, details = restore(model_rgb, row['text_boxes'], net, 'cuda', blend_source=model_rgb)
        output = OUT/'final'/(source.stem+'.png')
        assert cv2.imwrite(str(output), cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        factor = min(1, 960/max(result.shape[:2]))
        preview = cv2.resize(result, (round(result.shape[1]*factor), round(result.shape[0]*factor)), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(OUT/'previews'/(row['id']+'_result.jpg')), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        processed_preview = cv2.resize(processed, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(OUT/'previews'/(row['id']+'_preprocessed.jpg')), processed_preview)
        item = dict(id=row['id'], name=source.name, source=str(source), source_sha256=row['source_sha256'],
                    output=str(output), output_sha256=sha(output), width=row['width'], height=row['height'],
                    text_boxes=row['text_boxes'], detection_sha256=row['detection_sha256'],
                    preprocessing=dict(mode='provided_image', source=str(preprocessed_path), source_sha256=preprocessed_sha),
                    model_input_pixel_sha256=input_sha, blend_base='provided_preprocessing',
                    blend_base_pixel_sha256=input_sha,
                    checkpoint=str(CHECKPOINT), checkpoint_sha256=EXPECTED_SHA,
                    seconds=time.perf_counter()-started, **details)
        save_json(OUT/'receipts'/(source.stem+'.json'), item)
        records.append(item)
        save_json(OUT/f'progress_{rank}.json', dict(completed=len(records), expected=len(upstream['results'][:limit][rank::shards])))
        print('RESTORED', row['id'], source.name, round(item['seconds'], 2), flush=True)
    save_json(OUT/f'summary_{rank}.json', dict(status='passed', images=len(records)))


def supervise():
    """@luna-comment
    Input: completed first-image smoke and matched280-image source batch.
    Output: persistent worker status and hashed all280 completion summary.
    Key Steps: launch three owned workers on GPU1/2/3, monitor explicit failures,
    validate unique outputs and hashes, atomically publish pass only after all exits0.
    Key Params: three shards, two CPU threads per worker, no detector/preprocessor rerun.
    State/Dependencies: detached coordinator; isolated logs/output, no unrelated jobs touched.
    """
    from tools.auto_hx_v3_native import save_json, sha
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rank in range(3):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank+1),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        with (OUT/f'worker_{rank}.log').open('ab') as stream:
            child = subprocess.Popen([sys.executable, '-u', __file__, '--mode', 'worker', '--rank', str(rank), '--shards', '3'],
                                     cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        jobs.append(child)
    save_json(OUT/'launch.json', dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), pids=[p.pid for p in jobs]))
    while True:
        states = [p.poll() for p in jobs]
        failed = any(code not in (None, 0) for code in states)
        save_json(OUT/'status.json', dict(status='failed' if failed else 'running',
                  completed=len(list((OUT/'receipts').glob('*.json'))), expected=280,
                  exit_codes=states, updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        if failed:
            for child in jobs:
                if child.poll() is None:child.terminate()
            raise RuntimeError(states)
        if all(code == 0 for code in states):break
        time.sleep(15)
    upstream = json.loads((BASE/'summary.json').read_text())
    records = [json.loads((OUT/'receipts'/(Path(row['name']).stem+'.json')).read_text()) for row in upstream['results']]
    assert len(records) == len({r['id'] for r in records}) == 280
    for row, result in zip(upstream['results'], records):
        assert row['source_sha256'] == result['source_sha256'] and result['outside_unchanged']
        assert result['blend_base'] == 'provided_preprocessing'
        assert result['blend_base_pixel_sha256'] == result['model_input_pixel_sha256']
        assert sha(result['preprocessing']['source']) == result['preprocessing']['source_sha256']
        assert row['text_boxes'] == result['text_boxes']
        assert [row['width'], row['height']] == [result['width'], result['height']]
        assert result['output_sha256'] == sha(result['output'])
    save_json(OUT/'summary.json', dict(status='passed', images=280, checkpoint=str(CHECKPOINT),
              checkpoint_sha256=EXPECTED_SHA, preprocessing_source=str(PREPROCESSED), blend_base='provided_preprocessing', results=records))
    save_json(OUT/'status.json', dict(status='passed', completed=280, expected=280,
              exit_codes=states, completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print('PASS all280 matched V3 results', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'run'], default='run')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=280)
    args = parser.parse_args()
    if args.mode == 'run':supervise()
    else:
        assert 0 <= args.rank < args.shards
        worker(args.rank, args.shards, args.limit)
