"""Evaluate the V5 frozen-preprocessed H96 checkpoint on the matched camera batch."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/mojito_camera_v3_newdeg_20261016')
OUT = BASE.with_name('mojito_camera_v5_strict_h96_20261016')
TRAIN = Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/experiments/displayfit_4arm_20261014')
CHECKPOINT = TRAIN/'displayfit_v5_r1_bglf_uv_strict_combined_h96_250k_20261014/models/net_g_250000.pth'
EXPECTED_SHA = 'de3af40f67646e5335cfdfe6b794ea75ce6f666e1396fac305550a12a3fc9403'


def worker(rank, shards, limit):
    """@luna-comment
    Input: shard index/count and verified completed V3 camera assets.
    Output: native V5 PNGs, previews and atomic per-image/checkpoint receipts.
    Key Steps: strict V5 load, hash-match preprocessed RGB, shared native restore,
    preserve original outside boxes, record matched-input/source evidence.
    Key Params: final250K, canvas96x480/stride64x448/batch8/16px full-box feather.
    State/Dependencies: read-only V3 assets, original training YAML, isolated CUDA process.
    """
    import hashlib
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3_native_regions import bootstrap
    bootstrap()
    importlib.import_module('realesrgan.archs.basic_text_line_r4_student_y4uv_v5_arch')
    from basicsr.archs import build_network
    from tools.auto_hx_v3_native import restore, sha, save_json
    assert sha(CHECKPOINT) == EXPECTED_SHA
    settings = yaml.safe_load((TRAIN/'configs/train_strict_combined_h96_250k_20261014.yml').read_text())
    assert settings['network_g']['type'] == 'BasicTextLineR4StudentY4UVV5'
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
        processed = cv2.imread(str(BASE/'preprocessed'/(source.stem+'.png')))
        assert bgr is not None and processed is not None and bgr.shape == processed.shape
        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        model_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        input_sha = hashlib.sha256(model_rgb.tobytes()).hexdigest()
        assert input_sha == row['preprocessing']['model_input_pixel_sha256']
        result, details = restore(model_rgb, row['text_boxes'], net, 'cuda', blend_source=original_rgb)
        output = OUT/'final'/(source.stem+'.png')
        assert cv2.imwrite(str(output), cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        factor = min(1, 960/max(result.shape[:2]))
        preview = cv2.resize(result, (round(result.shape[1]*factor), round(result.shape[0]*factor)), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(OUT/'previews'/(row['id']+'_result.jpg')), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        item = dict(id=row['id'], name=source.name, source=str(source), source_sha256=row['source_sha256'],
                    output=str(output), output_sha256=sha(output), width=row['width'], height=row['height'],
                    text_boxes=row['text_boxes'], detection_sha256=row['detection_sha256'],
                    preprocessing=row['preprocessing'], model_input_pixel_sha256=input_sha,
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
    Key Steps: launch four owned workers on GPU0/3, monitor explicit failures,
    validate unique outputs and hashes, atomically publish pass only after all exits0.
    Key Params: four shards, two CPU threads per worker, no detector/preprocessor rerun.
    State/Dependencies: detached coordinator; isolated logs/output, no unrelated jobs touched.
    """
    from tools.auto_hx_v3_native import save_json, sha
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rank in range(4):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(0 if rank%2 == 0 else 3),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        with (OUT/f'worker_{rank}.log').open('ab') as stream:
            child = subprocess.Popen([sys.executable, '-u', __file__, '--mode', 'worker', '--rank', str(rank), '--shards', '4'],
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
        assert row['preprocessing']['model_input_pixel_sha256'] == result['model_input_pixel_sha256']
        assert result['output_sha256'] == sha(result['output'])
    save_json(OUT/'summary.json', dict(status='passed', images=280, checkpoint=str(CHECKPOINT),
              checkpoint_sha256=EXPECTED_SHA, matched_preprocessing=True, results=records))
    save_json(OUT/'status.json', dict(status='passed', completed=280, expected=280,
              exit_codes=states, completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    print('PASS all280 matched V5 results', flush=True)


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
