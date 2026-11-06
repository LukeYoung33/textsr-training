"""Reuse Hisense68 inputs and historical outputs; evaluate one frozen R1_ori checkpoint."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA / 'hisense68_filtered_ori_430k_20261106'
OLD = DATA / 'hisense_l772_two_models_20261105'
TRAIN = DATA / 'filtered_trusted_no_mojito_three_500k_20261105'
ARM = 'filtered_ori_430k'


def save(path, data):
    """@luna-comment Input:owned JSON path/data. Output:atomic metadata. Key Steps:mkdir/write/replace.
    Key Params:UTF8. State/Dependencies:owned output only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def prepare():
    """@luna-comment Input:verified430K model/config and accepted68 cached inputs.
    Output:frozen bindings. Key Steps:hash/copy model, validate cache receipts, reuse selection.
    Key Params:no preprocessing/OCR. State/Dependencies:training and historical data read-only.
    """
    from tools.run_uniform20_steep_scurve import digest
    declared = json.loads((TRAIN / 'CONFIG_BINDINGS.json').read_text())['runs']['ori']
    receipt = json.loads((TRAIN / 'experiments' / declared['name'] / 'training_states/430000.verified.json').read_text())
    assert receipt['iter'] == 430000 and digest(receipt['model']) == receipt['model_sha256']
    assert digest(receipt['state']) == receipt['state_sha256']
    assert digest(declared['config']) == declared['config_sha256']
    (OUT / 'models').mkdir(parents=True, exist_ok=True)
    model, config = OUT / 'models/model.pth', OUT / 'models/config.yml'
    shutil.copy2(receipt['model'], model)
    shutil.copy2(declared['config'], config)
    binding = dict(config=str(config), config_sha256=digest(config), checkpoint=str(model),
                   checkpoint_sha256=digest(model), iteration=430000, source_checkpoint=receipt['model'])
    shutil.copy2(OLD / 'selection.json', OUT / 'selection.json')
    rows = json.loads((OUT / 'selection.json').read_text())['samples']
    assert len(rows) == 68
    for row in rows:
        cached = OLD / 'pre_receipts' / (row['id'] + '.json')
        r = json.loads(cached.read_text())
        assert digest(row['input']) == r['input_sha256']
        target = OUT / 'pre_receipts' / (row['id'] + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, target)
    save(OUT / 'MODEL_BINDINGS.json', {ARM: binding})
    save(OUT / 'RUN_BINDING.json', dict(iteration=430000, samples=68, reused_preprocessing=68,
         new_ocr=0, old_results_rechecked=False, selection_sha256=digest(OUT / 'selection.json'),
         stitcher_sha256=digest(ROOT / 'tools/restore_crop8_block256.py'),
         post_sha256=digest(ROOT / 'tools/native_y_curve_four_images.py')))


def worker(rank):
    """@luna-comment Input:disjoint0/1 shard. Output:34 raw and34 post images.
    Key Steps:delegate unchanged accepted worker with new output root. Key Params:256/224/crop8/smooth16.
    State/Dependencies:one assigned CUDA GPU; original data read-only.
    """
    from tools import train_v3_native_regions as native
    native.ROOT = Path('/home/guang.yang/workspace/Real-ESRGAN')
    from tools import evaluate_zoom_plate_two_models as core
    core.OUT = OUT
    core.worker(ARM, rank)


def run():
    """@luna-comment Input:authorized68image evaluation. Output:audited136 new outputs and progress.
    Key Steps:exclusive lock,freeze,spawn two shards,monitor/check receipts. Key Params:GPU0/1,no trainer signals.
    State/Dependencies:owned child jobs and output metadata only.
    """
    import fcntl
    from PIL import Image
    from tools.run_uniform20_steep_scurve import digest
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (OUT / 'MODEL_BINDINGS.json').exists():
        prepare()
    started = time.time()
    children = []
    for rank in range(2):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (OUT / f'worker{rank}.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--rank', str(rank)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        children.append(child)
    save(OUT / 'CONTROLLER.json', dict(pid=os.getpid(), workers=[p.pid for p in children]))
    while True:
        done = len(list((OUT / 'receipts' / ARM).rglob('*.json')))
        elapsed = time.time() - started
        save(OUT / 'PROGRESS.json', dict(completed=done, total=68, elapsed_seconds=elapsed,
             eta_seconds=elapsed / done * (68 - done) if done else None))
        print(f'{done}/68 elapsed={elapsed:.0f}s', flush=True)
        failed = [(p.pid, p.poll()) for p in children if p.poll() not in (None, 0)]
        if failed:
            save(OUT / 'FAILED.json', dict(processes=failed))
            raise RuntimeError(failed)
        if all(p.poll() == 0 for p in children):
            break
        time.sleep(10)
    binding = json.loads((OUT / 'MODEL_BINDINGS.json').read_text())[ARM]
    for row in json.loads((OUT / 'selection.json').read_text())['samples']:
        r = json.loads((OUT / 'receipts' / ARM / (row['id'] + '.json')).read_text())
        assert r['binding'] == binding and r['restoration']['stitch_policy'] == 'crop8_smooth16'
        for folder, key in [('model', 'model_sha256'), ('post', 'post_sha256')]:
            path = OUT / folder / ARM / (row['id'] + '.png')
            assert digest(path) == r[key]
            with Image.open(path) as image:
                assert image.size == (row['width'], row['height'])
                image.verify()
    save(OUT / 'COMPLETE.json', dict(passed=True, samples=68, outputs=136, old_results_rechecked=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    run() if args.mode == 'run' else worker(args.rank)
