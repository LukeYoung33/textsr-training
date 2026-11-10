"""Frozen V3L final500K evaluation on the existing Mojito32 and Titians75 sets."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
TRAIN = DATA / 'v3l_s22_scratch_20261107'
KIT = TRAIN / 'kit'
OUT = DATA / 'v3l_final500k_mojito32_titians75_20261109'
sys.path.insert(0, str(KIT))


def digest(path):
    """@luna-comment Input: immutable file. Output: streamed SHA256; no mutations."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    """@luna-comment Input: owned output path/JSON. Output: atomic publication; no partial receipts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def prepare():
    """@luna-comment Freeze final weights, exact prior subsets and cached inputs without preprocessing.
    Input: eight completed runs. Output: hash-bound copies and two selections; reject changed sources.
    """
    assert json.loads((TRAIN / 'COMPLETE.json').read_text())['passed']
    plan = json.loads((TRAIN / 'PLAN.json').read_text())
    bindings = {}
    models = OUT / 'models'
    models.mkdir(parents=True, exist_ok=True)
    for run in plan['runs']:
        receipt = json.loads((TRAIN / (run['name'] + '.complete.json')).read_text())
        assert receipt['exit_code'] == 0
        assert digest(receipt['model']) == receipt['sha256']
        assert digest(run['config']) == run['config_sha256']
        arm = f"v3l{run['ratio']}_{run['loss']}"
        ckpt, config = models / (arm + '.pth'), models / (arm + '.yml')
        shutil.copy2(receipt['model'], ckpt)
        shutil.copy2(run['config'], config)
        bindings[arm] = dict(checkpoint=str(ckpt), checkpoint_sha256=digest(ckpt),
            config=str(config), config_sha256=digest(config), source_checkpoint=receipt['model'], iteration=500000,
            label=f"V3L-{run['ratio']}x | {run['loss']} | 500K", stitcher_sha256=digest(KIT / 'tools/restore_crop8_block256.py'))
        assert bindings[arm]['checkpoint_sha256'] == receipt['sha256']
    mojito = json.loads((DATA / 'mojito280_subset32_20261104/selection.json').read_text())['samples']
    ranges = [(26,30),(36,40),(61,65),(96,100),(111,120),(161,165),(186,190),(226,235),(241,255),(281,290)]
    ids = [f'{n:03d}' for lo, hi in ranges for n in range(lo, hi + 1)]
    camera = {r['id']: r for r in json.loads((DATA / 'titians_two_500k_20261105/selection.json').read_text())['samples']}
    selections = dict(mojito32=mojito, titians75=[camera[i] for i in ids])
    assert len(mojito) == 32 and len(ids) == len(set(ids)) == 75
    for dataset, rows in selections.items():
        root = OUT / dataset
        root.mkdir(exist_ok=True)
        for row in rows:
            assert digest(row['input']) == row['input_sha256']
            assert row['orientation'] in (1, 6)
            assert Path(row.get('lq', row['source'])).is_file()
            if row['orientation'] == 6:
                from PIL import Image
                target = root / 'display' / (row['id'] + '.png')
                target.parent.mkdir(exist_ok=True)
                with Image.open(row['input']) as image:
                    image.transpose(Image.Transpose.ROTATE_270).save(target)
                row['display_input'] = str(target)
        save(root / 'selection.json', dict(samples=rows))
        save(root / 'MODEL_BINDINGS.json', bindings)
        save(root / 'RUN_BINDING.json', dict(selection_sha256=digest(root / 'selection.json'),
             post_core_sha256=digest(KIT / 'tools/native_y_curve_four_images.py')))
        if not (root / 'models').exists():
            (root / 'models').symlink_to(models, target_is_directory=True)
    save(OUT / 'BINDINGS.json', bindings)
    save(OUT / 'READY.json', dict(passed=True, inputs=107, arms=8, outputs=1712, launcher_sha256=digest(__file__)))


def worker(arm):
    """@luna-comment Input: one frozen arm/GPU. Output: both subsets through unchanged restore/post.
    Key Steps: register standalone V3L then delegate accepted workers; strict load/hash gates remain active.
    """
    from tools.sg_wh_training_runtime import bootstrap
    bootstrap()
    from tools import evaluate_v3pre_three_mojito280 as inference
    from tools import evaluate_latest_six_mojito280 as post
    for dataset in ('mojito32', 'titians75'):
        root = OUT / dataset
        inference.OUT = root
        post.OUT = root
        inference.worker(arm)
        for row in json.loads((root / 'selection.json').read_text())['samples']:
            post.post((arm, row))
            print(json.dumps(dict(dataset=dataset, arm=arm, post_id=row['id'])), flush=True)


def publish(states, started):
    """@luna-comment Input: process states/time. Output: PDM monitor and progressive gallery manifests.
    Key Steps: count receipts; expose only produced variants; label incomplete results explicitly.
    """
    bindings = json.loads((OUT / 'BINDINGS.json').read_text())
    runs = []
    for dataset in ('mojito32', 'titians75'):
        root = OUT / dataset
        rows = json.loads((root / 'selection.json').read_text())['samples']
        models = [dict(key='lq', label='原始 LQ'), dict(key='v3_input', label='V3 预处理输入')]
        for arm, binding in bindings.items():
            count = sum(len(list((root / folder / arm / 'receipts').glob('*.json'))) for folder in ('results', 'post'))
            state = states.get(arm, '等待')
            runs.append(dict(name=dataset + ' | ' + binding['label'], current=count, total=len(rows)*2,
                        status=state, details=['原始输出 + 固化后处理；复用 V3 缓存']))
            for suffix, label in [('_raw', '原始输出'), ('_post', '固化后处理')]:
                models.append(dict(key=arm + suffix, label=binding['label'] + ' | ' + label))
        gallery = []
        for row in rows:
            images = dict(v3_input=row.get('display_input', row['input']).replace('/AI_DNS/', 'Y:/', 1))
            for arm in bindings:
                for folder, suffix in [('results', '_raw'), ('post', '_post')]:
                    image = root / folder / arm / 'final' / (row['id'] + '.png')
                    if (root / folder / arm / 'receipts' / (row['id'] + '.json')).exists():
                        images[arm + suffix] = str(image).replace('/AI_DNS/', 'Y:/', 1)
            gallery.append(dict(id=row['id'], name=row['name'], width=row['width'], height=row['height'],
                lq=row.get('lq', row['source']).replace('/AI_DNS/', 'Y:/', 1), images=images))
        save(root / 'manifest.json', dict(title='V3L 500K · ' + dataset, models=models, rows=gallery,
             defaults='lq,v3_input,v3l16_L1_R1_ori_raw,v3l128_L1_R1_ori_raw',
             notes='8组最终500K；256/224/batch8/crop8/smooth16；保留原始与固化Y后处理。缺失选项表示尚未生成，全部完成以COMPLETE.json为准。'))
    done = sum(r['current'] for r in runs)
    eta = (time.time() - started) / done * (1712 - done) if done else None
    save(OUT / 'monitor.json', dict(updated=datetime.now(timezone.utc).isoformat(), runs=runs,
         datasets=[dict(label='总输出', current=done, total=1712, status='全部完成' if (OUT / 'COMPLETE.json').exists() else '处理中',
                        eta=f'{eta / 3600:.1f} 小时（初估）' if eta else None)]))


def audit():
    """@luna-comment Validate every new image hash/geometry/binding before issuing completion."""
    from PIL import Image
    bindings = json.loads((OUT / 'BINDINGS.json').read_text())
    count = 0
    for dataset in ('mojito32', 'titians75'):
        root = OUT / dataset
        rows = json.loads((root / 'selection.json').read_text())['samples']
        selection_hash = digest(root / 'selection.json')
        for arm, binding in bindings.items():
            assert digest(binding['checkpoint']) == binding['checkpoint_sha256']
            for row in rows:
                raw = json.loads((root / 'results' / arm / 'receipts' / (row['id'] + '.json')).read_text())
                post = json.loads((root / 'post' / arm / 'receipts' / (row['id'] + '.json')).read_text())
                assert raw['binding'] == binding and raw['selection_sha256'] == selection_hash
                assert raw['input_sha256'] == row['input_sha256']
                assert raw['restoration']['stitch_policy'] == 'crop8_smooth16'
                assert post['source_sha256'] == raw['final_sha256']
                assert post['core_sha256'] == digest(KIT / 'tools/native_y_curve_four_images.py')
                for folder, expected in [('results', raw['final_sha256']), ('post', post['output_sha256'])]:
                    path = root / folder / arm / 'final' / (row['id'] + '.png')
                    assert digest(path) == expected
                    with Image.open(path) as image:
                        assert image.size == (row['width'], row['height'])
                        image.verify()
                    count += 1
    assert count == 1712
    save(OUT / 'COMPLETE.json', dict(passed=True, outputs=count, finished=time.time()))


def run():
    """@luna-comment Durable four-GPU queue, explicit failure receipts and final automatic audit."""
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    save(OUT / 'CONTROLLER.json', dict(pid=os.getpid(), started=started))
    states = {}
    try:
        if not (OUT / 'READY.json').exists():
            prepare()
        pending = list(json.loads((OUT / 'BINDINGS.json').read_text()))
        active = {}
        while pending or active:
            for arm, (process, gpu) in list(active.items()):
                code = process.poll()
                if code is not None:
                    states[arm] = '完成' if code == 0 else '异常退出'
                    if code:
                        raise RuntimeError(f'{arm} exited {code}')
                    del active[arm]
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            busy = {gpu for process, gpu in active.values()}
            for line in inventory.splitlines():
                gpu, free, util = map(int, line.split(','))
                if not pending or gpu in busy or free < 12000 or util > 10:
                    continue
                arm = pending.pop(0)
                log = OUT / 'logs' / (arm + '.log')
                log.parent.mkdir(exist_ok=True)
                with log.open('ab') as stream:
                    process = subprocess.Popen([sys.executable, '-B', '-u', __file__, 'worker', '--arm', arm], cwd=KIT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1'),
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[arm] = process, gpu
                states[arm] = '运行中'
                save(OUT / 'launch' / (arm + '.json'), dict(pid=process.pid, gpu=gpu))
            publish(states, started)
            if active or pending:
                time.sleep(20)
        audit()
        publish(states, started)
    except Exception as error:
        save(OUT / 'FAILED.json', dict(error=repr(error), at=time.time()))
        if (OUT / 'BINDINGS.json').exists():
            publish(states, started)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--arm')
    args = parser.parse_args()
    worker(args.arm) if args.mode == 'worker' else run()
