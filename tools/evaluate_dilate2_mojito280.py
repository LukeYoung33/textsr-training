"""Final500K original-GT width2 V3M evaluation with accepted Mojito280 inputs."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import fcntl
import html
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_uniform20_steep_scurve import save, digest

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE = DATA / 'v3m_original_dilate2_l1_500k_20261103'
OUT = DATA / 'v3m_original_dilate2_l1_mojito280_20261104'
BASE = DATA / 'v3pre_three_mojito280_20261028'
ARM = 'v3m_dilate2_l1'


def prepare():
    """@luna-comment Input:frozen source config and final500K weights. Output:copied hash-bound model and280 input selection. Key Steps:source binding validation,copy/hash,unique inventory. Key Params:final500000 only. State/Dependencies:immutable previous evaluation/source;isolated OUT."""
    import yaml
    declaration = json.loads((SOURCE/'CONFIG_BINDINGS.json').read_text())['runs'][ARM]
    config = Path(declaration['config'])
    assert digest(config) == declaration['config_sha256']
    cfg = yaml.safe_load(config.read_text())
    assert cfg['train']['total_iter'] == 500000
    checkpoint = SOURCE/'experiments'/cfg['name']/'models/net_g_500000.pth'
    before = digest(checkpoint)
    (OUT/'models').mkdir(parents=True, exist_ok=True)
    model = OUT/'models'/(ARM+'.pth')
    conf = OUT/'models'/(ARM+'.yml')
    shutil.copy2(checkpoint, model)
    shutil.copy2(config, conf)
    assert digest(model) == before == digest(checkpoint)
    binding = dict(config=str(conf), config_sha256=digest(conf), checkpoint=str(model),
                   checkpoint_sha256=before, source_checkpoint=str(checkpoint), iteration=500000,
                   label='V3M 原始GT笔画+2px L1-only 500K',
                   stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    shutil.copy2(BASE/'selection.json', OUT/'selection.json')
    samples = json.loads((OUT/'selection.json').read_text())['samples']
    assert len(samples) == len({s['id'] for s in samples}) == 280
    save(OUT/'MODEL_BINDINGS.json', {ARM: binding})
    save(OUT/'RUN_BINDING.json', dict(selection_sha256=digest(OUT/'selection.json'),
         worker_sha256=digest(ROOT/'tools/evaluate_v3pre_three_mojito280.py'),
         post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'),
         original_gt_training_excludes_all_mojito=True, raw_outputs=280, post_outputs=280))


def progress(started, phase, error=None):
    """@luna-comment Input:start and stage. Output:Chinese progress/phase ETA JSON andHTML. Key Steps:count atomic receipts,separate raw/post rates. Key Params:280 perstage,15s refresh. State/Dependencies:owned results only;no subjective quality scoring."""
    raw = len(list((OUT/'results'/ARM/'receipts').glob('*.json')))
    post = len(list((OUT/'post'/ARM/'receipts').glob('*.json')))
    now = datetime.now(timezone(timedelta(hours=8)))
    elapsed = time.time()-started
    eta = elapsed/max(raw, 1)*(280-raw) if raw and raw < 280 else None
    state = dict(phase=phase, raw=raw, post=post, total_images=280, updated=now.isoformat(),
                 elapsed_seconds=elapsed, raw_eta_seconds=eta, error=error)
    save(OUT/'PROGRESS.json', state)
    text = '<meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>Mojito280 测试进度</title><style>body{font:18px/1.7 sans-serif;margin:30px}progress{width:600px}</style><h1>V3M 原始GT笔画+2px · L1-only · 500K</h1>'
    text += '<p>'+html.escape(phase)+'</p><p>原始输出 '+str(raw)+'/280；固化后处理 '+str(post)+'/280</p><progress max="560" value="'+str(raw+post)+'"></progress>'
    text += '<p>推理阶段预计剩余：'+(f'{eta/60:.1f} 分钟，之后进行后处理及校验' if eta is not None else '采样中或推理阶段已结束')+'</p><p>V3缓存输入；256/224/batch8/crop8/smooth16。此模型训练集不含Mojito280。</p>'
    text += '<p>更新时间：'+now.strftime('%m-%d %H:%M:%S')+'</p>'
    if error:
        text += '<pre>'+html.escape(error)+'</pre>'
    tmp = OUT/'progress.tmp'
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(OUT/'index.html')


def worker():
    """@luna-comment Input:prepared model/input bindings. Output:280 raw PNGs with accepted receipts. Key Steps:rebind accepted worker to own root and strict CUDA inference. Key Params:unchanged accepted stitching. State/Dependencies:assigned GPU only;no training mutation."""
    from tools import evaluate_v3pre_three_mojito280 as accepted
    accepted.OUT = OUT
    accepted.worker(ARM)


def finish(started):
    """@luna-comment Input:280 raw finals. Output:frozen post280 andfull hash/shape audit. Key Steps:accepted CPU post,strict receipt checks and exactinventory. Key Params:Ypost.75/2/24;no GT accuracy claim. State/Dependencies:owned outputs andimmutable source model/input."""
    from PIL import Image
    from tools import evaluate_latest_six_mojito280 as core
    core.OUT = OUT
    samples = json.loads((OUT/'selection.json').read_text())['samples']
    binding = json.loads((OUT/'MODEL_BINDINGS.json').read_text())[ARM]
    run = json.loads((OUT/'RUN_BINDING.json').read_text())
    for entry in samples:
        core.post((ARM, entry))
        progress(started, '固化后处理')
    expected = {s['id'] for s in samples}
    for category in ['results', 'post']:
        for folder, extension in [('final', 'png'), ('receipts', 'json')]:
            assert {p.stem for p in (OUT/category/ARM/folder).glob('*.'+extension)} == expected
    records = []
    for entry in samples:
        identity = entry['id']
        raw = OUT/'results'/ARM/'final'/(identity+'.png')
        post = OUT/'post'/ARM/'final'/(identity+'.png')
        r = json.loads((OUT/'results'/ARM/'receipts'/(identity+'.json')).read_text())
        p = json.loads((OUT/'post'/ARM/'receipts'/(identity+'.json')).read_text())
        assert r['binding'] == binding and r['selection_sha256'] == run['selection_sha256']
        assert r['input_sha256'] == entry['input_sha256'] == digest(entry['input'])
        assert digest(raw) == r['final_sha256'] == p['source_sha256']
        assert digest(post) == p['output_sha256'] and p['core_sha256'] == run['post_core_sha256']
        assert r['restoration']['stitch_policy'] == 'crop8_smooth16'
        for path in [raw, post]:
            with Image.open(path) as im:
                assert im.size == (entry['width'], entry['height'])
                im.verify()
        records.append(dict(id=identity, raw=str(raw), post=str(post), raw_sha256=r['final_sha256'], post_sha256=p['output_sha256']))
    assert digest(binding['checkpoint']) == digest(binding['source_checkpoint']) == binding['checkpoint_sha256']
    assert digest(binding['config']) == binding['config_sha256']
    save(OUT/'OUTPUTS.json', dict(samples=records))
    save(OUT/'AUDIT.json', dict(passed=True, images=280, raw=280, post=280, records=records))
    save(OUT/'COMPLETE.json', dict(passed=True, images=280, outputs=560, at=time.time()))
    progress(started, '全部完成，完整性校验通过')


def run():
    """@luna-comment Input:authorized final500K test. Output:resident singleGPU job/progress/audit. Key Steps:exclusive lock,freeze,least-utilized memorygate,workerexit thenpost/audit. Key Params:6GiB free minimum,no trainer signals. State/Dependencies:NVIDIA inventory;separate process/log."""
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    save(OUT/'CONTROLLER.json', dict(pid=os.getpid(), started=started))
    try:
        if not (OUT/'MODEL_BINDINGS.json').exists():
            prepare()
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        available = [tuple(map(int, row.split(','))) for row in inventory.splitlines()]
        candidates = [row for row in available if row[1] >= 6144]
        assert candidates, 'No GPU has 6GiB free'
        gpu = min(candidates, key=lambda row: row[2])[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        with (OUT/'worker.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(OUT/'LAUNCH.json', dict(pid=child.pid, gpu=gpu))
        while child.poll() is None:
            progress(started, '模型推理中')
            time.sleep(15)
        assert child.returncode == 0, 'Inference failed: see worker.log'
        finish(started)
    except Exception as error:
        save(OUT/'FAILED.json', dict(error=repr(error)))
        progress(started, '异常停止', repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'worker'])
    args = parser.parse_args()
    worker() if args.mode == 'worker' else run()
