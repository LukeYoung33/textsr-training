"""Final five V3M controls on fixed32; validate newly produced outputs only."""
from pathlib import Path
import argparse
import fcntl
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
OUT = DATA/'v3m_five_controls_mojito32_20261104'
ARMS = {
    'v3m_full_noguard': 'V3M · Full 无guard',
    'v3m_mf_uv': 'V3M · L1＋Missing/False＋UV',
    'v3m_mf_uv_mix50': 'V3M · L1＋Missing/False＋UV＋50%GT退化混合',
    'v3m_mf_only': 'V3M · L1＋Missing/False',
    'v3m_uv_only': 'V3M · L1＋UV',
}


def prepare():
    """@luna-comment Input:five frozen source configs/final500K weights. Output:hash-bound copies andfixed32selection. Key Steps:strictconfig/iterationhash,copy,uniqueIDs. Key Params:no queuedL1mix50. State/Dependencies:immutable sourceexperiments;ownedOUT."""
    import yaml
    bindings = {}
    (OUT/'models').mkdir(exist_ok=True)
    for arm, label in ARMS.items():
        folder = 'v3m_aux_factorial_500k_20261103' if arm in ['v3m_mf_only', 'v3m_uv_only'] else 'v3m_loss_ablation_500k_20261103'
        source = DATA/folder
        declared = json.loads((source/'CONFIG_BINDINGS.json').read_text())['runs'][arm]
        config = Path(declared['config'])
        assert digest(config) == declared['config_sha256']
        cfg = yaml.safe_load(config.read_text())
        assert cfg['train']['total_iter'] == 500000
        checkpoint = source/'experiments'/cfg['name']/'models/net_g_500000.pth'
        model = OUT/'models'/(arm+'.pth')
        conf = OUT/'models'/(arm+'.yml')
        before = digest(checkpoint)
        shutil.copy2(checkpoint, model)
        shutil.copy2(config, conf)
        assert digest(model) == before == digest(checkpoint)
        bindings[arm] = dict(config=str(conf), config_sha256=digest(conf), checkpoint=str(model),
            checkpoint_sha256=before, source_checkpoint=str(checkpoint), iteration=500000,
            label=label, stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    selection = DATA/'mojito280_subset32_20261104/selection.json'
    shutil.copy2(selection, OUT/'selection.json')
    samples = json.loads(selection.read_text())['samples']
    assert len(samples) == len({e['id'] for e in samples}) == 32
    save(OUT/'MODEL_BINDINGS.json', bindings)
    save(OUT/'RUN_BINDING.json', dict(selection_sha256=digest(OUT/'selection.json'),
         post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'), outputs=320))


def progress(started, phase, error=None):
    """@luna-comment Input:runtime stage/start. Output:atomic counts/ETA andChinese progresspage. Key Steps:count onlynewreceipts. Key Params:5x32x2=320;15srefresh. State/Dependencies:isolatedOUT,noexistingresultreads."""
    counts = {arm: {k: len(list((OUT/folder/arm/'receipts').glob('*.json'))) for k, folder in [('raw', 'results'), ('post', 'post')]} for arm in ARMS}
    raw = sum(c['raw'] for c in counts.values())
    done = sum(sum(c.values()) for c in counts.values())
    elapsed = time.time()-started
    state = dict(phase=phase, counts=counts, completed=done, total=len(ARMS)*64, elapsed_seconds=elapsed,
                 raw_eta_seconds=elapsed/max(raw, 1)*(len(ARMS)*32-raw) if raw else None, error=error)
    save(OUT/'PROGRESS.json', state)
    body = '<meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>对照测试 Mojito32</title><style>body{font:18px sans-serif;padding:30px}article{padding:12px}</style><h1>对照测试 · Mojito32</h1><p>'+phase+'</p>'
    for arm, label in ARMS.items():
        body += '<article>'+label+'：原始 '+str(counts[arm]['raw'])+'/32，后处理 '+str(counts[arm]['post'])+'/32</article>'
    if error:
        body += '<pre>'+error+'</pre>'
    tmp = OUT/'index.tmp'
    tmp.write_text(body, encoding='utf-8')
    tmp.replace(OUT/'index.html')


def worker(arm):
    """@luna-comment Input:one preparedarm. Output:32 raw plus32 acceptedpostPNGs. Key Steps:acceptedstrictCUDAworker thenfrozenCPUpost. Key Params:256/224/batch8/crop8/smooth16,Ypost.75/2/24. State/Dependencies:one assignedGPU;isolatednewoutputs."""
    from tools import evaluate_v3pre_three_mojito280 as accepted
    from tools import evaluate_latest_six_mojito280 as core
    accepted.OUT = OUT
    core.OUT = OUT
    accepted.worker(arm)
    for entry in json.loads((OUT/'selection.json').read_text())['samples']:
        core.post((arm, entry))


def audit():
    """@luna-comment Input:new320outputs only. Output:hash/geometry/inputidentity audit anddeliveryinventory. Key Steps:exact32IDs/arm,bindings/sourcehashes,acceptedstitch/postchecks. Key Params:old18optionsneverrevalidated. State/Dependencies:PIL andnewreceipts only."""
    from PIL import Image
    samples = json.loads((OUT/'selection.json').read_text())['samples']
    bindings = json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    run = json.loads((OUT/'RUN_BINDING.json').read_text())
    ids = {s['id'] for s in samples}
    records = []
    for arm, binding in bindings.items():
        assert digest(binding['checkpoint']) == digest(binding['source_checkpoint']) == binding['checkpoint_sha256']
        assert digest(binding['config']) == binding['config_sha256']
        for folder in ['results', 'post']:
            assert {p.stem for p in (OUT/folder/arm/'final').glob('*.png')} == ids
            assert {p.stem for p in (OUT/folder/arm/'receipts').glob('*.json')} == ids
        for entry in samples:
            identity = entry['id']
            raw = OUT/'results'/arm/'final'/(identity+'.png')
            post = OUT/'post'/arm/'final'/(identity+'.png')
            r = json.loads((OUT/'results'/arm/'receipts'/(identity+'.json')).read_text())
            p = json.loads((OUT/'post'/arm/'receipts'/(identity+'.json')).read_text())
            assert r['binding'] == binding and r['selection_sha256'] == run['selection_sha256']
            assert r['input_sha256'] == entry['input_sha256']
            assert r['restoration']['stitch_policy'] == 'crop8_smooth16'
            assert digest(raw) == r['final_sha256'] == p['source_sha256']
            assert digest(post) == p['output_sha256'] and p['core_sha256'] == run['post_core_sha256']
            for path in [raw, post]:
                with Image.open(path) as im:
                    assert im.size == (entry['width'], entry['height'])
                    im.verify()
            records.append(dict(id=identity, arm=arm, raw=str(raw), post=str(post),
                 raw_sha256=r['final_sha256'], post_sha256=p['output_sha256']))
    save(OUT/'OUTPUTS.json', dict(samples=records))
    save(OUT/'AUDIT.json', dict(passed=True, images_per_arm=32, arms=len(ARMS), outputs=len(ARMS)*64, old_results_rechecked=False))
    save(OUT/'COMPLETE.json', dict(passed=True, outputs=len(ARMS)*64, at=time.time()))


def run():
    """@luna-comment Input:authorizedfivearmfixed32test. Output:residentparallelrun/completionaudit. Key Steps:exclusivefreeze,utilizationordered6GiBgate,maxoneownedworker/GPU,join/check/audit. Key Params:4workersmax,15spoll. State/Dependencies:existingtrainersnot signaled."""
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    save(OUT/'CONTROLLER.json', dict(pid=os.getpid(), started=started))
    try:
        if not (OUT/'MODEL_BINDINGS.json').exists():
            prepare()
        pending = list(ARMS)
        active = {}
        while pending or active:
            for arm, (p, gpu) in list(active.items()):
                if p.poll() is not None:
                    assert p.returncode == 0, (arm, p.returncode)
                    del active[arm]
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            rows = sorted([tuple(map(int, line.split(','))) for line in inventory.splitlines()], key=lambda row: row[2])
            busy = {gpu for p, gpu in active.values()}
            for gpu, free, util in rows:
                if not pending or gpu in busy or free < 6144:
                    continue
                arm = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
                (OUT/'logs').mkdir(exist_ok=True)
                with (OUT/'logs'/(arm+'.log')).open('ab') as log:
                    p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--arm', arm], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                save(OUT/'launch'/(arm+'.json'), dict(pid=p.pid, gpu=gpu))
                active[arm] = p, gpu
            progress(started, '并行推理与固化后处理')
            time.sleep(15)
        progress(started, '仅校验新增'+str(len(ARMS)*64)+'张输出')
        audit()
        progress(started, '全部完成')
    except Exception as error:
        save(OUT/'FAILED.json', dict(error=repr(error)))
        progress(started, '异常停止', repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--arm', choices=list(ARMS))
    args = parser.parse_args()
    worker(args.arm) if args.mode == 'worker' else run()
