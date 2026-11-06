"""Durable measured-ETA timer, data gates and three pretrain-to-finetune chains."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timedelta
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_v3pre_training import OUT, DATA, TZ, sha, save

REPO = '/home/guang.yang/workspace/Real-ESRGAN'
ARMS = ('full', 'text15', 'adapt_l1')
LABELS = dict(full='V3M Full', text15='V3M Text x1.5', adapt_l1='V3M adaptive-L1 cap4')


def monitor(phase, **extra):
    """@luna-comment Input:controller phase/details. Output:combined live progress/loss dashboard.
    Key Steps:read preparation/child states, embed distinct arm histories and ETA timer.
    Key Params:30s refresh/no HTTP dependency. State/Dependencies:observer-owned dashboard.
    """
    from tools.offline_loss_charts import read_loss_history, render_loss_charts
    path = OUT / 'TRAINING_STATUS.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    if not phase.endswith('failed'):
        state.pop('error', None)
    state.update(phase=phase, updated_at=datetime.now(TZ).isoformat(), **extra)
    save(path, state)
    prep = json.loads((OUT / 'STATUS.json').read_text()) if (OUT / 'STATUS.json').exists() else {}
    progress = {}
    histories = {}
    models = {}
    for stage in ('pretrain', 'finetune'):
        bank = OUT / stage / 'bank256'
        progress[stage] = {name: json.loads((bank/name).read_text()) for name in ('progress.json', 'COMPLETE.json', 'LINE_REGIONS_COMPLETE.json') if (bank/name).exists()}
        for arm in ARMS:
            logfile = OUT / stage / 'training/logs' / (arm+'.log')
            history = read_loss_history(logfile)
            if history:
                histories[stage+' | '+LABELS[arm]] = history
                with logfile.open('rb') as stream:
                    stream.seek(max(0, logfile.stat().st_size-32768))
                    tail = stream.read().decode('utf-8', errors='replace')
                estimates = re.findall(r'eta:\s*([^,\n]+)', tail)
                models[stage+'_'+arm] = dict(label=LABELS[arm], iteration=history[-1]['step'],
                    total_iter=250000 if stage == 'pretrain' else 150000,
                    eta=estimates[-1] if estimates else 'warming up')
    state = dict(preparation=prep, training=state, banks=progress, models=models)
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>V3-preprocessed V3M training</title><style>body{font:15px system-ui;margin:24px;background:#f4f7fa}pre{white-space:pre-wrap}</style><h1>V3 preprocessing / three V3M training chains</h1><button onclick="location.reload()">Refresh</button><p>Pretrain250K then own finetune150K. Full / Text x1.5 / adaptive-L1cap4. All timestamps UTC+08.</p><pre>'
    (OUT / 'dashboard.html').write_text(page+html.escape(json.dumps(state, indent=2))+'</pre>'+render_loss_charts(histories), encoding='utf-8')


def sidecars(bank, view, workers=12):
    """@luna-comment Input:completed bank/source view. Output:all source-bound line sidecars.
    Key Steps:parallel per-parent geometry replay then verified complete marker.
    Key Params:bitpacked union masks; source shards immutable. State/Dependencies:owned bank.
    """
    from tools.v3pre_line_regions import build
    jobs = []
    for split in ('train', 'val'):
        rows = [json.loads(line) for line in (view/'manifests'/(split+'.jsonl')).read_text().splitlines() if line.strip()]
        lookup = {(r['dataset'], r['sample_id']):r for r in rows}
        manifest = json.loads((bank/(split+'.json')).read_text())
        jobs += [(str(bank), shard, lookup[shard['dataset'], shard['sample_id']]) for shard in manifest['shards']]
    started = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 20 == 0:
                save(bank/'line_regions/progress.json', dict(done=len(records), total=len(jobs), seconds=time.time()-started))
    save(bank/'LINE_REGIONS_COMPLETE.json', dict(passed=True, count=len(records), seconds=time.time()-started,
         source_build_id=json.loads((bank/'COMPLETE.json').read_text())['build_id'], records=records))


def bank_stage(stage):
    """@luna-comment Input:completed preprocessed stage. Output:complete bank+line sidecars.
    Key Steps:accepted builder none-mode, actual source hash/shape checks; sidecar gate.
    Key Params:12workers, same recipes/sampling. State/Dependencies:source completion required.
    """
    bank, view = OUT/stage/'bank256', OUT/stage/'source_view'
    if not (bank/'COMPLETE.json').exists():
        with (OUT/(stage+'_bank.log')).open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(ROOT/'tools/build_vosr_block256_dataset.py'),
                '--source', str(view), '--output', str(bank), '--workers', '12', '--preprocess-mode', 'none'],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    if not (bank/'LINE_REGIONS_COMPLETE.json').exists():
        sidecars(bank, view)
    return stage


def configs():
    """@luna-comment Input:confirmed250K/150K matrix. Output:shared fresh init and six versioned YAMLs.
    Key Steps:retain historical Full/Text recipes and schedules; adaptive changes only
    model/dataset/base criterion. Key Params:seed20260828/batch4/V3M4/8/16/32.
    State/Dependencies:CPU torch/bootstrap; no optimizer/checkpoint resume.
    """
    import torch
    import yaml
    from tools.train_v3pre_area import bootstrap
    bootstrap(REPO)
    from basicsr.archs import build_network
    reference = DATA/'v3m_vosr2u_20261026/training_20261026/configs'
    baseline = yaml.safe_load((reference/'full_v3m.yml').read_text())
    initial = OUT/'initial/v3m_shared.pth'
    initial.parent.mkdir(exist_ok=True)
    if not initial.exists():
        torch.manual_seed(20260828)
        model = build_network(copy.deepcopy(baseline['network_g']))
        torch.save(dict(params=model.state_dict(), initialization='fresh_seeded_20260828'), initial)
    records = {}
    for stage in ('pretrain', 'finetune'):
        run = OUT/stage/'training'
        for gpu, arm in enumerate(ARMS):
            path = (reference/('text15_v3m.yml' if arm == 'text15' else 'full_v3m.yml')) if stage == 'pretrain' else DATA/'six_models_mojito_finetune_20261027/training/configs'/('text15_v3m.yml' if arm == 'text15' else 'full_v3m.yml')
            cfg = yaml.safe_load(path.read_text())
            cfg['v3pre_target_domain'] = ('VOSR2u train / older GT_VOSR583 validation' if stage == 'pretrain' else 'Mojito VOSR teacher380/67 scene split')
            cfg['name'] = 'v3pre_'+stage+'_'+arm+'_20261027'
            cfg['path'].update(pretrain_network_g=str(initial if stage == 'pretrain' else OUT/'pretrain/training/experiments'/('v3pre_pretrain_'+arm+'_20261027')/'models/net_g_250000.pth'), strict_load_g=True, resume_state=None, experiments_root=str(run/'experiments'))
            cfg['dist_params']['port'] = 33500+gpu+(10 if stage == 'finetune' else 0)
            cfg['datasets']['train']['num_worker_per_gpu'] = 2
            for split in ('train', 'val'):
                bank = OUT/stage/'bank256'
                cfg['datasets'][split].update(root=str(bank), manifest=str(bank/('train_balanced.json' if split == 'train' else 'val.json')))
            if arm == 'adapt_l1':
                cfg['model_type'] = 'V3PreAreaModel'
                cfg['train']['y4uv_native_opt'] = dict(type='AreaAdaptiveNativeL1', cap=4., loss_weight=1., eps=1e-6)
                for split in ('train', 'val'):
                    cfg['datasets'][split]['type'] = 'V3PreAreaDataset'
            assert cfg['train']['total_iter'] == (250000 if stage == 'pretrain' else 150000)
            assert cfg['datasets']['train']['batch_size_per_gpu'] == 4
            target = run/'configs'/(arm+'.yml')
            target.parent.mkdir(parents=True, exist_ok=True)
            content = yaml.safe_dump(cfg, sort_keys=False)
            if target.exists():
                assert target.read_text() == content
            else:
                target.write_text(content)
            records[stage+'_'+arm] = dict(config=str(target), sha256=sha(target), gpu=gpu, iterations=cfg['train']['total_iter'])
    save(OUT/'CONFIG_BINDINGS.json', dict(shared_initial_sha256=sha(initial), configs=records))
    for stage in ('pretrain', 'finetune'):
        root = OUT/stage/'training/configs'
        full = yaml.safe_load((root/'full.yml').read_text())['train']
        adaptive = yaml.safe_load((root/'adapt_l1.yml').read_text())['train']
        assert {k for k in full if full[k] != adaptive[k]} == {'y4uv_native_opt'}


def pilot():
    """@luna-comment Input:first completed preprocessed parents. Output:timed pilot bank and CUDA checks.
    Key Steps:select32train+4diagnostic val distinct parents, build exact pipeline,
    region sidecars and three actual CUDA smokes; measure data-stage throughput.
    Key Params:pilot is engineering-only, not a validation-quality dataset.
    State/Dependencies:does not save training weights or launch formal runs.
    """
    import yaml
    ready = sorted((OUT/'pretrain/source_view/receipts/train').glob('*/*.json'))
    assert len(ready) >= 36
    view, bank = OUT/'pilot/source_view', OUT/'pilot/bank256'
    training = ready[:32]
    identities = {json.loads(p.read_text())['row']['source_identity'] for p in training}
    validation = [p for p in ready[32:] if json.loads(p.read_text())['row']['source_identity'] not in identities][:4]
    assert len(validation) == 4, 'Need four engineering validation parents from other capture groups'
    for split, selected in [('train', training), ('val', validation)]:
        path = view/'manifests'/(split+'.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(json.loads(p.read_text())['row'])+'\n' for p in selected))
    if not (OUT/'PILOT_TIMING.json').exists():
        started = time.time()
        with (OUT/'pilot_bank.log').open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(ROOT/'tools/build_vosr_block256_dataset.py'), '--source', str(view), '--output', str(bank), '--workers', '4', '--preprocess-mode', 'none'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        bank_seconds = time.time()-started
        started = time.time()
        sidecars(bank, view, 4)
        sidecar_seconds = time.time()-started
        complete = json.loads((bank/'COMPLETE.json').read_text())
        size = sum(v['bytes'] for v in complete['splits'].values())
        save(OUT/'PILOT_TIMING.json', dict(passed=True, parents=36, workers=4, bank_seconds=bank_seconds,
             bank_bytes=size, sidecar_seconds=sidecar_seconds, note='engineering subset; original train parents used only for smoke'))
    configs()
    for arm in ARMS:
        cfg = yaml.safe_load((OUT/'pretrain/training/configs'/(arm+'.yml')).read_text())
        for split in ('train', 'val'):
            cfg['datasets'][split].update(root=str(bank), manifest=str(bank/('train_balanced.json' if split == 'train' else 'val.json')))
        cfg['path']['experiments_root'] = str(OUT/'pilot/experiments')
        target = OUT/'pilot/configs'/(arm+'.yml')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        with (OUT/('pilot_'+arm+'.log')).open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(ROOT/'tools/smoke_v3pre_training.py'), str(target), str(OUT/'pilot/smoke'/(arm+'.json'))], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'), stdout=stream, stderr=subprocess.STDOUT, check=True)
    save(OUT/'PILOT_COMPLETE.json', dict(passed=True, completed_at=datetime.now(TZ).isoformat()))


def launch(stage, arm, gpu):
    """@luna-comment Input:ready stage/arm/GPU. Output:detached child with hash-bound launch receipt.
    Key Steps:require bank/sidecar/config/init/smoke; no output overwrite; invoke
    explicit bootstrap. Key Params:strict weights-only initialization, no resume.
    State/Dependencies:controller owns child PIDs; existing jobs untouched.
    """
    import yaml
    bank, run = OUT/stage/'bank256', OUT/stage/'training'
    assert json.loads((bank/'COMPLETE.json').read_text())['complete']
    assert json.loads((bank/'LINE_REGIONS_COMPLETE.json').read_text())['passed']
    assert json.loads((OUT/'PILOT_COMPLETE.json').read_text())['passed']
    config = run/'configs'/(arm+'.yml')
    cfg = yaml.safe_load(config.read_text())
    binding = json.loads((OUT/'CONFIG_BINDINGS.json').read_text())
    assert sha(config) == binding['configs'][stage+'_'+arm]['sha256']
    initial = Path(cfg['path']['pretrain_network_g'])
    assert initial.is_file()
    if stage == 'pretrain':
        assert sha(initial) == binding['shared_initial_sha256']
    else:
        with (OUT/('finetune_smoke_'+arm+'.log')).open('ab') as stream:
            subprocess.run([sys.executable, '-u', str(ROOT/'tools/smoke_v3pre_training.py'),
                str(config), str(run/'smoke'/(arm+'.json'))], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='2'),
                stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert json.loads((run/'smoke'/(arm+'.json')).read_text())['config_sha256'] == sha(config)
    free = int(subprocess.check_output(['nvidia-smi', '--id='+str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    assert free > 4096
    assert not (run/'experiments'/cfg['name']).exists()
    command = [sys.executable, '-u', str(ROOT/'tools/train_v3pre_area.py'), '--repo-root', REPO, '-opt', str(config)]
    path = run/'logs'/(arm+'.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        proc = subprocess.Popen(command, cwd=run, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
    save(run/'launch'/(arm+'.json'), dict(pid=proc.pid, command=command, gpu=gpu, stage=stage, arm=arm,
        started_at=datetime.now(TZ).isoformat(), config_sha256=sha(config), initial_sha256=sha(initial)))
    return proc


def controller():
    """@luna-comment Input:measured pilot/preparation progress. Output:durable one-shot timer and six gated trainings.
    Key Steps:estimate full bank time, schedule UTC+08 start, wait for actual data,
    smoke real banks, launch three250K, own exit0/final checkpoint gates each150K.
    Key Params:readiness wins over ETA; failed stages stop and remain explicit.
    State/Dependencies:single controller lock, own child processes, persistent dashboard.
    """
    import fcntl
    lock = (OUT/'training_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not (OUT/'PILOT_COMPLETE.json').exists():
            monitor('engineering_pilot')
            pilot()
        timing = json.loads((OUT/'PILOT_TIMING.json').read_text())
        prep = json.loads((OUT/'STATUS.json').read_text())
        # Conservative no-parallel-speedup assumption; scale by known old-bank bytes.
        sizes = []
        for b in (DATA/'vosr_2u_block256_v1_20261023/bank_raw_v1', DATA/'six_models_mojito_finetune_20261027/bank_raw_v1'):
            c = json.loads((b/'COMPLETE.json').read_text())
            sizes.append(sum(s['bytes'] for s in c['splits'].values()))
        bank_estimate = timing['bank_seconds'] * sum(sizes) / timing['bank_bytes']
        regions_estimate = timing['sidecar_seconds'] * 5903 / 36
        target = datetime.fromisoformat(prep['preprocessing_eta']) + timedelta(seconds=1.25*(bank_estimate+regions_estimate)+300)
        target = datetime.fromtimestamp(math.ceil(target.timestamp()/600)*600, TZ)
        schedule = dict(passed=True, target=target.isoformat(), preprocessing_eta=prep['preprocessing_eta'],
            bank_estimate_seconds=bank_estimate, sidecar_estimate_seconds=regions_estimate, safety_factor=1.25,
            policy='resident one-shot UTC+08 timer; data/smoke/GPU gates; no partial-data launch',
            controller_pid=os.getpid(), created_at=datetime.now(TZ).isoformat())
        save(OUT/'SCHEDULE.json', schedule)
        while not (OUT/'PREPROCESS_COMPLETE.json').exists():
            assert json.loads((OUT/'STATUS.json').read_text())['phase'] != 'preprocessing_failed'
            monitor('waiting_preprocessing', schedule=schedule)
            time.sleep(20)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(bank_stage, stage) for stage in ('pretrain', 'finetune')]
            while not all(f.done() for f in futures):
                for future in futures:
                    if future.done():
                        future.result()
                monitor('building_banks_and_line_regions', schedule=schedule)
                time.sleep(20)
            for future in futures:
                future.result()
        for arm in ARMS:
            run = OUT/'pretrain/training'
            receipt = run/'smoke'/(arm+'.json')
            if receipt.exists():
                checked = json.loads(receipt.read_text())
                if checked.get('passed') and checked.get('config_sha256') == sha(run/'configs'/(arm+'.yml')):
                    continue
            with (OUT/('pretrain_smoke_'+arm+'.log')).open('ab') as stream:
                subprocess.run([sys.executable, '-u', str(ROOT/'tools/smoke_v3pre_training.py'), str(run/'configs'/(arm+'.yml')), str(run/'smoke'/(arm+'.json'))], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='2'), stdout=stream, stderr=subprocess.STDOUT, check=True)
        while datetime.now(TZ) < target:
            monitor('ready_waiting_timer', schedule=schedule)
            time.sleep(min(20, (target-datetime.now(TZ)).total_seconds()))
        children = {(stage, arm):None for stage in ('pretrain', 'finetune') for arm in ARMS}
        for gpu, arm in enumerate(ARMS):
            children['pretrain', arm] = launch('pretrain', arm, gpu)
        while True:
            states = {}
            for gpu, arm in enumerate(ARMS):
                pre = children['pretrain', arm]
                code = pre.poll()
                states['pretrain_'+arm] = dict(pid=pre.pid, exit_code=code)
                assert code in (None, 0), states
                if code == 0 and children['finetune', arm] is None:
                    checkpoint = OUT/'pretrain/training/experiments'/('v3pre_pretrain_'+arm+'_20261027')/'models/net_g_250000.pth'
                    assert checkpoint.is_file()
                    children['finetune', arm] = launch('finetune', arm, gpu)
                ft = children['finetune', arm]
                if ft is not None:
                    rc = ft.poll()
                    assert rc in (None, 0)
                    states['finetune_'+arm] = dict(pid=ft.pid, exit_code=rc)
                    if rc == 0:
                        assert (OUT/'finetune/training/experiments'/('v3pre_finetune_'+arm+'_20261027')/'models/net_g_150000.pth').is_file()
            monitor('training', schedule=schedule, children=states)
            if all(children['finetune', arm] is not None and children['finetune', arm].poll() == 0 for arm in ARMS):
                monitor('all_six_trainings_complete', children=states)
                save(OUT/'TRAINING_COMPLETE.json', dict(passed=True, children=states))
                break
            time.sleep(30)
    except BaseException as error:
        monitor('controller_failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pilot', 'controller', 'configs'))
    args = parser.parse_args()
    dict(pilot=pilot, controller=controller, configs=configs)[args.mode]()
