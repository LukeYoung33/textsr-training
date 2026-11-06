"""Independent three-arm VOSR2u full-loss repeat with unchanged accepted training paths."""
import argparse
from datetime import datetime,timedelta,timezone
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
BASE=DATA/'vosr2u_full_three_20261026'
RUN=BASE/'training_20261026'
BANK=DATA/'vosr_2u_block256_v1_20261023/bank_raw_v1'
REFERENCE=DATA/'original_gt_vosr2u_l1_six_20261023/training_20261023'
ARMS=tuple('vosr2u_full_'+a for a in ('v3','v5','sr'))
GPUS=dict(zip(ARMS,(0,1,2)))


def bind():
    """@luna-comment
    Input: owned run/arm/GPU constants. Output: existing smoke/launch helper with process-local bindings.
    Key Steps: override output/bank/mapping globals only. Key Params: three full-loss arms, no L1 prefix.
    State/Dependencies: isolated module state; existing source and other processes unchanged.
    """
    from tools import manage_original_gt_l1_matrix as manager
    manager.BASE,manager.RUN,manager.BANK=BASE,RUN,BANK
    manager.ARMS,manager.GPUS=ARMS,GPUS
    return manager


def prepare():
    """@luna-comment
    Input: complete VOSR2u bank and approved full-loss architecture configs. Output: immutable configs/provenance.
    Key Steps: verify counts/mode, clone originals, change data/run identity only, bind identical init hashes.
    Key Params:250K/batch4/full native+auxiliary losses; original architecture warm starts/fresh seeded SR.
    State/Dependencies: YAML/digests and immutable NAS; refuses incompatible existing configs or resume.
    """
    import yaml
    from tools.build_offline_augmented_dataset import digest
    manager=bind()
    complete=json.loads((BANK/'COMPLETE.json').read_text())
    assert complete['complete'] and complete['shape_hw']==[256,256]
    assert complete['splits']['train']['parents']==4873 and complete['splits']['train']['samples']==478078
    assert complete['splits']['val']['parents']==583 and complete['splits']['val']['samples']==16387
    provenance={}
    for index,arm in enumerate(ARMS):
        arch=arm.rsplit('_',1)[1]
        reference=REFERENCE/'configs'/('original_full_'+arch+'.yml')
        cfg=yaml.safe_load(reference.read_text())
        cfg['name']=arm+'_block256_250k_20261026'
        cfg['path'].update(experiments_root=str(RUN/'experiments'),resume_state=None)
        cfg['dist_params']['port']=32010+index
        for split in ('train','val'):
            manifest=BANK/('train_balanced.json' if split=='train' else 'val.json')
            metadata=json.loads(manifest.read_text())
            assert metadata['build_id']==complete['build_id'] and metadata['preprocess_mode']=='none'
            cfg['datasets'][split].update(root=str(BANK),manifest=str(manifest))
        original=yaml.safe_load(reference.read_text())
        assert cfg['train']==original['train'] and cfg['network_g']==original['network_g']
        assert cfg['manual_seed']==original['manual_seed'] and cfg['train']['total_iter']==250000
        assert cfg['path']['pretrain_network_g']==original['path']['pretrain_network_g']
        assert cfg['train']['cp_weight']>0 and cfg['train']['uv_strong_weight']>0 and cfg['train']['sdf_boundary_weight']>0
        path=RUN/'configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True)
        text=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        provenance[arm]=dict(config_sha256=digest(path),initial_checkpoint=cfg['path']['pretrain_network_g'],
            initial_sha256=digest(cfg['path']['pretrain_network_g']),baseline_config=str(reference),baseline_sha256=digest(reference),
            bank_build_id=complete['build_id'],validation_target='old_GT_VOSR_583_pairs',gpu=GPUS[arm],initialization='same_architecture_specific_original_start',losses='unchanged_full')
    manager.save(RUN/'provenance.json',provenance)


def monitor():
    """@luna-comment
    Input: owned launch/log/config records. Output: live loss/ETA status JSON/HTML every30seconds.
    Key Steps: verify PID/config identity, parse recent timings, distinguish completed/failed states and plot losses.
    Key Params:250K; old583 validation target explicit;30s/manual refresh. State/Dependencies: read-only job probes.
    """
    import fcntl,yaml
    from tools.manage_vosr_block256 import estimate_remaining
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    manager=bind()
    lock=(RUN/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manager.save(RUN/'monitor.json',dict(pid=os.getpid()))
    while True:
        now=datetime.now(timezone(timedelta(hours=8)))
        state=dict(updated_at=now.isoformat(),arms={},loss_history={})
        for arm in ARMS:
            path=RUN/'launch'/(arm+'.json');status,step,matches='preparing',0,[]
            item=dict(gpu=GPUS[arm],status=status,iteration=0)
            if path.exists():
                receipt=json.loads(path.read_text());log=Path(receipt['log'])
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-262144));text=stream.read().decode(errors='replace')
                matches=re.findall(r'iter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)\s*\(([\d.]+)\)',text)
                step=int(matches[-1][0].replace(',','')) if matches else 0
                command=Path('/proc')/str(receipt['pid'])/'cmdline'
                alive=command.exists() and str(RUN/'configs'/(arm+'.yml')).encode() in command.read_bytes()
                status=('running' if step else 'starting') if alive else 'exited'
                if 'Traceback (most recent call last)' in text:status='failed'
                cfg=yaml.safe_load((RUN/'configs'/(arm+'.yml')).read_text())
                final=RUN/'experiments'/cfg['name']/'models/net_g_250000.pth'
                if not alive and 'End of training' in text and final.exists():status,step='completed',250000
                item.update(pid=receipt['pid'],iteration=step,status=status)
                state['loss_history'][arm]=read_loss_history(log)
            item.update(estimate_remaining(step,status,[m[1] for m in matches],now))
            state['arms'][arm]=item
        manager.save(RUN/'STATUS.json',state)
        page='<!doctype html><meta charset="utf-8"><title>VOSR2u full loss: V3/V5/CompactX2SR</title><style>body{font:15px system-ui;margin:24px;background:#f4f7fa}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd;padding:10px}button{padding:10px}</style><h1>VOSR2u full loss: V3 / V5 / Compact X2SR</h1><p>New independent250K runs ·batch4 ·256 blocks ·no tonal preprocessing. SR: RGB128 features, original RGB256 LQ skip. Same architecture-specific initialization as prior L1 arms; no optimizer resume.</p><p>Train: latest VOSR2u4873 parents. Validation: OLD GT_VOSR583 parents.</p><button onclick="location.reload()">Refresh now</button><p>Auto refresh30s · '+state['updated_at']+'</p><table><tr><th>Arm</th><th>GPU/PID</th><th>Status</th><th>Iteration</th><th>Approx. remaining</th><th>Finish Beijing</th></tr>'
        for arm,item in state['arms'].items():
            seconds=item['estimated_remaining_seconds'];duration=f'{seconds/3600:.1f}h' if seconds is not None else 'pending'
            page+=f'<tr><td>{arm}</td><td>{item["gpu"]}/{item.get("pid","-")}</td><td>{item["status"]}</td><td>{item["iteration"]:,}/250,000</td><td>{duration}</td><td>{html.escape(item["estimated_finish_at"] or "pending")}</td></tr>'
        page+='</table><p>ETA uses recent logged speed; validation/save/load variability can extend completion.</p>'+render_loss_charts(state['loss_history'])+'<script>setTimeout(()=>location.reload(),30000)</script>'
        temp=RUN/'STATUS.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(RUN/'STATUS.html')
        if all(r['status']=='completed' for r in state['arms'].values()):return
        time.sleep(30)


def run():
    """@luna-comment
    Input: explicit three-arm full-loss repeat request. Output: strictly smoked detached jobs and observer.
    Key Steps: isolated lock/configs, real two-update/validation pilots, independent launches and monitor.
    Key Params:GPU0/1/2, threads2; no bank build or existing-job signals. State/Dependencies: Linux/CUDA/NAS.
    """
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True);RUN.mkdir(exist_ok=True)
    manager=bind()
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        (RUN/'smoke').mkdir(exist_ok=True)
        for arm in ARMS:
            assert not (RUN/'launch'/(arm+'.json')).exists(),'Already launched; inspect before any restart'
            with (RUN/'smoke'/(arm+'.log')).open('ab') as stream:
                subprocess.run([sys.executable,'-u',__file__,'smoke','--arm',arm],stdout=stream,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[arm]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            manager.launch(arm)
            print('launched '+arm,flush=True)
        with (BASE/'monitor.log').open('ab') as stream:
            process=subprocess.Popen([sys.executable,'-u',__file__,'monitor'],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        manager.save(BASE/'PIPELINE.json',dict(phase='three_launched',monitor_pid=process.pid,arms=list(ARMS)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','smoke','monitor'));parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.action=='smoke':bind().smoke(args.arm)
    else:{'run':run,'monitor':monitor}[args.action]()
