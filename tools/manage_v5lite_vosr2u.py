"""Launch two isolated VOSR2u V5Lite full/text1.5 jobs."""
import argparse
import copy
from datetime import datetime,timezone,timedelta
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
BASE=DATA/'v5lite_vosr2u_20261026'
RUN=BASE/'training_20261026'
BANK=DATA/'vosr_2u_block256_v1_20261023/bank_raw_v1'
REFERENCE=DATA/'original_gt_vosr2u_l1_six_20261023/training_20261023/configs/original_full_v3.yml'
REPO='/home/guang.yang/workspace/Real-ESRGAN'
ARMS=('full_v5lite','text15_v5lite')
GPUS=dict.fromkeys(ARMS,3)


def bind():
    """@luna-comment
    Input: owned run constants. Output: accepted smoke/save helper with isolated process globals.
    Key Steps: bind paths/arms/GPUs only. Key Params: two native256 arms.
    State/Dependencies: existing manager implementation; no external job mutation.
    """
    from tools import manage_original_gt_l1_matrix as manager
    manager.BASE,manager.RUN,manager.BANK,manager.ARMS,manager.GPUS=BASE,RUN,BANK,ARMS,GPUS
    return manager


def prepare():
    """@luna-comment
    Input: accepted complete VOSR2u bank/full recipe. Output: two configs and one shared fresh V5Lite state.
    Key Steps: validate bank; initialize once; clone full recipe and exact previous text1.5 modifications.
    Key Params:250K/batch4/seed20260828, GPU3shared, no resume or benchmark weights.
    State/Dependencies: accepted registration/data/optimizer pipeline; isolated NAS outputs only.
    """
    import torch,yaml
    from tools.train_v5lite_block256 import bootstrap
    from tools.build_offline_augmented_dataset import digest
    bootstrap(REPO)
    from basicsr.archs import build_network
    manager=bind();complete=json.loads((BANK/'COMPLETE.json').read_text())
    assert complete['complete'] and complete['shape_hw']==[256,256]
    assert complete['splits']['train']['samples']==478078 and complete['splits']['val']['samples']==16387
    original=yaml.safe_load(REFERENCE.read_text())
    network=dict(type='BasicTextLineR4StudentY4UVV5Lite',canvas_hw=[256,256],in_channel=6,out_channel=6,
        local_channels=12,stage_channels=[4,16],haar_switch=True,local_haar_switch=True,learnable_haar=False,grouped_phase_pack=True)
    torch.manual_seed(original['manual_seed']);model=build_network(copy.deepcopy(network))
    initial=RUN/'initial/v5lite.pth';initial.parent.mkdir(parents=True,exist_ok=True);assert not initial.exists()
    torch.save({'params':model.state_dict(),'initialization':'fresh_seeded_no_training'},initial)
    initial_sha=digest(initial);provenance={}
    for index,arm in enumerate(ARMS):
        cfg=copy.deepcopy(original);cfg['name']='vosr2u_'+arm+'_block256_250k_20261026'
        cfg['network_g']=copy.deepcopy(network)
        cfg['path'].update(pretrain_network_g=str(initial),strict_load_g=True,resume_state=None,experiments_root=str(RUN/'experiments'))
        cfg['dist_params']['port']=32210+index
        for split in ('train','val'):
            manifest=BANK/('train_balanced.json' if split=='train' else 'val.json')
            metadata=json.loads(manifest.read_text());assert metadata['build_id']==complete['build_id'] and metadata['preprocess_mode']=='none'
            cfg['datasets'][split].update(root=str(BANK),manifest=str(manifest))
        boost=arm.startswith('text15')
        if boost:
            cfg['train']['y4uv_native_opt'].update(type='TextBoostNativeMaskedL1Loss',text_multiplier=1.5)
            cfg['train']['uv_strong_opt'].update(type='TextBoostUVStrongSupervisionLoss',text_multiplier=1.5)
            for key in ('cp_weight','recall_aware_weight_missing','sdf_boundary_weight'):cfg['train'][key]*=1.5
        changes={k:dict(before=original['train'].get(k),after=v) for k,v in cfg['train'].items() if v!=original['train'].get(k)}
        assert set(changes)==({'y4uv_native_opt','uv_strong_opt','cp_weight','recall_aware_weight_missing','sdf_boundary_weight'} if boost else set())
        path=RUN/'configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        provenance[arm]=dict(config_sha256=digest(path),initial_sha256=initial_sha,initial_checkpoint=str(initial),
            initialization='fresh_seeded_no_optimizer_resume',loss_changes=changes,gpu=3,bank_build_id=complete['build_id'],
            validation_target='old_GT_VOSR_583_pairs',network=network)
    assert provenance[ARMS[0]]['initial_sha256']==provenance[ARMS[1]]['initial_sha256']
    manager.save(RUN/'provenance.json',provenance)


def smoke(arm):
    """@luna-comment
    Input: configured arm. Output: accepted actual CUDA update/validation smoke receipt.
    Key Steps: pre-register V5Lite/textboost; invoke unchanged full-loss smoke helper.
    Key Params: iter0/8000, CP/background/multiline/border real tiles. State/Dependencies: CUDA/data/registries.
    """
    from tools.train_v5lite_block256 import bootstrap
    bootstrap(REPO)
    bind().smoke(arm)


def launch(arm):
    """@luna-comment
    Input: passing smoke/config/hash provenance. Output: detached dedicated training process and identity receipt.
    Key Steps: verify strict init/hash/free GPU/output absence; spawn only owned run.
    Key Params: GPU3 shared,2 CPU threads, no resume. State/Dependencies: Linux/CUDA; never stops another job.
    """
    import yaml
    from tools.build_offline_augmented_dataset import digest
    manager=bind();path=RUN/'configs'/(arm+'.yml');cfg=yaml.safe_load(path.read_text())
    receipt=json.loads((RUN/'smoke'/(arm+'.json')).read_text())
    provenance=json.loads((RUN/'provenance.json').read_text())[arm]
    assert receipt['passed'] and receipt['config_sha256']==digest(path)==provenance['config_sha256']
    assert digest(cfg['path']['pretrain_network_g'])==provenance['initial_sha256']
    assert not (RUN/'experiments'/cfg['name']).exists() and not (RUN/'launch'/(arm+'.json')).exists()
    gpu=GPUS[arm];free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>4096
    command=[sys.executable,'-u',str(ROOT/'tools/train_v5lite_block256.py'),'--repo-root',REPO,'-opt',str(path)]
    log=RUN/'logs'/(arm+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('xb') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=str(RUN),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),start_new_session=True)
    manager.save(RUN/'launch'/(arm+'.json'),dict(arm=arm,pid=process.pid,gpu=gpu,command=command,log=str(log),config_sha256=digest(path)))


def monitor():
    """@luna-comment
    Input: owned launch/log/configs. Output: atomic live status/loss/ETA JSON/HTML every30s.
    Key Steps: validate process config identity; parse loss and recent speed; render controls/charts.
    Key Params:250K, full/text1.5 labels and old583 validation explicit. State/Dependencies: read-only job inspection.
    """
    import fcntl,yaml
    from tools.manage_vosr_block256 import estimate_remaining
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    manager=bind();lock=(RUN/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manager.save(RUN/'monitor.json',dict(pid=os.getpid()))
    while True:
        now=datetime.now(timezone(timedelta(hours=8)));state=dict(updated_at=now.isoformat(),arms={},loss_history={})
        for arm in ARMS:
            receipt=json.loads((RUN/'launch'/(arm+'.json')).read_text());log=Path(receipt['log'])
            with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-262144));text=stream.read().decode(errors='replace')
            matches=re.findall(r'iter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)\s*\(([\d.]+)\)',text)
            step=int(matches[-1][0].replace(',','')) if matches else 0
            command=Path('/proc')/str(receipt['pid'])/'cmdline'
            alive=command.exists() and str(RUN/'configs'/(arm+'.yml')).encode() in command.read_bytes()
            status=('running' if step else 'starting') if alive else 'exited'
            if 'Traceback (most recent call last)' in text:status='failed'
            cfg=yaml.safe_load((RUN/'configs'/(arm+'.yml')).read_text())
            if not alive and 'End of training' in text and (RUN/'experiments'/cfg['name']/'models/net_g_250000.pth').exists():status,step='completed',250000
            item=dict(gpu=GPUS[arm],pid=receipt['pid'],status=status,iteration=step)
            item.update(estimate_remaining(step,status,[m[1] for m in matches],now));state['arms'][arm]=item
            state['loss_history'][arm]=read_loss_history(log)
        manager.save(RUN/'STATUS.json',state)
        page='<!doctype html><meta charset="utf-8"><title>V5Lite VOSR2u</title><style>body{font:15px system-ui;margin:24px;background:#f4f7fa}td,th{padding:10px;border:1px solid #ccd}table{border-collapse:collapse}</style><h1>V5Lite - VOSR2u full-loss comparison</h1><p>Fresh seeded initialization;256 patches; batch4;250K. V5Lite local12/context4-16; one phase stage. Text1.5 strengthens foreground native/UV and CP/missing/SDF only; background/ring terms unchanged.</p><p>Train:VOSR2u4873 parents. Validation:OLD GT_VOSR583 parents.</p><button onclick="location.reload()">Refresh now</button><p>Auto refresh30s - '+state['updated_at']+'</p><table><tr><th>Arm</th><th>GPU/PID</th><th>Status</th><th>Iteration</th><th>Remaining</th><th>Finish Beijing</th></tr>'
        for arm,item in state['arms'].items():
            seconds=item['estimated_remaining_seconds'];remaining=f'{seconds/3600:.1f}h' if seconds is not None else 'pending'
            page+=f'<tr><td>{arm}</td><td>{item["gpu"]}/{item["pid"]}</td><td>{item["status"]}</td><td>{item["iteration"]:,}/250,000</td><td>{remaining}</td><td>{html.escape(item["estimated_finish_at"] or "pending")}</td></tr>'
        page+='</table><p>ETA uses recent logged speed; later ramp/validation/save cost may extend completion.</p>'+render_loss_charts(state['loss_history'])+'<script>setTimeout(()=>location.reload(),30000)</script>'
        temp=RUN/'STATUS.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(RUN/'STATUS.html')
        if all(x['status']=='completed' for x in state['arms'].values()):return
        time.sleep(30)


def run():
    """@luna-comment
    Input: user-authorized two-arm training plan. Output: real smoked jobs plus detached observer.
    Key Steps: exclusive controller lock; prepare; both pilots before launch; publish observer receipt.
    Key Params: no existing-run restart, no optimizer resume. State/Dependencies: Linux/CUDA/NAS.
    """
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True);RUN.mkdir(exist_ok=True);manager=bind()
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare();(RUN/'smoke').mkdir(exist_ok=True)
        for arm in ARMS:
            with (RUN/'smoke'/(arm+'.log')).open('xb') as stream:
                subprocess.run([sys.executable,'-u',__file__,'smoke','--arm',arm],stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[arm]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        for arm in ARMS:launch(arm);print('LAUNCHED',arm,flush=True)
        with (BASE/'monitor.log').open('ab') as stream:
            process=subprocess.Popen([sys.executable,'-u',__file__,'monitor'],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        manager.save(BASE/'PIPELINE.json',dict(phase='two_launched',monitor_pid=process.pid,arms=ARMS))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('run','smoke','monitor'));parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();{'run':run,'smoke':lambda:smoke(args.arm),'monitor':monitor}[args.action]()
