"""One-shot server-resident six-model150K finetune with complete-data/time gates."""
import argparse
import copy
from datetime import datetime,timezone,timedelta
import html
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE=DATA/'lenovo_mojito_vosr_finetune_20261026'
OUT=DATA/'six_models_mojito_finetune_20261027'
RUN=OUT/'training'
BANK=OUT/'bank_raw_v1'
REPO='/home/guang.yang/workspace/Real-ESRGAN'
TARGET=datetime.fromisoformat('2026-10-27T02:00:00+08:00')
TZ=timezone(timedelta(hours=8))
LABELS={'full_v3':'V3 | Full','full_v3m':'V3M | Full','text15_v3':'V3 | Text x1.5',
        'text15_v3m':'V3M | Text x1.5','full_v5lite':'V5Lite | Full','text15_v5lite':'V5Lite | Text x1.5'}
GPUS=dict(zip(LABELS,(0,1,2,3,0,1)))


def save(path,value):
    """@luna-comment Input: owned path/JSON. Output: atomic UTF8 metadata.
    Key Steps: sibling replacement. Key Params: finite JSON. State/Dependencies: owned output only.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def status(phase,**extra):
    """@luna-comment Input: scheduler phase/details. Output: live status JSON/HTML.
    Key Steps: publish exact time and labels; preserve data gating evidence. Key Params:30s refresh.
    State/Dependencies: own output, inline chart helper when histories supplied.
    """
    from tools.offline_loss_charts import render_loss_charts
    state=dict(phase=phase,scheduled_at=TARGET.isoformat(),updated_at=datetime.now(TZ).isoformat(),total_iter=150000,**extra)
    save(OUT/'STATUS.json',state)
    page='<!doctype html><meta charset="utf-8"><title>Six-model Mojito finetune</title><style>body{font:15px system-ui;margin:24px;background:#f4f7fa}pre{white-space:pre-wrap}</style><h1>Six-model Mojito finetune</h1><button onclick="location.reload()">Refresh</button><pre>'+html.escape(json.dumps({k:v for k,v in state.items() if k!='loss_history'},indent=2))+'</pre>'
    if 'loss_history' in extra:page+=render_loss_charts(extra['loss_history'])
    temp=OUT/'STATUS.html.tmp';temp.write_text(page+'<script>setTimeout(()=>location.reload(),30000)</script>',encoding='utf-8');temp.replace(OUT/'STATUS.html')


def prepare_configs():
    """@luna-comment Input: audited six final250K bindings. Output: frozen initial checkpoints and150K configs.
    Key Steps: hash originals/snapshots, preserve network/loss recipes, set new data/output/LR/schedule only.
    Key Params: Adam1e-5,45K/90K/135K, batch4; fresh optimizer. State/Dependencies: yaml and read-only old results.
    """
    import yaml
    from tools.build_camera_2x2_viewer import digest
    bindings=json.loads((DATA/'five_plus_v3_camera280_20261026/AUDIT.json').read_text())['bindings']
    provenance={}
    for i,arm in enumerate(LABELS):
        original=bindings[arm];cfg=yaml.safe_load(Path(original['config']).read_text())
        assert digest(original['config'])==original['config_sha256'] and digest(original['checkpoint'])==original['checkpoint_sha256']
        initial=OUT/'initial'/(arm+'.pth');initial.parent.mkdir(parents=True,exist_ok=True)
        if initial.exists():assert digest(initial)==original['checkpoint_sha256']
        else:shutil.copy2(original['checkpoint'],initial)
        before=copy.deepcopy(cfg['train']);cfg['name']='mojito_ft150k_'+arm+'_20261027'
        cfg['path'].update(pretrain_network_g=str(initial),strict_load_g=True,resume_state=None,experiments_root=str(RUN/'experiments'))
        cfg['train']['total_iter']=150000;cfg['train']['optim_g']['lr']=1e-5
        cfg['train']['scheduler']['milestones']=[45000,90000,135000]
        changes={k for k in before if before[k]!=cfg['train'][k]};assert changes=={'total_iter','optim_g','scheduler'}
        cfg['dist_params']['port']=32510+i
        for split in ('train','val'):
            cfg['datasets'][split].update(root=str(BANK),manifest=str(BANK/('train_balanced.json' if split=='train' else 'val.json')),num_worker_per_gpu=2)
        assert cfg['datasets']['train']['batch_size_per_gpu']==4
        path=RUN/'configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True)
        text=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        provenance[arm]=dict(source=original,initial_checkpoint=str(initial),initial_sha256=digest(initial),config_sha256=digest(path),gpu=GPUS[arm],label=LABELS[arm])
    save(OUT/'MODEL_BINDINGS.json',provenance)


def preflight():
    """@luna-comment Input: prepared configs/snapshots. Output: six strict CPU load confirmations.
    Key Steps: explicit extensions, instantiate configured networks, exact state_dict loads.
    Key Params: no optimizer/train/CUDA allocation. State/Dependencies: torch/yaml and source-bound checkpoints.
    """
    import torch,yaml
    from tools.train_v5lite_block256 import bootstrap
    if REPO in sys.path:sys.path.remove(REPO)
    sys.path.insert(0,REPO)
    bootstrap(REPO)
    from basicsr.archs import build_network
    checked=[]
    for arm in LABELS:
        cfg=yaml.safe_load((RUN/'configs'/(arm+'.yml')).read_text())
        model=build_network(copy.deepcopy(cfg['network_g']))
        model.load_state_dict(torch.load(cfg['path']['pretrain_network_g'],map_location='cpu')['params'],strict=True)
        checked.append(dict(arm=arm,strict_load=True,network=cfg['network_g'],total_iter=cfg['train']['total_iter'],lr=cfg['train']['optim_g']['lr']))
        del model
    save(OUT/'PREFLIGHT.json',dict(passed=True,models=checked))


def data_ready():
    """@luna-comment Input: teacher delivery root. Output: complete-delivery boolean.
    Key Steps: require final marker/report and verified scene-disjoint447-pair delivery.
    Key Params: no partial outputs. State/Dependencies: read-only publisher metadata.
    """
    if not (SOURCE/'DELIVERY_READY.json').exists() or not (SOURCE/'delivery_report.json').exists():return False
    report=json.loads((SOURCE/'delivery_report.json').read_text())
    return report.get('status')=='complete' and report.get('total_pairs')==447 and all(report.get(k) for k in ('scene_groups_disjoint','source_hashes_verified','target_hashes_verified','all_dimensions_verified'))


def prepare_bank():
    """@luna-comment Input: complete teacher train/val manifests. Output: hash-verified native256 bank.
    Key Steps: retain scene split, map file roles, verify hashes, build accepted offline grids/augmentations.
    Key Params:4 CPU workers, no tonal processing or new split. State/Dependencies: immutable delivery, own adapter/bank.
    """
    from tools.build_camera_2x2_viewer import digest
    assert data_ready()
    view=OUT/'source_view';(view/'manifests').mkdir(parents=True,exist_ok=True)
    groups={};ids=set();counts={}
    mapping={'lq':'LQ','gt':'GT_VOSR','ocr':'OCR','text_mask':'text_mask','valid_mask':'valid_mask'}
    for split in ('train','val'):
        rows=[json.loads(line) for line in (SOURCE/'manifests'/(split+'.jsonl')).read_text().splitlines() if line.strip()]
        groups[split]={r['scene_group'] for r in rows};counts[split]=len(rows);adapted=[]
        for row in rows:
            sid=row['sample_id'];assert sid not in ids;ids.add(sid)
            files={k:str(SOURCE/row['files'][v]) for k,v in mapping.items()}
            hashes={k:digest(v) for k,v in files.items()}
            assert hashes['lq']==row['sha256'] and hashes['gt']==row['gt_sha256']
            adapted.append(dict(sample_id=sid,dataset='lenovo_mojito',source_identity=row['scene_group'],files=files,output_sha256=hashes))
        path=view/'manifests'/(split+'.jsonl');text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in adapted)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
    assert len(ids)==447 and counts['train']>0 and counts['val']>0 and not groups['train']&groups['val']
    save(OUT/'DATA_BINDING.json',dict(source=str(SOURCE),delivery_sha256=digest(SOURCE/'DELIVERY_READY.json'),report_sha256=digest(SOURCE/'delivery_report.json'),parents=counts,scene_groups_disjoint=True))
    if not (BANK/'COMPLETE.json').exists():
        with (OUT/'bank_build.log').open('ab') as stream:
            subprocess.run([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(view),'--output',str(BANK),'--workers','4','--preprocess-mode','none'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    complete=json.loads((BANK/'COMPLETE.json').read_text())
    assert complete['complete'] and complete['shape_hw']==[256,256]
    for split in ('train','val'):assert complete['splits'][split]['parents']==counts[split] and complete['splits'][split]['samples']>0


def smoke(arm):
    """@luna-comment Input: arm with final config/bank. Output: actual strict CUDA two-update/validation receipt.
    Key Steps: register extensions; bind accepted real-data smoke to owned paths; run without saving pilot weights.
    Key Params: iter0/8000 and four validation blocks. State/Dependencies: accepted manager, CUDA, complete bank.
    """
    from tools.train_v5lite_block256 import bootstrap
    if REPO in sys.path:sys.path.remove(REPO)
    sys.path.insert(0,REPO)
    bootstrap(REPO)
    from tools import manage_original_gt_l1_matrix as manager
    manager.RUN,manager.BANK=RUN,BANK
    manager.smoke(arm)


def launch():
    """@luna-comment Input: six passing smoke receipts/time gate. Output: six detached training processes.
    Key Steps: validate hashes and fresh owned output, start dedicated extension bootstrap and record exact identities.
    Key Params:150K, GPUs0/1/2/3 mapped six arms; no optimizer resume. State/Dependencies: own jobs only.
    """
    from tools.build_camera_2x2_viewer import digest
    assert datetime.now(TZ)>=TARGET
    provenance=json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    for arm in LABELS:
        path=RUN/'configs'/(arm+'.yml');receipt=json.loads((RUN/'smoke'/(arm+'.json')).read_text())
        assert receipt['passed'] and receipt['config_sha256']==digest(path)==provenance[arm]['config_sha256']
        assert digest(provenance[arm]['initial_checkpoint'])==provenance[arm]['initial_sha256']
        assert not (RUN/'launch'/(arm+'.json')).exists()
        entry='train_v5lite_block256.py' if arm.endswith('v5lite') else 'train_v3m_block256.py'
        command=[sys.executable,'-u',str(ROOT/'tools'/entry),'--repo-root',REPO,'-opt',str(path)]
        log=RUN/'logs'/(arm+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,cwd=RUN,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[arm]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        save(RUN/'launch'/(arm+'.json'),dict(pid=proc.pid,arm=arm,gpu=GPUS[arm],command=command,log=str(log),started_at=datetime.now(TZ).isoformat()))


def monitor():
    """@luna-comment Input: six owned launch receipts/logs. Output: correctly named150K loss/status/ETA page.
    Key Steps: inspect exact config PID, parse actual logged steps/speed, compute remaining150K, render histories.
    Key Params:30s refresh; no invented data. State/Dependencies: read-only owned jobs and chart helper.
    """
    from tools.offline_loss_charts import read_loss_history
    while True:
        arms={};histories={}
        for arm,label in LABELS.items():
            receipt=json.loads((RUN/'launch'/(arm+'.json')).read_text());log=Path(receipt['log'])
            with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-262144));text=stream.read().decode(errors='replace')
            matches=re.findall(r'iter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)\s*\(([\d.]+)\)',text)
            step=int(matches[-1][0].replace(',','')) if matches else 0
            path=Path('/proc')/str(receipt['pid'])/'cmdline'
            alive=path.exists() and str(RUN/'configs'/(arm+'.yml')).encode() in path.read_bytes()
            state='running' if alive else ('completed' if 'End of training' in text else 'exited')
            if 'Traceback (most recent call last)' in text:state='failed'
            if state=='completed':step=150000
            remaining=max(0,150000-step)*statistics.median([float(m[1]) for m in matches[-20:]]) if matches else None
            arms[label]=dict(gpu=GPUS[arm],pid=receipt['pid'],status=state,iteration=step,remaining_seconds=remaining)
            histories[label]=read_loss_history(log)
        status('training',arms=arms,loss_history=histories)
        if all(v['status'] in ('completed','failed','exited') for v in arms.values()):return
        time.sleep(30)


def run():
    """@luna-comment Input: authorized fixed schedule. Output: persistent one-shot gated preparation/launch/monitor.
    Key Steps: exclusive lock, freeze/preflight, wait complete data, build ahead, wait exact time/idle GPUs, smoke/launch.
    Key Params: Beijing02:00; retry prerequisites30s, three idle checks; errors published and stop safely.
    State/Dependencies: server process survives SSH disconnect; never stops external jobs or runs twice.
    """
    import fcntl
    OUT.mkdir(exist_ok=True);lock=(OUT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'launch').exists(),'Never duplicate an existing launch'
    try:
        prepare_configs();preflight()
        save(OUT/'SCHEDULE.json',dict(pid=os.getpid(),scheduled_at=TARGET.isoformat(),timezone='Asia/Shanghai',one_shot=True,arms=LABELS,total_iter=150000,dataset=str(SOURCE),wait_if_not_ready=True))
        while not data_ready():status('waiting_for_complete_dataset');time.sleep(30)
        status('preparing_block_bank');prepare_bank()
        preflight_time=TARGET-timedelta(minutes=5)
        while datetime.now(TZ)<preflight_time:status('waiting_for_preflight_time');time.sleep(min(30,max(.1,(preflight_time-datetime.now(TZ)).total_seconds())))
        idle=0
        while idle<3:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
            available=[tuple(map(int,line.split(','))) for line in lines]
            idle=idle+1 if len(available)==4 and all(free>16000 and util<20 for free,util in available) else 0
            status('waiting_for_gpu_headroom',idle_checks=idle,gpu_free_util=available)
            if idle<3:time.sleep(30)
        status('cuda_preflight');(RUN/'smoke').mkdir(exist_ok=True)
        for arm in LABELS:
            with (RUN/'smoke'/(arm+'.log')).open('xb') as stream:
                subprocess.run([sys.executable,'-u',__file__,'smoke','--arm',arm],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[arm]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        while datetime.now(TZ)<TARGET:status('waiting_for_scheduled_time');time.sleep(min(30,max(.1,(TARGET-datetime.now(TZ)).total_seconds())))
        launch();monitor()
    except Exception as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('run','preflight','smoke'));parser.add_argument('--arm',choices=tuple(LABELS));args=parser.parse_args()
    # Bootstrap before metadata/viewer imports append legacy search paths. Parent
    # bootstrap modules prepend staging paths, so late path promotion is insufficient.
    from tools.train_v5lite_block256 import bootstrap
    bootstrap(REPO)
    if args.action=='run':run()
    elif args.action=='preflight':prepare_configs();preflight()
    else:
        if args.arm is None:parser.error('smoke requires --arm')
        smoke(args.arm)
