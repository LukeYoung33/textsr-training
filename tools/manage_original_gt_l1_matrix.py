"""Launch matched original-GT/full-loss and VOSR2u/L1 V3,V5,CompactSR arms."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
BASE = DATA/'original_gt_vosr2u_l1_six_20261023'
NEW = DATA/'vosr_2u_block256_v1_20261023'
SR = DATA/'vosr2u_compact_sr_x2_20261023'
RUN = BASE/'training_20261023'
BANK = BASE/'bank_original_gt_v1'
REPO = '/home/guang.yang/workspace/Real-ESRGAN'
ARCHES = ('v3', 'v5', 'sr')
ARMS = tuple(group+'_'+arch for group in ('original_full','vosr2u_l1') for arch in ARCHES)
GPUS = dict(original_full_v3=3, original_full_v5=0, original_full_sr=1,
            vosr2u_l1_v3=0, vosr2u_l1_v5=2, vosr2u_l1_sr=2)


def save(path, value):
    """@luna-comment
    Input: owned JSON path and serializable value. Output: atomic UTF8 receipt.
    Key Steps: create parent, write sibling, replace. Key Params: indented JSON.
    State/Dependencies: isolated matrix paths only; no source data mutation.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def prepare():
    """@luna-comment
    Input: accepted new-target manifests/configs and original-target lineage. Output: original source view/six configs.
    Key Steps: preserve exact parents/splits/LQ assets, substitute original GT, clone architecture baselines.
    Key Params: same250K/batch4/seed/init; L1 disables auxiliary options/weights; no optimizer resume.
    State/Dependencies: immutable NAS sources and yaml; refuses changed existing config/source publication.
    """
    import yaml
    from tools.build_offline_augmented_dataset import digest
    view=BASE/'source_view'
    (view/'manifests').mkdir(parents=True,exist_ok=True)
    populations={}
    for split,count in [('train',4873),('val',583)]:
        rows=[json.loads(x) for x in (NEW/'source_view/manifests'/(split+'.jsonl')).read_text().splitlines()]
        assert len(rows)==count
        populations[split]={r['source_identity'] for r in rows}
        for row in rows:
            lineage=row['original_gt_lineage']
            assert Path(lineage['path']).is_file()
            row['files']['gt']=lineage['path']
            row['output_sha256']['gt']=lineage['sha256']
            row['target_version']='original_GT_20261014'
            row.pop('target_refresh',None)
        path=view/'manifests'/(split+'.jsonl')
        text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
    assert not populations['train'] & populations['val']
    supplemental=NEW/'source_view/OCR_ppocrv6_medium'
    if supplemental.exists() and not (view/supplemental.name).exists():
        (view/supplemental.name).symlink_to(supplemental,target_is_directory=True)
    save(view/'COMPLETE.json',dict(complete=True,train=4873,val=583,target='original_GT',capture_overlap=0))
    provenance={}
    for number,arm in enumerate(ARMS):
        arch=arm.rsplit('_',1)[1]
        reference=(SR/'training_20261023/configs/07_v5_compact_sr2x.yml' if arch=='sr' else
            NEW/'training_comparison_20261023/configs'/('06_v3_block256_raw.yml' if arch=='v3' else '05_v5_block256_raw.yml'))
        cfg=yaml.safe_load(reference.read_text())
        baseline=copy.deepcopy(cfg)
        cfg['name']=arm+'_block256_250k_20261023'
        cfg['path'].update(experiments_root=str(RUN/'experiments'),resume_state=None)
        cfg['dist_params']['port']=31810+number
        if arm.startswith('vosr2u_l1'):
            train=cfg['train']
            for key in list(train):
                if key.endswith('_opt') and key not in ('y4uv_native_opt','yuv444_reference_opt'):
                    train[key]=None
                if key.endswith('_weight') or key.startswith('recall_aware_weight_'):
                    train[key]=0.
            train.update(train_reference_loss=False,loss_mode='native',stroke_prob_enabled=False)
        else:
            for split in ('train','val'):
                cfg['datasets'][split].update(root=str(BANK),manifest=str(BANK/('train_balanced.json' if split=='train' else 'val.json')))
            assert cfg['train']==baseline['train']
        assert cfg['network_g']==baseline['network_g'] and cfg['manual_seed']==baseline['manual_seed']
        assert cfg['train']['optim_g']==baseline['train']['optim_g'] and cfg['train']['scheduler']==baseline['train']['scheduler']
        assert cfg['train']['total_iter']==250000 and cfg['datasets']['train']['batch_size_per_gpu']==4
        path=RUN/'configs'/(arm+'.yml')
        path.parent.mkdir(parents=True,exist_ok=True)
        text=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        provenance[arm]=dict(config_sha256=digest(path),baseline_config=str(reference),baseline_sha256=digest(reference),
            initial_checkpoint=cfg['path']['pretrain_network_g'],initial_sha256=digest(cfg['path']['pretrain_network_g']),
            gpu=GPUS[arm],target='original_GT_train_and_val' if arm.startswith('original') else 'VOSR2u_train_old_GT_VOSR_val',
            losses='unchanged_full' if arm.startswith('original') else 'native_masked_channel_weighted_L1_only')
    for arch in ARCHES:
        assert provenance['original_full_'+arch]['initial_sha256']==provenance['vosr2u_l1_'+arch]['initial_sha256']
    save(RUN/'provenance.json',provenance)


def smoke(arm):
    """@luna-comment
    Input: one strict matrix config and complete matching bank. Output: real CUDA two-update/validation receipt.
    Key Steps: select CP/background/multiline/border tiles, verify packing/HR skip, update and check losses/weights.
    Key Params: iter0 and8000; L1 total exactly native, auxiliary modules absent; full CP path nonzero.
    State/Dependencies: existing bootstrap/reader/model; never saves pilot weights to formal initialization.
    """
    import numpy as np
    import torch
    from tools.train_vosr2u_sr import bootstrap
    from tools.build_offline_augmented_dataset import digest
    bootstrap(REPO)
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.data.offline_augmented_region_dataset import OfflineAugmentedRegionDataset
    from realesrgan.data.offline_sr2u_dataset import OfflineSR2uDataset
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    path=RUN/'configs'/(arm+'.yml')
    sys.argv=[sys.argv[0],'-opt',str(path)]
    opt,_=parse_options(REPO,is_train=True)
    opt.update(dist=False,rank=0,world_size=1)
    sr=arm.endswith('_sr')
    cls=OfflineSR2uDataset if sr else OfflineAugmentedRegionDataset
    dataset=cls(opt['datasets']['train'])
    bank=Path(opt['datasets']['train']['root'])
    refs={}
    for si,shard in enumerate(dataset.shards):
        for ri,record in enumerate(json.loads((bank/shard['metadata']).read_text())):
            for key,yes in [('cp',bool(record['cp_lines'])),('background',record['background_only']),
                            ('multiline',record['visible_ocr_rows']>3),('border',record['real_hw']!=[256,256])]:
                if yes:refs.setdefault(key,(si,ri))
        if len(refs)==4:break
    assert len(refs)==4
    dataset.index=np.asarray([refs[k] for k in ('cp','background','multiline','border')],np.uint32)
    samples=[dataset[i] for i in range(4)]
    batch=dataset.collate_fn(samples)
    assert list(batch['lq_rgb'].shape)==[4,3,128 if sr else 256,128 if sr else 256]
    model=build_model(opt)
    model.feed_data(batch)
    torch.testing.assert_close(model.lq,rgb_to_y4uv(batch['lq_rgb'].cuda()),atol=0,rtol=0)
    torch.testing.assert_close(model.gt,rgb_to_y4uv(batch['gt_rgb'].cuda()),atol=0,rtol=0)
    if sr:
        torch.testing.assert_close(model.original_lq,rgb_to_y4uv(batch['original_lq_rgb'].cuda()),atol=0,rtol=0)
        torch.testing.assert_close(model._forward_v5(model.lq),model.original_lq,atol=0,rtol=0)
    if arm.startswith('vosr2u_l1'):
        assert all(getattr(model,k,None) is None for k in ('cri_cp','cri_grad','cri_uv_strong','cri_bg_lf','cri_sdf_boundary','cri_cldice','cri_stroke_core','cri_skel_fbeta','cri_stroke_prob','cri_keep','cri_normal_profile','cri_stroke_width','cri_stroke_overshoot','cri_text_gestalt'))
    before={k:v.detach().clone() for k,v in model.net_g.named_parameters() if v.requires_grad}
    model.optimize_parameters(0)
    model.optimize_parameters(8000)
    logs={k:float(v) for k,v in model.get_current_log().items()}
    assert all(np.isfinite(v) for v in logs.values())
    changed=[k for k,v in model.net_g.named_parameters() if k in before and not torch.equal(v.detach(),before[k])]
    assert changed and tuple(model.output.shape)==(4,6,128,128)
    training_output_shape=list(model.output.shape)
    if arm.startswith('vosr2u_l1'):
        assert abs(logs['l_total']-logs['l_native'])<1e-7
        assert all(v==0 for k,v in logs.items() if k.endswith('_scale'))
    else:
        assert logs['cp_accepted_lines']>0 and logs['cp_scale']>0 and model.sdf_boundary_weight>0
        assert 'l_sdf_boundary' in logs and logs['uv_strong_scale']>0
    val=cls(opt['datasets']['val'])
    val.index=val.index[:4]
    loader=torch.utils.data.DataLoader(val,batch_size=2,collate_fn=val.collate_fn)
    model.nondist_validation(loader,8000,None,False)
    torch.cuda.synchronize()
    save(RUN/'smoke'/(arm+'.json'),dict(passed=True,config_sha256=digest(path),strict_load=True,
        lr_rgb=list(batch['lq_rgb'].shape),target=list(batch['gt_rgb'].shape),native_output=training_output_shape,
        original_hr_skip=sr,updates=2,changed_parameter_count=len(changed),losses=logs,validation_samples=4,
        maximum_memory_bytes=torch.cuda.max_memory_allocated()))


def launch(arm):
    """@luna-comment
    Input: arm with matching successful smoke. Output: independent detached process/identity receipt.
    Key Steps: verify config/init, memory headroom and output absence; spawn explicit bootstrap entry.
    Key Params: fixed GPU mapping, threads2, no resume. State/Dependencies: Linux CUDA/processes; never signals jobs.
    """
    import yaml
    from tools.build_offline_augmented_dataset import digest
    path=RUN/'configs'/(arm+'.yml')
    check=json.loads((RUN/'smoke'/(arm+'.json')).read_text())
    assert check['passed'] and check['config_sha256']==digest(path)
    cfg=yaml.safe_load(path.read_text())
    provenance=json.loads((RUN/'provenance.json').read_text())[arm]
    assert digest(cfg['path']['pretrain_network_g'])==provenance['initial_sha256']
    assert not (RUN/'experiments'/cfg['name']).exists() and not (RUN/'launch'/(arm+'.json')).exists()
    gpu=GPUS[arm]
    free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>4096
    entry='train_vosr2u_sr.py' if arm.endswith('_sr') else 'train_vosr_block256.py'
    command=[sys.executable,'-u',str(ROOT/'tools'/entry),'--repo-root',REPO,'-opt',str(path)]
    log=RUN/'logs'/(arm+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('xb') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            cwd=str(RUN),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),start_new_session=True)
    save(RUN/'launch'/(arm+'.json'),dict(arm=arm,pid=process.pid,gpu=gpu,command=command,log=str(log),config_sha256=digest(path)))


def verify_bank():
    """@luna-comment
    Input: completed original and new-target banks. Output: independent index/LQ/mask/target replay evidence.
    Key Steps: compare counts/categories/indices; sample all families and replay baseline/augmented original targets.
    Key Params: source hashes checked by builder; GT changes only, all block geometry/metadata equal.
    State/Dependencies: read-only mmap/OpenCV and existing common recipe logic; no source edits.
    """
    import cv2
    import numpy as np
    from tools import build_offline_augmented_dataset as common
    common.PREPROCESS_MODE='none'
    reference=NEW/'bank_raw_v1'
    complete=json.loads((BANK/'COMPLETE.json').read_text())
    old=json.loads((reference/'COMPLETE.json').read_text())
    assert complete['complete'] and complete['shape_hw']==[256,256]
    replay=[]
    for split in ('train','val'):
        assert complete['splits'][split]['samples']==old['splits'][split]['samples']
        assert complete['splits'][split]['categories']==old['splits'][split]['categories']
        assert np.array_equal(np.load(BANK/(split+'.index.npy')),np.load(reference/(split+'.index.npy')))
        manifest=json.loads((BANK/(split+'.json')).read_text())
        rows={r['sample_id']:r for r in map(json.loads,(BASE/'source_view/manifests'/(split+'.jsonl')).read_text().splitlines())}
        seen=set()
        for shard in manifest['shards']:
            if shard['dataset'] in seen:continue
            seen.add(shard['dataset'])
            array=np.load(BANK/shard['array'],mmap_mode='r')
            original=np.load(reference/shard['array'],mmap_mode='r')
            metadata=json.loads((BANK/shard['metadata']).read_text())
            assert metadata==json.loads((reference/shard['metadata']).read_text())
            assert np.array_equal(array[...,:3],original[...,:3]) and np.array_equal(array[...,6:],original[...,6:])
            target=cv2.imread(rows[shard['sample_id']]['files']['gt'])
            assert common.digest(rows[shard['sample_id']]['files']['gt'])==rows[shard['sample_id']]['output_sha256']['gt']
            for ri in sorted({0,len(array)-1}):
                record=metadata[ri]
                gt=common.photometry(target,record['recipe']) if record['category']=='photometry' else target
                matrix=np.asarray(record['matrix'])
                if not np.array_equal(matrix,np.eye(3)):
                    gt=cv2.warpPerspective(gt,matrix,(gt.shape[1],gt.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                x,y=record['xy'];hh,ww=record['real_hw']
                gt=np.pad(gt[y:y+hh,x:x+ww],((0,256-hh),(0,256-ww),(0,0)),mode='edge')
                assert np.array_equal(gt,array[ri,...,3:6])
                replay.append(dict(split=split,sample_id=shard['sample_id'],tile=ri,category=record['category']))
    assert np.array_equal(np.load(BANK/'train_balanced.index.npy'),np.load(reference/'train_balanced.index.npy'))
    save(BASE/'BANK_VERIFIED.json',dict(passed=True,counts=complete['splits'],replay=replay))


def monitor():
    """@luna-comment
    Input: owned launch/log/build receipts. Output: atomic six-arm progress/loss/ETA HTML and JSON every30s.
    Key Steps: validate PID/config command, parse complete log lines, show waiting states and target versions.
    Key Params:250K; recent-speed ETA, old validation explicit; manual refresh. State/Dependencies: read-only jobs.
    """
    import fcntl
    import yaml
    from tools.manage_vosr_block256 import estimate_remaining
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    lock=(RUN/'monitor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(RUN/'monitor.json',dict(pid=os.getpid()))
    while True:
        now=datetime.now(timezone(timedelta(hours=8)))
        state=dict(updated_at=now.isoformat(),arms={},loss_history={})
        for arm in ARMS:
            launch_file=RUN/'launch'/(arm+'.json')
            status,step,matches='waiting_for_offline_bank',0,[]
            item=dict(gpu=GPUS[arm],iteration=0,status=status)
            if launch_file.exists():
                receipt=json.loads(launch_file.read_text())
                log=Path(receipt['log'])
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-262144));text=stream.read().decode(errors='replace')
                matches=re.findall(r'iter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)\s*\(([\d.]+)\)',text)
                step=int(matches[-1][0].replace(',','')) if matches else 0
                cmd=Path('/proc')/str(receipt['pid'])/'cmdline'
                alive=cmd.exists() and str(RUN/'configs'/(arm+'.yml')).encode() in cmd.read_bytes()
                status=('running' if step else 'starting') if alive else 'exited'
                if 'Traceback (most recent call last)' in text:status='failed'
                config=yaml.safe_load((RUN/'configs'/(arm+'.yml')).read_text())
                final=RUN/'experiments'/config['name']/'models/net_g_250000.pth'
                if not alive and 'End of training' in text and final.exists():status,step='completed',250000
                item.update(pid=receipt['pid'],status=status,iteration=step)
                state['loss_history'][arm]=read_loss_history(log)
            elif (BASE/'PIPELINE.json').exists():
                phase=json.loads((BASE/'PIPELINE.json').read_text()).get('phase','')
                if phase=='failed':item['status']=status='controller_failed'
                elif not arm.startswith('original'):item['status']=status='preparing'
            item.update(estimate_remaining(step,status,[m[1] for m in matches],now))
            state['arms'][arm]=item
        progress=BANK/'progress.json'
        if progress.exists():state['offline_bank']=json.loads(progress.read_text())
        save(RUN/'STATUS.json',state)
        page='<!doctype html><meta charset="utf-8"><title>Original GT / VOSR2u L1 · 6 models</title><style>body{font:15px system-ui;margin:24px;background:#f3f6fa}td,th{padding:10px;border:1px solid #ccd}table{border-collapse:collapse;background:white}button{padding:10px}</style><h1>Original GT full loss / VOSR2u L1-only</h1><p>V3 / V5 / Compact X2 SR ·256 blocks (SR input128, original HR LQ skip) ·250K ·batch4 ·same architecture-specific initialization.</p><p>Original group: original GT train/val. L1 group: newest VOSR2u train, old GT_VOSR583 validation. Validation losses across different target groups are not directly comparable. No tonal preprocessing or online augmentation.</p><button onclick="location.reload()">Refresh now</button><p>Auto refresh30s · '+state['updated_at']+'</p>'
        if 'offline_bank' in state:page+='<p>Original-GT bank: '+html.escape(json.dumps(state['offline_bank']))+'</p>'
        page+='<table><tr><th>Arm</th><th>GPU/PID</th><th>Status</th><th>Iteration</th><th>Approx. remaining</th><th>Finish Beijing</th></tr>'
        for arm,item in state['arms'].items():
            seconds=item['estimated_remaining_seconds']
            duration=f'{seconds/3600:.1f} h' if seconds is not None else 'pending'
            page+=f'<tr><td>{arm}</td><td>{item["gpu"]}/{item.get("pid","-")}</td><td>{item["status"]}</td><td>{item["iteration"]:,}/250,000</td><td>{duration}</td><td>{html.escape(item["estimated_finish_at"] or "pending")}</td></tr>'
        page+='</table><p>ETA uses recent logged step speed; later validation/save/load changes can delay completion.</p>'+render_loss_charts(state['loss_history'])+'<script>setTimeout(()=>location.reload(),30000)</script>'
        temp=RUN/'STATUS.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(RUN/'STATUS.html')
        time.sleep(30)


def run():
    """@luna-comment
    Input: explicit six-arm authorization and immutable accepted baseline closure. Output: strictly smoked launches.
    Key Steps: lock, preserve live training commands, build original bank while launch L1 arms, verify then launch full.
    Key Params:24 CPU workers, GPU mapping/headroom; abort on failure with receipt, no auto restart/resume.
    State/Dependencies: Linux flock/subprocess/CUDA; isolated outputs; existing jobs never signaled.
    """
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True)
    RUN.mkdir(exist_ok=True)
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior={}
        for process in Path('/proc').glob('[0-9]*'):
            try:
                command=(process/'cmdline').read_bytes()
                if b'--repo-root' in command and b'-opt' in command and b'train_' in command:
                    prior[process.name]=command.hex()
            except (FileNotFoundError,PermissionError):pass
        save(BASE/'PRIOR_PROCESSES.json',prior)
        try:
            prepare()
            monitor_receipt=RUN/'monitor.json'
            monitor_pid=json.loads(monitor_receipt.read_text())['pid'] if monitor_receipt.exists() else None
            monitor_cmd=Path('/proc')/str(monitor_pid)/'cmdline'
            if monitor_pid is None or not monitor_cmd.exists():
                with (BASE/'monitor.log').open('ab') as stream:
                    observer=subprocess.Popen([sys.executable,'-u',__file__,'monitor'],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                monitor_pid=observer.pid
            else:assert str(__file__).encode() in monitor_cmd.read_bytes()
            save(BASE/'PIPELINE.json',dict(phase='building_original_launching_l1',controller_pid=os.getpid(),monitor_pid=monitor_pid))
            builder=None
            if not (BANK/'COMPLETE.json').exists():
                builder_receipt=BASE/'BUILDER.json'
                builder_pid=json.loads(builder_receipt.read_text())['pid'] if builder_receipt.exists() else None
                builder_cmd=Path('/proc')/str(builder_pid)/'cmdline'
                if builder_pid is not None and builder_cmd.exists():
                    assert str(BANK).encode() in builder_cmd.read_bytes() and b'build_vosr_block256_dataset.py' in builder_cmd.read_bytes()
                else:
                    with (BASE/'build.log').open('ab') as stream:
                        builder=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(BASE/'source_view'),'--output',str(BANK),'--workers','24','--preprocess-mode','none'],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,OFFLINE_PREPROCESS_MODE='none',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    builder_pid=builder.pid
                    save(BASE/'BUILDER.json',dict(pid=builder_pid))
            for group in ('vosr2u_l1','original_full'):
                if group=='original_full':
                    if builder is not None:assert builder.wait()==0
                    else:
                        while not (BANK/'COMPLETE.json').exists():
                            assert (Path('/proc')/str(builder_pid)/'cmdline').exists(),'Existing builder exited before completion'
                            time.sleep(5)
                    verify_bank()
                for arch in ARCHES:
                    arm=group+'_'+arch
                    if (RUN/'launch'/(arm+'.json')).exists():
                        receipt=json.loads((RUN/'launch'/(arm+'.json')).read_text())
                        assert str(RUN/'configs'/(arm+'.yml')).encode() in (Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes()
                        continue
                    with (RUN/'smoke'/(arm+'.log')).open('ab') as stream:
                        subprocess.run([sys.executable,'-u',__file__,'smoke','--arm',arm],stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[arm]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                    launch(arm)
                    print('launched '+arm,flush=True)
            preserved={}
            for pid,command in prior.items():
                path=Path('/proc')/pid/'cmdline'
                if path.exists():assert path.read_bytes().hex()==command;preserved[pid]='same_command'
                else:preserved[pid]='completed_without_signal'
            save(BASE/'PIPELINE.json',dict(phase='six_launched',controller_pid=os.getpid(),monitor_pid=monitor_pid,prior=preserved))
        except BaseException as error:
            save(BASE/'PIPELINE.json',dict(phase='failed',controller_pid=os.getpid(),error=repr(error)))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','smoke','monitor','verify-bank'))
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    (RUN/'smoke').mkdir(parents=True,exist_ok=True)
    if args.action=='smoke':smoke(args.arm)
    else:{'run':run,'monitor':monitor,'verify-bank':verify_bank}[args.action]()
