"""Build, check, start and monitor the additional full-image block V3/V5 pair."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import html
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data/vosr_fullimage_block256_v1_20261021')
PREVIOUS=Path('/AI_DNS/guang.yang/TextSR/data/adaptive_vosr_preprocessed_augmented_v1_20261021')
RUN=DATA/'training_comparison_20261021'
REPO='/home/guang.yang/workspace/Real-ESRGAN'
MODE=os.environ.get('BLOCK256_PREPROCESS_MODE','none')
SUFFIX='raw' if MODE=='none' else 'preprocessed'
ARMS=('05_v5_block256_'+SUFFIX,'06_v3_block256_'+SUFFIX)
BANK=DATA/('bank_raw_v1' if MODE=='none' else 'bank_both_v1')


def save(path,value):
    """@luna-comment
    Input:owned JSON path/value. Output:atomic receipt. Key Steps:write sibling then replace.
    Key Params:UTF8. State/Dependencies:isolated block run only.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def prepare():
    """@luna-comment
    Input:complete block bank and approved architecture reference configs. Output:two immutable configs.
    Key Steps:preserve source/loss/optimizer/warm starts; select256 canvas and polygon CP adapter.
    Key Params:250K,batch4,same seed; train/val matching preprocessing mode.
    State/Dependencies:read-only previous configs/checkpoints; own run outputs.
    """
    import yaml
    from tools.build_offline_augmented_dataset import digest
    complete=json.loads((BANK/'COMPLETE.json').read_text())
    if not complete['complete'] or complete['shape_hw']!=[256,256]:raise ValueError('Block bank not ready')
    provenance={}
    for arm,reference,gpu in zip(ARMS,('03_v5_raw','04_v3_raw'),(2,3)):
        cfg=yaml.safe_load((PREVIOUS/'training_comparison_20261021/configs'/(reference+'.yml')).read_text())
        cfg['name']='vosr_'+arm+'_250k_20261021'
        cfg['model_type']=cfg['train']['model_type']='Block256RegionModel'
        cfg['network_g']['canvas_hw']=[256,256]
        cfg['train']['cp_opt']['type']='CRNNTextLineContentLoss'
        cfg['path']['experiments_root']=str(RUN/'experiments')
        cfg['logger']['print_freq']=100
        for split in ('train','val'):
            cfg['datasets'][split].update(name='vosr_block256_'+split,root=str(BANK),canvas_hw=[256,256],
                                         manifest=str(BANK/('train_balanced.json' if split=='train' else 'val.json')))
        path=RUN/'configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True)
        payload=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists() and path.read_text()!=payload:raise ValueError('Refusing changed block config')
        path.write_text(payload)
        provenance[arm]=dict(gpu=gpu,config_sha256=digest(path),checkpoint=cfg['path']['pretrain_network_g'],
                             checkpoint_sha256=digest(cfg['path']['pretrain_network_g']),build_id=complete['build_id'],mode=MODE)
    save(RUN/'provenance.json',provenance)


def smoke(arm):
    """@luna-comment
    Input:selected256 arm. Output:strict-load/actual optimizer/full-block routing receipt.
    Key Steps:select actual text/background/multiline/border examples, exact pack check,
      run post-ramp active losses and verify finite logged losses, parameter change and output shape.
    Key Params:iter8000; four actual256 patches. State/Dependencies:CUDA; no checkpoint saved.
    """
    from tools.train_vosr_block256 import bootstrap
    bootstrap(REPO)
    import numpy as np
    import torch
    from basicsr.utils.options import parse_options
    from basicsr.models import build_model
    from realesrgan.data.offline_augmented_region_dataset import OfflineAugmentedRegionDataset
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv
    from tools.build_offline_augmented_dataset import digest
    path=RUN/'configs'/(arm+'.yml');sys.argv=[sys.argv[0],'-opt',str(path)]
    opt,_=parse_options(REPO,is_train=True);opt.update(dist=False,rank=0,world_size=1)
    dataset=OfflineAugmentedRegionDataset(opt['datasets']['train']);refs={}
    for si,shard in enumerate(dataset.shards):
        for ri,record in enumerate(json.loads((BANK/shard['metadata']).read_text())):
            keys=['ordinary']
            if record['background_only']:keys.append('background')
            if record['visible_ocr_rows']>3:keys.append('multiline')
            if record['cp_lines']:keys.append('cp')
            if record['real_hw']!=[256,256]:keys.append('border')
            for key in keys:refs.setdefault(key,(si,ri))
        if len(refs)==5:break
    chosen=[refs[k] for k in ('cp','background','multiline','border','ordinary') if k in refs][:4]
    while len(chosen)<4:chosen.append(chosen[0])
    dataset.index=np.asarray(chosen,np.uint32)
    samples=[dataset[i] for i in range(4)];batch=dataset.collate_fn(samples)
    assert list(batch['lq_rgb'].shape)==[4,3,256,256]
    model=build_model(opt);model.feed_data(batch)
    torch.testing.assert_close(model.lq,rgb_to_y4uv(batch['lq_rgb'].cuda()),rtol=0,atol=0)
    tracked=next(p for p in model.net_g.parameters() if p.requires_grad);before=tracked.detach().clone()
    model.optimize_parameters(8000);torch.cuda.synchronize()
    logs={key:float(value) for key,value in model.get_current_log().items()}
    assert all(np.isfinite(value) for value in logs.values()) and not torch.equal(before,tracked.detach())
    assert tuple(model.output.shape)==(4,6,128,128)
    save(RUN/'smoke'/(arm+'.json'),dict(passed=True,arm=arm,config_sha256=digest(path),strict_load=True,
         optimizer_changed_weights=True,exact_full_block_input=True,rgb_shape=[4,3,256,256],losses=logs,
         available_sample_types=list(refs),cp_lines=[len(s['cp_lines']) for s in samples],
         max_memory_bytes=torch.cuda.max_memory_allocated()))


def launch(arm):
    """@luna-comment
    Input:arm with passed matching GPU smoke. Output:detached exact-command PID receipt.
    Key Steps:reject existing outputs, verify memory headroom, spawn on approved GPU2/3.
    Key Params:no resume,threads2; previous jobs untouched. State/Dependencies:CUDA/process/filesystem.
    """
    import yaml
    from tools.build_offline_augmented_dataset import digest
    path=RUN/'configs'/(arm+'.yml');checked=json.loads((RUN/'smoke'/(arm+'.json')).read_text())
    assert checked['passed'] and checked['config_sha256']==digest(path)
    gpu=2+ARMS.index(arm)
    free=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<4096:raise RuntimeError('Insufficient free GPU memory for the additional job')
    cfg=yaml.safe_load(path.read_text())
    if (RUN/'launch'/(arm+'.json')).exists() or (RUN/'experiments'/cfg['name']).exists():raise ValueError('Block arm already launched')
    log=RUN/'logs'/(arm+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'tools/train_vosr_block256.py'),'--repo-root',REPO,'-opt',str(path)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('xb') as stream:
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True,cwd=str(RUN))
    save(RUN/'launch'/(arm+'.json'),dict(arm=arm,pid=process.pid,gpu=gpu,log=str(log),command=cmd,config_sha256=digest(path)))


def estimate_remaining(iteration, status, timings, now, total=250000):
    """@luna-comment
    Input: current step/status, recent logged seconds-per-step, aware current time and total steps.
    Output: approximate remaining seconds, Beijing completion timestamp and smoothed iteration speed.
    Key Steps: reject inactive estimates, median last20 finite positive timings, extrapolate.
    Key Params:250K steps; excludes future validation/checkpoint/load variability.
    State/Dependencies:statistics/math/datetime; no process or training state is modified.
    """
    import math
    values = [float(x) for x in timings[-20:] if math.isfinite(float(x)) and float(x) > 0]
    speed = statistics.median(values) if values else None
    remaining = None
    if status == 'completed':
        remaining = 0
    elif status == 'running' and iteration > 0 and speed is not None:
        remaining = math.ceil(max(0, total - iteration) * speed)
    finish = (now + timedelta(seconds=remaining)).astimezone(timezone(timedelta(hours=8))).isoformat() if remaining is not None else None
    return dict(estimated_remaining_seconds=remaining, estimated_finish_at=finish,
                eta_seconds_per_iter=speed, eta_method='median_last20_logged_iteration_times')


def monitor():
    """@luna-comment
    Input:two owned launch receipts/logs. Output:live HTML/JSON with actual progress and loss charts.
    Key Steps:probe receipt PIDs, parse steps/losses, estimate remaining time, mark completion after exit/end log.
    Key Params:30second refresh; median20 iteration times; completion does not check a final checkpoint.
    State/Dependencies:read-only process/log probes; PID identity is not revalidated by this observer.
    """
    import fcntl
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    with (RUN/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(RUN/'monitor.json',dict(pid=os.getpid()))
        while True:
            state=dict(updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),arms={},loss_history={})
            cards=[]
            for arm in ARMS:
                receipt=json.loads((RUN/'launch'/(arm+'.json')).read_text());log=Path(receipt['log'])
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-262144));text=stream.read().decode(errors='replace')
                matches=re.findall(r'iter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)\s*\(([\d.]+)\)',text)
                step,seconds,wait=matches[-1] if matches else ('0','0','0')
                try:os.kill(receipt['pid'],0);status='running' if int(step.replace(',','')) else 'starting'
                except ProcessLookupError:status='exited'
                if 'Traceback (most recent call last)' in text:status='failed'
                if status=='exited' and 'End of training' in text:status='completed';step='250000'
                item=dict(pid=receipt['pid'],gpu=receipt['gpu'],iteration=int(step.replace(',','')),status=status,seconds_per_iter=float(seconds),data_seconds=float(wait))
                item.update(estimate_remaining(item['iteration'],status,[m[1] for m in matches],datetime.now(timezone(timedelta(hours=8)))))
                state['arms'][arm]=item;state['loss_history'][arm]=read_loss_history(log)
                cards.append(f'<article><h2>{html.escape(arm)}</h2><p>{item["iteration"]:,} / 250,000 · {status}</p><p>GPU {item["gpu"]} · {seconds} seconds/step</p></article>')
                remaining=item['estimated_remaining_seconds']
                minutes=(remaining+59)//60 if remaining is not None else None
                duration=(f'{minutes//60}小时 {minutes%60}分' if minutes is not None else '暂不可估算')
                finish=(datetime.fromisoformat(item['estimated_finish_at']).strftime('%m-%d %H:%M') if remaining is not None else '—')
                cards[-1]=cards[-1].replace('</article>',f'<p><strong>预计剩余：{duration}</strong></p><p>预计完成：{finish}（北京时间）</p><small>按近期速度估算，验证、存盘及负载变化可能延后。</small></article>')
            state['both_advancing']=all(a['status']=='running' for a in state['arms'].values())
            save(RUN/'STATUS.json',state)
            page='<!doctype html><meta charset="utf-8"><title>V3/V5 full-image256 training</title><style>body{font:15px Segoe UI,sans-serif;margin:32px;background:#f3f6fa;color:#172638}main{max-width:1180px;margin:auto}.cards{display:flex;gap:20px}article{background:white;padding:24px;border:1px solid #dce4ed;border-radius:12px;flex:1}button{padding:12px;background:#267970;color:white;border:0;border-radius:6px}</style><main><h1>V3 / V5 · Full-image256x256 blocks</h1><p>Real LQ / GT_VOSR pairs. No text-line crop or normalization. Preprocessing mode: '+MODE+'.</p><button onclick="refreshPage()">Refresh now</button><p>Updated: '+state['updated_at']+' · auto refresh30s</p><section class="cards">'+''.join(cards)+'</section>'+render_loss_charts(state['loss_history'])+'</main><script>/* @luna-comment Input:button/timer. Output:cache-busted page reload. Key Steps:replace current URL. Key Params:30000ms timer. State/Dependencies:browser location. */function refreshPage(){const u=new URL(location.href);u.searchParams.set("_refresh",Date.now());location.replace(u.href);}setTimeout(refreshPage,30000);</script>'
            temp=RUN/'STATUS.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(RUN/'STATUS.html')
            if state['both_advancing'] and not (RUN/'BOTH_ARMS_ADVANCING.json').exists():save(RUN/'BOTH_ARMS_ADVANCING.json',state)
            if all(a['status']=='completed' for a in state['arms'].values()):return
            time.sleep(30)


def run():
    """@luna-comment
    Input:approved additional-pair invocation. Output:complete bank, smoke-checked jobs and monitor.
    Key Steps:single-instance lock, offline build, prepare/smoke/launch each arm then detach monitor.
    Key Params:16CPU workers; CUDA2/3 shared only if headroom permits; previous jobs never stopped.
    State/Dependencies:own root/status/logs and existing immutable VOSR source view.
    """
    import fcntl
    DATA.mkdir(parents=True,exist_ok=True);RUN.mkdir(exist_ok=True)
    with (DATA/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            save(DATA/'PIPELINE.json',dict(phase='building',pid=os.getpid(),mode=MODE))
            with (DATA/'build.log').open('ab') as stream:
                subprocess.run([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),
                    '--source',str(PREVIOUS/'source_view'),'--output',str(BANK),'--workers','16','--preprocess-mode',MODE],
                    stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            prepare();save(DATA/'PIPELINE.json',dict(phase='smoking_and_launching',pid=os.getpid(),mode=MODE))
            for arm,gpu in zip(ARMS,(2,3)):
                with (DATA/(arm+'_smoke.log')).open('ab') as stream:
                    subprocess.run([sys.executable,str(__file__),'smoke','--arm',arm],stdout=stream,stderr=subprocess.STDOUT,
                                   check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                launch(arm)
            with (DATA/'monitor.log').open('ab') as stream:
                process=subprocess.Popen([sys.executable,str(__file__),'monitor'],stdout=stream,stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL,start_new_session=True)
            save(DATA/'PIPELINE.json',dict(phase='both_launched',monitor_pid=process.pid,mode=MODE))
        except Exception as error:
            save(DATA/'PIPELINE.json',dict(phase='failed',error=repr(error),mode=MODE));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('run','prepare','smoke','launch','monitor'))
    parser.add_argument('--arm',choices=ARMS);args=parser.parse_args()
    if args.action in ('smoke','launch'):
        if not args.arm:parser.error('--arm required')
        (smoke if args.action=='smoke' else launch)(args.arm)
    else:dict(run=run,prepare=prepare,monitor=monitor)[args.action]()
