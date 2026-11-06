"""Adopt running S2.2 trainers and start all pending recipes across four GPUs."""
from pathlib import Path
from datetime import datetime
import json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.schedule_full_s22_training import OUT,TZ,ready,publish
from tools.run_uniform20_steep_scurve import save,digest


def alive(pid,config):
    """@luna-comment Input:recordedPID and exact owned config path. Output:true only matching live non-zombie trainer. Key Steps:read proc cmdline/state;check trainer and config arguments. Key Params:no PID-only adoption. State/Dependencies:read-only Linux proc."""
    path=Path('/proc')/str(pid)
    try:
        args=(path/'cmdline').read_bytes().split(b'\0')
        return str(config).encode() in args and str(ROOT/'tools/train_full_s22_strategy.py').encode() in args and (path/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False


def memory():
    """@luna-comment Input:currentGPU inventory. Output:index/free/usedMiB rows. Key Steps:query nvidia-smi CSV. Key Params:4known3090 cards. State/Dependencies:read-only driver."""
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(parts[0]):dict(free=int(parts[1]),used=int(parts[2])) for line in text.splitlines() if (parts:=line.split(','))}


def monitor(records,launches,children):
    """@luna-comment Input:immutable config records and adopted/newPID mapping. Output:running/paused monitor,GPU/ETA/done/failure receipts. Key Steps:read explicit pause policy;check process/checkpoint;publish losses;incremental recent-speed/validation ETA;embed forecast cards. Key Params:no implicit restart on failure. State/Dependencies:owned logs/receipts and read-only ETA cache."""
    states={};failed=[]
    policy_path=OUT/'RUN_POLICY.json';policy=json.loads(policy_path.read_text()) if policy_path.exists() else {};paused={name for name in policy.get('paused',[]) if not (OUT/'training/done'/(name+'.json')).exists()}
    for name,record in records.items():
        launch=launches[name];pid=launch['pid'];running=alive(pid,record['config'])
        final=OUT/'training/experiments'/record['name']/'models/net_g_250000.pth'
        done=OUT/'training/done'/(name+'.json')
        returncode=children[name].poll() if name in children else None
        if name in paused:
            states[name]=dict(pid=pid,gpu=launch['gpu'],state='paused');continue
        if not running and not done.exists():
            if final.exists() and returncode in (None,0):save(done,dict(passed=True,final=str(final),final_sha256=digest(final),completed_epoch=time.time(),adopted=name not in children))
            else:failed.append(dict(name=name,pid=pid,returncode=returncode))
        states[name]=dict(pid=pid,gpu=launch['gpu'],state='running' if running else 'complete' if done.exists() else 'failed')
    if failed:save(OUT/'TRAINING_FAILED.json',dict(errors=failed,epoch=time.time(),policy='Other concurrent runs continue;failed configs are not silently restarted'))
    counts={gpu:sum(r['state']=='running' and r['gpu']==gpu for r in states.values()) for gpu in range(4)}
    save(OUT/'CONCURRENCY_STATE.json',dict(runs=states,active=sum(counts.values()),per_gpu=counts,memory=memory(),updated_cst=datetime.now(TZ).isoformat()))
    publish('可信边缘抵消保护：Full / Text×1.5 / Adaptive-L1' if policy.get('mode')=='guard_only' else '按当前策略训练')
    from tools.s22_training_eta import read_timing,calculate,duration
    import html,statistics
    handoff=json.loads((OUT/'CONCURRENCY_HANDOFF.json').read_text());cutoff=datetime.fromisoformat(policy.get('updated_cst',handoff['started_cst'])).timestamp()
    timings={name:read_timing(OUT/'training/logs'/(name+'.log')) for name in records}
    estimates={};snapshot=json.loads((OUT/'MONITOR.json').read_text())
    for name,record in records.items():
        reference=timings[record['strategy']+'_full']['validations'] or timings['paired_full']['validations']
        fallback=statistics.median(reference[-3:]) if reference else 360.
        estimate=calculate(timings[name],snapshot['runs'][name]['step'],cutoff=cutoff,validation_fallback=fallback,complete=states[name]['state']=='complete');estimates[name]=estimate
        if name in paused:estimate=dict(remaining_seconds=None,finish_cst='已暂停',phase='paused',basis='进程挂起，保留内存状态',updated_cst=datetime.now(TZ).isoformat());estimates[name]=estimate
        snapshot['runs'][name]['state']=states[name]['state']
        snapshot['runs'][name]['eta']=estimate
    save(OUT/'TRAINING_ETA.json',dict(runs=estimates,updated_cst=datetime.now(TZ).isoformat(),method='Recent step durations plus observed periodic validation;updates every20s;contention may change'))
    save(OUT/'MONITOR.json',snapshot)
    page=OUT/'dashboard.html';body=page.read_text(encoding='utf-8')
    for name,estimate in estimates.items():
        heading='<h2>'+html.escape(snapshot['runs'][name]['label'])+'</h2>'
        phase='已暂停' if estimate['phase']=='paused' else '检查点／验证中' if estimate['phase']=='checkpoint_or_validation' else '训练中' if estimate['phase']=='training' else '已完成' if estimate['phase']=='complete' else '速度采样中'
        extra='<p><strong>预计完成：'+html.escape(estimate['finish_cst'])+'</strong><br>预计剩余：'+duration(estimate['remaining_seconds'])+' · '+phase+'</p><p class="muted">'+html.escape(estimate['basis'])+'<br>更新：'+html.escape(estimate['updated_cst'])+'；实际可能随共享资源变化。</p>'
        if name in paused:
            start=body.find(heading);end=body.find('</section>',start)
            if start>=0 and end>=0:body=body[:start]+body[start:end].replace('训练中','已暂停').replace('排队中','已暂停')+body[end:]
            extra='<p><strong>已暂停</strong> · 保留模型与优化器内存状态，停止GPU计算。</p>'
        body=body.replace(heading,heading+extra,1)
    summary='<section><h2>训练调度</h2><p>当前运行 '+str(sum(counts.values()))+' 组；已暂停 '+str(len(paused))+' 组。暂停组保留显存以便原地恢复。</p></section>'
    marker='<section><h2>全量数据处理</h2>';body=body.replace(marker,summary+marker,1)
    temporary=OUT/'dashboard.html.tmp';temporary.write_text(body,encoding='utf-8');temporary.replace(page)
    return states


def main():
    """@luna-comment Input:user-authorized handoff or monitor reattachment. Output:adopted existing trainers and any pending starts,shared monitor. Key Steps:strict readiness/config/smoke/init;terminate owned predecessor controller only;exclusive lock;balanced starts;PID/checkpoint/ETA supervision. Key Params:15jobs,4GPU,2s startup stagger,4GiB headroom. State/Dependencies:trainers never terminated/reset;original handoff receipt preserved."""
    import fcntl
    assert ready(OUT)
    binding=json.loads((OUT/'CONFIG_BINDINGS.json').read_text());records=binding['runs'];assert len(records)==15
    assert json.loads((OUT/'SMOKE_COMPLETE.json').read_text())['count']==15
    assert digest(binding['shared_initial'])==binding['shared_initial_sha256']
    for record in records.values():assert digest(record['config'])==record['sha256']
    launches={}
    for name,record in records.items():
        path=OUT/'training/launch'/(name+'.json')
        if path.exists():
            launch=json.loads(path.read_text());assert launch['config_sha256']==record['sha256']
            assert alive(launch['pid'],record['config']) or (OUT/'training/done'/(name+'.json')).exists()
            launches[name]=launch
    old=json.loads((OUT/'SCHEDULER_PROCESS.json').read_text());oldpid=old['pid'];cmdline=(Path('/proc')/str(oldpid)/'cmdline').read_bytes()
    assert any(str(ROOT/'tools'/name).encode() in cmdline for name in ['schedule_full_s22_training.py','run_s22_concurrent_training.py']),'Refusing unexpected controller'
    adopted={name:dict(**launch) for name,launch in launches.items()}
    handoff_name='CONCURRENCY_REATTACH.json' if (OUT/'CONCURRENCY_HANDOFF.json').exists() else 'CONCURRENCY_HANDOFF.json'
    save(OUT/handoff_name,dict(previous_controller=oldpid,preserved_trainers=adopted,started_cst=datetime.now(TZ).isoformat(),policy='Terminate only controller;trainers have separate sessions'))
    os.kill(oldpid,signal.SIGTERM)
    lock=(OUT/'scheduler.lock').open('a')
    for _ in range(100):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(.1)
    else:raise RuntimeError('Previous scheduler lock not released')
    save(OUT/'SCHEDULER_PROCESS.json',dict(pid=os.getpid(),started_cst=datetime.now(TZ).isoformat(),controller=str(Path(__file__)),mode='15concurrent'))
    children={};streams=[];counts={gpu:sum(launch['gpu']==gpu for launch in launches.values()) for gpu in range(4)}
    for name,record in records.items():
        if name in launches:continue
        policy_path=OUT/'RUN_POLICY.json'
        if policy_path.exists() and name in json.loads(policy_path.read_text()).get('paused',[]):continue
        inventory=memory();eligible=[gpu for gpu in range(4) if inventory[gpu]['free']>=4096]
        assert eligible,'Insufficient safe startup memory'
        gpu=min(eligible,key=lambda i:(counts[i],inventory[i]['used'],i))
        log=OUT/'training/logs'/(name+'.log');assert not log.exists(),'Refusing untracked log overwrite'
        stream=log.open('xb');streams.append(stream)
        command=[sys.executable,'-u',str(ROOT/'tools/train_full_s22_strategy.py'),'-opt',record['config']]
        process=subprocess.Popen(command,cwd=OUT/'training',stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        launch=dict(pid=process.pid,gpu=gpu,original_queue_gpu=record['gpu'],command=command,config_sha256=record['sha256'],started_cst=datetime.now(TZ).isoformat(),mode='concurrent')
        save(OUT/'training/launch'/(name+'.json'),launch);launches[name]=launch;children[name]=process;counts[gpu]+=1
        print('LAUNCHED',name,'GPU',gpu,'PID',process.pid,flush=True);time.sleep(2)
    launch_receipt='CONCURRENT_REATTACH_COMPLETE.json' if (OUT/'CONCURRENT_LAUNCH_COMPLETE.json').exists() else 'CONCURRENT_LAUNCH_COMPLETE.json'
    save(OUT/launch_receipt,dict(passed=True,total=15,preserved=len(adopted),new=len(children),per_gpu=counts,completed_cst=datetime.now(TZ).isoformat()))
    while True:
        states=monitor(records,launches,children)
        if not any(r['state']=='running' for r in states.values()):break
        time.sleep(20)
    assert all(r['state'] in ('complete','paused') for r in states.values()),'One or more concurrent runs failed'
    save(OUT/'TRAINING_COMPLETE.json',dict(passed=True,count=sum(r['state']=='complete' for r in states.values()),paused=[n for n,r in states.items() if r['state']=='paused'],mode='active_subset',completed_cst=datetime.now(TZ).isoformat()))
    for stream in streams:stream.close()


if __name__=='__main__':
    try:main()
    except Exception as error:save(OUT/'CONCURRENCY_FAILED.json',dict(error=repr(error),epoch=time.time()));raise
