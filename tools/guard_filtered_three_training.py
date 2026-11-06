"""Independent per-minute progress and verified-checkpoint guard for owned training."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,os,json,time,math,signal,subprocess,fcntl,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_filtered_three_training import OUT,LABELS,read,alive
from tools.prepare_r1_width_bank import save,digest
from tools.train_latest_vosr_500k import rate


def inspect(kind,current,child,previous,config,now):
    """@luna-comment Input:owned heartbeat/receipt/prior snapshot/config/time. Output:health assessment. Key Steps:check identity,advance,finite logs,LR/data policy,aux ramps and validation-aware staleness. Key Params:10min stall,30min validation grace. State/Dependencies:read-only proc check;no visual ranking."""
    step=current.get('iter',0);age=max(0,now-current.get('updated',child.get('started',now)));live=alive(child);issues=[];logs=current.get('logs',{})
    val_freq=int(config.get('val',{}).get('val_freq') or 25000);grace=1800 if step and step%val_freq==0 else 600
    if not live:issues.append('trainer_not_running')
    if age>grace:issues.append('iteration_stalled')
    if any(not math.isfinite(float(v)) for v in logs.values()):issues.append('nonfinite_loss_or_stat')
    if step and abs(float(current['lr'])-rate(step))>1e-12:issues.append('unexpected_learning_rate')
    if current.get('synthetic_count',0)!=0 or logs.get('gt_mixture_probability',0)!=0 or logs.get('input_paired_fraction',1)!=1:issues.append('input_policy_changed')
    if config['datasets']['train'].get('patch_contrast_alignment',False):issues.append('contrast_policy_changed')
    if logs and abs(float(logs['l_total'])-float(logs['l_native'])-float(logs.get('l_missing',0))-float(logs.get('l_false',0)))>1e-5:issues.append('loss_composition_mismatch')
    start=int(config['train'].get('aux_start_iter',0));ramp=int(config['train'].get('aux_ramp_iters',0));factor=0 if step<start else min(1.,(step-start+1)/ramp) if ramp>0 else 1.
    if kind!='l1':
        for term,target in [('missing',1.),('false',.25)]:
            a,b=logs.get('l_'+term+'_raw'),logs.get('l_'+term)
            if a is not None and b is not None and abs(float(b)-factor*target*float(a))>1e-5:issues.append('unexpected_'+term+'_schedule')
    return dict(kind=kind,iter=step,gpu=child.get('gpu'),pid=child.get('pid'),live=live,heartbeat_age_seconds=age,advance_since_previous_check=step-previous.get('iter',step),lr=current.get('lr'),issues=issues,auxiliary_warmup=kind!='l1' and factor==0)


def checkpoint(kind,binding,cache):
    """@luna-comment Input:owned checkpoint receipts/cache. Output:newest hash/finite verified model-state pair or None. Key Steps:validate immutable stat identity,hashes,CPU tensors and Adam;cache successful check. Key Params:weights-only never resumable. State/Dependencies:read-only CPU torch load,no CUDA allocation."""
    for path in sorted((OUT/'experiments'/binding['name']/'training_states').glob('*.verified.json'),key=lambda p:int(p.name.split('.')[0]),reverse=True):
        try:
            r=read(path);m=Path(r['model']);s=Path(r['state']);signature=dict(receipt_sha256=digest(path),model=[m.stat().st_size,m.stat().st_mtime_ns],state=[s.stat().st_size,s.stat().st_mtime_ns])
            if cache.get(kind,{}).get('signature')==signature:return r
            assert digest(m)==r['model_sha256'] and digest(s)==r['state_sha256']
            import torch
            params=torch.load(m,map_location='cpu')['params'];state=torch.load(s,map_location='cpu');assert state['iter']==r['iter'] and state['optimizers'] and state['mixture_synthetic_count']==0
            assert all(torch.isfinite(v).all() for v in params.values())
            assert all(torch.isfinite(v).all() for opt in state['optimizers'] for item in opt['state'].values() for v in item.values() if torch.is_tensor(v))
            cache[kind]=dict(signature=signature,iter=r['iter'],checked=time.time());return r
        except Exception as error:save(OUT/'guard_rejected_checkpoints'/(kind+'_'+path.name),dict(path=str(path),error=repr(error),updated=time.time()))
    return None


def recover(kind,child,good,assessment,history,now):
    """@luna-comment Input:unhealthy owned run and verified resume point. Output:bounded recovery action or explicit refusal. Key Steps:only exit/stall/nonfinite recoveries;validate exact command/group;terminate own trainer for controller resume. Key Params:max3/hour,no scratch reset or foreign signal. State/Dependencies:owned proc and journal."""
    if not assessment['issues'] or not set(assessment['issues'])<={'trainer_not_running','iteration_stalled','nonfinite_loss_or_stat'}:return None
    if good is None:return dict(kind=kind,action='blocked_no_verified_checkpoint',updated=now)
    recent=[r for r in history if r.get('kind')==kind and now-r['updated']<3600 and r.get('action')=='terminate_for_verified_resume']
    if len(recent)>=3:return dict(kind=kind,action='recovery_rate_limit_reached',updated=now)
    if alive(child):
        pid=child['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        assert str(ROOT/'tools/train_filtered_three.py') in cmd and str(OUT/'configs'/(kind+'_launch.yml')) in cmd and os.getpgid(pid)==pid
        os.killpg(pid,signal.SIGTERM);return dict(kind=kind,action='terminate_for_verified_resume',resume_iter=good['iter'],pid=pid,updated=now)
    return dict(kind=kind,action='controller_resume_expected',resume_iter=good['iter'],updated=now)


def publish(runs,actions,controller_live):
    """@luna-comment Input:health summaries/actions/controller state. Output:JSON/history and health-plus-training page. Key Steps:explicit status/warmup/advance/checkpoints,embed existing monitor. Key Params:60s cadence,no external messages. State/Dependencies:owned outputs only."""
    tz=timezone(timedelta(hours=8));now=time.time();healthy=controller_live and all(not r['issues'] for r in runs.values());snap=dict(updated=now,healthy=healthy,controller_live=controller_live,runs=runs,actions=actions,interval_seconds=60);save(OUT/'SUPERVISION.json',snap)
    with (OUT/'SUPERVISION_HISTORY.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
    body='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>三组训练持续巡检</title><style>body{font:16px/1.6 system-ui;margin:20px;background:#eef3f8;color:#17324d}.cards{display:flex;gap:12px}.card{background:white;padding:15px;border-radius:12px;flex:1}iframe{width:100%;height:1100px;border:0}</style><h1>三组训练 · 每分钟独立巡检</h1><p>'+('当前三组正常，未发现失败或停滞' if healthy else '有需要关注的状态，请查看详情')+' · '+datetime.now(tz).strftime('%m-%d %H:%M:%S')+'</p><div class="cards">'
    for kind,r in runs.items():
        state='已完成' if r.get('complete') else '正常训练（辅助loss预热）' if r.get('auxiliary_warmup') else '正常训练' if not r['issues'] else '需关注：'+', '.join(r['issues'])
        body+='<div class="card"><b>'+html.escape(LABELS[kind])+'</b><p>'+html.escape(state)+'</p><p>GPU '+str(r.get('gpu'))+' · '+str(r['iter'])+' / 500000</p><p>较上次推进 '+str(r.get('advance_since_previous_check',0))+' iter</p><p>已校验断点：'+str(r.get('verified_checkpoint_iter','暂无'))+'</p></div>'
    body+='</div><p>检查进程身份、迭代推进、loss/LR/输入策略与checkpoint更新；异常恢复仅操作本任务进程，且要求已校验断点。</p>'
    if actions:body+='<p>'+html.escape(json.dumps(actions,ensure_ascii=False))+'</p>'
    body+='<p><a href="index.html">训练详情与loss曲线</a> · <a href="SUPERVISION.json">原始巡检记录</a></p><iframe src="index.html"></iframe>';tmp=OUT/'supervision.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'supervision.html')


def run():
    """@luna-comment Input:authorized per-minute invocation. Output:ongoing supervision and limited safe recovery. Key Steps:exclusive lock,heartbeat/config/state checks,controller recovery,checkpoint audit and journal. Key Params:healthy jobs untouched,no idle-GPU condition. State/Dependencies:own cron/controller/trainer receipts."""
    lock=(OUT/'guard.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if read(OUT/'CANCELLED.json').get('cancelled'):return
    import yaml
    bindings=read(OUT/'CONFIG_BINDINGS.json')['runs'];previous=read(OUT/'SUPERVISION.json').get('runs',{});cache=read(OUT/'GUARD_CHECKPOINT_CACHE.json');history=read(OUT/'GUARD_ACTIONS.json').get('actions',[]);now=time.time();runs={};actions=[];controller_live=alive(read(OUT/'CONTROLLER.json'))
    if not controller_live and not read(OUT/'COMPLETE.json').get('passed'):
        with (OUT/'guard_controller_recovery.log').open('ab') as stream:p=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/run_filtered_three_training.py')],cwd=OUT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        actions.append(dict(action='restart_owned_controller',pid=p.pid,updated=now))
    for kind,binding in bindings.items():
        assert digest(binding['config'])==binding['config_sha256'];config=yaml.safe_load(Path(binding['config']).read_text());child=read(OUT/'children'/('train_'+kind+'.json'));current=read(OUT/'audits'/kind/'RUN_STATUS.json');complete=read(OUT/'done'/(kind+'.json')).get('passed',False)
        r=dict(kind=kind,iter=500000,gpu=child.get('gpu'),pid=child.get('pid'),live=alive(child),issues=[],complete=True,advance_since_previous_check=0) if complete else inspect(kind,current,child,previous.get(kind,{}),config,now)
        good=checkpoint(kind,binding,cache);r['verified_checkpoint_iter']=good['iter'] if good else None
        if current.get('iter',0)>=10000 and (good is None or current['iter']-good['iter']>=10000):r['issues'].append('checkpoint_not_advancing')
        if not complete and current.get('iter',0)>=500000 and good and good['iter']==500000 and alive(child):r['issues']=[];r['final_validation']=True
        action=recover(kind,child,good,r,history,now)
        if action:actions.append(action)
        runs[kind]=r
    save(OUT/'GUARD_CHECKPOINT_CACHE.json',cache)
    if actions:save(OUT/'GUARD_ACTIONS.json',dict(actions=(history+actions)[-1000:]))
    publish(runs,actions,controller_live or bool(read(OUT/'COMPLETE.json').get('passed')))
    print(json.dumps(dict(healthy=all(not r['issues'] for r in runs.values()),runs={k:dict(iter=r['iter'],issues=r['issues'],live=r['live']) for k,r in runs.items()},actions=actions)),flush=True)


if __name__=='__main__':run()
