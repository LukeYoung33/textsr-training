"""Read-only recent-speed and validation-aware training completion estimates."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import re,time,statistics
TZ=timezone(timedelta(hours=8))
_CACHE={}
_STAMP=re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})')
_TRAIN=re.compile(r'\biter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)')
_VAL=re.compile(r'Validation .*? iter (\d+):')


def read_timing(path):
    """@luna-comment Input:append-only original training log. Output:bounded step samples,validation durations/pending checkpoint event. Key Steps:incremental bytes with incomplete-line retention;parse CST stamps;pair save/validation. Key Params:2000steps/12validation histories retained. State/Dependencies:process-local observer cache,no log writes."""
    path=Path(path)
    if not path.exists():return dict(points=[],validations=[],pending=None)
    stat=path.stat();key=str(path);identity=(stat.st_dev,stat.st_ino)
    state=_CACHE.get(key)
    if not state or state['identity']!=identity or stat.st_size<state['offset']:
        state=dict(identity=identity,offset=0,partial=b'',points=[],validations=[],pending=None);_CACHE[key]=state
    with path.open('rb') as stream:
        stream.seek(state['offset']);lines=(state['partial']+stream.read()).split(b'\n');state['offset']=stream.tell();state['partial']=lines.pop()
    for raw in lines:
        line=raw.decode('utf-8',errors='replace');match=_STAMP.match(line)
        if not match:continue
        epoch=datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=TZ).timestamp()
        step=_TRAIN.search(line)
        if step:state['points'].append(dict(epoch=epoch,step=int(step[1].replace(',','')),seconds=float(step[2])))
        if 'Saving models and training states.' in line and state['points']:
            state['pending']=dict(epoch=epoch,step=state['points'][-1]['step'])
        validation=_VAL.search(line)
        if validation and state['pending'] and int(validation[1])==state['pending']['step']:
            state['validations'].append(max(0,epoch-state['pending']['epoch']));state['pending']=None
    state['points']=state['points'][-2000:];state['validations']=state['validations'][-12:]
    return state


def calculate(timing,step,total=250000,freq=25000,now=None,cutoff=0,validation_fallback=360.,complete=False):
    """@luna-comment Input:parsed timing/currentstep,target/frequency/time/cutoff. Output:forecast seconds/CST and explicit uncertainty/basis. Key Steps:median recent step durations;add future and pending validation;completed zero. Key Params:15min/last20samples,validation last3median,60s overrun floor. State/Dependencies:pure calculation;no training mutation."""
    now=time.time() if now is None else now
    if complete:return dict(remaining_seconds=0,finish_epoch=now,finish_cst='已完成',phase='complete',basis='完成标记',updated_cst=datetime.fromtimestamp(now,TZ).isoformat())
    points=timing['points'];step=max(step,points[-1]['step'] if points else step)
    recent=[p for p in points if p['epoch']>=max(cutoff,now-900) and p['seconds']>0][-20:]
    basis='最近并发训练速度'
    if len(recent)<3:
        recent=recent or [p for p in points[-20:] if p['seconds']>0];basis='并发采样较少，临时估算'
    if not recent:return dict(remaining_seconds=None,finish_cst='采样中',phase='warming_up',basis='等待训练日志',updated_cst=datetime.fromtimestamp(now,TZ).isoformat())
    seconds=statistics.median(p['seconds'] for p in recent)
    val=statistics.median(timing['validations'][-3:]) if timing['validations'] else validation_fallback
    future=max(0,total//freq-step//freq);pending=timing.get('pending');pending_seconds=0;phase='training'
    if pending and pending['step']>=step and pending['step']%freq==0:
        elapsed=max(0,now-pending['epoch']);pending_seconds=max(60,val-elapsed);val=max(val,elapsed);phase='checkpoint_or_validation'
    remaining=max(0,total-step)*seconds+future*val+pending_seconds;finish=now+remaining
    return dict(step=step,remaining_seconds=remaining,finish_epoch=finish,finish_cst=datetime.fromtimestamp(finish,TZ).strftime('%m-%d %H:%M'),phase=phase,seconds_per_iter=seconds,speed_samples=len(recent),future_validations=future,pending_validation_seconds=pending_seconds,validation_seconds=val,basis=basis+'；包含周期验证',updated_cst=datetime.fromtimestamp(now,TZ).strftime('%m-%d %H:%M:%S'))


def duration(seconds):
    """@luna-comment Input:remaining seconds/None. Output:Chinese rounded duration. Key Steps:round up to minute;split hours/minutes. Key Params:finite nonnegative values. State/Dependencies:none."""
    if seconds is None:return '采样中'
    minutes=max(0,int((seconds+59)//60));return str(minutes//60)+'小时'+str(minutes%60)+'分钟' if minutes>=60 else str(minutes)+'分钟'
