"""Durable complete-GT-before-training controller for two matched 500K controls."""
from pathlib import Path
import sys,os,json,time,copy,subprocess,fcntl,html,traceback
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_mask_repaired_vosr_mix50 import OUT,DATA,save,digest
TRAINER=ROOT/'tools/train_mask_repaired_mix50.py'
LABELS={'l1':'V3M · 修复GT · L1-only · 50%GT退化','r1_ori':'V3M · 修复GT · L1+R1_ori · 50%GT退化'}


def read(path):
    """@luna-comment Input:owned JSON path. Output:document or empty. Key Steps:read atomic snapshots. Key Params:missing permitted,corruption raises. State/Dependencies:filesystem."""
    p=Path(path);return json.loads(p.read_text()) if p.exists() else {}


def alive(record):
    """@luna-comment Input:owned launch receipt. Output:identity-checked process state. Key Steps:match proc start token and nonzombie status. Key Params:no PID-only trust. State/Dependencies:Linux proc."""
    try:
        stat=Path(f'/proc/{record["pid"]}/stat').read_text().split();return stat[2]!='Z' and stat[21]==record['start_token']
    except (OSError,KeyError):return False


def configure():
    """@luna-comment Input:old hash-bound native-L1 recipe/init. Output:two frozen matched configurations. Key Steps:replace targets/model only;enable original MF in second arm. Key Params:scratch500K,mix50,sharedseed,schedule1e-4to5e-6. State/Dependencies:owned configs only."""
    import yaml
    old=DATA/'latest_vosr_four_500k_20261102';binding=read(old/'CONFIG_BINDINGS.json')['runs']['v3m_l1']
    assert digest(binding['config'])==binding['config_sha256'];base=yaml.safe_load(Path(binding['config']).read_text())
    assert digest(base['path']['pretrain_network_g'])==binding['initial_sha256']
    configs={}
    for index,arm in enumerate(LABELS):
        cfg=copy.deepcopy(base);cfg['name']='v3m_repaired_'+arm+'_mix50_500k_20261104';cfg['model_type']='MaskRepairedMix50';cfg['train']['model_type']='MaskRepairedMix50'
        cfg['path'].update(experiments_root=str(OUT/'experiments'),resume_state=None)
        cfg['train'].update(total_iter=500000,recall_aware_weight_missing=1. if arm=='r1_ori' else 0.,recall_aware_weight_false=.25 if arm=='r1_ori' else 0.,recall_aware_opt=dict(ring_radius=3,margin=.03,temperature=.01,min_polarity_contrast=.03) if arm=='r1_ori' else None)
        cfg['train']['optim_g']['lr']=1e-4;cfg['train']['scheduler']={'type':'LatestVOSR500K'}
        cfg['s22_strategy'].update(name='mixed50',audit_root=str(OUT/'audits'/arm));cfg['dist_params']['port']=36240+index
        cfg['logger'].update(save_checkpoint_freq=5000,print_freq=100,use_tb_logger=False)
        for split in ['train','val']:
            cfg['datasets'][split].update(root=str(OUT/'bank'),manifest=str(OUT/'bank'/(split+'.json')))
        cfg['v3pre_target_domain']='Train:VOSR+S2.2 repaired with M2/local original ink/adjacent background;Mojito region-only originals retained. Val:historical620 unchanged.'
        path=OUT/'configs'/(arm+'.yml');path.parent.mkdir(exist_ok=True);payload=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
        if path.exists():assert path.read_text()==payload,'Frozen recipe changed'
        else:path.write_text(payload)
        configs[arm]=dict(config=str(path),config_sha256=digest(path),initial=cfg['path']['pretrain_network_g'],initial_sha256=binding['initial_sha256'],name=cfg['name'])
    assert configs['l1']['initial_sha256']==configs['r1_ori']['initial_sha256']
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=configs,total_iter=500000,mixture=.5,r1_ori=dict(missing=1,false=.25,width_reweighting=False),schedule=[[0,1e-4],[200000,5e-5],[300000,2.5e-5],[400000,1e-5],[450000,5e-6]]))
    return configs


def publish(phase,extra=None):
    """@luna-comment Input:pipeline stage/live heartbeat. Output:Chinese page and unified-monitor compatible JSON. Key Steps:distinct GT/bank/queue/train status;rolling ETA,loss records. Key Params:no running/completion claims without receipts. State/Dependencies:owned output only."""
    now=time.time();cards=[];histories={};tz=timezone(timedelta(hours=8))
    for arm,label in LABELS.items():
        launch=read(OUT/'children'/('train_'+arm+'.json'));status=read(OUT/'audits'/arm/'TRAIN_STATUS.json');done=read(OUT/'done'/(arm+'.json'))
        step=500000 if done.get('passed') else status.get('iter',0);records={};p=OUT/'audits'/arm/'loss.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                r=json.loads(line);records[r['iter']]=dict(step=r['iter'],values=r['logs'],updated=r['updated'])
        history=sorted(records.values(),key=lambda x:x['step']);histories[label]=history
        eta=None
        # Short and long windows use the same strictly measured wall-time slope.
        if len(history)>2 and now-status.get('updated',0)<300:
            window=history[-50:];a,b=window[0],window[-1]
            if b['step']>a['step']:eta=(500000-step)*(b['updated']-a['updated'])/(b['step']-a['step'])
        state='已完成' if done.get('passed') else '训练中' if alive(launch) else '数据完成后自动启动' if not (OUT/'DATA_READY.json').exists() else '等待空闲GPU / 启动检查'
        finish=datetime.fromtimestamp(now+eta,tz).strftime('%m-%d %H:%M') if eta is not None else '开训后按实测速度更新'
        cards.append(dict(label=label,status=state,current=step,total=500000,gpu=launch.get('gpu'),details=['LR '+str(status.get('lr',1e-4)),'预计完成：'+finish,'从头训练；50% GT轻度退化；'+('Missing=1 / False=0.25，无粗细重加权' if arm=='r1_ori' else '仅原生L1')]))
    data=read(OUT/'DATA_STATUS.json');bank=read(OUT/'bank/progress.json');side=read(OUT/'bank/line_regions/progress.json')
    mapping={'repairing_gt':'逐笔画修复GT','bank':'生成离线256训练块','finalize':'生成文字区域缓存与有效采样索引','ready':'数据已完成'}
    stage=data.get('phase','准备输入清单');current=data.get('completed',0);total=data.get('total',8352);finish=data.get('estimated_finish')
    if phase=='bank':stage='bank';current=bank.get('completed',0);total=bank.get('total',8352);elapsed=bank.get('seconds',0);finish=None
    elif phase=='finalize':stage='finalize';current=side.get('done',0);total=side.get('total',8352);elapsed=side.get('seconds',0);finish=None
    else:elapsed=0
    if stage in ['bank','finalize'] and current>0:finish=datetime.fromtimestamp(now+(total-current)*elapsed/current,tz).strftime('%m-%d %H:%M')
    if (OUT/'DATA_READY.json').exists():stage='ready';current=total=8352;finish='已完成'
    dataset=dict(label='全量GT修复 → 离线训练数据',status=mapping.get(stage,stage),current=current,total=total,details=['8,109张逐笔画修复；Mojito训练243张保留原GT','当前阶段预计完成：'+(finish or '首批完成后更新'),'复用V3输入、mask、OCR；不重跑教师模型','训练mask：修复来源用M2，Mojito保留原区域mask；验证620张不变'])
    save(OUT/'monitor.json',dict(updated=datetime.now(tz).isoformat(),runs=cards,datasets=[dataset],lossHistory=histories,phase=phase,extra=extra))
    save(OUT/'CONTROLLER_STATUS.json',dict(pid=os.getpid(),phase=phase,updated=now,extra=extra))
    body='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>修复GT · 两组500K</title><style>body{font:17px/1.6 system-ui;background:#eef3f8;color:#17324d;max-width:1250px;margin:30px auto}section{display:flex;gap:18px}article{background:white;padding:24px;border-radius:14px;flex:1;min-width:0}progress{width:100%;height:18px}small{color:#567}</style><h1>修复GT · V3M 两组500K训练</h1><p>数据全量完成并通过检查后自动开训；两组分配不同空闲GPU。</p>'
    for group in [[dataset],cards]:
        body+='<section>'
        for c in group:body+='<article><h2>'+html.escape(c['label'])+'</h2><b>'+html.escape(c['status'])+'</b>　GPU '+str(c.get('gpu','—'))+'<p>'+str(c['current'])+' / '+str(c['total'])+'</p><progress max="'+str(c['total'])+'" value="'+str(c['current'])+'"></progress><p>'+'</p><p>'.join(html.escape(v) for v in c['details'])+'</p></article>'
        body+='</section><br>'
    if extra:body+='<p>'+html.escape(str(extra))+'</p>'
    body+='<small>更新：'+datetime.now(tz).strftime('%m-%d %H:%M:%S')+' · 20秒刷新 · <a href="monitor.json">原始状态</a> · <a href="SOURCE_BINDING.json">数据来源</a></small>'
    tmp=OUT/'index.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'index.html')


def launch(stage,command,gpu=None):
    """@luna-comment Input:owned stage command/GPU. Output:adopted or detached child. Key Steps:identity-check,isolated env,receipt with start token. Key Params:one owned child per stage. State/Dependencies:Linux process group/logs."""
    path=OUT/'children'/(stage+'.json');record=read(path)
    if alive(record):return record
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',REAL_ESRGAN_DEFER_FINAL_VALIDATION='1');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    (OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'logs'/(stage+'.log')).open('ab') as stream:p=subprocess.Popen(command,cwd=OUT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,start_token=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=time.time(),gpu=gpu,command=command)
    save(path,record);return record


def stage(name,completion):
    """@luna-comment Input:data stage/completion receipt. Output:checked stage completion. Key Steps:launch/adopt,heartbeat every20s,require successful receipt. Key Params:partial state resumable. State/Dependencies:known preparation script."""
    if completion.exists():return
    record=launch('data_'+name,[sys.executable,'-u',str(ROOT/'tools/prepare_mask_repaired_vosr_mix50.py'),'--stage',name])
    while alive(record):publish(name);time.sleep(20)
    assert completion.exists(),'Stage '+name+' failed; see logs/data_'+name+'.log'


def resume_recipe(arm,binding):
    """@luna-comment Input:frozen arm/binding. Output:launch config with verified latest optimizer pair. Key Steps:rehash config/init;descending paired checkpoint validation;refuse progress without usable state. Key Params:own experiment only. State/Dependencies:torch/yaml checkpoint reads."""
    import yaml,torch
    assert digest(binding['config'])==binding['config_sha256'] and digest(binding['initial'])==binding['initial_sha256']
    cfg=yaml.safe_load(Path(binding['config']).read_text());experiment=OUT/'experiments'/binding['name'];found=False
    for path in sorted((experiment/'training_states').glob('*.verified.json'),key=lambda p:int(p.name.split('.')[0]),reverse=True):
        try:
            r=read(path);assert digest(r['model'])==r['model_sha256'] and digest(r['state'])==r['state_sha256']
            state=torch.load(r['state'],map_location='cpu');assert state['iter']==r['iter'] and 'mix_rng' in state
            cfg['path']['resume_state']=r['state'];found=True;break
        except Exception as error:save(OUT/(arm+'_checkpoint_rejected.json'),dict(path=str(path),error=repr(error)))
    if not found and read(OUT/'audits'/arm/'TRAIN_STATUS.json').get('iter',0)>3:raise RuntimeError('Progress exists without resumable checkpoint: '+arm)
    target=OUT/'configs'/(arm+'_launch.yml');target.write_text(yaml.safe_dump(cfg,sort_keys=False));return target


def run():
    """@luna-comment Input:authorized automatic pipeline. Output:full repaired dataset followed by matched two training runs. Key Steps:exclusive lock,data gates,CUDA smoke,idle-GPU assignment,verified completion/retry. Key Params:20/30s polling,never foreign-job termination. State/Dependencies:owned receipts/configs only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),start_token=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],started=time.time()))
    configs=configure()
    failures=0
    while True:
        try:
            stage('gt',OUT/'GT_COMPLETE.json');stage('bank',OUT/'bank/COMPLETE.json');stage('finalize',OUT/'DATA_READY.json')
            ready=read(OUT/'DATA_READY.json');assert ready['passed'] and digest(OUT/'bank/train.json')==ready['train_sha256'] and digest(OUT/'bank/val.json')==ready['val_sha256']
            from tools.launch_v3m_aux_factorial import gpu_inventory,choose_idle_gpu
            while True:
                for arm,binding in configs.items():
                    if read(OUT/'done'/(arm+'.json')).get('passed'):continue
                    child=read(OUT/'children'/('train_'+arm+'.json'))
                    if alive(child):continue
                    final=OUT/'experiments'/binding['name']/'training_states/500000.verified.json'
                    if final.exists():
                        r=read(final);assert digest(r['model'])==r['model_sha256'] and digest(r['state'])==r['state_sha256']
                        import torch
                        state=torch.load(r['state'],map_location='cpu');assert state['iter']==500000 and abs(state['optimizers'][0]['param_groups'][0]['lr']-5e-6)<1e-12
                        save(OUT/'done'/(arm+'.json'),dict(passed=True,checkpoint=r['model'],checkpoint_sha256=r['model_sha256'],iter=500000));continue
                    free,busy=gpu_inventory();owned={read(OUT/'children'/('train_'+a+'.json')).get('gpu') for a in LABELS if alive(read(OUT/'children'/('train_'+a+'.json')))}
                    gpu=choose_idle_gpu(free,busy,owned)
                    if gpu is None:continue
                    smoke=OUT/(arm+'_SMOKE.json')
                    if not smoke.exists():
                        import yaml
                        cfg=yaml.safe_load(Path(binding['config']).read_text());cfg['name']+='_smoke';cfg['s22_strategy']['audit_root']=str(OUT/'smoke_audits'/arm)
                        path=OUT/'configs'/(arm+'_smoke.yml');path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                        record=launch('smoke_'+arm,[sys.executable,'-u',str(TRAINER),'--smoke-config',str(path),'--smoke-output',str(smoke)],gpu)
                        while alive(record):publish('CUDA启动检查',dict(arm=arm,gpu=gpu));time.sleep(20)
                        assert read(smoke).get('passed'),'CUDA smoke failed: '+arm
                    path=resume_recipe(arm,binding);launch('train_'+arm,[sys.executable,'-u',str(TRAINER),'-opt',str(path)],gpu)
                if all(read(OUT/'done'/(a+'.json')).get('passed') for a in LABELS):save(OUT/'COMPLETE.json',dict(passed=True,arms=2,total_iter=500000));publish('全部完成');return
                publish('训练 / 等待空闲GPU');time.sleep(30)
        except Exception as error:
            failures+=1;record=dict(error=repr(error),traceback=traceback.format_exc(),attempt=failures,updated=time.time());save(OUT/'LAST_ERROR.json',record);publish('异常，等待自动重试',record);time.sleep(min(60,10*failures))


if __name__=='__main__':run()
