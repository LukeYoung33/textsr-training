"""Isolated two-arm scratch guard experiment with per-patch GT contrast matching."""
from pathlib import Path
from datetime import datetime
import copy,html,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.schedule_full_s22_training import OUT as SOURCE,ready,TZ
from tools.run_uniform20_steep_scurve import save,digest
OUT=SOURCE.parent/'s22_patch_contrast_guard_20261029'
ARMS={'full_sdf':(0,'Full + SDF'),'full_nosdf':(3,'Full without SDF')}


def prepare():
    """@luna-comment Input:completed original bank/shared init and historical SDF JSON. Output:two immutable configs plus provenance. Key Steps:clone guardFull;change isolated paths/contrast flag/SDF only;bind hashes. Key Params:scratch250K,batch4,same seed/init/data,guard ratio.5. State/Dependencies:read-only existing data;owned output only."""
    import yaml
    assert ready(SOURCE)
    base=yaml.safe_load((SOURCE/'training/configs/guard_full.yml').read_text())
    sdf=json.loads((OUT/'SDF_REFERENCE.json').read_text())
    assert base['train']['sdf_boundary_weight']==0 and base['paired_probability']==1
    records={}
    for arm,(gpu,label) in ARMS.items():
        cfg=copy.deepcopy(base);cfg['name']='s22_patchcontrast_guard_'+arm+'_20261029'
        cfg['s22_patch_contrast_experiment']=True;cfg['datasets']['train']['patch_contrast_alignment']=True
        cfg['datasets']['val']['patch_contrast_alignment']=False
        cfg['path']['experiments_root']=str(OUT/'experiments');cfg['path']['resume_state']=None
        cfg['s22_strategy']['audit_root']=str(OUT/'audits'/arm)
        if arm=='full_sdf':cfg['train'].update(sdf)
        path=OUT/'configs'/(arm+'.yml');path.parent.mkdir(exist_ok=True)
        assert not path.exists();path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke/audits'/arm)
        sp=OUT/'configs'/(arm+'_smoke.yml');sp.write_text(yaml.safe_dump(smoke,sort_keys=False))
        records[arm]=dict(config=str(path),sha256=digest(path),smoke_config=str(sp),gpu=gpu,label=label,name=cfg['name'],initial=cfg['path']['pretrain_network_g'],initial_sha256=digest(cfg['path']['pretrain_network_g']))
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=records,source_manifest_sha256=digest(SOURCE/'source_view/manifests/train.jsonl'),trainer_sha256=digest(ROOT/'tools/train_full_s22_strategy.py'),contrast_sha256=digest(ROOT/'tools/patch_contrast_alignment.py'),validation='Original583 older GT_VOSR with no contrast matching;target domain differs from training'))
    return records


def publish(records,processes,phase):
    """@luna-comment Input:exact run records/owned child states. Output:Chinese two-run dashboard with distinct losses,contrast diagnostics and validation-aware ETA. Key Steps:read audits/logs;derive process states;atomic HTML/JSON. Key Params:30s refresh,no forecast for stopped/complete. State/Dependencies:observer only;immutable datasets untouched."""
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    from tools.s22_training_eta import read_timing,calculate,duration
    histories={};states={};cards=[]
    for arm,record in records.items():
        path=OUT/'audits'/arm/'STATUS.json';status=json.loads(path.read_text()) if path.exists() else {}
        step=status.get('iter',0);process=processes.get(arm);code=process.poll() if process else None
        final=OUT/'experiments'/record['name']/'models/net_g_250000.pth'
        state='complete' if process and code==0 and final.exists() else 'failed' if process and code is not None else 'running' if process else 'waiting'
        history=read_loss_history(OUT/'logs'/(arm+'.log'))
        if history:histories[record['label']]=history
        timing=read_timing(OUT/'logs'/(arm+'.log'));eta=calculate(timing,step,complete=state=='complete')
        states[arm]=dict(label=record['label'],state=state,step=250000 if state=='complete' else step,gpu=record['gpu'],pid=process.pid if process else None,returncode=code,eta=eta,logs=status.get('logs',{}))
        logs=status.get('logs',{})
        text={'running':'\u8bad\u7ec3\u4e2d','complete':'\u5df2\u5b8c\u6210','failed':'\u5931\u8d25','waiting':'\u68c0\u67e5\u4e2d'}[state]
        cards.append('<section><h2>'+html.escape(record['label'])+'</h2><p>'+text+' | GPU '+str(record['gpu'])+' | '+str(states[arm]['step'])+' / 250000</p><progress max="250000" value="'+str(states[arm]['step'])+'"></progress><p>\u9884\u8ba1\u5b8c\u6210: '+html.escape(eta['finish_cst'])+' | \u5269\u4f59: '+duration(eta['remaining_seconds'])+'</p><p>\u5bf9\u6bd4\u5ea6\u589e\u76ca: '+str(round(logs.get('contrast_gain',1),3))+' | \u6821\u6b63\u6bd4\u4f8b: '+str(round(logs.get('contrast_applied',0),3))+' | \u88c1\u526a\u6bd4\u4f8b: '+str(round(logs.get('contrast_clipping_fraction',0),4))+'</p></section>')
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>GT patch contrast guard</title><style>body{font:16px/1.7 system-ui;margin:24px;background:#edf2f8}section{background:white;padding:20px;margin:14px 0;border-radius:12px}progress{width:100%}</style><h1>\u5757\u7ea7GT\u5bf9\u6bd4\u5ea6\u5bf9\u9f50 | \u53ef\u4fe1\u8fb9\u7f18\u62b5\u6d88\u4fdd\u62a4</h1><p>'+html.escape(phase)+' | '+datetime.now(TZ).isoformat()+'</p><p>V3M | 4873\u5bf9 | 256x256 | \u4ece\u5934\u8bad\u7ec3250K | \u540c\u4e00\u521d\u59cb\u5316 | \u4ec5SDF\u5f00\u5173\u4e0d\u540c</p><p>\u9a8c\u8bc1\u96c6583\u5f20\u4fdd\u7559\u65e7GT_VOSR\uff0c\u4e0d\u505a\u5bf9\u6bd4\u5ea6\u6821\u6b63\uff1b\u4e0e\u8bad\u7ec3GT\u57df\u4e0d\u540c\u3002</p>'+''.join(cards)+render_loss_charts(histories)+'</html>'
    temp=OUT/'dashboard.html.tmp';temp.write_text(body,encoding='utf-8');temp.replace(OUT/'dashboard.html')
    save(OUT/'MONITOR.json',dict(phase=phase,runs=states,updated_cst=datetime.now(TZ).isoformat()))
    return states


def main():
    """@luna-comment Input:user-authorized isolated experiment root. Output:two gated detached trainers and persistent monitoring/complete receipts. Key Steps:exclusive lock;configs or explicit failed-prelaunch recovery;CUDA smokes;launch GPU0/3;observe exact owned children. Key Params:never resume/reset old jobs,no implicit failure restart. State/Dependencies:shared ready bank and source trainer;owned logs/audits/checkpoints only."""
    import fcntl
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'CONFIG_BINDINGS.json').exists():
        assert '--recover-startup' in sys.argv and (OUT/'FAILED.json').exists(),'Refusing duplicate experiment startup'
        assert not (OUT/'LAUNCH_COMPLETE.json').exists() and not any(OUT.glob('*_LAUNCH.json')),'Cannot recover a formally launched experiment'
        binding=json.loads((OUT/'CONFIG_BINDINGS.json').read_text());records=binding['runs'];stamp=str(int(time.time()))
        for arm,record in records.items():
            assert digest(record['config'])==record['sha256'];record['gpu']=ARMS[arm][0]
            log=OUT/'smoke'/(arm+'.log')
            if log.exists():log.rename(log.with_name(log.name+'.failed_startup_'+stamp))
        (OUT/'FAILED.json').rename(OUT/('FAILED_STARTUP_'+stamp+'.json'))
        binding['runs']=records;save(OUT/'CONFIG_BINDINGS.json',binding)
        save(OUT/'STARTUP_RECOVERY.json',dict(reused_configs=True,formal_training_previously_started=False,gpus={a:r['gpu'] for a,r in records.items()},epoch=time.time()))
    else:records=prepare()
    processes={};streams=[]
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started_cst=datetime.now(TZ).isoformat()))
    for folder in ['logs','smoke']: (OUT/folder).mkdir(exist_ok=True)
    publish(records,processes,'CUDA smoke checks')
    for arm,record in records.items():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(record['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        command=[sys.executable,'-u',str(ROOT/'tools/train_full_s22_strategy.py')]
        with (OUT/'smoke'/(arm+'.log')).open('xb') as stream:
            result=subprocess.run(command+['--smoke-config',record['smoke_config'],'--smoke-output',str(OUT/'smoke'/(arm+'.json'))],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=OUT)
        assert result.returncode==0,arm+' smoke failed'
        receipt=json.loads((OUT/'smoke'/(arm+'.json')).read_text());assert receipt['passed'] and receipt['updates']==9 and receipt['initial_exact'] and receipt['no_sdf']==(arm=='full_nosdf')
    save(OUT/'SMOKE_COMPLETE.json',dict(passed=True,count=2,completed_cst=datetime.now(TZ).isoformat()))
    for arm,record in records.items():
        assert digest(record['config'])==record['sha256']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(record['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        stream=(OUT/'logs'/(arm+'.log')).open('xb');streams.append(stream)
        command=[sys.executable,'-u',str(ROOT/'tools/train_full_s22_strategy.py'),'-opt',record['config']]
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=OUT,start_new_session=True,env=env);processes[arm]=process
        save(OUT/(arm+'_LAUNCH.json'),dict(pid=process.pid,gpu=record['gpu'],config_sha256=record['sha256'],command=command,started_cst=datetime.now(TZ).isoformat()))
    save(OUT/'LAUNCH_COMPLETE.json',dict(passed=True,count=2,pids={a:p.pid for a,p in processes.items()},started_cst=datetime.now(TZ).isoformat()))
    while True:
        states=publish(records,processes,'Only Full with SDF / Full without SDF')
        if not any(r['state']=='running' for r in states.values()):break
        time.sleep(20)
    assert all(r['state']=='complete' for r in states.values()),'An experiment trainer failed'
    save(OUT/'TRAINING_COMPLETE.json',dict(passed=True,runs=states,completed_cst=datetime.now(TZ).isoformat()))
    for stream in streams:stream.close()


if __name__=='__main__':
    try:main()
    except Exception as error:save(OUT/'FAILED.json',dict(error=repr(error),epoch=time.time()));raise
