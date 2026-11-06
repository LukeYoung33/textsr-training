"""Persistent data-gated V3M L1/R1 experiment controller and observation producer."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,signal,shutil,html,traceback
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_r1_width_bank import OUT,OLD,SOURCE,DATA,save,digest
PYTHON=sys.executable;NAME='v3m_dilate2_l1_r1_width_500k_20261103'
TRAINER=ROOT/'tools/train_r1_width_500k.py';EXPERIMENT=OUT/'experiments'/NAME
LABEL='V3M · 原始GT+2px / 新mask · L1+粗细R1'


def read(path):
    """@luna-comment Input:owned JSON path. Output:document orempty when absent. Key Steps:readatomicpublishedJSON. Key Params:malformednotignored. State/Dependencies:filesystem."""
    path=Path(path);return json.loads(path.read_text()) if path.exists() else {}


def alive(record):
    """@luna-comment Input:owned child pid/start token. Output:liveness withidentity validation. Key Steps:procstat token andzombiestate. Key Params:nevertrustPIDalone. State/Dependencies:Linuxproc read-only."""
    try:
        stat=Path(f'/proc/{record["pid"]}/stat').read_text().split();return stat[2]!='Z' and stat[21]==record['start_token']
    except (FileNotFoundError,KeyError,ProcessLookupError):return False


def publish(phase,extra=None):
    """@luna-comment Input:pipelinephase anddetails. Output:atomic8765-compatible JSON plusstandalonepage. Key Steps:mergeowneddata/trainingprogress andreal lossrecords,compute recentETA. Key Params:no inferredcompletion. State/Dependencies:ownedstatusonly."""
    data=read(OUT/'DATA_STATUS.json');train=read(OUT/'audits/TRAIN_STATUS.json');history={}
    path=OUT/'audits/loss.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:r=json.loads(line);history[r['iter']]=dict(step=r['iter'],values=r['logs'],updated=r['updated'])
            except (ValueError,KeyError):continue
    records=sorted(history.values(),key=lambda r:r['step']);eta=None
    if len(records)>2:
        window=records[-50:];delta=window[-1]['step']-window[0]['step'];seconds=window[-1]['updated']-window[0]['updated']
        if delta>0 and seconds>0:eta=(500000-train.get('iter',0))*seconds/delta
    now=datetime.now(timezone(timedelta(hours=8)));finish=(now+timedelta(seconds=eta)).strftime('%m-%d %H:%M') if eta is not None else '开训后按实测速度更新'
    if phase=='已完成':finish='已完成'
    details=['目标500K；LR1e-4→5e-6；从相同随机初始化训练','L1 + R1 missing1/false0.25；当前batch平方根权重，上限4','学习率：'+str(train.get('lr','1e-4（待启动）')),'预计训练完成：'+finish]
    if extra:details.append(json.dumps(extra,ensure_ascii=False))
    data_finish='已完成' if data.get('completed')==7385 else str(data.get('estimated_finish','采样中'))
    snap=dict(updated=now.isoformat(),runs=[dict(label=LABEL,status=phase,current=train.get('iter',0),total=500000,details=details)],datasets=[dict(label='原始GT+2px同步mask · 离线粗细分组',status=data.get('phase','准备中'),current=data.get('completed',0),total=7385,details=['8个CPU进程；复用GT/LQ，不跑OCR','预计完成：'+data_finish])],lossHistory={LABEL:records},gpu=str(read(OUT/'children/train.json').get('gpu','数据就绪后自动选择')))
    save(OUT/'monitor.json',snap);save(OUT/'CONTROLLER_STATUS.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),extra=extra))
    body='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><style>body{font:18px/1.7 system-ui;margin:30px;background:#eef3f8}article{background:white;padding:24px;margin:16px;border-radius:12px}progress{width:100%}</style><h1>'+LABEL+'</h1>'
    for group,title in [('datasets','数据处理'),('runs','训练')]:
        body+='<h2>'+title+'</h2>'
        for c in snap[group]:body+='<article><h3>'+html.escape(c['label'])+'</h3><b>'+html.escape(c['status'])+'</b><p>'+str(c['current'])+' / '+str(c['total'])+'</p><progress value="'+str(c['current'])+'" max="'+str(c['total'])+'"></progress><p>'+'</p><p>'.join(html.escape(v) for v in c['details'])+'</p></article>'
    body+='<p>更新：'+snap['updated']+'</p><a href="delivery/MANIFEST.json">最终交付清单（完成后）</a>';tmp=OUT/'index.html.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'index.html')


def configure():
    """@luna-comment Input:priorL1recipe andfrozenrandominit. Output:immutable trainingconfig/binding. Key Steps:enableonlyR1,specializedclasses,newpaths;preservedata/schedule. Key Params:500K,checkpoint5K,val25K. State/Dependencies:ownconfigs,oldrecipeunchanged."""
    import yaml
    cfg=yaml.safe_load((OLD.parent/'configs/v3m_dilate2_l1.yml').read_text());cfg['name']=NAME;cfg['model_type']='R1WidthModel';cfg['train']['model_type']='R1WidthModel'
    cfg['path'].update(experiments_root=str(OUT/'experiments'),resume_state=None);cfg['train'].update(recall_aware_weight_missing=1.,recall_aware_weight_false=.25,recall_aware_opt=dict(ring_radius=3,margin=.03,temperature=.01,min_polarity_contrast=.03))
    cfg['logger'].update(save_checkpoint_freq=5000,print_freq=100);cfg['s22_strategy'].update(name='paired',audit_root=str(OUT/'audits'));cfg['dist_params']['port']=35965
    cfg['v3pre_target_domain']='Original GT width+2px and synchronized new text masks; offline augmented width bins; native L1 plus current-batch capped sqrt R1'
    for split in ['train','val']:cfg['datasets'][split].update(type='R1WidthDataset',root=str(OUT/'bank'),manifest=str(OUT/'bank'/(split+'.json')))
    cfg['r1_width_balancing']=dict(mode='current_batch',exponent=.5,epsilon=1e-4,cap=4.,mask_version=str(SOURCE),bins=[2,4])
    path=OUT/'config.yml';text=yaml.safe_dump(cfg,sort_keys=False)
    if path.exists():assert path.read_text()==text,'Frozen config mismatch'
    else:path.write_text(text)
    bind=dict(config=str(path),config_sha256=digest(path),initial=cfg['path']['pretrain_network_g'],initial_sha256=digest(cfg['path']['pretrain_network_g']))
    save(OUT/'CONFIG_BINDING.json',bind);return cfg


def gpu():
    """@luna-comment Input:currentGPUtelemetry. Output:headroomGPUorNone. Key Steps:filter8GiBfree/util<90,sortutil. Key Params:nojobtermination. State/Dependencies:nvidia-smi read-only."""
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    candidates=[tuple(map(int,line.split(','))) for line in rows.splitlines()]
    candidates=[r for r in candidates if r[1]>=8192 and r[2]<90];return min(candidates,key=lambda r:(r[2],-r[1]))[0] if candidates else None


def child(stage,command,completion,gpu_id=None):
    """@luna-comment Input:ownedstagecommand/completion gate. Output:finishedchild orraisedfailure. Key Steps:adoptverifiedPIDorlaunch,heartbeat,monitorhungownedprocess,receiptgate. Key Params:onehourwithoutprogress,neverforeignkill. State/Dependencies:ownedLinuxprocessgroup andlogs."""
    record=read(OUT/'children'/(stage+'.json'));p=None
    if not alive(record):
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',REAL_ESRGAN_DEFER_FINAL_VALIDATION='1');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        if gpu_id is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu_id)
        (OUT/'logs').mkdir(exist_ok=True)
        with (OUT/'logs'/(stage+'.log')).open('ab') as stream:p=subprocess.Popen(command,cwd=OUT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        token=Path(f'/proc/{p.pid}/stat').read_text().split()[21];record=dict(pid=p.pid,start_token=token,gpu=gpu_id,command=command,started=time.time());save(OUT/'children'/(stage+'.json'),record)
    while alive(record):
        publish({'data':'离线处理，完成后自动开训','smoke':'训练启动与恢复检查','train':'训练中','final':'500K完成，正在验证交付'}[stage],dict(stage=stage,gpu=record.get('gpu')))
        progress=OUT/'audits/TRAIN_STATUS.json' if stage=='train' else OUT/'DATA_STATUS.json' if stage=='data' else OUT/'logs'/(stage+'.log')
        stamp=max(progress.stat().st_mtime if progress.exists() else 0,record['started'])
        if time.time()-stamp>3600:
            assert os.getpgid(record['pid'])==record['pid'];os.killpg(record['pid'],signal.SIGTERM);raise RuntimeError('Owned stage stalled for one hour: '+stage)
        time.sleep(20)
        if p is not None:p.poll()
    if p is not None and p.wait()!=0:raise RuntimeError('Stage failed: '+stage+'; see logs/'+stage+'.log')
    if not completion.exists():raise RuntimeError('Stage exited without completion: '+stage)


def resume_config(cfg):
    """@luna-comment Input:frozen recipe. Output:perlaunch config fromlatestverifiedpair. Key Steps:hashcheckdescendingreceipts,validate torchstate,bindresume;rejectforeignweights. Key Params:fallbackearlierverified checkpoint. State/Dependencies:readoldcheckpoints,writeownedlaunchconfig."""
    import torch,yaml
    candidates=sorted((EXPERIMENT/'training_states').glob('*.verified.json'),key=lambda p:int(p.name.split('.')[0]),reverse=True)
    for path in candidates:
        try:
            r=read(path);assert digest(r['model'])==r['model_sha256'] and digest(r['state'])==r['state_sha256'];s=torch.load(r['state'],map_location='cpu');assert s['iter']==r['iter'] and s['optimizers']
            assert all(torch.isfinite(v).all() for o in s['optimizers'] for item in o['state'].values() for v in item.values() if torch.is_tensor(v))
            params=torch.load(r['model'],map_location='cpu')['params'];assert all(torch.isfinite(v).all() for v in params.values())
            cfg['path']['resume_state']=r['state'];save(OUT/'RESUME.json',dict(iter=s['iter'],state=r['state'],updated=time.time()));break
        except Exception as error:save(OUT/'CHECKPOINT_REJECTED.json',dict(path=str(path),error=repr(error),updated=time.time()))
    else:
        if (EXPERIMENT/'training_states').exists() and any((EXPERIMENT/'training_states').glob('*.state')):raise RuntimeError('No verified resumable checkpoint; refusing silent scratch restart')
    path=OUT/'launch.yml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));return path


def run():
    """@luna-comment Input:explicitunattendedexperiment. Output:data-gatedtraining andverifieddelivery. Key Steps:lock,persistentretry,adoptofflineworker,smoke,resume500K,finalexport. Key Params:retry30..900seconds,cronrecovery. State/Dependencies:ownedtasks only."""
    import yaml
    OUT.mkdir(exist_ok=True);lock=(OUT/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if (OUT/'COMPLETE.json').exists():publish('已完成');return
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),start_token=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],started=time.time()))
    failures=0
    while True:
        try:
            cfg=configure()
            if not (OUT/'DATA_READY.json').exists():
                status=read(OUT/'DATA_STATUS.json');pid=status.get('pid');proc=Path(f'/proc/{pid}/cmdline')
                if pid and proc.exists() and b'prepare_r1_width_bank.py' in proc.read_bytes():save(OUT/'children/data.json',dict(pid=pid,start_token=Path(f'/proc/{pid}/stat').read_text().split()[21],started=time.time(),gpu=None))
                child('data',[PYTHON,'-u',str(ROOT/'tools/prepare_r1_width_bank.py')],OUT/'DATA_READY.json')
            assert read(OUT/'DATA_READY.json')['passed']
            selected=gpu()
            while selected is None:publish('数据已就绪，等待GPU资源');time.sleep(30);selected=gpu()
            if not (OUT/'SMOKE.json').exists():
                smoke=__import__('copy').deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke_audit');sp=OUT/'smoke.yml';sp.write_text(yaml.safe_dump(smoke,sort_keys=False))
                child('smoke',[PYTHON,'-u',str(TRAINER),'--smoke-config',str(sp),'--smoke-output',str(OUT/'SMOKE.json')],OUT/'SMOKE.json',selected)
            assert read(OUT/'SMOKE.json')['passed']
            final=EXPERIMENT/'training_states/500000.verified.json'
            if not final.exists():
                launch=resume_config(cfg);child('train',[PYTHON,'-u',str(TRAINER),'-opt',str(launch)],final,selected)
            # An adopted train process may still be completing its terminal validation.
            record=read(OUT/'children/train.json')
            while alive(record):publish('500K已保存，等待训练末次验证结束');time.sleep(20)
            if not (OUT/'FINAL_VALIDATION.json').exists():child('final',[PYTHON,'-u',str(TRAINER),'--final-config',str(OUT/'config.yml')],OUT/'FINAL_VALIDATION.json',selected)
            check=read(OUT/'FINAL_VALIDATION.json');receipt=read(final)
            assert check['passed'] and check['iter']==500000 and check['lr']==5e-6 and check['checkpoint_sha256']==receipt['model_sha256']
            assert digest(receipt['model'])==receipt['model_sha256']
            delivery=OUT/'delivery';delivery.mkdir(exist_ok=True)
            for src,dest in [(Path(receipt['model']),delivery/'V3M_L1_R1_width2_500K.pth'),(OUT/'config.yml',delivery/'config.yml'),(OUT/'SOURCE_BINDING.json',delivery/'SOURCE_BINDING.json'),(OUT/'FINAL_VALIDATION.json',delivery/'VALIDATION.json')]:
                temp=dest.with_suffix(dest.suffix+'.tmp');shutil.copyfile(src,temp);assert digest(src)==digest(temp);temp.replace(dest)
            for source in [TRAINER,Path(__file__),ROOT/'tools/prepare_r1_width_bank.py',OUT/'DATA_READY.json',OUT/'SMOKE.json']:
                dest=delivery/source.name;tmp=dest.with_suffix(dest.suffix+'.tmp');shutil.copyfile(source,tmp);assert digest(source)==digest(tmp);tmp.replace(dest)
            manifest=dict(passed=True,total_iter=500000,final_lr=5e-6,files={p.name:digest(p) for p in delivery.iterdir() if p.is_file() and p.name!='MANIFEST.json'},checkpoint=str(delivery/'V3M_L1_R1_width2_500K.pth'),updated=time.time())
            save(delivery/'MANIFEST.json',manifest);save(OUT/'COMPLETE.json',manifest);publish('已完成');return
        except Exception as error:
            failures+=1;delay=min(900,30*2**min(failures,5));record=dict(error=repr(error),traceback=traceback.format_exc(),attempt=failures,retry_seconds=delay,updated=time.time());save(OUT/'LAST_ERROR.json',record);publish('异常后等待自动重试',record);time.sleep(delay)


if __name__=='__main__':
    if '--publish' in sys.argv:publish('离线处理，完成后自动开训')
    else:run()
