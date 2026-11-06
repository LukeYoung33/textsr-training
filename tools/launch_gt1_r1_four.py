"""Data-gated four confirmed500K recipes with distinct telemetry and automatic launch."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import copy,fcntl,html,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'gt1_r1_mixture_four_500k_20261104';BASE=DATA/'latest_vosr_four_500k_20261102'
ARMS={
 'gt1_l1_mix50':('gt1','none','V3M · 原始GT+1px · L1-only · GT退化50%'),
 'gt1_r1ori_mix50':('gt1','ori','V3M · 原始GT+1px · L1+R1_ori · GT退化50%'),
 'gt1_r1norm_mix50':('gt1','norm','V3M · 原始GT+1px · L1+R1_norm · GT退化50%'),
 'vosr_s22_r1norm_mix50':('vosr','norm','V3M · VOSR-S2.2 · L1+R1_norm · GT退化50%')}
TRAINER=ROOT/'tools/train_gt1_r1_four.py';PREP=ROOT/'tools/prepare_four_r1_sources.py'


def read(path):
    """@luna-comment Input:optionalownedJSON. Output:parsedorabsentdict. Key Steps:atomicreader. Key Params:no malformed-filefallback. State/Dependencies:readonlyownedmetadata."""
    p=Path(path);return json.loads(p.read_text()) if p.exists() else {}


def configure():
    """@luna-comment Input:confirmed4recipes/sourcefrozenL1Fullconfigs. Output:immutableconfigs/sharedinitbinding. Key Steps:cloneL1,onlyR1switches,allmix.5/contrastoff,matched7385bankpaths. Key Params:500K,1e-4to5e-6,batch4. State/Dependencies:previousconfigsreadonly."""
    import yaml
    declared=read(BASE/'CONFIG_BINDINGS.json')['runs'];b=declared['v3m_l1'];f=declared['v3m_guard']
    assert digest(b['config'])==b['config_sha256'] and digest(f['config'])==f['config_sha256']
    base=yaml.safe_load(Path(b['config']).read_text());full=yaml.safe_load(Path(f['config']).read_text());bindings={}
    assert digest(base['path']['pretrain_network_g'])==b['initial_sha256']
    for n,(arm,(domain,kind,label)) in enumerate(ARMS.items()):
        c=copy.deepcopy(base);model='GT1NormMixtureModel' if kind=='norm' else 'GT1MixtureModel'
        c['name']=arm+'_500k_20261104';c['model_type']=model;c['train']['model_type']=model
        c['path'].update(experiments_root=str(OUT/'experiments'),resume_state=None)
        c['s22_strategy'].update(name='paired',audit_root=str(OUT/'audits'/arm));c['gt1_mix']=dict(probability=.5,r1=kind)
        c['v3pre_target_domain']=domain+' matched7385;contrastoff;GTmildmixture50;validationold620unchanged'
        c['dist_params']['port']=36110+n
        for split in ['train','val']:
            c['datasets'][split].update(root=str(OUT/domain/'bank'),manifest=str(OUT/domain/'bank'/(split+'.json')),patch_contrast_alignment=False)
            c['datasets'][split]['type']='R1WidthDataset' if kind=='norm' else 'S22Dataset'
        if kind!='none':
            for key in ['recall_aware_opt','recall_aware_weight_missing','recall_aware_weight_false']:c['train'][key]=copy.deepcopy(full['train'][key])
            assert c['train']['recall_aware_weight_missing']==1 and c['train']['recall_aware_weight_false']==.25
        assert c['train']['total_iter']==500000 and c['train']['optim_g']['lr']==1e-4
        assert all(v==0 for k,v in c['train'].items() if k.endswith('_weight'))
        path=OUT/'configs'/(arm+'.yml');path.parent.mkdir(exist_ok=True);text=yaml.safe_dump(c,sort_keys=False)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        smoke=copy.deepcopy(c);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke_audits'/arm);sp=OUT/'configs'/(arm+'_smoke.yml');sp.write_text(yaml.safe_dump(smoke,sort_keys=False))
        bindings[arm]=dict(name=c['name'],label=label,domain=domain,r1=kind,config=str(path),config_sha256=digest(path),smoke=str(sp),initial=c['path']['pretrain_network_g'],initial_sha256=b['initial_sha256'])
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=bindings,all_from_scratch=True,all_probability=.5,all_contrast_off=True,all_train_parents=7385,initial_sha256=b['initial_sha256']))
    return bindings


def publish(phase):
    """@luna-comment Input:liveownedtraining/datareceipts. Output:Chinese4cards/loss/ETA and8765monitorJSON. Key Steps:separatedata/width/trainstages,actualtelemetry,rollingwallETA. Key Params:4distinctlabels,noaggregate-lossqualityclaim. State/Dependencies:readonlylogs;ownobserveroutputs."""
    from tools.offline_loss_charts import render_loss_charts
    tz=timezone(timedelta(hours=8));now=datetime.now(tz);rows=[];histories={};status={};data=[]
    for domain in ['gt1','vosr']:
        p=OUT/domain;bank=read(p/'train_bank/progress.json');lines=read(p/'train_bank/line_regions/progress.json');width=read(p/'WIDTH_STATUS.json')
        ready=bool(read(p/'DATA_READY.json').get('passed'));wr=bool(read(p/'WIDTH_READY.json').get('passed'))
        data.append(dict(label=domain+' 数据与粗细分组',status='数据就绪；粗细分组'+('就绪' if wr else '待完成') if ready else str(read(p/'DATA_STATUS.json').get('phase','准备中')),current=7385 if ready else bank.get('completed',0),total=7385,details=['patch：'+str(bank.get('completed',7385 if ready else 0))+'/7385','文字几何缓存：'+str(lines.get('done',0))+'/7385','粗细分组：'+str(width.get('completed',0))+'/7385']))
    for arm,(domain,kind,label) in ARMS.items():
        p=OUT/'audits'/arm;current=read(p/'RUN_STATUS.json');launch=read(OUT/'launch'/(arm+'.json'));history=[]
        if (p/'history.jsonl').exists():
            for line in (p/'history.jsonl').read_text().splitlines():
                r=json.loads(line);history.append(dict(step=r['iter'],values=r['logs'],updated=r['updated']))
        histories[label]=history;iteration=current.get('iter',0);eta=None
        if len(history)>2:
            window=history[-100:];delta=window[-1]['step']-window[0]['step'];seconds=window[-1]['updated']-window[0]['updated']
            if delta>0 and seconds>0:eta=(500000-iteration)*seconds/delta
        finish=(now+timedelta(seconds=eta)).strftime('%m-%d %H:%M') if eta is not None else '开训后估算'
        done=bool(read(OUT/'done'/(arm+'.json')).get('passed'))
        state='已完成' if done else '训练中' if (OUT/'launch'/(arm+'.json')).exists() else '启动检查中' if (OUT/'smoke_launch'/(arm+'.json')).exists() else '等待数据／粗细分组'
        details=['LR：'+str(current.get('lr','1e-4')),'GPU：'+str(launch.get('gpu','待分配')),'预计完成：'+('已完成' if done else finish),'GT退化50%；对比度对齐关闭；7385来源']
        rows.append(dict(label=label,status=state,current=iteration,total=500000,details=details))
        status[arm]=dict(iter=iteration,phase=state,gpu=launch.get('gpu'),finish=finish,lr=current.get('lr',1e-4))
    save(OUT/'STATUS.json',dict(updated=now.isoformat(),phase=phase,runs=status,datasets=data))
    save(OUT/'monitor.json',dict(updated=now.isoformat(),runs=rows,datasets=data,lossHistory=histories))
    body='<meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>GT+1与VOSR四组500K</title><style>body{font:17px/1.7 system-ui;background:#eef3f8;margin:25px;color:#18314b}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}article{padding:20px;background:white;border-radius:12px}progress{width:100%}</style><h1>V3M · GT+1／VOSR · R1对照 · 四组500K</h1><p>'+html.escape(phase)+' · '+now.strftime('%m-%d %H:%M:%S')+' 北京时间</p><p>同一随机初始化；初始LR1e-4→5e-6；四组GT轻度退化均为50%。R1_ori原始Missing/False；R1_norm粗细归一限幅。总loss不直接用于效果排名。</p>'
    for title,items in [('训练',rows),('数据准备',data)]:
        body+='<h2>'+title+'</h2><div class="grid">'
        for r in items:body+='<article><h3>'+html.escape(r['label'])+'</h3><b>'+html.escape(r['status'])+'</b><p>'+str(r['current'])+' / '+str(r['total'])+'</p><progress max="'+str(r['total'])+'" value="'+str(r['current'])+'"></progress><p>'+'</p><p>'.join(html.escape(v) for v in r['details'])+'</p></article>'
        body+='</div>'
    body+=render_loss_charts(histories);tmp=OUT/'index.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'index.html')


def spawn(command,log,gpu=None):
    """@luna-comment Input:knownownedcommand/log/optionalGPU. Output:detachedPopen. Key Steps:boundedCPUenv/CUDAassignment andappendlog. Key Params:noothersignal;2OMPthreads. State/Dependencies:ownedchildonly."""
    log.parent.mkdir(parents=True,exist_ok=True);env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with log.open('ab') as stream:return subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)


def main():
    """@luna-comment Input:userconfirmed4recipes/allmix50. Output:residentdatagated4trainingprocesses/completion. Key Steps:freezeconfigs/teachersubset,adoptexistingbankbuilder,CPU8sequentialwidthprep,4balancedCUDAstartupgates,monitor/checkfinalpairs. Key Params:neverstopotherjobs,oneownedworker/GPU. State/Dependencies:ownedlocks/receipts;failclosedonbadsmoke/data."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not list((OUT/'launch').glob('*.json')),'Existing launch receipts: inspect/adopt before restarting controller'
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=time.time()));bindings=configure()
    subprocess.run([sys.executable,'-u',str(PREP),'teacher_subset'],cwd=ROOT,check=True)
    processes={};width=None;width_domain=None;builder=None
    while len(list((OUT/'done').glob('*.json')))<4:
        for domain in ['gt1','vosr']:
            assert not (OUT/domain/'FAILED.json').exists(),str(OUT/domain/'FAILED.json')
        ready=bool(read(OUT/'gt1/DATA_READY.json').get('passed'))
        if not ready:
            r=read(OUT/'gt1/CONTROLLER.json');cmd=Path('/proc')/str(r.get('pid'))/'cmdline'
            if not cmd.exists():
                if builder is not None:assert builder.poll() is None,'GT1 builder ended without DATA_READY'
                builder=spawn([sys.executable,'-u',str(ROOT/'tools/prepare_gt1_four_training.py')],OUT/'gt1/controller.log');save(OUT/'gt1/CONTROLLER.json',dict(pid=builder.pid,started=time.time()))
        if width is not None and width.poll() is not None:
            assert width.returncode==0 and read(OUT/width_domain/'WIDTH_READY.json').get('passed'),width_domain+' width preparation failed'
            width=None;width_domain=None
        if ready and width is None:
            for domain in ['gt1','vosr']:
                if not read(OUT/domain/'WIDTH_READY.json').get('passed'):
                    width_domain=domain;width=spawn([sys.executable,'-u',str(PREP),'width_'+domain],OUT/domain/'width.log');save(OUT/domain/'WIDTH_CONTROLLER.json',dict(pid=width.pid,started=time.time()));break
        for arm,entry in list(processes.items()):
            p,gpu,stage=entry
            if p.poll() is None:continue
            assert p.returncode==0,(arm,stage,p.returncode)
            if stage=='smoke':
                assert read(OUT/'smoke'/(arm+'.json')).get('passed')
                r=bindings[arm];assert digest(r['config'])==r['config_sha256'] and digest(r['initial'])==r['initial_sha256']
                child=spawn([sys.executable,'-u',str(TRAINER),'-opt',r['config']],OUT/'logs'/(arm+'.log'),gpu)
                save(OUT/'launch'/(arm+'.json'),dict(pid=child.pid,gpu=gpu,config=r['config']));processes[arm]=(child,gpu,'train')
            else:
                final=OUT/'experiments'/bindings[arm]['name']/'training_states/500000.verified.json';receipt=read(final)
                assert receipt['iter']==500000 and digest(receipt['model'])==receipt['model_sha256'] and digest(receipt['state'])==receipt['state_sha256']
                save(OUT/'done'/(arm+'.json'),dict(passed=True,checkpoint=receipt['model'],checkpoint_sha256=receipt['model_sha256']));del processes[arm]
        busy={gpu for p,gpu,stage in processes.values()}
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        free=sorted([tuple(map(int,row.split(','))) for row in inventory.splitlines()],key=lambda v:v[2])
        for arm,(domain,kind,label) in ARMS.items():
            if arm in processes or (OUT/'done'/(arm+'.json')).exists():continue
            if not read(OUT/domain/'DATA_READY.json').get('passed'):continue
            if kind=='norm' and not read(OUT/domain/'WIDTH_READY.json').get('passed'):continue
            gpu=next((g for g,m,u in free if g not in busy and m>=6144),None)
            if gpu is None:continue
            r=bindings[arm];p=spawn([sys.executable,'-u',str(TRAINER),'--smoke-config',r['smoke'],'--smoke-output',str(OUT/'smoke'/(arm+'.json'))],OUT/'logs'/(arm+'_smoke.log'),gpu)
            processes[arm]=(p,gpu,'smoke');busy.add(gpu);save(OUT/'smoke_launch'/(arm+'.json'),dict(pid=p.pid,gpu=gpu))
        publish('数据就绪后自动启动；粗细分组分阶段准备');time.sleep(20)
    save(OUT/'COMPLETE.json',dict(passed=True,arms=4,total_iter=500000,all_gt_degradation=.5,all_contrast_off=True));publish('四组全部完成')


if __name__=='__main__':
    try:main()
    except Exception as error:
        save(OUT/'FAILED.json',dict(error=repr(error)));publish('控制器异常：'+repr(error));raise
