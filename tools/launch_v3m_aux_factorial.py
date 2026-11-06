"""Concurrent MF-only/UV-only scratch controls and persistent idle-GPU L1-mix50 queue."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,os,copy,json,time,subprocess,fcntl,re,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import launch_latest_vosr_500k as monitor
from tools.run_uniform20_steep_scurve import save,digest
BASE=monitor.OUT;OUT=BASE.parent/'v3m_aux_factorial_500k_20261103'
LABELS={'v3m_mf_only':'V3M · L1＋Missing/False · 真实LQ','v3m_uv_only':'V3M · L1＋UV · 真实LQ','v3m_l1_mix50':'V3M · L1-only · 50%GT轻退化'}
QUEUED='v3m_l1_mix50';PRIORITIES=('v3m_mf_only','v3m_uv_only')
monitor.OUT=OUT;monitor.LABELS=LABELS;original_publish=monitor.publish


def choose_idle_gpu(free,compute_busy,owned_busy):
    """@luna-comment Input:GPUfreeMiB andbusysets. Output:first genuinely idleGPU orNone. Key Steps:excludebothcomputeusers andownedstartingworkers. Key Params:6GiB minimum. State/Dependencies:pure queuegate."""
    return next((g for g in sorted(free) if free[g]>=6144 and g not in compute_busy and g not in owned_busy),None)


def gpu_inventory():
    """@luna-comment Input:live NVIDIA inventory. Output:freeMiB map andcompute-busyGPUset. Key Steps:UUID-to-index binding thenallcomputePIDs. Key Params:read-only queries. State/Dependencies:nvidia-smi,unknownusers respected."""
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    free={};uuids={}
    for line in text.splitlines():
        index,uuid,mem=[v.strip() for v in line.split(',')];free[int(index)]=int(mem);uuids[uuid]=int(index)
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={uuids[line.split(',')[0].strip()] for line in text.splitlines() if ',' in line and line.split(',')[0].strip() in uuids}
    return free,busy


def owned_busy(processes):
    """@luna-comment Input:owned Popen map. Output:busyGPUs includingpre-CUDAworkers. Key Steps:ownnewPIDs plusprior3identity-checkedtrainers. Key Params:noPIDreuseguessing. State/Dependencies:read-onlyproc/launchreceipts."""
    blocked=set()
    for arm,p in processes.items():
        if p.poll() is None:blocked.add(monitor.read(OUT/'launch'/(arm+'.json'))['gpu'])
    old=BASE.parent/'v3m_loss_ablation_500k_20261103'
    for path in (old/'launch').glob('*.json'):
        r=monitor.read(path);cmd=Path('/proc')/str(r.get('pid'))/'cmdline'
        try:payload=cmd.read_bytes()
        except OSError:continue
        if str(r.get('config')).encode() in payload:blocked.add(r['gpu'])
    return blocked


def estimate_start(states):
    """@luna-comment Input:currentprioritystates. Output:earliest predictedGPUreleaseorunknown. Key Steps:maximumfinishperGPU thenminimumacrossGPUs;includeprior3. Key Params:Beijinglogtimes;noqueuedcompletionguess. State/Dependencies:read-onlymonitorstatus."""
    tz=timezone(timedelta(hours=8));now=datetime.now(tz);groups={}
    old=monitor.read(BASE.parent/'v3m_loss_ablation_500k_20261103/STATUS.json').get('runs',{})
    for arm,s in {**old,**states}.items():
        if arm==QUEUED or s.get('phase')!='训练中' or s.get('gpu') is None:continue
        match=re.search(r'(\d\d-\d\d \d\d:\d\d)',s.get('finish',''))
        if match:
            stamp=datetime.strptime(str(now.year)+'-'+match[1],'%Y-%m-%d %H:%M').replace(tzinfo=tz)
            if stamp>=now:groups.setdefault(s['gpu'],[]).append(stamp)
    return min(max(v) for v in groups.values()).strftime('%m-%d %H:%M 北京时间') if groups else '等待运行任务估时'


def publish():
    """@luna-comment Input:logs andpersistedqueue. Output:3cards withaccuratequeuedstartversuscompletionETA. Key Steps:reuseobserver;replacequeuedcardandstatus;updateestimatedrelease. Key Params:queuepoll30s. State/Dependencies:ownedmetadataonly."""
    original_publish();status=monitor.read(OUT/'STATUS.json');queue=monitor.read(OUT/'QUEUE.json');states=status.get('runs',{})
    for state in states.values():
        if state['finish']=='采样中':state['finish']='等待训练日志估时'
    waiting=queue.get('state') in ['waiting_gpu','checking'] and not (OUT/'launch'/(QUEUED+'.json')).exists()
    eta=estimate_start(states)
    if waiting:
        states[QUEUED].update(phase='排队中，等待训练GPU空闲',queued=True,finish=eta,gpu=None)
        queue['estimated_start']=eta;save(OUT/'QUEUE.json',queue)
    save(OUT/'STATUS.json',status);path=OUT/'index.html';text=path.read_text(encoding='utf-8')
    text=text.replace('最新VOSR · 四组500K','V3M · 辅助loss补充对照与队列').replace('V3M / V5 · guard / L1-only · 从头训练500K','V3M · Missing/False、UV与混合退化对照')
    text=text.replace('采样中','等待训练日志估时').replace('等待固定快照patch构建和启动检查','等待启动检查')
    if waiting:
        card='<article><h2>'+LABELS[QUEUED]+'</h2><b>排队中，等待训练GPU空闲</b><p>0 / 500,000 · LR 1e-4</p><progress max="500000" value="0"></progress><p>预计启动：'+html.escape(eta)+'</p><p>GPU完全空闲后自动启动；完成时间待开训后估算。</p></article>'
        text=re.sub(r'<article><h2>'+re.escape(LABELS[QUEUED])+r'</h2>.*?</article>',lambda _:card,text,count=1,flags=re.S)
    tmp=OUT/'index.queue.tmp';tmp.write_text(text,encoding='utf-8');tmp.replace(path)


def prepare():
    """@luna-comment Input:completedL1/fullbasebindings. Output:two priorityrecipesandonefrozenqueuedrecipe. Key Steps:cloneL1,enableonlyMF/UVorinputmix;strictdata/init/schedulechecks. Key Params:500K,same1e-4to5e-6. State/Dependencies:originalartifactsimmutable."""
    import yaml
    declared=json.loads((BASE/'CONFIG_BINDINGS.json').read_text())['runs'];configs={}
    for key in ['v3m_l1','v3m_guard']:
        d=declared[key];assert digest(d['config'])==d['config_sha256'];configs[key]=yaml.safe_load(Path(d['config']).read_text())
    base=configs['v3m_l1'];full=configs['v3m_guard'];records={}
    assert digest(base['path']['pretrain_network_g'])==declared['v3m_l1']['initial_sha256']==declared['v3m_guard']['initial_sha256']
    for index,(arm,label) in enumerate(LABELS.items()):
        cfg=copy.deepcopy(base);mixed=arm==QUEUED;kind='LatestVOSRMix50' if mixed else 'S22Model';cfg['name']=arm+'_500k_20261103';cfg['model_type']=kind;cfg['train']['model_type']=kind
        cfg['s22_strategy'].update(name='mixed50' if mixed else 'paired',audit_root=str(OUT/'audits'/arm));cfg['path'].update(experiments_root=str(OUT/'experiments'),resume_state=None);cfg['dist_params']['port']=35940+index
        if arm=='v3m_mf_only':
            for k in ['recall_aware_opt','recall_aware_weight_missing','recall_aware_weight_false']:cfg['train'][k]=copy.deepcopy(full['train'][k])
        if arm=='v3m_uv_only':
            for k in ['uv_strong_opt','uv_strong_weight']:cfg['train'][k]=copy.deepcopy(full['train'][k])
        assert cfg['datasets']==base['datasets'] and cfg['network_g']==base['network_g'] and cfg['manual_seed']==base['manual_seed']
        assert cfg['train']['total_iter']==500000 and cfg['train']['optim_g']==base['train']['optim_g'] and cfg['train']['scheduler']==base['train']['scheduler']
        path=OUT/'configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True);text=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        sm=copy.deepcopy(cfg);sm['name']+='_smoke';sm['s22_strategy']['audit_root']=str(OUT/'smoke_audits'/arm);sp=OUT/'configs'/(arm+'_smoke.yml');sp.write_text(yaml.safe_dump(sm,sort_keys=False))
        records[arm]=dict(label=label,name=cfg['name'],config=str(path),config_sha256=digest(path),initial=cfg['path']['pretrain_network_g'],initial_sha256=digest(cfg['path']['pretrain_network_g']),smoke=str(sp),priority=arm in PRIORITIES,planned_gpu=3 if arm in PRIORITIES else None)
    for name in ['DATA_READY.json','SOURCE_ELIGIBILITY.json']:
        link=OUT/name
        if not link.exists():link.symlink_to(BASE/name)
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=records,bank_ready_sha256=digest(BASE/'DATA_READY.json'),initialization='same frozen seeded V3M random state;fresh Adam',schedule=[(0,1e-4),(200000,5e-5),(300000,2.5e-5),(400000,1e-5),(450000,5e-6)]))
    save(OUT/'QUEUE.json',dict(arm=QUEUED,state='checking',condition='no compute PIDs and no live owned training on GPU;freeVRAM>=6144MiB',created=time.time()))
    return records


def launch(arm,r,gpu):
    """@luna-comment Input:frozenrecipeandchosenGPU. Output:detachedformalPopen/launchreceipt. Key Steps:rehashinit/config;isolatedlog/CUDAenv. Key Params:noresume,500K. State/Dependencies:ownedsingleworker."""
    assert digest(r['config'])==r['config_sha256'] and digest(r['initial'])==r['initial_sha256'];(OUT/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    with (OUT/'logs'/(arm+'.log')).open('xb') as f:p=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/train_v3m_aux_factorial.py'),'-opt',r['config']],cwd=OUT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    save(OUT/'launch'/(arm+'.json'),dict(pid=p.pid,gpu=gpu,config=r['config'],initial_sha256=r['initial_sha256']));return p


def main():
    """@luna-comment Input:authorizedprioritytwoandqueuedone. Output:residentexact-oncequeueandthreeeventual500Kjobs. Key Steps:lock/allsmokes,priorityGPU3launch,realidleGPUgate,completionmonitor. Key Params:30spoll,neverstopoldjobs. State/Dependencies:CUDA/ownedprocessreceipts."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not list((OUT/'launch').glob('*.json'))
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=time.time()));records=prepare()
    for arm,r in records.items():
        save(OUT/'CONTROLLER_STATUS.json',dict(phase='启动检查',arm=arm));env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        monitor.run_checked([sys.executable,'-u',str(ROOT/'tools/train_v3m_aux_factorial.py'),'--smoke-config',r['smoke'],'--smoke-output',str(OUT/(arm+'_smoke.json'))],OUT/(arm+'_smoke.log'),env)
        assert monitor.read(OUT/(arm+'_smoke.json'))['passed']
    processes={}
    for arm in PRIORITIES:
        while gpu_inventory()[0][3]<6144:publish();time.sleep(30)
        processes[arm]=launch(arm,records[arm],3)
    save(OUT/'QUEUE.json',dict(arm=QUEUED,state='waiting_gpu',condition='wait for genuinely idle training GPU',ready=time.time()))
    pending=True
    while pending or processes:
        for arm,p in list(processes.items()):
            if p.poll() is not None:
                final=OUT/'experiments'/records[arm]['name']/'models/net_g_500000.pth';assert p.returncode==0 and final.exists(),arm+' failed'
                save(OUT/'done'/(arm+'.json'),dict(passed=True,checkpoint=str(final)));del processes[arm]
        if pending:
            try:free,busy=gpu_inventory();gpu=choose_idle_gpu(free,busy,owned_busy(processes))
            except subprocess.CalledProcessError:gpu=None
            if gpu is not None:
                processes[QUEUED]=launch(QUEUED,records[QUEUED],gpu);pending=False;save(OUT/'QUEUE.json',dict(arm=QUEUED,state='running',gpu=gpu,pid=processes[QUEUED].pid,started=time.time()))
        save(OUT/'CONTROLLER_STATUS.json',dict(phase='优先两组训练，L1混合排队' if pending else '全部已启动'));publish();time.sleep(30)
    save(OUT/'COMPLETE.json',dict(passed=True,arms=3,total_iter=500000));save(OUT/'QUEUE.json',dict(arm=QUEUED,state='complete'));publish()


monitor.publish=publish
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'FAILED.json',dict(error=repr(e)));save(OUT/'CONTROLLER_STATUS.json',dict(phase='异常停止',error=repr(e)));publish();raise
