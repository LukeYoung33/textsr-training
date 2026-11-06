"""Data-gated four-arm scratch500K controller and live overview."""
from pathlib import Path
import sys,os,json,copy,time,subprocess,re,html,fcntl,math
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_latest_vosr_500k import OUT,DATA,LENOVO
from tools.run_uniform20_steep_scurve import save,digest
LABELS={'v3m_guard':'V3M · guard + 对比度对齐 + noCP','v3m_l1':'V3M · L1-only + 对比度对齐 + noCP','v5_guard':'V5 · guard + 对比度对齐 + noCP','v5_l1':'V5 · L1-only + 对比度对齐 + noCP'}


def prepare():
    """@luna-comment Input:confirmed four scratch recipes. Output:architecture-shared seed snapshots/configs. Key Steps:CPU seeded network build;freeze same architecture init;clone matched losses and schedule. Key Params:seed20260828,500K,1e-4. State/Dependencies:owned artifacts only."""
    import torch,yaml,random,numpy as np
    from tools.train_latest_vosr_500k import bootstrap
    bootstrap();from basicsr.archs import build_network
    base=yaml.safe_load((DATA/'mojito280_guard_finetune_20261030/configs/contrast_nocp.yml').read_text())
    v5=yaml.safe_load((DATA/'mojito280_v5_guard_ft_20261030/config.yml').read_text())['network_g'];records={}
    for arch,network in [('v3m',base['network_g']),('v5',v5)]:
        initial=OUT/'initial'/(arch+'_scratch.pth');initial.parent.mkdir(exist_ok=True)
        if not initial.exists():
            torch.manual_seed(base['manual_seed']);random.seed(base['manual_seed']);np.random.seed(base['manual_seed']);net=build_network(copy.deepcopy(network));torch.save({'params':net.state_dict()},initial);del net
        for loss in ['guard','l1']:
            arm=arch+'_'+loss;cfg=copy.deepcopy(base);cfg['name']='latest_vosr_'+arm+'_500k_20261102';cfg['network_g']=network
            cfg['path'].update(pretrain_network_g=str(initial),resume_state=None,experiments_root=str(OUT/'experiments'))
            cfg['train']['total_iter']=500000;cfg['train']['optim_g']['lr']=1e-4;cfg['train']['scheduler']={'type':'LatestVOSR500K'}
            cfg['s22_strategy']['audit_root']=str(OUT/'audits'/arm);cfg['dist_params']['port']=35910+len(records)
            cfg['v3pre_target_domain']='Train: completed V3LQ/fullVOSR/S2.2 union;validation:olderGT583 plus Mojito37 S2.2, original split retained'
            for split in ['train','val']:
                cfg['datasets'][split].update(root=str(OUT/'bank'),manifest=str(OUT/'bank'/(split+'.json')),num_worker_per_gpu=2)
            if loss=='l1':
                cfg['s22_strategy']['name']='paired'
                for key in list(cfg['train']):
                    if key.endswith('_weight') or key.startswith('recall_aware_weight_'):cfg['train'][key]=0.
                    if key.endswith('_opt') and key not in ['y4uv_native_opt','yuv444_reference_opt']:cfg['train'][key]=None
            path=OUT/'configs'/(arm+'.yml');path.parent.mkdir(exist_ok=True);text=yaml.safe_dump(cfg,sort_keys=False)
            if path.exists():assert path.read_text()==text
            else:path.write_text(text)
            smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke_audits'/arm);sm=OUT/'configs'/(arm+'_smoke.yml');sm.write_text(yaml.safe_dump(smoke,sort_keys=False))
            records[arm]=dict(name=cfg['name'],config=str(path),config_sha256=digest(path),smoke=str(sm),initial=str(initial),initial_sha256=digest(initial),gpu=len(records),label=LABELS[arm])
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=records,initialization='architecture-specific seeded random;not pretrained',schedule=[(0,1e-4),(200000,5e-5),(300000,2.5e-5),(400000,1e-5),(450000,5e-6)]))
    return records


def read(path):
    """@luna-comment Input:optional atomicJSON. Output:dict. Key Steps:missing safe. Key Params:read-only. State/Dependencies:monitor."""
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def publish():
    """@luna-comment Input:owned state/logs. Output:Chinese overview with500K progress,LR andETA. Key Steps:actual PID state,rolling time samples,data counters,unique loss labels. Key Params:30srefresh,stale300s suppressETA. State/Dependencies:read-only training."""
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    tz=timezone(timedelta(hours=8));now=datetime.now(tz);cards=[];histories={};states={}
    for arm,label in LABELS.items():
        audit=read(OUT/'audits'/arm/'STATUS.json');launch=read(OUT/'launch'/(arm+'.json'));step=audit.get('iter',0);phase='等待固定快照patch构建和启动检查';eta='采样中';lr='1e-4';log=OUT/'logs'/(arm+'.log')
        live=False
        if launch:
            proc=Path('/proc')/str(launch['pid']);live=proc.exists() and (proc/'stat').read_text().split(') ',1)[1].split()[0] not in ['Z','T','t'];phase='训练中' if live else '进程未运行'
        if (OUT/'done'/(arm+'.json')).exists():phase='已完成';step=500000;eta='完成'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000000));tail=f.read().decode(errors='replace')
            points=re.findall(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO: .*?iter:\s*([\d,]+), lr:\(([^,]+),',tail)[-200:]
            if points:
                lr=points[-1][2];last=datetime.strptime(points[-1][0],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=tz)
                if live and (now-last).total_seconds()>300:eta='进度超过5分钟未更新'
                elif live and len(points)>1:
                    first=datetime.strptime(points[0][0],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=tz);n=int(points[-1][1].replace(',',''));n0=int(points[0][1].replace(',',''))
                    if n>n0:finish=last+timedelta(seconds=(last-first).total_seconds()/(n-n0)*(500000-n));eta=finish.strftime('%m-%d %H:%M')+' 北京时间'
            histories[label]=read_loss_history(log)
        states[arm]=dict(iter=step,phase=phase,lr=lr,finish=eta,gpu=launch.get('gpu'))
        cards.append(f'<article><h2>{label}</h2><b>{phase}</b> · GPU {launch.get("gpu","—")}<p>{step:,} / 500,000 · LR {lr}</p><progress max="500000" value="{step}"></progress><p>预计完成：{eta}</p></article>')
    gt=read(LENOVO/'STATUS.json');data=read(OUT/'DATA_STATUS.json');bank=read(OUT/'lenovo_bank/progress.json');controller=read(OUT/'CONTROLLER_STATUS.json');elig=read(OUT/'SOURCE_ELIGIBILITY.json')
    detail=dict(controller=controller,data=data,lenovo_gt=gt.get('verified_gt'),lenovo_total=3236,bank=bank,eligible_train=elig.get('train_counts'),data_ready=read(OUT/'DATA_READY.json').get('passed',False))
    save(OUT/'STATUS.json',dict(updated=now.isoformat(),runs=states,preparation=detail))
    text='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>最新VOSR · 四组500K</title><style>body{font:16px/1.7 system-ui;background:#eef3f8;color:#19314b;margin:25px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}article{background:white;padding:20px;border-radius:12px}progress{width:100%}pre{white-space:pre-wrap}</style><h1>V3M / V5 · guard / L1-only · 从头训练500K</h1><p>同结构共享随机初始化 · 对比度对齐 · 无CP · 有效文字patch · 初始1e-4，200K/300K/400K/450K降至5e-5/2.5e-5/1e-5/5e-6</p><p>'+now.strftime('%Y-%m-%d %H:%M:%S')+' 北京时间</p><article><h2>数据与启动状态</h2><pre>'+html.escape(json.dumps(detail,ensure_ascii=False,indent=2))+'</pre></article><div class="grid">'+''.join(cards)+'</div>'+render_loss_charts(histories)
    tmp=OUT/'index.tmp';tmp.write_text(text,encoding='utf-8');tmp.replace(OUT/'index.html')


def run_checked(command,log,env=None):
    """@luna-comment Input:known owned command. Output:checked completion. Key Steps:background subprocess and heartbeat. Key Params:15s poll. State/Dependencies:fail stops formal launch."""
    with log.open('ab') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        while p.poll() is None:publish();time.sleep(15)
        assert p.returncode==0,(command,p.returncode,str(log))


def main():
    """@luna-comment Input:explicit authorized four-arm run. Output:persistent data gate and four isolated500K jobs. Key Steps:exclusive lock;freeze scratch weights;prepare complete data;all CUDA gates;capacity-bound launches;monitor500K artifacts. Key Params:one arm perGPU,no old-job termination. State/Dependencies:server resident."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not list((OUT/'launch').glob('*.json')),'Refuse duplicate launch';save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=time.time()))
    records=prepare();save(OUT/'CONTROLLER_STATUS.json',dict(phase='准备已完成GT的固定快照'));publish()
    run_checked([sys.executable,'-u',str(ROOT/'tools/prepare_latest_vosr_500k.py')],OUT/'data.log')
    ready=read(OUT/'DATA_READY.json');assert ready['passed']
    for arm,r in records.items():
        save(OUT/'CONTROLLER_STATUS.json',dict(phase='启动检查',arm=arm))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        run_checked([sys.executable,'-u',str(ROOT/'tools/train_latest_vosr_500k.py'),'--smoke-config',r['smoke'],'--smoke-output',str(OUT/(arm+'_smoke.json'))],OUT/(arm+'_smoke.log'),env)
        assert read(OUT/(arm+'_smoke.json'))['passed']
    processes={}
    for arm,r in records.items():
        assert digest(r['config'])==r['config_sha256'] and digest(r['initial'])==r['initial_sha256']
        while True:
            free=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
            if free[r['gpu']]>=10000:break
            save(OUT/'CONTROLLER_STATUS.json',dict(phase='等待GPU显存',gpu=r['gpu']));publish();time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        (OUT/'logs').mkdir(exist_ok=True)
        with (OUT/'logs'/(arm+'.log')).open('xb') as f:p=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/train_latest_vosr_500k.py'),'-opt',r['config']],env=env,cwd=OUT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        processes[arm]=p;save(OUT/'launch'/(arm+'.json'),dict(pid=p.pid,gpu=r['gpu'],config=r['config'],initial_sha256=r['initial_sha256']))
    save(OUT/'CONTROLLER_STATUS.json',dict(phase='四组训练中'))
    while processes:
        for arm,p in list(processes.items()):
            if p.poll() is not None:
                final=OUT/'experiments'/records[arm]['name']/'models/net_g_500000.pth';assert p.returncode==0 and final.exists(),arm+' failed'
                save(OUT/'done'/(arm+'.json'),dict(passed=True,checkpoint=str(final)));del processes[arm]
        publish();time.sleep(30)
    save(OUT/'COMPLETE.json',dict(passed=True,arms=4,total_iter=500000));save(OUT/'CONTROLLER_STATUS.json',dict(phase='全部完成'));publish()


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'CONTROLLER_STATUS.json',dict(phase='异常停止，请检查日志',error=repr(e)));save(OUT/'FAILED.json',dict(error=repr(e)));publish();raise
