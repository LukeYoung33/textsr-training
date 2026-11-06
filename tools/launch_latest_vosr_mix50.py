"""Two isolated scratch500K Full-no-guard mild-GT mixture controls."""
from pathlib import Path
import sys,os,copy,json,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import launch_latest_vosr_500k as monitor
from tools.run_uniform20_steep_scurve import save,digest
BASE=monitor.OUT;OUT=BASE.parent/'latest_vosr_mix50_500k_20261102'
LABELS={'v3m_mix50':'V3M · Full无guard · 50% GT轻度退化','v5_mix50':'V5 · Full无guard · 50% GT轻度退化'}
monitor.OUT=OUT;monitor.LABELS=LABELS


def prepare():
    """@luna-comment Input:frozen four-arm config/bank. Output:two matched Full configs. Key Steps:clone guard recipes;disable guard only;share original scratch initialization. Key Params:500K,same LR/data/aux. State/Dependencies:read-only original experiment."""
    import yaml
    assert json.loads((BASE/'DATA_READY.json').read_text())['passed'];records={}
    for arch,gpu in [('v3m',1),('v5',2)]:
        arm=arch+'_mix50';cfg=yaml.safe_load((BASE/'configs'/(arch+'_guard.yml')).read_text())
        cfg['name']='latest_vosr_'+arm+'_500k_20261102';cfg['model_type']='LatestVOSRMix50';cfg['train']['model_type']='LatestVOSRMix50'
        cfg['path']['experiments_root']=str(OUT/'experiments');cfg['s22_strategy'].update(name='mixed50',audit_root=str(OUT/'audits'/arm))
        cfg['dist_params']['port']=35920+len(records)
        path=OUT/'configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True);text=yaml.safe_dump(cfg,sort_keys=False)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke_audits'/arm)
        sm=OUT/'configs'/(arm+'_smoke.yml');sm.write_text(yaml.safe_dump(smoke,sort_keys=False))
        records[arm]=dict(config=str(path),config_sha256=digest(path),initial_sha256=digest(cfg['path']['pretrain_network_g']),initial=cfg['path']['pretrain_network_g'],smoke=str(sm),gpu=gpu,name=cfg['name'])
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=records,bank_ready_sha256=digest(BASE/'DATA_READY.json'),mixture=dict(probability=.5,sigma=[.3,.9],resize=[.8,1.],noise_sigma_255=[0,2]),loss='Full auxiliary losses retained;edge cancellation guard off;SDF/CP off'))
    return records


def main():
    """@luna-comment Input:authorized two-arm extension. Output:resident checked launch and live ETA. Key Steps:lock,freeze configs,two CUDA smokes,capacity gates,launch,monitor. Key Params:GPU1/2 coexist,no existing job termination. State/Dependencies:owned processes only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not list((OUT/'launch').glob('*.json'))
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=time.time()));records=prepare()
    command=[sys.executable,'-u',str(ROOT/'tools/train_latest_vosr_mix50.py')]
    for arm,r in records.items():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        save(OUT/'CONTROLLER_STATUS.json',dict(phase='启动检查',arm=arm))
        monitor.run_checked(command+['--smoke-config',r['smoke'],'--smoke-output',str(OUT/(arm+'_smoke.json'))],OUT/(arm+'_smoke.log'),env)
        assert monitor.read(OUT/(arm+'_smoke.json'))['passed']
    processes={}
    for arm,r in records.items():
        assert digest(r['config'])==r['config_sha256'] and digest(r['initial'])==r['initial_sha256']
        while True:
            free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
            if free[r['gpu']]>=6000:break
            save(OUT/'CONTROLLER_STATUS.json',dict(phase='等待GPU显存',gpu=r['gpu']));monitor.publish();time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        (OUT/'logs').mkdir(exist_ok=True)
        with (OUT/'logs'/(arm+'.log')).open('xb') as f:p=subprocess.Popen(command+['-opt',r['config']],env=env,cwd=OUT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        processes[arm]=p;save(OUT/'launch'/(arm+'.json'),dict(pid=p.pid,gpu=r['gpu'],config=r['config']))
    save(OUT/'CONTROLLER_STATUS.json',dict(phase='两组训练中'))
    while processes:
        for arm,p in list(processes.items()):
            if p.poll() is not None:
                final=OUT/'experiments'/records[arm]['name']/'models/net_g_500000.pth';assert p.returncode==0 and final.exists(),arm+' failed'
                save(OUT/'done'/(arm+'.json'),dict(passed=True,checkpoint=str(final)));del processes[arm]
        monitor.publish();time.sleep(30)
    save(OUT/'COMPLETE.json',dict(passed=True,arms=2,total_iter=500000));monitor.publish()


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'FAILED.json',dict(error=repr(e)));save(OUT/'CONTROLLER_STATUS.json',dict(phase='异常停止',error=repr(e)));monitor.publish();raise
