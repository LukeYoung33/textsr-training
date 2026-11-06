"""One recommended native-L1 control using completed dilated original aligned GTs."""
from pathlib import Path
import sys,os,json,subprocess,time,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import launch_latest_vosr_500k as monitor
from tools.prepare_dilate2_l1_bank import OUT,BASE,DATA
from tools.run_uniform20_steep_scurve import save,digest
ARM='v3m_dilate2_l1';monitor.OUT=OUT;monitor.LABELS={ARM:'V3M · 原始GT笔画+2px · L1-only'};original_publish=monitor.publish


def publish():
    """@luna-comment Input:data/trainingstate. Output:singlecontrolprogress/dataETA. Key Steps:reuseobserver;correctlabel andexposebank/sidecarprogress. Key Params:7385train/620val. State/Dependencies:ownedmetadataonly."""
    original_publish();data=monitor.read(OUT/'DATA_STATUS.json');bank=monitor.read(OUT/'train_bank/progress.json');lines=monitor.read(OUT/'train_bank/line_regions/progress.json')
    state=monitor.read(OUT/'STATUS.json');state['preparation'].update(data=data,bank=bank,line_regions=lines,train_count=7385,val_count=620,excluded_mojito=243)
    if not (OUT/'launch'/(ARM+'.json')).exists():state['runs'][ARM].update(phase=data.get('phase','准备独立训练数据'),finish='数据就绪后自动启动')
    save(OUT/'STATUS.json',state);path=OUT/'index.html';text=path.read_text(encoding='utf-8')
    text=text.replace('最新VOSR · 四组500K','V3M · 原始GT+2px · L1-only').replace('V3M / V5 · guard / L1-only · 从头训练500K','V3M · 原始对齐GT笔画+2px · L1-only · 500K')
    text=text.replace('等待固定快照patch构建和启动检查',data.get('phase','准备独立训练数据')).replace('采样中','等待训练日志估时')
    info='<p>7385训练图，不含Mojito243；复用原620验证图。真实LQ沿用V3缓存，新GT无额外预处理。独立patch数据构建完成后自动开训。</p><pre>'+json.dumps(dict(data=data,bank=bank,line_regions=lines),ensure_ascii=False,indent=2)+'</pre>'
    text=text.replace('<div class="grid">',info+'<div class="grid">',1);tmp=OUT/'index.corrected.tmp';tmp.write_text(text,encoding='utf-8');tmp.replace(path)


def prepare():
    """@luna-comment Input:completedoldL1recipe andnewGTcompletion. Output:onefrozenL1config/sharedrandominitbinding. Key Steps:changebankandtargetlineageonly;preserveloss/seed/optimizer/schedule. Key Params:500K,1e-4to5e-6,batch4,noaux. State/Dependencies:oldartifactsread-only."""
    import yaml
    d=json.loads((BASE/'CONFIG_BINDINGS.json').read_text())['runs']['v3m_l1'];assert digest(d['config'])==d['config_sha256'];cfg=yaml.safe_load(Path(d['config']).read_text());oldtrain=cfg['train'].copy()
    cfg['name']=ARM+'_500k_20261103';cfg['path'].update(experiments_root=str(OUT/'experiments'),resume_state=None);cfg['s22_strategy'].update(name='paired',audit_root=str(OUT/'audits'/ARM));cfg['dist_params']['port']=35955
    cfg['v3pre_target_domain']='Train7385 original aligned GT plus width2;existing V3 LQ;Mojito243 excluded;validation620 unchanged'
    for split in ['train','val']:cfg['datasets'][split].update(root=str(OUT/'bank'),manifest=str(OUT/'bank'/(split+'.json')))
    assert cfg['train']==oldtrain and cfg['train']['total_iter']==500000 and cfg['train']['optim_g']['lr']==1e-4
    assert digest(cfg['path']['pretrain_network_g'])==d['initial_sha256'];assert all(v==0 for k,v in cfg['train'].items() if k.endswith('_weight') or k.startswith('recall_aware_weight_'))
    path=OUT/'configs'/(ARM+'.yml');path.parent.mkdir(parents=True,exist_ok=True);text=yaml.safe_dump(cfg,sort_keys=False)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    cfg['name']+='_smoke';cfg['s22_strategy']['audit_root']=str(OUT/'smoke_audit');smoke=OUT/'configs/smoke.yml';smoke.write_text(yaml.safe_dump(cfg,sort_keys=False))
    result=dict(label=monitor.LABELS[ARM],name=ARM+'_500k_20261103',config=str(path),config_sha256=digest(path),initial=cfg['path']['pretrain_network_g'],initial_sha256=digest(cfg['path']['pretrain_network_g']),smoke=str(smoke),gpu=2)
    save(OUT/'CONFIG_BINDINGS.json',dict(runs={ARM:result},base_config_sha256=d['config_sha256'],source_gt_complete_sha256=digest(DATA/'original_aligned_gt_dilate2_20261103/COMPLETE.json'),initialization='same previous V3M frozen random state;fresh Adam'));return result


def main():
    """@luna-comment Input:userrequestednewGTcontrol. Output:residentdata-gated500Ktraining. Key Steps:exclusivefreeze,buildindependentdata,realL1smoke,resourceheadroomlaunch,completionmonitor. Key Params:GPU2co-location>=8GiBfree andutil<=80. State/Dependencies:neverstopoldjobsorchangequeuepolicy."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'launch'/(ARM+'.json')).exists()
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=time.time()));r=prepare();save(OUT/'CONTROLLER_STATUS.json',dict(phase='准备新GT训练数据'))
    monitor.run_checked([sys.executable,'-u',str(ROOT/'tools/prepare_dilate2_l1_bank.py')],OUT/'data.log')
    assert monitor.read(OUT/'DATA_READY.json')['passed'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    save(OUT/'CONTROLLER_STATUS.json',dict(phase='L1训练启动检查'))
    monitor.run_checked([sys.executable,'-u',str(ROOT/'tools/train_latest_vosr_500k.py'),'--smoke-config',r['smoke'],'--smoke-output',str(OUT/'smoke.json')],OUT/'smoke.log',env)
    assert monitor.read(OUT/'smoke.json')['passed'];assert digest(r['config'])==r['config_sha256'] and digest(r['initial'])==r['initial_sha256']
    while True:
        free,util=[int(s.strip()) for s in subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
        if free>=8192 and util<=80:break
        save(OUT/'CONTROLLER_STATUS.json',dict(phase='等待GPU2资源余量'));publish();time.sleep(30)
    (OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'logs'/(ARM+'.log')).open('xb') as f:p=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/train_latest_vosr_500k.py'),'-opt',r['config']],cwd=OUT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(OUT/'launch'/(ARM+'.json'),dict(pid=p.pid,gpu=2,config=r['config']));save(OUT/'CONTROLLER_STATUS.json',dict(phase='训练中'))
    while p.poll() is None:publish();time.sleep(30)
    final=OUT/'experiments'/r['name']/'models/net_g_500000.pth';assert p.returncode==0 and final.exists()
    save(OUT/'done'/(ARM+'.json'),dict(passed=True,checkpoint=str(final)));save(OUT/'COMPLETE.json',dict(passed=True,total_iter=500000));publish()


monitor.publish=publish
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'FAILED.json',dict(error=repr(e)));save(OUT/'CONTROLLER_STATUS.json',dict(phase='异常停止',error=repr(e)));publish();raise
