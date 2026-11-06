"""Isolated high-learning-rate guard versus native-L1 comparison."""
from pathlib import Path
import os,sys,json,copy,subprocess,time,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest


def main():
    """@luna-comment Input:approved matched pair. Output:two smoke-gated150K runs. Key Steps:clone noCP recipe;remove auxiliary terms only forL1;verify actual CUDA updates;monitor both. Key Params:lr1e-3,GPU1/3,fresh optimizer. State/Dependencies:read-only shared bank and initial checkpoint."""
    import yaml
    import tools.monitor_mojito_finetune_eta as monitor
    out=Path('/AI_DNS/guang.yang/TextSR/data/mojito_v3m_lr1e3_pair_20261030');out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not list((out/'launch').glob('*.json'))
    base=out.parent/'mojito280_guard_finetune_20261030/configs/contrast_nocp.yml'
    template=yaml.safe_load(base.read_text());bindings={}
    # Isolated smoke variant additionally verifies that the L1 objective is the entire loss.
    text=(ROOT/'tools/train_full_s22_strategy.py').read_text()
    old="    assert logs['uv_strong_scale']>0"
    assert text.count(old)==1
    text=text.replace(old,"    if opt['train'].get('uv_strong_weight',0)>0:assert logs['uv_strong_scale']>0\n    else:\n        assert model.strategy=='paired' and not model.calibration\n        assert abs(logs['l_total']-logs['l_native'])<1e-7\n        assert all(v==0 for k,v in logs.items() if k.endswith('_scale'))")
    trainer=ROOT/'tools/train_mojito_lr1e3_pair_snapshot.py';assert not trainer.exists();trainer.write_text(text)
    for i,arm in enumerate(['guard','l1_only']):
        cfg=copy.deepcopy(template);cfg['name']='mojito_v3m_lr1e3_'+arm+'_20261030'
        cfg['path'].update(experiments_root=str(out/'experiments'),resume_state=None)
        cfg['train']['optim_g']['lr']=1e-3;cfg['s22_strategy']['audit_root']=str(out/'audits'/arm);cfg['dist_params']['port']=35840+i
        if arm=='l1_only':
            cfg['s22_strategy']['name']='paired'
            for k in list(cfg['train']):
                if k.endswith('_weight') or k.startswith('recall_aware_weight_'):cfg['train'][k]=0.
                if k.endswith('_opt') and k not in ['y4uv_native_opt','yuv444_reference_opt']:cfg['train'][k]=None
        config=out/(arm+'.yml');config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(out/'smoke_audits'/arm)
        sm=out/(arm+'_smoke.yml');sm.write_text(yaml.safe_dump(smoke,sort_keys=False))
        bindings[arm]=dict(config=str(config),config_sha256=digest(config),smoke=str(sm),name=cfg['name'],initial_sha256=digest(cfg['path']['pretrain_network_g']))
    save(out/'BINDINGS.json',bindings);processes={}
    monitor.OUT=out;monitor.LABELS={'guard':'V3M guard + 对比度对齐 + noCP | LR 1e-3','l1_only':'V3M L1-only + 对比度对齐 + noCP | LR 1e-3'}
    for arm,gpu in [('guard',1),('l1_only',3)]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        with (out/(arm+'_smoke.log')).open('ab') as log:
            subprocess.run([sys.executable,'-u',str(trainer),'--smoke-config',bindings[arm]['smoke'],'--smoke-output',str(out/(arm+'_smoke.json'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((out/(arm+'_smoke.json')).read_text())['passed']
        (out/'logs').mkdir(exist_ok=True)
        with (out/'logs'/(arm+'.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-u',str(trainer),'-opt',bindings[arm]['config']],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        processes[arm]=p;save(out/'launch'/(arm+'.json'),dict(pid=p.pid,gpu=gpu))
    while processes:
        for arm,p in list(processes.items()):
            if p.poll() is not None:
                assert p.returncode==0 and (out/'experiments'/bindings[arm]['name']/'models/net_g_150000.pth').exists()
                save(out/'done'/(arm+'.json'),dict(passed=True));del processes[arm]
        monitor.update()
        page=out/'eta_dashboard.html';s=page.read_text().replace('三组 finetune','LR 1e-3 两组 finetune');page.write_text(s)
        time.sleep(30)


if __name__=='__main__':main()
