"""Isolated V5 pretrained control for the Mojito noCP contrast guard recipe."""
from pathlib import Path
import sys,json,os,subprocess,copy,time,shutil,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest


def main():
    """@luna-comment Input:approved V5 block250K and existing noCP recipe. Output:smoke-gated150K training and ETA page. Key Steps:freeze weights/config;strict nine-update smoke;launch fresh optimizer;monitor owned process. Key Params:GPU2,batch4,243/37 split. State/Dependencies:existing training untouched."""
    import yaml
    import tools.monitor_mojito_finetune_eta as monitor
    data=Path('/AI_DNS/guang.yang/TextSR/data');out=data/'mojito280_v5_guard_ft_20261030';out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'launch/v5.json').exists(),'Existing training must not be relaunched'
    source=data/'vosr_2u_block256_v1_20261023/training_comparison_20261023'
    old=yaml.safe_load((source/'configs/05_v5_block256_raw.yml').read_text())
    checkpoint=source/'experiments'/old['name']/'models/net_g_250000.pth';sha=digest(checkpoint)
    initial=out/'initial.pth';shutil.copy2(checkpoint,initial);assert digest(initial)==sha==digest(checkpoint)
    base=data/'mojito280_guard_finetune_20261030/configs/contrast_nocp.yml';cfg=yaml.safe_load(base.read_text())
    cfg['name']='mojito280_v5_guard_contrast_nocp_ft150k_20261030';cfg['network_g']=old['network_g']
    cfg['path'].update(pretrain_network_g=str(initial),experiments_root=str(out/'experiments'),resume_state=None)
    cfg['s22_strategy']['audit_root']=str(out/'audits/v5');cfg['dist_params']['port']=35735
    config=out/'config.yml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(out/'smoke_audit')
    smokeconfig=out/'smoke.yml';smokeconfig.write_text(yaml.safe_dump(smoke,sort_keys=False))
    save(out/'BINDING.json',dict(source_checkpoint=str(checkpoint),checkpoint_sha256=sha,source_recipe=str(base),source_recipe_sha256=digest(base),config_sha256=digest(config)))
    monitor.OUT=out;monitor.LABELS={'v5':'V5 · guard + 对比度对齐 + noCP'}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    trainer=ROOT/'tools/train_full_s22_strategy.py'
    with (out/'smoke.log').open('ab') as log:
        subprocess.run([sys.executable,'-u',str(trainer),'--smoke-config',str(smokeconfig),'--smoke-output',str(out/'smoke.json')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    receipt=json.loads((out/'smoke.json').read_text());assert receipt['passed'] and receipt['no_cp'] and receipt['no_sdf'] and receipt['initial_exact'] and receipt['updates']==9
    (out/'logs').mkdir(exist_ok=True)
    with (out/'logs/v5.log').open('xb') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(trainer),'-opt',str(config)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(out/'launch/v5.json',dict(pid=proc.pid,gpu=2,config=str(config)))
    while proc.poll() is None:monitor.update();time.sleep(30)
    final=out/'experiments'/cfg['name']/'models/net_g_150000.pth'
    assert proc.returncode==0 and final.exists()
    save(out/'done/v5.json',dict(passed=True,checkpoint=str(final)));monitor.update()


if __name__=='__main__':main()
