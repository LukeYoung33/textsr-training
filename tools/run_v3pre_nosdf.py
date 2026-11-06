"""Independent scratch250K three-arm no-SDF controller; previous runs untouched."""
import copy
from datetime import datetime,timezone,timedelta
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.train_v3_edge_pilot import save,digest
SOURCE=Path('/AI_DNS/guang.yang/TextSR/data/v3pre_v3m_three_20261027')
OUT=SOURCE.with_name('v3pre_v3m_nosdf_250k_20261028')
REPO='/home/guang.yang/workspace/Real-ESRGAN'
ARMS={'full':'V3M Full｜无 SDF','text15':'V3M Text×1.5｜无 SDF','adapt_l1':'V3M Adaptive-L1 cap4｜无 SDF'}


def prepare():
    """@luna-comment Input:three accepted pretrain configs/complete V3 bank. Output:strict shared fresh init and isolated no-SDF YAMLs. Key Steps:clone corresponding arm;alter only SDF and run bindings;seed build fresh network;hash receipts. Key Params:250K,seed20260828,batch4. State/Dependencies:CPU bootstrap; no trained weights or optimizer resume."""
    import torch
    import yaml
    from tools.train_v3pre_area import bootstrap
    bootstrap(REPO)
    from basicsr.archs import build_network
    bank=SOURCE/'pretrain/bank256'
    assert json.loads((bank/'COMPLETE.json').read_text())['complete']
    assert json.loads((bank/'LINE_REGIONS_COMPLETE.json').read_text())['passed']
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'CONFIG_BINDINGS.json').exists(),'Refusing to overwrite a configured experiment'
    reference=yaml.safe_load((SOURCE/'pretrain/training/configs/full.yml').read_text())
    torch.manual_seed(reference['manual_seed'])
    network=build_network(copy.deepcopy(reference['network_g']))
    initial=OUT/'initial/v3m_fresh_shared.pth';initial.parent.mkdir(exist_ok=True)
    torch.save(dict(params=network.state_dict(),initialization='fresh_seeded_20260828_no_optimizer'),initial)
    records={}
    for gpu,arm in enumerate(ARMS):
        source=SOURCE/'pretrain/training/configs'/(arm+'.yml')
        old=yaml.safe_load(source.read_text());cfg=copy.deepcopy(old)
        cfg['name']='v3pre_nosdf_'+arm+'_20261028'
        cfg['path'].update(pretrain_network_g=str(initial),resume_state=None,strict_load_g=True,experiments_root=str(OUT/'training/experiments'))
        cfg['train'].update(sdf_boundary_weight=0.,sdf_boundary_opt=None)
        cfg['dist_params']['port']=33880+gpu
        assert {key for key in old['train'] if old['train'][key]!=cfg['train'][key]}=={'sdf_boundary_weight','sdf_boundary_opt'}
        assert cfg['train']['total_iter']==250000 and cfg['datasets']==old['datasets'] and cfg['network_g']==old['network_g']
        assert cfg['datasets']['train']['batch_size_per_gpu']==4
        path=OUT/'training/configs'/(arm+'.yml');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        records[arm]=dict(config=str(path),config_sha256=digest(path),reference_sha256=digest(source),gpu=gpu,name=cfg['name'],total_iter=250000)
    save(OUT/'CONFIG_BINDINGS.json',dict(fresh_initial_sha256=digest(initial),fresh_initial=str(initial),seed=reference['manual_seed'],bank_receipt_sha256=digest(bank/'COMPLETE.json'),line_receipt_sha256=digest(bank/'LINE_REGIONS_COMPLETE.json'),sdf_removed=True,arms=records))
    return records


def publish(phase,processes=None):
    """@luna-comment Input:controller phase/children. Output:distinct loss curves, progress/ETA and JSON snapshot. Key Steps:read actual logs;render Chinese cards and standalone charts;never infer final completion. Key Params:250K,30s refresh. State/Dependencies:owned logs only."""
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    states={};histories={}
    for arm,label in ARMS.items():
        logfile=OUT/'training/logs'/(arm+'.log');history=read_loss_history(logfile)
        if history:histories[label.replace('｜',' | ')]=history
        text=''
        if logfile.exists():
            with logfile.open('rb') as stream:
                stream.seek(max(0,logfile.stat().st_size-32768));text=stream.read().decode('utf-8',errors='replace')
        eta=re.findall(r'eta:\s*([^,\n]+)',text)
        states[arm]=dict(label=label,step=history[-1]['step'] if history else 0,total=250000,eta=eta[-1] if eta else '等待稳定日志',exit_code=processes[arm].poll() if processes and arm in processes else None,last_loss=history[-1] if history else None)
    timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
    save(OUT/'STATUS.json',dict(phase=phase,updated_at=timestamp,arms=states))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>V3M 无SDF三组预训练</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei";margin:28px;background:#f4f7fa;color:#172334}section{padding:18px;background:white;border-radius:12px;margin:16px 0}progress{width:100%;height:24px}pre{white-space:pre-wrap}h2{font-size:20px}</style><h1>V3M + V3 preprocess：无 SDF 三组预训练</h1><p>Full / Text×1.5 / Adaptive-L1 cap4；相同全新随机初始化；256×256 patch，batch=4，各250K iter。SDF 权重为0且模块未构建。其他对应 loss、数据和学习率计划沿用上一轮。</p>'
    page+='<p>状态：'+html.escape(phase)+'　更新时间：'+timestamp+'　<button onclick="location.reload()">刷新</button></p>'
    for arm,s in states.items():page+=f'<section><h2>{s["label"]}</h2><progress value="{s["step"]}" max="250000"></progress><p>{s["step"]:,} / 250,000 iter　ETA：{html.escape(str(s["eta"]))}　退出码：{s["exit_code"]}</p></section>'
    page+=render_loss_charts(histories).replace(",['l_sdf_boundary','SDF boundary']",'')+'</html>'
    temporary=OUT/'dashboard.html.tmp';temporary.write_text(page,encoding='utf-8');temporary.replace(OUT/'dashboard.html')


def main():
    """@luna-comment Input:explicit user-authorized controller invocation. Output:three detached scratch runs and durable monitor. Key Steps:fresh configs;actual CUDA no-SDF smokes;hash/GPU gates;unchanged trainer;check final250K weights. Key Params:GPU0/1/2 and smokeGPU3, no finetune. State/Dependencies:existing workloads never terminated."""
    OUT.mkdir(parents=True,exist_ok=True);processes={};streams=[]
    try:
        configs=prepare();publish('真实 CUDA 无SDF训练检查中')
        for arm,record in configs.items():
            folder=OUT/'smoke';folder.mkdir(exist_ok=True)
            with (folder/(arm+'.log')).open('xb') as stream:
                subprocess.run([sys.executable,'-u',str(ROOT/'tools/smoke_v3pre_nosdf.py'),record['config'],str(folder/(arm+'.json'))],stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        save(OUT/'SMOKE_COMPLETE.json',dict(passed=True,arms=list(ARMS)))
        for arm,record in configs.items():
            assert digest(record['config'])==record['config_sha256']
            assert json.loads((OUT/'smoke'/(arm+'.json')).read_text())['config_sha256']==record['config_sha256']
            free=int(subprocess.check_output(['nvidia-smi','--id='+str(record['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>4096
            folder=OUT/'training/logs';folder.mkdir(parents=True,exist_ok=True)
            stream=(folder/(arm+'.log')).open('xb');streams.append(stream)
            command=[sys.executable,'-u',str(ROOT/'tools/train_v3pre_area.py'),'--repo-root',REPO,'-opt',record['config']]
            proc=subprocess.Popen(command,cwd=OUT/'training',stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(record['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            processes[arm]=proc;save(OUT/'training/launch'/(arm+'.json'),dict(pid=proc.pid,command=command,gpu=record['gpu'],config_sha256=record['config_sha256'],started_at=datetime.now(timezone.utc).isoformat()))
        while any(p.poll() is None for p in processes.values()):
            publish('三组250K预训练进行中',processes);time.sleep(15)
        assert all(p.returncode==0 for p in processes.values()),'A training job failed; inspect logs'
        finals={arm:str(OUT/'training/experiments'/r['name']/'models/net_g_250000.pth') for arm,r in configs.items()}
        assert all(Path(p).exists() for p in finals.values())
        save(OUT/'COMPLETE.json',dict(passed=True,finals=finals));publish('三组250K全部完成',processes)
    except Exception as error:
        save(OUT/'FAILED.json',dict(error=repr(error)));publish('执行异常，请查看日志',processes);raise
    finally:
        for stream in streams:stream.close()


if __name__=='__main__':main()
