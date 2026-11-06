"""Detached bounded pilot controller and readable Chinese progress dashboard."""
from datetime import datetime, timezone, timedelta
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_v3_edge_pilot import OUT, save, digest
CONTINUATION='--continue-2k' in sys.argv
if CONTINUATION:OUT=OUT.with_name('v3_edge_cont10k_20261028')
TOTAL=10000 if CONTINUATION else 2000
ARMS={'baseline':'原 loss 续训基准','identity':'实验一：GT 恒等监督','guard':'实验二：可信边缘抵消保护'}


def dashboard(phase,processes=None):
    """@luna-comment Input:controller phase/process handles. Output:Chinese live HTML and snapshot. Key Steps:read independent progress, show losses/ETA, validate sample hashes at matched checkpoints. Key Params:10s server update/30s page refresh. State/Dependencies:owned files only."""
    records={}
    for arm in ARMS:
        folder=OUT/arm
        records[arm]={name:json.loads((folder/name).read_text()) for name in ['status.json','CALIBRATION.json','FAILED.json','COMPLETE.json'] if (folder/name).exists()}
        if processes and arm in processes:records[arm]['exit_code']=processes[arm].poll()
    now=datetime.now(timezone(timedelta(hours=8))).isoformat()
    save(OUT/'STATUS.json',dict(phase=phase,updated_at=now,arms=records))
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>V3M 两组边缘诊断实验</title><style>body{font:16px/1.6 system-ui,"Microsoft YaHei";max-width:1250px;margin:30px auto;background:#f4f7fa;color:#172334}section{background:white;border-radius:12px;padding:20px;margin:16px 0}progress{width:100%;height:24px}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #ddd;text-align:left;padding:8px}pre{white-space:pre-wrap;word-break:break-all}small{color:#526171}</style>'
    page+='<h1>V3M：两组短程强约束诊断</h1><p>同一 Full250K 权重与 Adam 状态；原生 V3 预处理数据；三组相同样本顺序、终点学习率和辅助 loss 状态。每组 2K iter，保存 0 / 100 / 500 / 1K / 2K。新增梯度初始目标约为原生重建项的 50%。</p>'
    if CONTINUATION:page=page.replace('同一 Full250K 权重与 Adam 状态','共同 Full250K 起点；恢复各组自己的 2K 权重与 Adam 状态').replace('每组 2K iter，保存 0 / 100 / 500 / 1K / 2K。','各自从 2K 权重和 Adam 状态继续到累计 10K iter，保存 2K / 4K / 6K / 10K，沿用原固定 loss 权重。')
    page+='<p>状态：'+html.escape(phase)+'　更新时间：'+now+'　<button onclick="location.reload()">立即刷新</button></p>'
    for arm,label in ARMS.items():
        record=records[arm];state=record.get('status.json',{});step=state.get('step',2000 if CONTINUATION else 0)
        eta=state.get('eta_seconds');eta_text=f'{eta/60:.1f} 分钟' if eta is not None else '等待实测'
        page+=f'<section><h2>{label}</h2><progress value="{step}" max="{TOTAL}"></progress><p>{step} / {TOTAL} iter　剩余约 {eta_text}　每步 {state.get("seconds_per_step",0):.3f} 秒</p>'
        if 'FAILED.json' in record:page+='<pre>'+html.escape(json.dumps(record['FAILED.json'],ensure_ascii=False))+'</pre>'
        page+='<table><tr><th>观测项</th><th>当前值</th></tr>'
        labels={'l_native':'原生配对重建 loss','l_total':'总 loss（含新增项）','extra_raw':'新增约束原始值','extra_weighted':'新增约束加权值','weight':'固定标量权重','support_fraction':'可信边缘/恒等边缘覆盖率','projection':'可信边缘残差投影','identity_native':'GT 恒等原生误差','cp_scale':'CP 权重','l_cp':'CP loss','l_sdf_boundary':'SDF loss'}
        for key,label2 in labels.items():
            if key in state.get('logs',{}):page+=f'<tr><td>{label2}</td><td>{state["logs"][key]:.7g}</td></tr>'
        page+='</table><details><summary>校准和运行明细</summary><pre>'+html.escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre></details></section>'
    page+='<p>这是方向诊断，不代表生产推荐。变锐、halo、假笔画及背景噪声需要共同审阅；没有自动扩大训练。</p></html>'
    temp=OUT/'dashboard.html.tmp';temp.write_text(page,encoding='utf-8');temp.replace(OUT/'dashboard.html')


def main():
    """@luna-comment Input:explicit controller and optional continuation flag. Output:three detached bounded jobs after actual CUDA smokes. Key Steps:smoke each arm;verify starting provenance/order or resumed prefix;launch GPU0/1/2;publish ETA;fail on exit. Key Params:2K pilot or own2K-to10K continuation,engineering GPU3. State/Dependencies:existing workloads never stopped."""
    OUT.mkdir(parents=True,exist_ok=True)
    trainer=Path(__file__).with_name('train_v3_edge_pilot.py')
    processes={};streams=[]
    try:
        dashboard('真实 CUDA 路径检查中')
        for arm in ARMS:
            folder=OUT/'engineering_smoke';folder.mkdir(exist_ok=True)
            with (folder/(arm+'.log')).open('wb') as stream:
                subprocess.run([sys.executable,'-u',str(trainer),'--arm',arm,'--steps',str(2002 if CONTINUATION else 2),'--smoke',*(['--continue-2k'] if CONTINUATION else [])],stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        smokes=[json.loads((OUT/'engineering_smoke'/arm/'COMPLETE.json').read_text()) for arm in ARMS]
        assert len({v['sequence_sha256'] for v in smokes})==1,'Paired sample order mismatch'
        assert all(v['logs']['cp_scale']>0 for v in smokes),'Original auxiliary loss schedule inactive'
        provenance=[json.loads((OUT/'engineering_smoke'/arm/'PROVENANCE.json').read_text()) for arm in ARMS]
        if CONTINUATION:
            assert len({tuple(p['learning_rates']) for p in provenance})==1
            assert all(p['start_step']==2000 and p['calibration_reused'] for p in provenance)
            gates=[json.loads((OUT/'engineering_smoke'/arm/'RESUME_GATE.json').read_text()) for arm in ARMS]
            assert all(g['passed'] for g in gates) and len({g['prefix_sha256'] for g in gates})==1
        else:assert len({(p['checkpoint_sha256'],p['optimizer_sha256'],tuple(p['learning_rates'])) for p in provenance})==1
        save(OUT/'SMOKE_GATE.json',dict(passed=True,matched_sequence=smokes[0]['sequence_sha256'],provenance=provenance,trainer_sha256=digest(trainer)))
        for gpu,arm in enumerate(ARMS):
            stream=(OUT/(arm+'.log')).open('wb');streams.append(stream)
            command=[sys.executable,'-u',str(trainer),'--arm',arm,'--steps',str(TOTAL),*(['--continue-2k'] if CONTINUATION else [])]
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),start_new_session=True)
            processes[arm]=process
            save(OUT/(arm+'_launch.json'),dict(pid=process.pid,gpu=gpu,command=command))
        while any(p.poll() is None for p in processes.values()):
            dashboard('短程实验进行中',processes);time.sleep(10)
        assert all(p.returncode==0 for p in processes.values()),'One or more pilot jobs failed; inspect per-arm logs'
        finals=[json.loads((OUT/arm/'COMPLETE.json').read_text()) for arm in ARMS]
        assert len({v['sequence_sha256'] for v in finals})==1
        save(OUT/'COMPLETE.json',dict(passed=True,paired_order_matches=True,arms=finals))
        dashboard('全部短程实验完成；等待效果审阅',processes)
    except Exception as error:
        save(OUT/'CONTROLLER_FAILED.json',dict(error=repr(error)));dashboard('执行异常，请查看日志',processes);raise
    finally:
        for stream in streams:stream.close()


if __name__=='__main__':main()
