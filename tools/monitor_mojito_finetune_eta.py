"""Read-only training observer with rolling wall-clock completion forecasts."""
from pathlib import Path
import json, re, time, os, fcntl
from datetime import datetime, timedelta, timezone
from tools.schedule_mojito280_guard_finetune import OUT, LABELS
from tools.offline_loss_charts import read_loss_history, render_loss_charts


def update():
    """@luna-comment Input: owned logs/launch receipts. Output: ETA HTML/JSON. Key Steps: estimate from last200 timestamped records; suppress stale or dead process forecasts. Key Params:150K target,300s stale threshold,CST. State/Dependencies:read-only training observer; separate monitor artifacts."""
    cards=[]; histories={}; estimates={};tz=timezone(timedelta(hours=8))
    for arm,label in LABELS.items():
        log=OUT/'logs'/(arm+'.log');points=[]
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-2000000));raw=f.read().decode('utf-8',errors='replace')
            for stamp,step in re.findall(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO: .*?\biter:\s*([\d,]+),',raw):
                points.append((datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=tz),int(step.replace(',',''))))
        points=points[-200:];step=points[-1][1] if points else 0
        launch=OUT/'launch'/(arm+'.json');live=False
        if launch.exists():live=Path('/proc/'+str(json.loads(launch.read_text())['pid'])).exists()
        detail='正在采集训练速度';seconds=None;finish=None
        if (OUT/'done'/(arm+'.json')).exists():step=150000;detail='训练已完成'
        elif not live:detail='训练进程未运行，暂无可靠预计时间'
        elif points and (datetime.now(tz)-points[-1][0]).total_seconds()>300:detail='进度超过5分钟未更新，暂不估计'
        elif len(points)>1 and points[-1][1]>points[0][1]:
            rate=(points[-1][0]-points[0][0]).total_seconds()/(points[-1][1]-points[0][1])
            seconds=max(0,150000-step)*rate;finish=points[-1][0]+timedelta(seconds=seconds)
            remaining=max(0,(finish-datetime.now(tz)).total_seconds());minutes=round(remaining/60)
            detail=f'预计剩余：{minutes//60}小时{minutes%60}分钟<br>预计完成：{finish:%m月%d日 %H:%M}（北京时间）'
        estimates[arm]=dict(iter=step,remaining_seconds=seconds,estimated_finish=finish.isoformat() if finish else None,live=live)
        cards.append(f'<section><h2>{label}</h2><p>{step:,} / 150,000</p><progress max="150000" value="{step}"></progress><p>{detail}</p></section>')
        histories[label]=read_loss_history(log)
    body='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Mojito280训练监控</title><style>body{font:17px/1.7 system-ui;background:#eef3f8;color:#19314b;margin:28px}section{background:white;padding:20px;border-radius:12px;margin:16px 0}progress{width:100%}</style><h1>Mojito280 · 三组 finetune 150K</h1><p>243张训练 / 37张验证 · 共用普通 guard250K 初始化</p><p>按近期最多200条训练记录的实际耗时估计；验证、保存及GPU负载变化会影响完成时间。每30秒刷新。</p><p>更新时间：'+datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')+'</p>'+''.join(cards)+render_loss_charts(histories)
    tmp=OUT/'eta_dashboard.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'eta_dashboard.html')
    tmp=OUT/'ETA.tmp';tmp.write_text(json.dumps(estimates));tmp.replace(OUT/'ETA.json')


if __name__=='__main__':
    lock=(OUT/'eta_monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        update();time.sleep(30)
