"""Read-only unified dashboard for the four contrast/guard training arms."""
from pathlib import Path
from datetime import datetime
import html,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_s22_patch_contrast_guard import SOURCE,TZ
from tools.run_uniform20_steep_scurve import save
from tools.offline_loss_charts import read_loss_history,render_loss_charts
from tools.s22_training_eta import duration
DATA=SOURCE.parent
OUT=DATA/'s22_guard_contrast_overview_20261029'
GROUPS=[('cp_all',DATA/'s22_patch_contrast_guard_20261029',True),('nocp_text',DATA/'s22_nocp_text_guard_20261029',False)]


def publish():
    """@luna-comment Input:two authoritative monitor snapshots and append-only logs. Output:atomic four-run overview and one shared four-series chart. Key Steps:read status/ETA;label SDF/CP/filter explicitly;show source ages;render histories. Key Params:90s stale warning,250K target. State/Dependencies:read-only sources;owned overview only;no training control."""
    runs={};histories={};cards=[];now=datetime.now(TZ);ages={}
    filtered=json.loads((GROUPS[1][1]/'FILTER_COMPLETE.json').read_text())
    for group,source,cp in GROUPS:
        snapshot=json.loads((source/'MONITOR.json').read_text());age=max(0,(now-datetime.fromisoformat(snapshot['updated_cst'])).total_seconds());ages[group]=age
        for arm in ['full_sdf','full_nosdf']:
            sdf=arm=='full_sdf'
            label=('\u542bCP\uff5c\u5168\u90e8patch' if cp else '\u65e0CP\uff5c\u4ec5\u6709\u6548\u6587\u672cpatch')+' | '+('Full\uff08\u542bSDF\uff09' if sdf else 'Full\uff08\u65e0SDF\uff09')
            record=dict(snapshot['runs'][arm]);record.update(label=label,cp=cp,sdf=sdf,filter_empty_text=not cp,source_updated_cst=snapshot['updated_cst'],source_age_seconds=age)
            key=group+'_'+arm;runs[key]=record
            history=read_loss_history(source/'logs'/(arm+'.log'))
            if history:histories[label]=history
            state={'running':'\u8bad\u7ec3\u4e2d','complete':'\u5df2\u5b8c\u6210','failed':'\u5931\u8d25','waiting':'\u7b49\u5f85'}.get(record['state'],record['state'])
            eta=record.get('eta',{});logs=record.get('logs',{})
            warning='<p class="warning">\u6570\u636e\u6e90\u5df2\u8d85\u8fc790\u79d2\u672a\u66f4\u65b0</p>' if age>90 else ''
            cards.append('<section class="run-card"><h2>'+html.escape(label)+'</h2><p>'+html.escape(state)+' | GPU '+str(record.get('gpu'))+' | PID '+str(record.get('pid'))+'</p><progress value="'+str(record['step'])+'" max="250000"></progress><p><strong>'+str(record['step'])+' / 250000</strong></p><p>\u9884\u8ba1\u5b8c\u6210: '+html.escape(eta.get('finish_cst','--'))+'<br>\u9884\u8ba1\u5269\u4f59: '+duration(eta.get('remaining_seconds'))+'</p><p class="muted">\u5757\u7ea7GT\u5bf9\u6bd4\u5ea6\u589e\u76ca: '+str(round(logs.get('contrast_gain',1),3))+'<br>\u6570\u636e\u66f4\u65b0: '+html.escape(snapshot['updated_cst'])+'</p>'+warning+'</section>')
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>\u53ef\u4fe1\u8fb9\u7f18\u4fdd\u62a4\uff5c4\u7ec4\u8bad\u7ec3\u603b\u89c8</title><style>body{font:16px/1.7 system-ui;margin:24px;background:#f2f5fa;color:#172334}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:16px}section{background:white;border-radius:12px;padding:20px}h2{font-size:19px}progress{width:100%;height:20px}.muted{font-size:13px;color:#637083}.warning{color:#b04420}@media(max-width:760px){.grid{grid-template-columns:1fr}}</style><h1>\u53ef\u4fe1\u8fb9\u7f18\u4fdd\u62a4 | \u56db\u7ec4\u8bad\u7ec3\u603b\u89c8</h1><p>V3M | GT\u5757\u7ea7\u5bf9\u6bd4\u5ea6\u5bf9\u9f50 | \u4ece\u5934\u8bad\u7ec3250K | batch4 | 256x256</p><p>\u6709CP\u7ec4\u4fdd\u7559\u5168\u90e8patch\uff1b\u65e0CP\u7ec4\u4ec5\u4fdd\u7559\u6709\u6548\u6587\u672cpatch\uff08'+str(filtered['kept_unique'])+' / '+str(filtered['source_unique'])+'\uff09\u3002\u8fd9\u8f6e\u540c\u65f6\u6539\u53d8CP\u548c\u91c7\u6837\uff0c\u7528\u4e8e\u6bd4\u8f83\u7ec4\u5408\u6548\u679c\u3002</p><p class="muted">\u66f4\u65b0: '+now.isoformat()+' | \u9875\u976230\u79d2\u5237\u65b0\uff1b\u9a8c\u8bc1\u96c6583\u5f20\u4fdd\u6301\u65e7GT_VOSR\uff0c\u4e0d\u8fdb\u884c\u6b64\u6b21\u6821\u6b63\u3002</p><div class="grid">'+''.join(cards)+'</div>'+render_loss_charts(histories)+'</html>'
    temporary=OUT/'dashboard.html.tmp';temporary.write_text(body,encoding='utf-8');temporary.replace(OUT/'dashboard.html')
    save(OUT/'MONITOR.json',dict(runs=runs,source_ages=ages,updated_cst=now.isoformat(),series_count=len(histories)))
    return runs


def main():
    """@luna-comment Input:existing four-arm experiment sources. Output:persistent observer-only unified dashboard. Key Steps:exclusive lock;write PID receipt;publish every20s. Key Params:no trainer/controller restart or source writes. State/Dependencies:Linux flock,owned overview artifacts only."""
    import fcntl
    OUT.mkdir(exist_ok=True);lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'OBSERVER.json',dict(pid=os.getpid(),started_cst=datetime.now(TZ).isoformat()))
    while True:
        try:publish()
        except Exception as error:save(OUT/'OBSERVER_ERROR.json',dict(error=repr(error),epoch=time.time()))
        time.sleep(20)


if __name__=='__main__':main()
