"""Read-only unified monitor for the five approved VOSR2u training runs."""
import argparse
from datetime import datetime,timezone,timedelta
import html
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.offline_loss_charts import render_loss_charts
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'vosr2u_five_monitor_20261026'
SOURCES={'v3m':DATA/'v3m_vosr2u_20261026/training_20261026/STATUS.json',
         'v5lite':DATA/'v5lite_vosr2u_20261026/training_20261026/STATUS.json'}
ARMS=(('v3m','full_v3m','V3M','Full'),('v3m','text15_v3','V3','Text x1.5'),
      ('v3m','text15_v3m','V3M','Text x1.5'),('v5lite','full_v5lite','V5Lite','Full'),
      ('v5lite','text15_v5lite','V5Lite','Text x1.5'))


def publish(path,payload):
    """@luna-comment Input: owned destination/text. Output: atomic published file.
    Key Steps: write sibling temporary then replace. Key Params: UTF8.
    State/Dependencies: unified monitor output only; source files untouched.
    """
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(payload,encoding='utf-8');temporary.replace(path)


def snapshot():
    """@luna-comment Input: two atomic source status files. Output: five-arm unified state and standalone HTML.
    Key Steps: read source timestamps/arms, flag age>120s, combine unchanged loss histories, render table/charts.
    Key Params: exactly five source bindings; no invented zero values for missing data.
    State/Dependencies: read-only source JSON and accepted offline chart renderer; no process mutations.
    """
    now=datetime.now(timezone(timedelta(hours=8)));sources={};errors={}
    for key,path in SOURCES.items():
        try:sources[key]=json.loads(path.read_text())
        except (OSError,ValueError) as error:sources[key]={};errors[key]=str(error)
    state=dict(updated_at=now.isoformat(),arms={},loss_history={},sources={})
    for key,source in sources.items():
        timestamp=source.get('updated_at');age=None
        if timestamp:
            try:age=max(0,(now-datetime.fromisoformat(timestamp)).total_seconds())
            except (TypeError,ValueError):pass
        state['sources'][key]=dict(path=str(SOURCES[key]),updated_at=timestamp,age_seconds=age,stale=age is None or age>120,error=errors.get(key))
    for source_key,arm,model,loss in ARMS:
        source=sources[source_key];item=dict(source.get('arms',{}).get(arm,{}))
        item.update(model=model,loss=loss,source=source_key,source_stale=state['sources'][source_key]['stale'])
        item.setdefault('status','unavailable');state['arms'][arm]=item
        state['loss_history'][model+' | '+loss]=source.get('loss_history',{}).get(arm,[])
    page='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VOSR2u - Five training runs</title><style>body{font:15px system-ui;margin:24px;background:#f4f7fa;color:#17283b}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #ccd;padding:10px;text-align:left}button{padding:10px 18px}.stale{color:#b45309}.scroll{overflow:auto}</style></head><body><h1>VOSR2u - Five training runs</h1><p>Fresh initialization;256x256 patches;batch4;250K steps. V3M:4/8/16/32. V5Lite:local12/context4-16. All five share the same VOSR2u training bank and OLD583 validation targets.</p><button onclick="location.reload()">Refresh now</button><p>Auto refresh30s - '+state['updated_at']+'</p><div class="scroll"><table><thead><tr><th>Model</th><th>Loss</th><th>GPU / PID</th><th>Status</th><th>Step</th><th>Remaining</th><th>Estimated finish (Beijing)</th></tr></thead><tbody>'
    for arm,item in state['arms'].items():
        seconds=item.get('estimated_remaining_seconds');remaining=f'{seconds/3600:.1f}h' if seconds is not None else 'pending'
        step=item.get('iteration');step=f'{step:,} / 250,000' if step is not None else 'pending'
        status=html.escape(item['status'])+(' / source stale' if item['source_stale'] else '')
        page+=f'<tr data-arm="{arm}"><td>{item["model"]}</td><td>{item["loss"]}</td><td>{item.get("gpu","-")} / {item.get("pid","-")}</td><td class="{"stale" if item["source_stale"] else ""}">{status}</td><td>{step}</td><td>{remaining}</td><td>{html.escape(item.get("estimated_finish_at") or "pending")}</td></tr>'
    page+='</tbody></table></div><p>Text x1.5 changes weighted loss scales. Curves show optimization progress; image quality requires separate validation. ETA follows recent logged speed and may change after auxiliary ramp, validation or checkpoint saves.</p>'
    for key,source in state['sources'].items():page+=f'<p class="{"stale" if source["stale"] else ""}">Source {key}: {html.escape(source["updated_at"] or "unavailable")}{" (stale)" if source["stale"] else ""}</p>'
    page+=render_loss_charts(state['loss_history'])+'<script>setTimeout(()=>location.reload(),30000)</script></body></html>'
    return state,page


def main(once=False):
    """@luna-comment Input: optional one-shot flag. Output: independent unified observer and atomic status files.
    Key Steps: lock owned observer, render/publish snapshots every30seconds.
    Key Params: no training/old-observer signals; once supports focused validation.
    State/Dependencies: Linux fcntl, source status publishers and own output directory only.
    """
    import fcntl,os
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    publish(OUT/'observer.json',json.dumps(dict(pid=os.getpid()),indent=2))
    while True:
        state,page=snapshot();publish(OUT/'STATUS.json',json.dumps(state,indent=2,allow_nan=False));publish(OUT/'STATUS.html',page)
        if once:return
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--once',action='store_true');args=parser.parse_args();main(args.once)
