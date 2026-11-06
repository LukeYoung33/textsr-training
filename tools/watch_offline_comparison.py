"""Publish standalone live HTML and matrix health throughout training."""
from datetime import datetime, timezone, timedelta
import html
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.manage_offline_comparison import ARMS, RUN, MATRIX_KIND, write_json
from tools.offline_loss_charts import read_loss_history, render_loss_charts


def render_html(state):
    """@luna-comment
    Input: current matrix snapshot dictionary. Output: standalone escaped HTML text.
    Key Steps: render progress cards/table and full loss charts; embed source time/reload controls.
    Key Params:30s refresh;120s stale warning;250K target steps.
    State/Dependencies: no external assets or browser fetch; file and HTTP compatible.
    """
    escape = lambda value: html.escape(str(value), quote=True)
    labels = {'running': 'Training', 'starting': 'Starting', 'completed': 'Completed',
              'waiting_for_offline_bank': 'Waiting for offline data', 'failed': 'Failed', 'exited': 'Exited'}
    cards = []
    for arm, entry in state['arms'].items():
        iteration = entry.get('iteration', 0)
        percent = min(100., max(0., iteration / 250000 * 100))
        status = entry['status']
        seconds = entry.get('seconds_per_iter')
        eta = '—' if not seconds or status != 'running' else f"~{max(0, (250000-iteration)*seconds/3600):.1f} h"
        speed = f'{seconds:.3f} s' if seconds else '—'
        data_time = f"{entry['data_seconds']:.3f} s" if 'data_seconds' in entry else '—'
        variant = ('Both-side preprocessing' if arm.endswith('_preprocessed') else 'No preprocessing') if MATRIX_KIND == 'preprocess_ab' else ('Real paired' if arm.endswith('_real') else 'Synthetic')
        title = ('V5' if '_v5_' in arm else 'V3') + ' · ' + variant
        tone = 'bad' if status in ('failed', 'exited') else ('ok' if status in ('running', 'completed') else 'wait')
        cards.append(f'''<article class="card"><div class="cardhead"><h2>{title}</h2><span class="badge {tone}">{escape(labels.get(status,status))}</span></div>
<div class="iteration">{iteration:,}<span> / 250,000 steps</span></div>
<div class="track" role="progressbar" aria-label="{escape(title)}" aria-valuenow="{iteration}" aria-valuemin="0" aria-valuemax="250000"><div style="width:{percent:.2f}%"></div></div>
<div class="progresslabel">{percent:.2f}%<span>GPU {escape(entry['gpu'])} · PID {escape(entry.get('pid','—'))}</span></div>
<dl><div><dt>Step time</dt><dd>{speed}</dd></div><div><dt>Data loading</dt><dd>{data_time}</dd></div><div><dt>Remaining*</dt><dd>{eta}</dd></div></dl>
<div class="arm">{escape(arm)}</div></article>''')
    done = state.get('verified_synthetic_parents', 0)
    build_percent = min(100., done / 4875 * 100)
    running = sum(s['status'] == 'running' for s in state['arms'].values())
    completed = sum(s['status'] == 'completed' for s in state['arms'].values())
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><title>V3 / V5 · Training monitor</title>
<style>
:root{color-scheme:light;font-family:Segoe UI,Arial,sans-serif;color:#172638;background:#f3f6fa}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:48px auto;padding:0 24px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:2px;color:#527082}header{display:flex;align-items:center;justify-content:space-between;gap:24px}h1{font-size:34px;letter-spacing:-1px;margin:9px 0 12px}header p,.muted{color:#647588;font-size:14px;line-height:1.6}.controls{text-align:right;flex-shrink:0}button{border:0;border-radius:10px;padding:13px 24px;background:#1c625d;color:white;font-weight:600;font-size:14px;cursor:pointer}button:hover{background:#144b47}button:focus-visible{outline:3px solid #4fa6de;outline-offset:3px}.countdown{display:block;margin-top:10px;font-size:12px;color:#65778b}.notice{margin:22px 0;padding:14px 18px;border:1px solid #d4e3df;background:#eaf4ef;border-radius:12px;color:#275d51;font-size:13px}.notice.stale{background:#fff2db;border-color:#ebd4a1;color:#795419}.summary{display:flex;gap:12px;flex-wrap:wrap;margin:25px 0}.summary span{font-size:13px;background:white;padding:10px 14px;border:1px solid #e1e8ed;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{background:white;border:1px solid #e0e7ee;border-radius:16px;padding:25px;box-shadow:0 4px 18px #213d5410}.cardhead{display:flex;justify-content:space-between;align-items:center;gap:10px}h2{font-size:18px;margin:0}.badge{font-size:11px;padding:6px 9px;border-radius:20px;white-space:nowrap}.ok{background:#e8f5ef;color:#21714f}.bad{background:#ffe9e9;color:#ac3030}.wait{background:#fff1d7;color:#866014}.iteration{font-size:36px;font-weight:650;letter-spacing:-1px;margin:26px 0 15px;font-variant-numeric:tabular-nums}.iteration span{font-size:14px;font-weight:400;color:#718196;letter-spacing:0}.track{height:7px;border-radius:8px;background:#eaf0f4;overflow:hidden}.track>div{height:100%;background:#327f78;border-radius:8px}.progresslabel{display:flex;justify-content:space-between;font-size:12px;color:#64778b;margin-top:10px}dl{display:grid;grid-template-columns:repeat(3,1fr);margin:24px 0 20px;gap:10px}dt{color:#758395;font-size:12px}dd{margin:7px 0 0;font-size:16px;font-weight:600}.arm{border-top:1px solid #edf0f3;padding-top:15px;font-size:11px;color:#8a96a4;letter-spacing:.4px}.build{margin-top:20px;background:white;border:1px solid #e0e7ee;border-radius:14px;padding:22px 25px}.buildhead{display:flex;justify-content:space-between;gap:20px;font-size:14px;margin-bottom:15px}.buildhead span{color:#687c8e}footer{font-size:12px;color:#768698;line-height:1.8;margin:24px 0}a{color:#326a84}#refreshError{color:#a13232}@media(max-width:700px){main{margin:25px auto;padding:0 16px}header{align-items:flex-start;flex-direction:column;gap:8px}.controls{text-align:left}.grid{grid-template-columns:1fr}.card{padding:20px}h1{font-size:28px}.iteration{font-size:32px}}
</style></head><body><main>
<header><div><div class="eyebrow">REAL-ESRGAN / EXPERIMENT MONITOR</div><h1>V3 / V5 training comparison</h1><p>Real paired inputs and offline synthetic degradation · Four matched training runs</p></div>
<div class="controls"><button id="refresh" type="button" onclick="refreshPage()">Refresh now ↻</button><span class="countdown">Auto refresh in <b id="countdown">30</b>s</span><span id="refreshError" role="alert"></span></div></header>
<div class="notice" id="freshness" role="status">Snapshot updated: @@UPDATED@@ (Beijing time)</div>
<div class="summary"><span>@@RUNNING@@ training · @@COMPLETED@@ completed</span><span>250K steps / run</span><span>Batch 4</span><span>Checkpoint + validation every 25K</span></div>
<section class="grid" aria-label="Training runs">@@CARDS@@</section>
@@LOSS_CHARTS@@
<section class="build"><div class="buildhead"><strong>Offline synthetic dataset</strong><span>@@DONE@@ / 4,875 verified parents · @@BUILD_PERCENT@@%</span></div><div class="track"><div style="width:@@BUILD_PERCENT@@%"></div></div><p class="muted">Queue: @@QUEUE@@</p></section>
<footer>* Remaining time uses the latest step duration; validation and checkpoint overhead can extend it.<br>Source data updates every 30 seconds. Browser refresh reloads the latest generated snapshot.<br><a href="STATUS.md">Markdown status</a> · <a href="STATUS.json">Raw status data</a></footer>
</main><script>
const sourceUpdatedAt = @@UPDATED_JSON@@;
let remaining = 30;
// @luna-comment Input: button/timer event. Output: cache-busted page navigation.
// Key Steps: update query and replace location. Key Params: timestamp query.
// State/Dependencies: current location; works without file-origin fetch.
function refreshPage(){const target=new URL(window.location.href);target.searchParams.set('_refresh',Date.now().toString());window.location.replace(target.href);}
// @luna-comment Input: timer tick. Output: countdown and stale-source warning.
// Key Steps: measure source age and reload at zero. Key Params:30s/120s.
// State/Dependencies: page DOM and embedded ISO timestamp.
function tick(){remaining-=1;document.getElementById('countdown').textContent=Math.max(0,remaining);const age=Date.now()-Date.parse(sourceUpdatedAt);if(age>120000){const notice=document.getElementById('freshness');notice.classList.add('stale');notice.textContent='Source data is '+Math.floor(age/60000)+' minutes old. Browser refresh does not guarantee a new backend snapshot.';}if(remaining<=0)refreshPage();}
setInterval(tick,1000);
</script></body></html>'''
    replacements = {'UPDATED': escape(state['updated_at']), 'UPDATED_JSON': json.dumps(state['updated_at_iso']),
                    'RUNNING': str(running), 'COMPLETED': str(completed), 'CARDS': ''.join(cards),
                    'DONE': f'{done:,}', 'BUILD_PERCENT': f'{build_percent:.1f}',
                    'LOSS_CHARTS': render_loss_charts(state.get('loss_history', {})),
                    'QUEUE': escape(state.get('queue', {}).get('phase', 'starting'))}
    for name, value in replacements.items():
        page = page.replace('@@' + name + '@@', value)
    if MATRIX_KIND == 'preprocess_ab':
        page = page.replace('Real paired inputs and offline synthetic degradation · Four matched training runs',
                            'Real LQ / GT_VOSR pairs · Both-side preprocessing versus no preprocessing')
        page = re.sub(r'<section class="build">.*?</section>',
                      '<section class="build"><strong>Two complete offline banks</strong><p>Same real pairs, augmentation recipes, native crops, masks and sampling weights. Preprocessing is applied to both LQ and GT_VOSR, or neither. No online preprocessing or synthetic degradation.</p></section>', page, flags=re.S)
    return page


def tail(path, size=131072):
    """@luna-comment
    Input: optional log path and byte bound. Output: bounded decoded log suffix.
    Key Steps: seek from end and decode. Key Params: 128KiB default.
    State/Dependencies: own run logs; no process mutation.
    """
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-size))
        return stream.read().decode('utf-8', errors='replace')


def snapshot():
    """@luna-comment
    Input: matrix receipts/logs. Output: atomic live JSON, Markdown and HTML status.
    Key Steps: probe recorded PIDs/latest iteration; incrementally read full losses and build ranks.
    Key Params: iteration>0 means advancing; completion uses the actual versioned config name.
    State/Dependencies: read-only process probes and immutable arm configs.
    """
    now = datetime.now(timezone(timedelta(hours=8)))
    result = dict(updated_at=now.strftime('%Y-%m-%d %H:%M:%S %z'), updated_at_iso=now.isoformat(), arms={}, build={}, loss_history={})
    queue_path = RUN / 'synthetic_queue.json'
    if queue_path.exists():
        result['queue'] = json.loads(queue_path.read_text())
    done = 0
    for rank in range(8):
        text = tail(RUN / 'synthetic_build_logs' / f'worker{rank}.log', 16384)
        records = []
        for line in text.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if 'ordinal' in record:
                records.append(record)
        if records:
            result['build'][str(rank)] = records[-1]
            done += records[-1]['ordinal'] // 8 + 1
    result['verified_synthetic_parents'] = done
    for arm in ARMS:
        path = RUN / 'launch' / (arm + '.json')
        state = dict(status='waiting_for_offline_bank', gpu=ARMS.index(arm), iteration=0)
        if path.exists():
            launch = json.loads(path.read_text())
            state.update(pid=launch['pid'], log=launch['log'])
            try:
                os.kill(launch['pid'], 0)
                alive = True
            except ProcessLookupError:
                alive = False
            text = tail(Path(launch['log']))
            result['loss_history'][arm] = read_loss_history(launch['log'])
            matches = re.findall(r'iter:\s*([\d,]+),.*?time \(data\):\s*([\d.]+)\s*\(([\d.]+)\)', text)
            if matches:
                iteration, seconds, data_seconds = matches[-1]
                state.update(iteration=int(iteration.replace(',', '')), seconds_per_iter=float(seconds), data_seconds=float(data_seconds))
            state['status'] = 'running' if alive and state['iteration'] > 0 else ('starting' if alive else 'exited')
            if 'Traceback (most recent call last)' in text:
                state['status'] = 'failed'
            import yaml
            config_path = RUN / 'configs' / (arm + '.yml')
            experiment_name = yaml.safe_load(config_path.read_text())['name']
            final = RUN / 'experiments' / experiment_name / 'models/net_g_250000.pth'
            if not alive and state['status'] != 'failed' and final.exists() and 'End of training' in text:
                state.update(status='completed', iteration=250000)
        result['arms'][arm] = state
    result['all_four_advancing'] = all(s['status'] == 'running' for s in result['arms'].values())
    write_json(RUN / 'STATUS.json', result)
    lines = ['# Live offline V3/V5 training matrix', '', 'Updated: ' + result['updated_at'], '',
             '| Arm | GPU | Status | Iteration | Seconds/step | PID |', '|---|---|---|---|---|---|']
    for arm, state in result['arms'].items():
        lines.append(f"| {arm} | {state['gpu']} | {state['status']} | {state['iteration']} | {state.get('seconds_per_iter', '')} | {state.get('pid', '')} |")
    lines += ['', f'Verified synthetic parents: {done}/4875.', '', 'Queue phase: ' + result.get('queue', {}).get('phase', 'starting'),
              '', 'Each arm: 250K steps, batch4, same real validation, checkpoints/validation every25K.',
              'Synthetic arms start only after complete target/mask equality verification and full-loss smoke.',
              'input_paired_fraction denotes the stored-input execution route, not real/synthetic origin.']
    if MATRIX_KIND == 'preprocess_ab':
        lines = lines[:6 + len(ARMS)] + ['', 'All arms use real LQ -> GT_VOSR pairs. Both-side frozen preprocessing versus neither.',
                             'Each arm:250K steps, batch4, corresponding fixed validation; no online preprocessing/degradation.']
    temp = RUN / 'STATUS.md.tmp'
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(RUN / 'STATUS.md')
    temp = RUN / 'STATUS.html.tmp'
    temp.write_text(render_html(result), encoding='utf-8')
    temp.replace(RUN / 'STATUS.html')
    return result


def main():
    """@luna-comment
    Input: detached observer invocation. Output: periodically refreshed health receipts.
    Key Steps: single-instance lock; snapshot every30s until all four complete.
    Key Params: retain startup/failure receipts; never restart or terminate training.
    State/Dependencies: own run files and Linux advisory lock only.
    """
    import fcntl
    with (RUN / 'status_observer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Status observer is already running')
        write_json(RUN / 'status_observer.json', dict(pid=os.getpid(), refresh_seconds=30, mode='through_completion'))
        while True:
            state = snapshot()
            if state['all_four_advancing'] and not (RUN / 'FOUR_ARMS_ADVANCING.json').exists():
                write_json(RUN / 'FOUR_ARMS_ADVANCING.json', state)
            if state.get('queue', {}).get('phase') == 'failed' or any(a['status'] in ('failed', 'exited') for a in state['arms'].values()):
                write_json(RUN / 'ATTENTION_REQUIRED.json', state)
            if all(a['status'] == 'completed' for a in state['arms'].values()):
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
