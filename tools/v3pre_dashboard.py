"""Readable local presentation of the existing server snapshot and loss charts."""
from datetime import datetime, timezone, timedelta
import html
import json
import re

ARMS = {'full': ('Full', '原始完整 loss'), 'text15': ('Text ×1.5', '文字相关 loss 权重提升 50%'),
        'adapt_l1': ('adapt-L1', '文字行面积自适应 L1 · cap=4')}
PHASES = {'engineering_pilot': '训练前检查', 'waiting_preprocessing': '完整图预处理中',
          'building_banks_and_line_regions': '训练数据准备中', 'ready_waiting_timer': '已就绪，等待定时启动',
          'training': '训练进行中', 'all_six_trainings_complete': '全部训练完成', 'controller_failed': '启动流程已暂停'}


def escape(value):
    """@luna-comment Input:display value. Output:escaped HTML. Key Steps:string escape.
    Key Params:quote-safe. State/Dependencies:no source mutation.
    """
    return html.escape(str(value), quote=True)


def progress(done, total):
    """@luna-comment Input:observed counts/positive total. Output:bounded accessible progress bar.
    Key Steps:calculate display percentage. Key Params:missing counts stay unknown.
    State/Dependencies:presentation only; never infers a completion marker.
    """
    percent = max(0., min(100., 100*done/total)) if total else 0
    return f'<div class="bar" role="progressbar" aria-valuenow="{percent:.1f}" aria-valuemin="0" aria-valuemax="100"><span style="width:{percent:.2f}%"></span></div>'


def bank_card(stage, label, state):
    """@luna-comment Input:bank snapshot. Output:compact parent and sidecar readiness card.
    Key Steps:use completion markers over stale progress; retain actual counts.
    Key Params:readiness needs bank and sidecar. State/Dependencies:no estimates fabricated.
    """
    value = state.get('banks', {}).get(stage, {})
    complete = value.get('COMPLETE.json', {})
    regions = value.get('LINE_REGIONS_COMPLETE.json', {})
    current = value.get('progress.json', {})
    ready = bool(complete.get('complete') and regions.get('passed'))
    if complete.get('complete'):
        done = total = sum(s.get('parents', 0) for s in complete.get('splits', {}).values())
        samples = sum(s.get('samples', 0) for s in complete.get('splits', {}).values())
    else:
        done, total, samples = current.get('completed', 0), current.get('total', 0), current.get('samples', 0)
    badge = '数据就绪' if ready else '标注打包中' if complete.get('complete') else '生成 patch 中' if total else '等待处理'
    return f'<article class="card data-card"><div class="card-head"><h3>{escape(label)}</h3><span class="badge {"good" if ready else "pending"}">{badge}</span></div><div class="count">{done:,}<small> / {total:,} 对</small></div>{progress(done,total)}<div class="meta">已生成 {samples:,} 个 patch</div><div class="checks"><span>{"✓" if complete.get("complete") else "○"} 图像数据</span><span>{"✓" if regions.get("passed") else "○"} 文字行标注</span></div></article>'


def model_card(arm, state):
    """@luna-comment Input:arm/live model and child states. Output:two-stage training card.
    Key Steps:show actual iteration/ETA/PID and explicit waiting/running/completed states.
    Key Params:250K then150K; no zero-iteration claim before logs exist.
    State/Dependencies:source snapshot only.
    """
    name, note = ARMS[arm]
    rows = []
    children = state.get('training', {}).get('children', {})
    for stage, title, total in [('pretrain', 'Pretrain', 250000), ('finetune', 'Finetune', 150000)]:
        key = stage+'_'+arm
        model = state.get('models', {}).get(key, {})
        child = children.get(key, {})
        finished = child.get('exit_code') == 0 and 'exit_code' in child
        step = total if finished else model.get('iteration')
        status = '已完成' if finished else '训练中' if model else '启动中' if child.get('pid') else '等待 Pretrain 完成' if stage == 'finetune' else '尚未启动'
        count = f'{step:,}' if step is not None else '—'
        eta = model.get('eta')
        timing = '已完成' if finished else '剩余 '+str(eta) if eta and eta != 'warming up' else '产生训练日志后显示 ETA'
        rows.append(f'<div class="stage-row"><div class="card-head"><strong>{title}</strong><span class="stage-status">{status}</span></div><div class="iterations">{count}<small> / {total:,} iter</small></div>{progress(step or 0,total)}<div class="meta">{escape(timing)}</div></div>')
    return f'<article class="card model-card"><div class="model-title"><span class="model-icon">M</span><div><h3>V3M · {name}</h3><div class="meta">{note}</div></div></div>{rows[0]}<div class="connector">↓ 各自加载 Pretrain 最终权重</div>{rows[1]}</article>'


def render_dashboard(source):
    """@luna-comment Input:trusted server HTML snapshot. Output:readable standalone dashboard.
    Key Steps:parse escaped JSON, compact large receipts, render status/cards and
    retain original loss section. Key Params:Chinese presentation30s refresh/source freshness.
    State/Dependencies:local mirror only; never changes timer/training state.
    """
    match = re.search(r'<pre>(.*?)</pre>', source, re.S)
    if match is None:
        raise ValueError('Missing source snapshot')
    state = json.loads(html.unescape(match.group(1)))
    train, prep = state.get('training', {}), state.get('preparation', {})
    phase = train.get('phase', 'unknown')
    failed = phase.endswith('failed')
    target = train.get('schedule', {}).get('target')
    target_text = datetime.fromisoformat(target).astimezone(timezone(timedelta(hours=8))).strftime('%m月%d日 %H:%M') if target else '尚未设定'
    updated = train.get('updated_at') or prep.get('updated_at')
    timestamp = datetime.fromisoformat(updated).timestamp() * 1000 if updated else 0
    label = PHASES.get(phase, phase)
    done, total = prep.get('completed_pairs', 0), prep.get('total_pairs', 0)
    prep_done = prep.get('phase') == 'preprocessing_complete'
    preprocessing = f'<article class="card data-card"><div class="card-head"><h3>GT / LQ · V3 预处理</h3><span class="badge {"good" if prep_done else "pending"}">{"已完成" if prep_done else "处理中"}</span></div><div class="count">{done:,}<small> / {total:,} 对</small></div>{progress(done,total)}<div class="meta">共 {prep.get("total_images",0):,} 张图 · {prep.get("workers",0)} 进程</div><div class="checks"><span>完整图处理 → 再切 patch</span></div></article>'
    error = f'<div class="notice danger"><strong>训练启动检查未通过</strong><p>数据进度与训练启动状态分开显示。当前尚未确认能按时启动，请查看下方异常详情。</p><details><summary>查看异常详情</summary><pre>{escape(train.get("error", "服务器报告失败，等待进一步信息"))}</pre></details></div>' if failed else '<div class="notice">到达计划时间后，仅在数据完整、训练检查通过且 GPU 可用时启动。每组 Pretrain 完成后自动接续自己的 Finetune。</div>'
    sections = source[source.find('<section class="loss-section"'):] if '<section class="loss-section"' in source else ''
    compact = {k:v for k,v in state.items() if k != 'banks'}
    compact['banks'] = {stage:{key:({k:v for k,v in value.items() if k != 'records'} if isinstance(value,dict) else value) for key,value in values.items()} for stage,values in state.get('banks',{}).items()}
    css = '''*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#1a2b42;font-family:system-ui,"Microsoft YaHei",sans-serif;font-size:14px}main{max-width:1280px;margin:auto;padding:34px 28px 48px}h1{font-size:30px;letter-spacing:-1px;margin:4px 0 8px}h2{font-size:19px;margin:30px 0 14px}h3{font-size:16px;margin:0}.eyebrow{font-size:11px;letter-spacing:2px;color:#65809a;font-weight:700}.subtitle,.meta{color:#697c90;font-size:12px;line-height:1.7}.top{display:flex;justify-content:space-between;align-items:center;gap:16px}.refresh{border:1px solid #d7e1eb;background:white;padding:10px 16px;border-radius:9px;color:#315576;cursor:pointer}.hero{margin-top:24px;padding:24px 28px;background:#142b45;color:white;border-radius:16px;display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:26px}.hero small{display:block;color:#aebfd1;margin-bottom:8px;font-size:12px}.hero strong{font-size:23px;line-height:1.4}.hero p{margin:8px 0 0;color:#b8cadb;font-size:12px}.hero .status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#51d7b5;margin-right:9px}.hero.failed .status-dot{background:#ff947f}.notice{margin:14px 0 24px;padding:14px 18px;border:1px solid #dbe6f1;background:#edf3f9;border-radius:10px;color:#526b84;line-height:1.7}.danger{border-color:#f3c4bb;background:#fff1ed;color:#9c422f}.notice p{margin:5px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.card{background:white;border:1px solid #dfe7ef;border-radius:14px;padding:20px;min-width:0}.card-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.badge{font-size:11px;padding:4px 8px;border-radius:6px;white-space:nowrap}.good{background:#e4f5ed;color:#217858}.pending{background:#edf2fa;color:#557492}.count{font-size:30px;font-weight:650;margin:22px 0 12px;font-variant-numeric:tabular-nums}.count small,.iterations small{font-size:12px;color:#8a99a8;font-weight:400}.bar{height:6px;background:#edf1f5;overflow:hidden;border-radius:6px;margin:10px 0}.bar span{display:block;height:100%;background:#388eaf;border-radius:6px}.checks{display:flex;gap:16px;margin-top:16px;color:#6a8192;font-size:11px}.model-title{display:flex;gap:12px;align-items:center;margin-bottom:22px}.model-icon{display:grid;place-items:center;background:#e8f1fa;color:#34719d;border-radius:10px;width:38px;height:38px;font-weight:700}.stage-row{border-top:1px solid #edf1f5;padding-top:15px}.stage-status{font-size:11px;color:#74879b}.iterations{font-size:21px;margin-top:12px;font-variant-numeric:tabular-nums}.connector{color:#91a1b1;font-size:10px;text-align:center;padding:14px 0}.freshness{color:#72869a;font-size:11px;margin-top:12px}.details{margin-top:26px;border:1px solid #dfe7ef;background:white;border-radius:12px;padding:16px}.details summary,summary{cursor:pointer}pre{white-space:pre-wrap;word-break:break-word;font-size:11px;max-height:440px;overflow:auto;color:#5c7085}.loss-heading h2{font-size:19px}.loss-section{margin-top:30px}.footer{font-size:11px;color:#8797a7;margin-top:20px}@media(max-width:850px){.hero{grid-template-columns:1fr 1fr}.hero>div:first-child{grid-column:1/-1}.grid{grid-template-columns:1fr}.top{align-items:flex-start}main{padding:22px 16px}.hero strong{font-size:20px}}'''
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>V3M 训练监控</title><style>'+css+'</style></head><body><main><header class="top"><div><div class="eyebrow">REMOTE58 / TRAINING</div><h1>V3M 训练监控</h1><div class="subtitle">V3 预处理 · 三组 loss 对照 · Pretrain → Finetune</div></div><button class="refresh" onclick="location.reload()">刷新状态 ↻</button></header><section class="hero '+('failed' if failed else '')+'"><div><small>当前阶段</small><strong><span class="status-dot"></span>'+escape(label)+'</strong><p>训练前的数据准备与检查均为独立阶段</p></div><div><small>Pretrain 计划启动 · 北京时间</small><strong>'+escape(target_text)+'</strong><p id="countdown">'+('检查未通过，启动受阻' if failed else '由服务器定时器控制')+'</p></div><div><small>每组训练安排</small><strong>250K → 150K</strong><p>共享随机初始化 · 各自接续微调</p></div></section><div class="freshness" id="freshness">源数据更新：'+escape(updated or '暂无')+' · 页面每 30 秒刷新</div>'+error+'<h2>数据准备</h2><section class="grid">'+preprocessing+bank_card('pretrain','Pretrain 数据',state)+bank_card('finetune','Finetune 数据',state)+'</section><h2>三组训练进度</h2><section class="grid">'+''.join(model_card(arm,state) for arm in ARMS)+'</section>'+sections+'<details class="details"><summary>运行详情与校验信息</summary><pre>'+escape(json.dumps(compact,ensure_ascii=False,indent=2))+'</pre></details><div class="footer">Pretrain 验证集沿用 583 张旧 GT_VOSR；Finetune 使用 Mojito 380 / 67 划分。Loss 曲线用于观察训练，不代表视觉质量排名。</div></main><script>const updatedAt='+json.dumps(timestamp)+';const scheduledAt='+json.dumps(target)+';const paused='+json.dumps(failed)+';/* @luna-comment Input:source and timer timestamps. Output:freshness/countdown text. Key Steps:check source age and remaining time. Key Params:120s stale threshold. State/Dependencies:local browser clock only. */function tick(){const age=(Date.now()-updatedAt)/1000;if(age>120){const e=document.getElementById("freshness");e.textContent="源数据已超过 "+Math.floor(age/60)+" 分钟未更新，请检查连接或服务器状态";e.style.color="#b35d39";}if(scheduledAt&&!paused){const seconds=Math.ceil((new Date(scheduledAt)-Date.now())/1000);document.getElementById("countdown").textContent=seconds>0?"距计划启动 "+Math.floor(seconds/3600)+" 小时 "+Math.floor(seconds%3600/60)+" 分钟":"已到计划时间，以实际训练状态为准";}}tick();setInterval(tick,1000);</script></body></html>'
