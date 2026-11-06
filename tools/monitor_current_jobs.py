"""Read-only unified Remote58 experiment monitor."""
from pathlib import Path
import json,time,subprocess,re,html,fcntl
from datetime import datetime,timedelta,timezone
from tools.offline_loss_charts import read_loss_history,render_loss_charts
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'current_jobs_monitor_20261030'
GROUPS=[('mojito280_guard_finetune_20261030',{'guard':'V3M guard | LR 1e-5','contrast':'V3M guard + 对比度对齐 | LR 1e-5','contrast_nocp':'V3M guard + 对比度对齐 + noCP | LR 1e-5'}),('mojito280_v5_guard_ft_20261030',{'v5':'V5 guard + 对比度对齐 + noCP | LR 1e-5'}),('mojito_v3m_lr1e3_pair_20261030',{'guard':'V3M guard + 对比度对齐 + noCP | LR 1e-3','l1_only':'V3M L1-only + 对比度对齐 + noCP | LR 1e-3'})]


def read(path):
    """@luna-comment Input:optional atomic JSON. Output:dict or empty. Key Steps:read safely across replacement. Key Params:read-only. State/Dependencies:NAS."""
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def update():
    """@luna-comment Input:owned receipts/logs and process inventory. Output:unified cards,ETA,loss and GT status. Key Steps:verify process state;derive rolling ETA;separate paused/stale/completed. Key Params:150K,300s stale,30s refresh. State/Dependencies:no job control."""
    tz=timezone(timedelta(hours=8));now=datetime.now(tz);cards=[];histories={};rows=[];known=set()
    for folder,arms in GROUPS:
        root=DATA/folder
        for arm,label in arms.items():
            launch=read(root/'launch'/(arm+'.json'));pid=launch.get('pid');proc=Path('/proc')/str(pid);known.add(str(pid))
            status='未启动';alive=False
            if proc.exists():
                state=(proc/'stat').read_text().split(') ',1)[1].split()[0];alive=state not in ['T','t','Z'];status='运行中' if alive else '已暂停' if state in ['T','t'] else '进程退出'
            elif pid:status='进程已退出'
            if (root/'USER_STOPPED.json').exists():status='已按要求停止';alive=False
            audit=read(root/'audits'/arm/'STATUS.json');step=audit.get('iter',0)
            if (root/'done'/(arm+'.json')).exists():status='已完成';step=150000
            log=root/'logs'/(arm+'.log');points=[]
            if log.exists():
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000000));text=f.read().decode('utf-8',errors='replace')
                for stamp,n in re.findall(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO: .*?\biter:\s*([\d,]+),',text):points.append((datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=tz),int(n.replace(',',''))))
                histories[label]=read_loss_history(log)
            eta='—';finish=None;points=points[-200:]
            if alive and points and (now-points[-1][0]).total_seconds()>300:status='进度超过5分钟未更新'
            elif alive and len(points)>1 and points[-1][1]>points[0][1]:
                seconds=(150000-points[-1][1])*(points[-1][0]-points[0][0]).total_seconds()/(points[-1][1]-points[0][1]);finish=points[-1][0]+timedelta(seconds=seconds);mins=max(0,round((finish-now).total_seconds()/60));eta=f'{mins//60}小时{mins%60}分钟 · {finish:%m-%d %H:%M}'
            row=dict(label=label,status=status,iter=step,pid=pid,gpu=launch.get('gpu'),eta=eta);rows.append(row)
            cards.append(f'<article><h3>{label}</h3><b>{status}</b> · GPU {launch.get("gpu","—")}<p>{step:,} / 150,000</p><progress max="150000" value="{step}"></progress><p>剩余 / 预计完成：{eta}</p></article>')
    gt=[]
    for folder,label in [('lenovo_validcrop_full_gt_s22_20261029','Lenovo 全量 GT'),('mojito280_reused_v3_vosr_s22_20261029','Mojito280 GT')]:
        root=DATA/folder;s=read(root/'STATUS.json');complete=read(root/'COMPLETE.json');phase=s.get('phase','未知');count=s.get('verified_gt',s.get('completed',s.get('verified',0)));total=s.get('total',280 if 'mojito' in folder else 3236)
        if complete.get('passed'):phase='已完成';count=complete.get('count',total)
        elif not Path('/proc/'+str(s.get('pid'))).exists():phase='未运行 / '+phase
        gt.append(f'<article><h3>{label}</h3><b>{html.escape(phase)}</b><p>{count} / {total}</p></article>')
    processes=subprocess.check_output(['ps','-eo','pid,ppid,stat,args'],text=True);extra={}
    for line in processes.splitlines():
        parts=line.split(None,3)
        if len(parts)!=4:continue
        pid,ppid,state,cmd=parts
        if ' -opt ' not in cmd or str(DATA) not in cmd:continue
        cfg=cmd.split(' -opt ',1)[1].split()[0]
        if any(folder in cfg for folder,_ in GROUPS):continue
        if any(folder in cfg for folder in ['latest_vosr_four_500k_20261102','latest_vosr_mix50_500k_20261102','v3m_loss_ablation_500k_20261103','v3m_aux_factorial_500k_20261103','v3m_original_dilate2_l1_500k_20261103']):continue
        if cfg not in extra or int(pid)<extra[cfg][0]:extra[cfg]=(int(pid),state)
    extras=''.join('<li>'+html.escape(Path(k).stem)+'：'+('已暂停' if 'T' in v[1] else '进程存在（详见原任务监控）')+'</li>' for k,v in extra.items())
    latest=read(DATA/'latest_vosr_four_500k_20261102/STATUS.json')
    mixed=read(DATA/'latest_vosr_mix50_500k_20261102/STATUS.json')
    latest.setdefault('runs',{}).update(mixed.get('runs',{}))
    ablation=read(DATA/'v3m_loss_ablation_500k_20261103/STATUS.json')
    ablation_bindings=read(DATA/'v3m_loss_ablation_500k_20261103/CONFIG_BINDINGS.json').get('runs',{})
    latest['runs']={**ablation.get('runs',{}),**latest['runs']}
    factorial=read(DATA/'v3m_aux_factorial_500k_20261103/STATUS.json')
    factorial_bindings=read(DATA/'v3m_aux_factorial_500k_20261103/CONFIG_BINDINGS.json').get('runs',{})
    latest['runs']={**factorial.get('runs',{}),**latest['runs']}
    dilate_root=DATA/'v3m_original_dilate2_l1_500k_20261103'
    dilated=read(dilate_root/'STATUS.json');dilated_bindings=read(dilate_root/'CONFIG_BINDINGS.json').get('runs',{})
    for state in dilated.get('runs',{}).values():
        if state.get('gpu') is None:
            preparation=read(dilate_root/'DATA_STATUS.json');bank=read(dilate_root/'train_bank/progress.json')
            state['preparing']=True;state['phase']=preparation.get('phase','准备独立训练数据')
            if state['phase']=='complete':state['phase']=read(dilate_root/'CONTROLLER_STATUS.json').get('phase','启动检查')
            if bank.get('completed',0)>0 and bank['completed']<bank.get('total',0):state['finish']=(now+timedelta(seconds=bank['seconds']/bank['completed']*(bank['total']-bank['completed']))).strftime('%m-%d %H:%M')+'（仅patch构建）'
    latest['runs']={**dilated.get('runs',{}),**latest['runs']}
    four_root=DATA/'gt1_r1_mixture_four_500k_20261104'
    four=read(four_root/'STATUS.json');four_bindings=read(four_root/'CONFIG_BINDINGS.json').get('runs',{})
    latest['runs']={**four.get('runs',{}),**latest['runs']}
    latest_cards=[]
    for arm,state in latest.get('runs',{}).items():
        label=arm.replace('v3m','V3M').replace('v5','V5').replace('_',' · ')
        if arm.endswith('_mix50'):label=label.replace('mix50','Full无guard · 50% GT轻度退化')
        if arm in ablation_bindings:label=ablation_bindings[arm]['label']
        if arm in factorial_bindings:label=factorial_bindings[arm]['label']
        if arm in dilated_bindings:label=dilated_bindings[arm]['label']
        if arm in four_bindings:label=four_bindings[arm]['label']
        time_label='预计启动' if state.get('queued',False) else '预计数据完成' if state.get('preparing',False) else '预计完成'
        latest_cards.append(f'<article><h3>{label} · 从头训练500K</h3><b>{html.escape(state["phase"])}</b><p>{state["iter"]:,} / 500,000 · LR {state["lr"]}</p><progress max="500000" value="{state["iter"]}"></progress><p>{time_label}：{html.escape(state["finish"])}</p></article>')
        folder='latest_vosr_mix50_500k_20261102' if arm.endswith('_mix50') else 'latest_vosr_four_500k_20261102'
        if arm in ablation_bindings:folder='v3m_loss_ablation_500k_20261103'
        if arm in factorial_bindings:folder='v3m_aux_factorial_500k_20261103'
        if arm in dilated_bindings:folder='v3m_original_dilate2_l1_500k_20261103'
        if arm in four_bindings:folder='gt1_r1_mixture_four_500k_20261104'
        log=DATA/folder/'logs'/(arm+'.log')
        if log.exists():histories[label+' | scratch500K']=read_loss_history(log)
    if latest_cards:
        cards=latest_cards+cards
        gt.append('<article><h3>新四组数据准备</h3><pre>'+html.escape(json.dumps(latest.get('preparation',{}),ensure_ascii=False,indent=2))+'</pre><a href="../latest_vosr_four_500k_20261102/index.html">打开四组详细监控</a></article>')
    if four:
        gt.insert(0,'<article><h3>GT+1／VOSR · R1四组500K</h3><p>四组均GT退化50%，对比度对齐关闭，同7385来源。</p><pre>'+html.escape(json.dumps(four.get('datasets',[]),ensure_ascii=False,indent=2))+'</pre><a href="../gt1_r1_mixture_four_500k_20261104/index.html">打开四组数据／训练／loss详细监控</a></article>')
    cache=read(DATA/'training_local_cache_20261103/STATUS.json')
    if cache:
        cache_text='完成' if cache.get('phase')=='complete' else '复制并校验中' if cache.get('phase')=='copying' else cache.get('phase','未知')
        gt.insert(0,'<article><h3>训练数据 → Remote58 本地盘</h3><b>'+html.escape(cache_text)+'</b><p>'+str(cache.get('completed_gib',0))+' / '+str(cache.get('total_gib',0))+' GiB · '+str(cache.get('percent',0))+'%</p><progress max="100" value="'+str(cache.get('percent',0))+'"></progress><p>当前：'+html.escape(str(cache.get('active_bank','')))+' · '+str(cache.get('mib_per_second',0))+' MiB/s</p><p>已切换：'+html.escape(', '.join(x['bank'] for x in cache.get('switched',[])) or '等待首份校验完成')+'</p><p>预计完成：'+html.escape(str(cache.get('estimated_finish','采样中；小文件阶段暂不估时')))+'</p><a href="../training_local_cache_20261103/index.html">打开复制详细进度</a></article>')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True)
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>TextSR 任务总览</title><style>body{font:16px/1.7 system-ui;background:#eff3f8;color:#18314b;margin:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}article{background:white;padding:20px;border-radius:12px}h3{margin-top:0}progress{width:100%}pre{white-space:pre-wrap}</style><h1>TextSR · 当前任务总览</h1><p>更新：'+now.strftime('%Y-%m-%d %H:%M:%S')+' 北京时间 · 每30秒刷新</p><p>预计时间根据近期训练日志计算；验证和GPU负载会影响速度。不同loss组合的总loss不直接代表效果优劣。</p><div class="grid">'+''.join(cards)+'</div>'+render_loss_charts(histories)+'<h2>数据处理</h2><div class="grid">'+''.join(gt)+'</div><h2>GPU状态</h2><p>编号 / 已用显存 / 总显存 / 利用率</p><pre>'+html.escape(gpu)+'</pre><details><summary>其他历史训练进程（暂停任务不计作运行）</summary><ul>'+extras+'</ul></details></html>'
    tmp=OUT/'index.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'index.html')
    tmp=OUT/'STATUS.tmp';tmp.write_text(json.dumps(dict(updated=now.isoformat(),runs=rows),ensure_ascii=False));tmp.replace(OUT/'STATUS.json')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:update();time.sleep(30)
