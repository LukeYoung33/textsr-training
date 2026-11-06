"""Measured data ETA, buffered durable timer and fifteen runs across four GPUs."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from concurrent.futures import ThreadPoolExecutor
import copy,html,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
from tools.prepare_full_s22_training import OUT,PARENT
TZ=timezone(timedelta(hours=8))
REPO='/home/guang.yang/workspace/Real-ESRGAN'
STRATEGIES=['paired','gt_degrade30','identity','guard','gt_degrade100']
ARMS=['full','text15','adapt_l1']
LABELS=dict(paired='真实配对',gt_degrade30='30%GT退化混合',identity='GT一致性监督',guard='可信边缘抵消保护',gt_degrade100='100%GT退化',full='Full',text15='Text×1.5',adapt_l1='Adaptive-L1 cap4')


def ready(root):
    """@luna-comment Input:owned experiment root. Output:strict full-data readiness boolean. Key Steps:require signed marker and bank/sidecar/target hashes/counts. Key Params:4873train,583val. State/Dependencies:read-only receipts;never time alone."""
    root=Path(root)
    if not (root/'DATA_READY.json').exists():return False
    receipt=json.loads((root/'DATA_READY.json').read_text())
    if not receipt.get('passed') or receipt.get('train_parents')!=4873 or receipt.get('val_parents')!=583:return False
    if not all((root/p).is_file() and digest(root/p)==receipt[key] for p,key in [('bank256/COMPLETE.json','bank_receipt_sha256'),('bank256/LINE_REGIONS_COMPLETE.json','line_receipt_sha256'),('TARGETS_COMPLETE.json','targets_receipt_sha256')]):return False
    bank=json.loads((root/'bank256/COMPLETE.json').read_text());line=json.loads((root/'bank256/LINE_REGIONS_COMPLETE.json').read_text());targets=json.loads((root/'TARGETS_COMPLETE.json').read_text())
    return bool(bank.get('complete') and bank.get('splits',{}).get('train',{}).get('parents')==4873 and bank.get('splits',{}).get('val',{}).get('parents')==583 and line.get('passed') and line.get('count')==5456 and line.get('source_build_id')==bank.get('build_id') and targets.get('passed') and targets.get('count')==4873)


def prepare_configs():
    """@luna-comment Input:accepted corresponding recipes. Output:15strict noSDF configs/shared fresh initialization. Key Steps:seed freshV3M;clone Full/Text/Adaptive;replace only run/data/SDF/strategy bindings. Key Params:250K,batch4,seed20260828. State/Dependencies:CPU bootstrap,owned configs;no prior weights/optimizer."""
    import torch,yaml
    from tools.train_v3pre_area import bootstrap
    bootstrap(REPO)
    from basicsr.archs import build_network
    source=PARENT.parent/'training/configs';initial=OUT/'initial/v3m_fresh_shared.pth';initial.parent.mkdir(exist_ok=True)
    reference=yaml.safe_load((source/'full.yml').read_text());torch.manual_seed(reference['manual_seed'])
    if not initial.exists():torch.save(dict(params=build_network(copy.deepcopy(reference['network_g'])).state_dict(),initialization='fresh_seeded_20260828'),initial)
    records={}
    for strategy_index,strategy in enumerate(STRATEGIES):
        gpu=strategy_index%4
        for ai,arm in enumerate(ARMS):
            cfg=yaml.safe_load((source/(arm+'.yml')).read_text());name=strategy+'_'+arm
            cfg['name']='s22_'+name+'_20261028';cfg['model_type']='S22Model'
            cfg['train'].update(model_type='S22Model',sdf_boundary_weight=0.,sdf_boundary_opt=None)
            cfg['path'].update(pretrain_network_g=str(initial),strict_load_g=True,resume_state=None,experiments_root=str(OUT/'training/experiments'))
            cfg['s22_strategy']=dict(name=strategy,gradient_ratio=.5,calibration_active_batches=8,audit_root=str(OUT/'training/audits'/name))
            cfg['v3pre_target_domain']='Train:V3LQ->fullVOSR->S2.2;val:olderGT_VOSR583 V3 cache'
            cfg['dist_params']['port']=34600+strategy_index*10+ai
            for split in ['train','val']:
                cfg['datasets'][split].update(type='S22Dataset',root=str(OUT/'bank256'),manifest=str(OUT/'bank256'/('train_balanced.json' if split=='train' else 'val.json')))
            assert cfg['train']['total_iter']==250000 and cfg['datasets']['train']['batch_size_per_gpu']==4
            assert cfg['network_g']==reference['network_g'] and cfg['paired_probability']==1
            path=OUT/'training/configs'/(name+'.yml');path.parent.mkdir(parents=True,exist_ok=True)
            content=yaml.safe_dump(cfg,sort_keys=False)
            if path.exists():assert path.read_text()==content
            else:path.write_text(content)
            records[name]=dict(strategy=strategy,arm=arm,gpu=gpu,config=str(path),sha256=digest(path),name=cfg['name'],iterations=250000)
    save(OUT/'CONFIG_BINDINGS.json',dict(shared_initial=str(initial),shared_initial_sha256=digest(initial),seed=reference['manual_seed'],no_sdf=True,runs=records))
    return records


def pilot_and_smoke(records):
    """@luna-comment Input:full15config bindings and first8new targets. Output:engineering bank and15actualCUDA smoke receipts. Key Steps:disjoint8train/4oldval pilot;accepted builder/sidecars;override only smoke data/audit paths. Key Params:GPU3 with free-memory gate,no saved smoke weights. State/Dependencies:full training configs never point to pilot."""
    import yaml
    receipts=sorted((OUT/'targets').glob('*/*/receipt.json'))
    assert len(receipts)>=8
    view=OUT/'pilot/source_view';bank=OUT/'pilot/bank256';(view/'manifests').mkdir(parents=True,exist_ok=True)
    rows=[json.loads(p.read_text())['row'] for p in receipts[:8]]
    val=[json.loads(line) for line in (PARENT/'manifests/val.jsonl').read_text().splitlines() if line.strip()][:4]
    assert not ({r['source_identity'] for r in rows}&{r['source_identity'] for r in val})
    for split,items in [('train',rows),('val',val)]:
        (view/'manifests'/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in items))
    if not (bank/'COMPLETE.json').exists():
        with (OUT/'pilot/bank.log').open('ab') as stream:subprocess.run([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(view),'--output',str(bank),'--workers','4','--preprocess-mode','none'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not (bank/'LINE_REGIONS_COMPLETE.json').exists():
        from tools.run_v3pre_training import sidecars
        sidecars(bank,view,4)
    for name,record in records.items():
        result=OUT/'smoke'/(name+'.json')
        if result.exists():assert json.loads(result.read_text())['passed'];continue
        cfg=yaml.safe_load(Path(record['config']).read_text())
        for split in ['train','val']:cfg['datasets'][split].update(root=str(bank),manifest=str(bank/('train_balanced.json' if split=='train' else 'val.json')))
        cfg['path']['experiments_root']=str(OUT/'smoke/experiments');cfg['s22_strategy']['audit_root']=str(OUT/'smoke/audits'/name)
        path=OUT/'smoke/configs'/(name+'.yml');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        while int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6000:time.sleep(15)
        with (OUT/'smoke'/(name+'.log')).open('ab') as stream:
            subprocess.run([sys.executable,'-u',str(ROOT/'tools/train_full_s22_strategy.py'),'--smoke-config',str(path),'--smoke-output',str(result)],stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    save(OUT/'SMOKE_COMPLETE.json',dict(passed=True,count=len(records),completed_epoch=time.time()))


def estimate_timer(data):
    """@luna-comment Input:measured data progress>=96completed and elapsed>=300s. Output:absolute launch plan with exactly30min user-requested margin. Key Steps:max remaining branch;add measured historical bank/sidecar;add1800s. Key Params:no extra rounding margin. State/Dependencies:historical timings,UTC+08 display."""
    assert data['teacher_done']>=96 and data['gt_done']>=64 and data['elapsed_seconds']>=300
    old=PARENT.parent/'bank256'
    bank_seconds=json.loads((old/'COMPLETE.json').read_text())['seconds'];line_seconds=json.loads((old/'LINE_REGIONS_COMPLETE.json').read_text())['seconds']
    remaining=max((4873-data['teacher_done'])/(data['teacher_done']/data['elapsed_seconds']),(4873-data['gt_done'])/(data['gt_done']/data['elapsed_seconds']))
    budget=bank_seconds+line_seconds;expected=time.time()+remaining+budget;scheduled=expected+1800
    return dict(state='armed',created_epoch=time.time(),measured_teacher_done=data['teacher_done'],measured_gt_done=data['gt_done'],measured_elapsed_seconds=data['elapsed_seconds'],teacher_per_hour=data['teacher_per_hour'],gt_per_hour=data['gt_per_hour'],historical_bank_seconds=bank_seconds,historical_sidecar_seconds=line_seconds,bank_budget_seconds=budget,estimated_data_ready_epoch=expected,estimated_data_ready_cst=datetime.fromtimestamp(expected,TZ).isoformat(),buffer_seconds=scheduled-expected,scheduled_epoch=scheduled,scheduled_cst=datetime.fromtimestamp(scheduled,TZ).isoformat(),readiness_required=True,smoke_required=True,late_policy='Wait for complete data and all checks;never launch partial bank')


def publish(phase):
    """@luna-comment Input:controller phase. Output:readable Chinese combined data/timer/15loss dashboard. Key Steps:read actual JSON/log tails;embed distinct strategy/loss curves. Key Params:30s refresh;CST. State/Dependencies:observer only,atomic owned HTML."""
    from tools.offline_loss_charts import read_loss_history,render_loss_charts
    data=json.loads((OUT/'DATA_STATUS.json').read_text()) if (OUT/'DATA_STATUS.json').exists() else {}
    timer=json.loads((OUT/'TIMER.json').read_text()) if (OUT/'TIMER.json').exists() else {}
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>S2.2全量数据与15组无SDF训练</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei";margin:25px;background:#f3f6fa;color:#172334}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}section{padding:18px;background:white;border-radius:12px;margin-bottom:16px}progress{width:100%;height:22px}.muted{color:#58687a}h2{font-size:20px}a{color:#17659a}</style><h1>S2.2新GT｜V3M 15组无SDF训练</h1><p>'+html.escape(phase)+' · '+datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S UTC+08')+'</p>'
    page+='<section><h2>全量数据处理</h2><p>4873张训练图：V3缓存LQ → 完整VOSR → S曲线斜率2.2；新GT不再重复预处理。</p><progress value="'+str(data.get('gt_done',0))+'" max="4873"></progress><p>教师 '+str(data.get('teacher_done',0))+'/4873 · 新GT '+str(data.get('gt_done',0))+'/4873 · 阶段：'+html.escape(data.get('phase','等待'))+'</p><p>实测教师 '+str(round(data.get('teacher_per_hour',0)))+'张/小时 · 新GT '+str(round(data.get('gt_per_hour',0)))+'张/小时</p><p class="muted">验证集583张保留原V3缓存与旧GT_VOSR，目标域与本轮新GT不同。</p></section>'
    page+='<section><h2>定时启动</h2><p>预计数据就绪：'+html.escape(timer.get('estimated_data_ready_cst','吞吐测量中'))+'</p><p>计划启动：<strong>'+html.escape(timer.get('scheduled_cst','等待稳定吞吐'))+'</strong> · 额外余量 '+str(round(timer.get('buffer_seconds',0)/3600,2))+'小时</p><p>到时仍需全量数据、训练bank、几何sidecar及15项CUDA检查全部就绪；若数据延迟，继续等待。</p></section><div class="grid">'
    histories={};states={}
    for strategy in STRATEGIES:
        for arm in ARMS:
            name=strategy+'_'+arm;label=LABELS[strategy]+' | '+LABELS[arm];history=read_loss_history(OUT/'training/logs'/(name+'.log'))
            if history:histories[label]=history
            launch=OUT/'training/launch'/(name+'.json');complete=OUT/'training/done'/(name+'.json');status=OUT/'training/audits'/name/'STATUS.json'
            step=json.loads(status.read_text()).get('iter',0) if status.exists() else history[-1]['step'] if history else 0
            condition='已完成' if complete.exists() else '训练中' if launch.exists() else '排队等待'
            states[name]=dict(label=label,step=step,total=250000,state=condition)
            page+='<section><h2>'+html.escape(label)+'</h2><progress value="'+str(step)+'" max="250000"></progress><p>'+condition+' · '+str(step)+' /250000</p><p class="muted">无SDF · 从头初始化 · batch4 · 256×256</p></section>'
    page+='</div>'
    for filename in ['DATA_FAILED.json','TRAINING_FAILED.json']:
        if (OUT/filename).exists():page+='<section><h2>异常</h2><pre>'+html.escape((OUT/filename).read_text())+'</pre></section>'
    page+=render_loss_charts(histories)+'</html>';temporary=OUT/'dashboard.html.tmp';temporary.write_text(page,encoding='utf-8');temporary.replace(OUT/'dashboard.html')
    save(OUT/'MONITOR.json',dict(phase=phase,data=data,timer=timer,runs=states,updated_epoch=time.time()))


def gpu_queue(gpu,records):
    """@luna-comment Input:oneGPU and all assigned loss recipes. Output:sequential detached original training runs. Key Steps:full readiness/init/config/smoke gates;wait GPU free;launch;verify final250K before successor. Key Params:GPU0 owns6recipes,others3;no termination/no partial data. State/Dependencies:owned PID/hash receipts."""
    for name,record in records.items():
        if record['gpu']!=gpu:continue
        assert ready(OUT) and json.loads((OUT/'SMOKE_COMPLETE.json').read_text())['count']==len(records)
        assert digest(record['config'])==record['sha256']
        while int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>2000:time.sleep(30)
        log=OUT/'training/logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        assert not (OUT/'training/launch'/(name+'.json')).exists(),'Refusing duplicate run'
        command=[sys.executable,'-u',str(ROOT/'tools/train_full_s22_strategy.py'),'-opt',record['config']]
        with log.open('xb') as stream:
            process=subprocess.Popen(command,cwd=OUT/'training',stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            save(OUT/'training/launch'/(name+'.json'),dict(pid=process.pid,gpu=gpu,command=command,config_sha256=record['sha256'],started_cst=datetime.now(TZ).isoformat()))
            code=process.wait();assert code==0,(name,code)
        final=OUT/'training/experiments'/record['name']/'models/net_g_250000.pth';assert final.exists()
        save(OUT/'training/done'/(name+'.json'),dict(passed=True,final=str(final),final_sha256=digest(final),completed_epoch=time.time()))


def main():
    """@luna-comment Input:authorized durable scheduling invocation. Output:buffered absolute timer and guarded15-run pipeline. Key Steps:exclusive lock;configs;measured ETA;pilot15smokes;wait timer+data+allchecks;four sequentialGPU queues (GPU0 owns two strategy blocks);monitor. Key Params:250K each,max4concurrent. State/Dependencies:owned process survivesSSH,full-data preparation separate."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'SCHEDULER_PROCESS.json',dict(pid=os.getpid(),started_cst=datetime.now(TZ).isoformat(),controller=str(Path(__file__))))
    records=prepare_configs();publish('测量全量处理吞吐')
    while not (OUT/'TIMER.json').exists():
        assert not (OUT/'DATA_FAILED.json').exists(),'Data preparation failed'
        data=json.loads((OUT/'DATA_STATUS.json').read_text()) if (OUT/'DATA_STATUS.json').exists() else {}
        if data.get('teacher_done',0)>=96 and data.get('gt_done',0)>=64 and data.get('elapsed_seconds',0)>=300:save(OUT/'TIMER.json',estimate_timer(data))
        publish('测量吞吐并准备定时任务');time.sleep(15)
    publish('定时任务已登记，执行15项工程检查');pilot_and_smoke(records)
    timer=json.loads((OUT/'TIMER.json').read_text())
    while time.time()<timer['scheduled_epoch'] or not ready(OUT):
        timer=json.loads((OUT/'TIMER.json').read_text())
        assert not (OUT/'DATA_FAILED.json').exists(),'Data preparation failed'
        publish('定时等待；全量数据就绪后启动' if time.time()<timer['scheduled_epoch'] else '已到定时时间，等待全量数据就绪');time.sleep(20)
    assert json.loads((OUT/'SMOKE_COMPLETE.json').read_text())['count']==len(records)
    bindings=json.loads((OUT/'CONFIG_BINDINGS.json').read_text());assert digest(bindings['shared_initial'])==bindings['shared_initial_sha256']
    save(OUT/'TIMER_TRIGGERED.json',dict(triggered_cst=datetime.now(TZ).isoformat(),scheduled_cst=timer['scheduled_cst'],data_ready_sha256=digest(OUT/'DATA_READY.json')))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(gpu_queue,gpu,records) for gpu in range(4)]
        while any(not f.done() for f in futures):
            for f in futures:
                if f.done() and f.exception() is not None:save(OUT/'TRAINING_FAILED.json',dict(error=repr(f.exception()),epoch=time.time(),policy='Failed GPU queue stopped;other owned queues continue'))
            publish('四路GPU训练队列运行中');time.sleep(20)
        for f in futures:f.result()
    save(OUT/'TRAINING_COMPLETE.json',dict(passed=True,count=len(records),completed_cst=datetime.now(TZ).isoformat()));publish('15组250K全部完成')


if __name__=='__main__':
    try:main()
    except Exception as error:save(OUT/'TRAINING_FAILED.json',dict(error=repr(error),epoch=time.time()));raise
