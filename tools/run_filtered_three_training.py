"""Shared-GPU data-gated three-arm real-paired scratch500K training controller."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import sys,os,json,time,copy,subprocess,fcntl,html,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_filtered_three_bank import OUT,DATA,save,digest
LABELS={'l1':'V3M · L1-only · 500K','ori':'V3M · L1+R1_ori · 500K','norm':'V3M · L1+R1_norm · 500K'}
TRAINER=ROOT/'tools/train_filtered_three.py'


def read(path):
    """@luna-comment Input:atomic owned JSON path. Output:document or absent dict. Key Steps:UTF8 load. Key Params:corruption raises. State/Dependencies:filesystem."""
    p=Path(path);return json.loads(p.read_text()) if p.exists() else {}


def alive(record):
    """@luna-comment Input:owned PID/start token. Output:identity-checked liveness. Key Steps:exclude zombies and PID reuse. Key Params:Linux proc only. State/Dependencies:read-only process state."""
    try:
        state=Path(f'/proc/{record["pid"]}/stat').read_text().split();return state[2]!='Z' and state[21]==record['start_token']
    except (OSError,KeyError):return False


def configure():
    """@luna-comment Input:frozen L1 base recipe and scratch weights. Output:three matched actual-data configs. Key Steps:no mixture/no contrast;only original versus accepted normalized R1;strict shared init/schedule. Key Params:500K,1e-4to5e-6,batch4,no other auxiliary losses. State/Dependencies:owned YAML/bindings only."""
    import yaml
    b=read(DATA/'latest_vosr_four_500k_20261102/CONFIG_BINDINGS.json')['runs']['v3m_l1'];assert digest(b['config'])==b['config_sha256'];base=yaml.safe_load(Path(b['config']).read_text());assert digest(base['path']['pretrain_network_g'])==b['initial_sha256']
    result={}
    for i,kind in enumerate(LABELS):
        c=copy.deepcopy(base);c['name']='filtered_trusted_v3m_'+kind+'_paired_500k_20261105';c['model_type']='FilteredNormModel' if kind=='norm' else 'FilteredPairedModel';c['train']['model_type']=c['model_type']
        c['path'].update(experiments_root=str(OUT/'experiments'),resume_state=None);c['s22_strategy'].update(name='paired',audit_root=str(OUT/'audits'/kind));c['filtered_control']=dict(kind=kind,synthetic_gt_probability=0,contrast_alignment=False,train_count=5284)
        c['train'].update(total_iter=500000,recall_aware_weight_missing=0. if kind=='l1' else 1.,recall_aware_weight_false=0. if kind=='l1' else .25,recall_aware_opt=None if kind=='l1' else dict(ring_radius=3,margin=.03,temperature=.01,min_polarity_contrast=.03))
        c['train']['optim_g']['lr']=1e-4;c['train']['scheduler']={'type':'LatestVOSR500K'};c['dist_params']['port']=36410+i;c['logger'].update(save_checkpoint_freq=5000,print_freq=100,use_tb_logger=False)
        for split in ['train','val']:c['datasets'][split].update(type='R1WidthDataset' if kind=='norm' else 'S22Dataset',root=str(OUT/'bank'),manifest=str(OUT/'bank'/(split+'.json')),patch_contrast_alignment=False)
        c['v3pre_target_domain']='Filtered trusted-background GT5284;no Mojito;real paired only;contrast off;historical583 non-Mojito validation'
        assert all(v==0 for k,v in c['train'].items() if k.endswith('_weight'))
        path=OUT/'configs'/(kind+'.yml');path.parent.mkdir(exist_ok=True);payload=yaml.safe_dump(c,sort_keys=False)
        if path.exists():assert path.read_text()==payload,'Frozen config mismatch'
        else:path.write_text(payload)
        sm=copy.deepcopy(c);sm['name']+='_smoke';sm['s22_strategy']['audit_root']=str(OUT/'smoke_audits'/kind);sp=OUT/'configs'/(kind+'_smoke.yml');sp.write_text(yaml.safe_dump(sm,sort_keys=False))
        result[kind]=dict(config=str(path),config_sha256=digest(path),smoke=str(sp),name=c['name'],initial=c['path']['pretrain_network_g'],initial_sha256=b['initial_sha256'])
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=result,all_from_scratch=True,shared_initial_sha256=b['initial_sha256'],total_iter=500000,synthetic_gt_probability=0,contrast_alignment=False,shared_gpu_authorized=True))
    return result


def publish(phase,extra=None):
    """@luna-comment Input:owned stage/training receipts. Output:Chinese three-run monitor with real LR/loss/stage ETA. Key Steps:separate bank/line/width/train gates;unique loss labels;live PID checks. Key Params:shared GPUs explicitly shown,no idle wait. State/Dependencies:owned JSON/HTML only."""
    now=time.time();tz=timezone(timedelta(hours=8));histories={};cards=[]
    for kind,label in LABELS.items():
        status=read(OUT/'audits'/kind/'RUN_STATUS.json');child=read(OUT/'children'/('train_'+kind+'.json'));done=read(OUT/'done'/(kind+'.json'));path=OUT/'audits'/kind/'history.jsonl';records={}
        if path.exists():
            for line in path.read_text().splitlines():
                r=json.loads(line);records[r['iter']]=dict(step=r['iter'],values=r['logs'],updated=r['updated'])
        history=sorted(records.values(),key=lambda r:r['step']);histories[label]=history;iteration=status.get('iter',0);eta=None
        if len(history)>2 and now-status.get('updated',0)<300:
            window=history[-50:];a,b=window[0],window[-1]
            if b['step']>a['step']:eta=(500000-iteration)*(b['updated']-a['updated'])/(b['step']-a['step'])
        finish=datetime.fromtimestamp(now+eta,tz).strftime('%m-%d %H:%M') if eta is not None else '开训后按实测速度更新'
        stage='已完成' if done.get('passed') else '训练中（与现有任务并行）' if alive(child) else '等待共享数据准备' if not (OUT/'DATA_READY.json').exists() else '等待匹配粗细标签' if kind=='norm' and not (OUT/'WIDTH_READY.json').exists() else '启动检查 / 准备开训'
        cards.append(dict(label=label,status=stage,current=500000 if done.get('passed') else iteration,total=500000,details=['GPU '+str(child.get('gpu','待分配')),'LR '+str(status.get('lr',1e-4)),'预计完成：'+('已完成' if done.get('passed') else finish),'从头训练；仅真实LQ；无GT退化、无对比度对齐']))
    datasets=[]
    for name,path,done,total in [('5284张离线训练块',OUT/'train_bank/progress.json',OUT/'train_bank/COMPLETE.json',5284),('文字区域标注缓存',OUT/'train_bank/line_regions/progress.json',OUT/'DATA_READY.json',5284),('R1_norm粗细标签',OUT/'WIDTH_STATUS.json',OUT/'WIDTH_READY.json',5284)]:
        r=read(path);n=total if done.exists() else r.get('completed',r.get('done',0));seconds=r.get('seconds',0);eta=(total-n)*seconds/n if n else None
        details=['预计完成：'+(datetime.fromtimestamp(now+eta,tz).strftime('%m-%d %H:%M') if eta is not None else '首批完成后更新')]
        datasets.append(dict(label=name,status='已完成' if done.exists() else '处理中' if n else '待处理',current=n,total=total,details=details))
    snapshot=dict(updated=datetime.now(tz).isoformat(),runs=cards,datasets=datasets,lossHistory=histories,phase=phase,extra=extra)
    save(OUT/'monitor.json',snapshot);save(OUT/'CONTROLLER_STATUS.json',dict(pid=os.getpid(),phase=phase,updated=now,extra=extra))
    body='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>新版GT三组500K</title><style>body{font:17px/1.7 system-ui;background:#eef3f8;color:#17324d;margin:25px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}article{background:white;border-radius:13px;padding:20px}progress{width:100%;height:19px}</style><h1>新版可信背景GT · V3M三组500K</h1><p>5284训练图 / 583验证图，均无Mojito；三组共享从头初始化。与已有任务共享GPU，不等空闲。</p><p>'+html.escape(phase)+'</p>'
    for title,items in [('训练',cards),('共享数据准备',datasets)]:
        body+='<h2>'+title+'</h2><div class="grid">'
        for r in items:body+='<article><h3>'+html.escape(r['label'])+'</h3><b>'+r['status']+'</b><p>'+str(r['current'])+' / '+str(r['total'])+'</p><progress max="'+str(r['total'])+'" value="'+str(r['current'])+'"></progress><p>'+'</p><p>'.join(html.escape(v) for v in r['details'])+'</p></article>'
        body+='</div>'
    if extra:body+='<p>'+html.escape(str(extra))+'</p>'
    body+='<p>更新：'+snapshot['updated']+'</p>';tmp=OUT/'index.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'index.html')


def spawn(stage,command,gpu=None):
    """@luna-comment Input:known stage command and optional shared GPU. Output:adopted or detached owned child. Key Steps:PID-token receipt,CPU limits,isolated logs,identity-safe recovery. Key Params:never signal other jobs. State/Dependencies:owned process groups/files."""
    path=OUT/'children'/(stage+'.json');r=read(path)
    if alive(r):return r
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',REAL_ESRGAN_DEFER_FINAL_VALIDATION='1');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    (OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'logs'/(stage+'.log')).open('ab') as stream:p=subprocess.Popen(command,cwd=OUT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    r=dict(pid=p.pid,start_token=Path(f'/proc/{p.pid}/stat').read_text().split()[21],started=time.time(),gpu=gpu,command=command);save(path,r);return r


def shared_gpu(owned):
    """@luna-comment Input:GPUs already assigned to new live arms. Output:least-loaded distinct GPU with startup headroom. Key Steps:count existing compute processes,then utilization/memory rank. Key Params:allow busy GPUs,minimum6GiB;no idle gate. State/Dependencies:nvidia-smi read-only."""
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);inventory={};uuids={}
    for line in lines.splitlines():
        i,u,free,util=[s.strip() for s in line.split(',')];inventory[int(i)]=dict(free=int(free),util=int(util),count=0);uuids[u]=int(i)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for line in processes.splitlines():
        uuid=line.split(',')[0].strip()
        if uuid in uuids:inventory[uuids[uuid]]['count']+=1
    candidates=[i for i,r in inventory.items() if i not in owned and r['free']>=6144]
    return min(candidates,key=lambda i:(inventory[i]['count'],inventory[i]['util'],-inventory[i]['free'])) if candidates else None


def launch_recipe(kind,binding):
    """@luna-comment Input:frozen recipe and verified owned checkpoint receipts. Output:per-launch fresh or resumed config. Key Steps:strict config/init hashes,descending paired-state validation;no silent scratch restart after progress. Key Params:optimizer and scheduler preserved. State/Dependencies:owned YAML/checkpoints."""
    import yaml,torch
    assert digest(binding['config'])==binding['config_sha256'] and digest(binding['initial'])==binding['initial_sha256'];c=yaml.safe_load(Path(binding['config']).read_text());found=False
    for p in sorted((OUT/'experiments'/binding['name']/'training_states').glob('*.verified.json'),key=lambda p:int(p.name.split('.')[0]),reverse=True):
        r=read(p)
        try:
            assert digest(r['model'])==r['model_sha256'] and digest(r['state'])==r['state_sha256'];state=torch.load(r['state'],map_location='cpu');assert state['iter']==r['iter'];c['path']['resume_state']=r['state'];found=True;break
        except Exception as error:save(OUT/(kind+'_checkpoint_rejected.json'),dict(path=str(p),error=repr(error)))
    if not found and read(OUT/'audits'/kind/'RUN_STATUS.json').get('iter',0)>3:raise RuntimeError('Progress without verified checkpoint: '+kind)
    path=OUT/'configs'/(kind+'_launch.yml');path.write_text(yaml.safe_dump(c,sort_keys=False));return path


def run():
    """@luna-comment Input:authorized shared-GPU experiment. Output:three real-paired500K runs with independent data gates and recovery. Key Steps:adopt bank worker;sidecars;parallel width prep;smoke then launch each available recipe on busy-allowed GPUs. Key Params:no old-job stop/no idle wait;20s heartbeat. State/Dependencies:owned lock/state only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if read(OUT/'CANCELLED.json').get('cancelled'):return
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),start_token=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],started=time.time()))
    bindings=configure();failures=0
    while True:
        try:
            if not (OUT/'train_bank/COMPLETE.json').exists():
                child=spawn('bank',[sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(OUT/'source_view'),'--output',str(OUT/'train_bank'),'--workers','8','--preprocess-mode','none'])
                while alive(child):publish('构建共享5284张离线训练数据');time.sleep(20)
                assert (OUT/'train_bank/COMPLETE.json').exists(),'Native bank failed'
            if not (OUT/'DATA_READY.json').exists():
                child=spawn('finalize',[sys.executable,'-u',str(ROOT/'tools/prepare_filtered_three_bank.py'),'finalize'])
                while alive(child):publish('复用OCR标注缓存并生成有效文字采样索引');time.sleep(20)
                assert read(OUT/'DATA_READY.json').get('passed'),'Shared data finalization failed'
            ready=read(OUT/'DATA_READY.json');assert ready['no_mojito_in_train_or_val'] and digest(OUT/'bank/train.json')==ready['train_sha256']
            if not (OUT/'WIDTH_READY.json').exists():spawn('widths',[sys.executable,'-u',str(ROOT/'tools/prepare_filtered_three_bank.py'),'widths'])
            for kind,binding in bindings.items():
                if read(OUT/'done'/(kind+'.json')).get('passed') or alive(read(OUT/'children'/('train_'+kind+'.json'))):continue
                if kind=='norm' and not (OUT/'WIDTH_READY.json').exists():continue
                final=OUT/'experiments'/binding['name']/'training_states/500000.verified.json'
                if final.exists():
                    import torch
                    r=read(final);assert digest(r['model'])==r['model_sha256'] and digest(r['state'])==r['state_sha256'];state=torch.load(r['state'],map_location='cpu');assert state['iter']==500000 and abs(state['optimizers'][0]['param_groups'][0]['lr']-5e-6)<1e-12;save(OUT/'done'/(kind+'.json'),dict(passed=True,checkpoint=r['model'],checkpoint_sha256=r['model_sha256']));continue
                owned={read(OUT/'children'/('train_'+k+'.json')).get('gpu') for k in LABELS if alive(read(OUT/'children'/('train_'+k+'.json')))}
                gpu=shared_gpu(owned)
                if gpu is None:continue
                smoke=OUT/'smoke'/(kind+'.json')
                if not smoke.exists():
                    child=spawn('smoke_'+kind,[sys.executable,'-u',str(TRAINER),'--smoke-config',binding['smoke'],'--smoke-output',str(smoke)],gpu)
                    while alive(child):publish('CUDA启动检查',dict(kind=kind,gpu=gpu));time.sleep(20)
                    assert read(smoke).get('passed'),'CUDA smoke failed: '+kind
                config=launch_recipe(kind,binding);spawn('train_'+kind,[sys.executable,'-u',str(TRAINER),'-opt',str(config)],gpu)
            if all(read(OUT/'done'/(k+'.json')).get('passed') for k in LABELS):save(OUT/'COMPLETE.json',dict(passed=True,arms=3,total_iter=500000));publish('三组全部完成');return
            publish('按数据就绪顺序并行开训，不等待GPU空闲');time.sleep(20)
        except Exception as error:
            failures+=1;r=dict(error=repr(error),traceback=traceback.format_exc(),updated=time.time(),attempt=failures);save(OUT/'LAST_ERROR.json',r);publish('异常后等待自动重试',r);time.sleep(min(60,10*failures))


if __name__=='__main__':run()
