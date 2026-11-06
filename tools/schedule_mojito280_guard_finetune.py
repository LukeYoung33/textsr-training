"""Server-resident, data-gated three-arm Mojito280 finetune timer."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy, json, os, sys, time, subprocess, shutil, html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_uniform20_steep_scurve import save, digest

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'mojito280_guard_finetune_20261030'
GT = DATA/'mojito280_reused_v3_vosr_s22_20261029'
VIEW = OUT/'source_view'
BANK = OUT/'bank256'
TZ = timezone(timedelta(hours=8))
TARGET = datetime.fromisoformat('2026-10-30T02:06:10+08:00').timestamp()
SOURCE_CONFIGS = {
    'guard': DATA/'full_s22_training_matrix_20261028/training/configs/guard_full.yml',
    'contrast': DATA/'s22_patch_contrast_guard_20261029/configs/full_nosdf.yml',
    'contrast_nocp': DATA/'s22_nocp_text_guard_20261029/configs/full_nosdf.yml',
}
LABELS = {'guard':'Full 无SDF · 可信边缘保护', 'contrast':'Full 无SDF · 可信边缘保护 + 对比度对齐',
          'contrast_nocp':'Full 无SDF · 可信边缘保护 + 对比度对齐 · 无CP / 有效文字patch'}
CHECKPOINT = DATA/'full_s22_training_matrix_20261028/training/experiments/s22_guard_full_20261028/models/net_g_250000.pth'


def publish(phase, **details):
    """@luna-comment Input:current phase/details. Output:atomic Chinese status page. Key Steps:read each owned audit and loss history;report timer and waits explicitly. Key Params:150K,UTC+08. State/Dependencies:owned monitor only."""
    from tools.offline_loss_charts import read_loss_history, render_loss_charts
    histories = {}; cards = []; states = {}
    for arm, label in LABELS.items():
        path = OUT/'audits'/arm/'STATUS.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        launch = OUT/'launch'/(arm+'.json'); done = OUT/'done'/(arm+'.json')
        status = '完成' if done.exists() else '已启动' if launch.exists() else '等待启动条件'
        step = 150000 if done.exists() else state.get('iter',0)
        states[arm] = dict(label=label, step=step, status=status)
        history = read_loss_history(OUT/'logs'/(arm+'.log'))
        if history: histories[label] = history
        cards.append('<section><h2>'+label+'</h2><p>'+status+' · '+str(step)+' / 150000</p><progress max="150000" value="'+str(step)+'"></progress></section>')
    state = dict(phase=phase, scheduled_cst=datetime.fromtimestamp(TARGET,TZ).isoformat(),
                 updated_cst=datetime.now(TZ).isoformat(), pid=os.getpid(), runs=states, **details)
    save(OUT/'STATUS.json',state)
    body = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Mojito280 三组 finetune</title><style>body{font:16px/1.7 system-ui;margin:28px;background:#eef3f8;color:#19314b}section{background:white;padding:20px;border-radius:12px;margin:16px 0}progress{width:100%}pre{white-space:pre-wrap}</style><h1>Mojito280 · 三组 finetune 150K</h1><p>计划启动：2026-10-30 02:06:10 北京时间；数据或 GPU 未就绪则等待。</p><p>243训练 / 37验证 · 三组共用普通 guard250K 初始化 · 新优化器 · V3 输入 + S2.2 GT</p><section><strong>'+html.escape(phase)+'</strong><pre>'+html.escape(json.dumps(details,ensure_ascii=False,indent=2))+'</pre><small>'+state['updated_cst']+'</small></section>'+''.join(cards)+render_loss_charts(histories)
    temp=OUT/'dashboard.tmp';temp.write_text(body,encoding='utf-8');temp.replace(OUT/'dashboard.html')


def prepare_configs():
    """@luna-comment Input:shared guard250K and three existing recipes. Output:immutable configs/common weight snapshot. Key Steps:clone losses;set150K/Adam1e-5 and45/90/135K schedule;bind independent outputs. Key Params:CP off only third,contrast off validation,fresh optimizer. State/Dependencies:no source modifications."""
    import yaml
    if not CHECKPOINT.exists(): return False
    initial=OUT/'initial/guard250k_shared.pth';initial.parent.mkdir(parents=True,exist_ok=True)
    sha=digest(CHECKPOINT)
    if not initial.exists(): shutil.copyfile(CHECKPOINT,initial)
    assert digest(initial)==sha
    records={}
    for i,(arm,path) in enumerate(SOURCE_CONFIGS.items()):
        cfg=yaml.safe_load(path.read_text());cfg['name']='mojito280_guard_ft150k_'+arm+'_20261030'
        cfg['path'].update(pretrain_network_g=str(initial),resume_state=None,strict_load_g=True,experiments_root=str(OUT/'experiments'))
        cfg['train']['total_iter']=150000;cfg['train']['optim_g']['lr']=1e-5
        cfg['train']['scheduler']['milestones']=[45000,90000,135000]
        cfg['s22_strategy']['audit_root']=str(OUT/'audits'/arm)
        cfg['dist_params']['port']=35720+i
        for split in ['train','val']:
            cfg['datasets'][split].update(root=str(BANK),manifest=str(BANK/('train_balanced.json' if split=='train' else 'val.json')),num_worker_per_gpu=2)
            cfg['datasets'][split]['patch_contrast_alignment']=split=='train' and arm!='guard'
        if arm=='contrast_nocp': cfg['datasets']['train']['manifest']=str(OUT/'filter/train_balanced.json')
        assert cfg['train']['sdf_boundary_weight']==0 and cfg['train']['sdf_boundary_opt'] is None
        assert cfg['s22_strategy']['name']=='guard' and cfg['paired_probability']==1
        assert (cfg['train']['cp_opt'] is None)==(arm=='contrast_nocp')
        target=OUT/'configs'/(arm+'.yml');target.parent.mkdir(exist_ok=True)
        text=yaml.safe_dump(cfg,sort_keys=False)
        if target.exists(): assert target.read_text()==text
        else: target.write_text(text)
        smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke/audits'/arm)
        smoke_path=OUT/'configs'/(arm+'_smoke.yml');smoke_path.write_text(yaml.safe_dump(smoke,sort_keys=False))
        records[arm]=dict(config=str(target),config_sha256=digest(target),smoke=str(smoke_path),name=cfg['name'],initial_sha256=sha,source_config=str(path),source_sha256=digest(path))
    save(OUT/'CONFIG_BINDINGS.json',dict(shared_checkpoint=str(initial),shared_sha256=sha,runs=records))
    return True


def prepare_parent(selected, record, original, geometry):
    """@luna-comment Input:completed teacher receipt/original masks/rotated OCR. Output:hash-bound native training row. Key Steps:reuse V3 and S2.2;rotate existing masks only forEXIF6;write provided polygons. Key Params:no OCR,no image preprocessing,unchanged243/37 split. State/Dependencies:owned source view."""
    import cv2, numpy as np
    cv2.setNumThreads(1)
    ident=selected['id'];folder=VIEW/'images'/ident;receipt=folder/'receipt.json'
    if receipt.exists(): return json.loads(receipt.read_text())['row']
    folder.mkdir(parents=True,exist_ok=True)
    lq=Path(record['source']);gt=GT/'results/steep_s'/(ident+'.png')
    assert digest(lq)==record['source_sha256'] and digest(gt)==record['arms']['steep_s']['sha256']
    shape=tuple(record['shape_hw']);assert shape==tuple(geometry['shape_hw'])
    assert cv2.imread(str(lq)).shape[:2]==cv2.imread(str(gt)).shape[:2]==shape
    files=dict(lq=str(lq),gt=str(gt));hashes=dict(lq=digest(lq),gt=digest(gt))
    for key in ['text_mask','valid_mask']:
        source=Path(original['files'][key]);assert digest(source)==original['output_sha256'][key]
        mask=cv2.imread(str(source),0);assert mask is not None
        if selected['orientation']==6: mask=cv2.rotate(mask,cv2.ROTATE_90_CLOCKWISE)
        assert mask.shape==shape
        path=folder/(key+'.png');assert cv2.imwrite(str(path),mask)
        files[key]=str(path);hashes[key]=digest(path)
    annotations=[]
    for i,(poly,score) in enumerate(zip(geometry['polygons'],geometry['scores'])):
        q=np.asarray(poly);assert np.isfinite(q).all()
        annotations.append(dict(instance_id=str(i),points=poly,bbox=[*q.min(0).tolist(),*q.max(0).tolist()],score=score,ignore=False,visible=True,text=''))
    ocr=folder/'ocr.json';save(ocr,dict(coordinate_system='displayed_v3_pixels',ocr=dict(annotations=annotations),provenance=geometry['source_annotation'],annotation_sha256=geometry['annotation_sha256']))
    files['ocr']=str(ocr);hashes['ocr']=digest(ocr)
    row=dict(dataset='lenovo_mojito',sample_id=selected['sample_id'],source_identity=original['source_identity'],files=files,output_sha256=hashes)
    save(receipt,dict(row=row,split=selected['split'],id=ident,teacher_source_sha256=record['source_sha256'],orientation=selected['orientation']))
    return row


def run_checked(command, log, phase, env=None):
    """@luna-comment Input:known command/log/phase. Output:checked subprocess completion with heartbeat. Key Steps:poll child while publishing state. Key Params:15s poll. State/Dependencies:only owned subprocess;failure stops launch."""
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,cwd=OUT)
        while proc.poll() is None:
            progress=json.loads((BANK/'progress.json').read_text()) if (BANK/'progress.json').exists() else {}
            publish(phase,child_pid=proc.pid,bank_progress=progress);time.sleep(15)
        assert proc.returncode==0,(phase,proc.returncode,str(log))


def build_sidecars_and_filter():
    """@luna-comment Input:complete243/37 patch bank. Output:line sidecars and noCP stable text-only index. Key Steps:reuse geometry builder and existing positive-support filter. Key Params:training only,retain sampling order/repeats. State/Dependencies:immutable bank and existing annotations."""
    import numpy as np
    from tools.run_v3pre_training import sidecars
    from tools.run_s22_nocp_text_guard import read_support,filter_sampling_index
    if not (BANK/'LINE_REGIONS_COMPLETE.json').exists(): sidecars(BANK,VIEW,2)
    path=BANK/'train_balanced.json';manifest=json.loads(path.read_text())
    flags=[read_support((BANK,shard)) for shard in manifest['shards']]
    target=OUT/'filter';target.mkdir(exist_ok=True)
    counts={}
    for key,name in [('index','train_balanced.index.npy'),('unique_index','train.index.npy')]:
        index=np.load(BANK/manifest[key],allow_pickle=False);filtered,_=filter_sampling_index(index,flags);assert len(filtered)>0
        np.save(target/name,filtered,allow_pickle=False);manifest[key]=str(target/name);counts[key]=dict(original=len(index),kept=len(filtered))
    save(target/'train_balanced.json',manifest);save(OUT/'FILTER_COMPLETE.json',dict(passed=True,counts=counts))
    bank=json.loads((BANK/'COMPLETE.json').read_text());lines=json.loads((BANK/'LINE_REGIONS_COMPLETE.json').read_text())
    assert bank['complete'] and bank['splits']['train']['parents']==243 and bank['splits']['val']['parents']==37
    assert lines['passed'] and lines['count']==280 and lines['source_build_id']==bank['build_id']
    save(OUT/'DATA_READY.json',dict(passed=True,train=243,val=37,bank_sha256=digest(BANK/'COMPLETE.json'),sidecars_sha256=digest(BANK/'LINE_REGIONS_COMPLETE.json'),filter_sha256=digest(target/'train_balanced.json')))


def main():
    """@luna-comment Input:confirmed timer/config/selection. Output:gated150K runs and truthful monitor. Key Steps:exclusive lock;incremental data;full bank;deadline;CUDA smokes;capacity-based launch;observe checkpoints. Key Params:no partial bank,no existing-job termination,no implicit failed-run restart. State/Dependencies:server persists independently of local PC."""
    import fcntl
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'FAILED.json').exists(),'Previous failure requires inspection'
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started_cst=datetime.now(TZ).isoformat()))
    save(OUT/'TIMER.json',dict(scheduled_epoch=TARGET,scheduled_cst=datetime.fromtimestamp(TARGET,TZ).isoformat(),late_policy='Wait for all data,shared weights,CUDA smoke and GPU capacity',total_iter=150000,train=243,val=37))
    if '--sidecars' in sys.argv: build_sidecars_and_filter();return
    while not prepare_configs(): publish('等待普通 guard250K 权重');time.sleep(30)
    if '--prepare-only' in sys.argv: publish('定时配置已准备');return
    selection=json.loads((OUT/'selection.json').read_text())['samples'];assert len(selection)==280
    cache=json.loads((GT/'OCR_CACHE.json').read_text());lookup={}
    for split in ['train','val']:
        path=DATA/'v3pre_v3m_three_20261027/finetune/source_view/manifests'/(split+'.jsonl')
        for line in path.read_text().splitlines():
            row=json.loads(line);lookup[row['sample_id']]=row
    while not (OUT/'SOURCE_READY.json').exists():
        exported=json.loads((GT/'EXPORT_MANIFEST.json').read_text());available={r['id']:r for r in exported['records']};prepared={'train':[],'val':[]}
        for selected in selection:
            if selected['id'] not in available:continue
            row=prepare_parent(selected,available[selected['id']],lookup[selected['sample_id']],cache[selected['id']]);prepared[selected['split']].append(row)
        count=sum(map(len,prepared.values()));publish('准备配对数据（复用 V3 和文字框）',prepared=count,total=280)
        complete=json.loads((GT/'COMPLETE.json').read_text()) if (GT/'COMPLETE.json').exists() else {}
        if count==280 and complete.get('passed') and complete.get('count')==280:
            assert len(prepared['train'])==243 and len(prepared['val'])==37
            assert not ({r['source_identity'] for r in prepared['train']}&{r['source_identity'] for r in prepared['val']})
            (VIEW/'manifests').mkdir(parents=True,exist_ok=True)
            for split,rows in prepared.items():(VIEW/'manifests'/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            save(OUT/'SOURCE_READY.json',dict(passed=True,train=243,val=37,gt_complete_sha256=digest(GT/'COMPLETE.json')));break
        time.sleep(30)
    if not (BANK/'COMPLETE.json').exists():run_checked([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(VIEW),'--output',str(BANK),'--workers','2','--preprocess-mode','none'],OUT/'bank.log','构建完整 256 patch 数据')
    if not (OUT/'DATA_READY.json').exists():
        # The helper does not take the controller lock; it only builds owned sidecars.
        code='from tools.schedule_mojito280_guard_finetune import build_sidecars_and_filter;build_sidecars_and_filter()'
        run_checked([sys.executable,'-c',"import sys;sys.path.insert(0,"+repr(str(ROOT))+");"+code],OUT/'sidecars.log','构建文字区域缓存与 noCP 索引')
    while time.time()<TARGET:publish('数据已就绪，等待定时启动',remaining_seconds=TARGET-time.time());time.sleep(max(0,min(30,TARGET-time.time())))
    binding=json.loads((OUT/'CONFIG_BINDINGS.json').read_text());pending=list(LABELS);processes={};streams=[]
    assert not any((OUT/'launch').glob('*.json')),'Refuse duplicate training launch'
    while pending or processes:
        for arm,proc in list(processes.items()):
            if proc.poll() is not None:
                record=binding['runs'][arm];final=OUT/'experiments'/record['name']/'models/net_g_150000.pth'
                assert proc.returncode==0 and final.exists(),arm+' training failed'
                save(OUT/'done'/(arm+'.json'),dict(passed=True,checkpoint=str(final),sha256=digest(final)));del processes[arm]
        occupied={json.loads((OUT/'launch'/(arm+'.json')).read_text())['gpu'] for arm in processes}
        capacity=[tuple(map(int,line.split(','))) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
        candidates=[x for x in capacity if x[0] not in occupied and x[1]>=7500]
        if pending and candidates:
            gpu,free=max(candidates,key=lambda x:x[1]);arm=pending[0];record=binding['runs'][arm]
            assert digest(record['config'])==record['config_sha256'] and digest(binding['shared_checkpoint'])==binding['shared_sha256']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            result=OUT/'smoke'/(arm+'.json')
            run_checked([sys.executable,'-u',str(ROOT/'tools/train_full_s22_strategy.py'),'--smoke-config',record['smoke'],'--smoke-output',str(result)],OUT/'smoke'/(arm+'.log'),'启动测试：'+LABELS[arm],env)
            smoke=json.loads(result.read_text());assert smoke['passed'] and smoke['updates']==9 and smoke['initial_exact'] and smoke['fresh_optimizer'] and smoke['no_sdf'] and smoke['no_cp']==(arm=='contrast_nocp')
            log=OUT/'logs'/(arm+'.log');log.parent.mkdir(exist_ok=True);stream=log.open('xb');streams.append(stream)
            command=[sys.executable,'-u',str(ROOT/'tools/train_full_s22_strategy.py'),'-opt',record['config']]
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=OUT,env=env,start_new_session=True)
            save(OUT/'launch'/(arm+'.json'),dict(pid=proc.pid,gpu=gpu,command=command,initial_sha256=binding['shared_sha256'],config_sha256=record['config_sha256'],started_cst=datetime.now(TZ).isoformat()))
            processes[arm]=proc;pending.pop(0)
        publish('训练中 / 等待可用 GPU',pending=pending,running=list(processes),gpu_capacity=capacity);time.sleep(20)
    save(OUT/'TRAINING_COMPLETE.json',dict(passed=True,arms=3,total_iter=150000));publish('三组训练完成')


if __name__=='__main__':
    try: main()
    except Exception as error:
        save(OUT/'FAILED.json',dict(error=repr(error),updated_cst=datetime.now(TZ).isoformat()));publish('任务异常，未继续启动',error=repr(error));raise
