"""Frozen final500K six-model Mojito280 inference, deployed postprocessing and viewer."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,os,json,time,shutil,subprocess,argparse,fcntl,html
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');BASE=DATA/'v3pre_three_mojito280_20261028';OUT=DATA/'latest_six_500k_mojito280_20261103'
LABELS={'v3m_guard':'V3M · Full + guard','v3m_l1':'V3M · L1-only','v5_guard':'V5 · Full + guard','v5_l1':'V5 · L1-only','v3m_mix50':'V3M · Full无guard + 50% GT轻退化','v5_mix50':'V5 · Full无guard + 50% GT轻退化'}


def prepare():
    """@luna-comment Input:six completed500K sources. Output:immutable model/config/input bindings. Key Steps:strict completion,hash-copy-hash,original selection reuse. Key Params:final500K only. State/Dependencies:owned output;original artifacts read-only."""
    import yaml
    assert not (OUT/'MODEL_BINDINGS.json').exists();(OUT/'models').mkdir(parents=True,exist_ok=True);bindings={}
    for arm,label in LABELS.items():
        source=DATA/('latest_vosr_mix50_500k_20261102' if arm.endswith('mix50') else 'latest_vosr_four_500k_20261102')
        assert json.loads((source/'COMPLETE.json').read_text())['passed']
        declared=json.loads((source/'CONFIG_BINDINGS.json').read_text())['runs'][arm];config=Path(declared['config'])
        assert digest(config)==declared['config_sha256'];cfg=yaml.safe_load(config.read_text());assert cfg['train']['total_iter']==500000
        checkpoint=source/'experiments'/cfg['name']/'models/net_g_500000.pth';sha=digest(checkpoint)
        target=OUT/'models'/(arm+'.pth');shutil.copy2(checkpoint,target);assert digest(target)==sha==digest(checkpoint)
        conf=OUT/'models'/(arm+'.yml');shutil.copy2(config,conf)
        bindings[arm]=dict(config=str(conf),config_sha256=digest(conf),checkpoint=str(target),checkpoint_sha256=sha,source_checkpoint=str(checkpoint),iteration=500000,label=label,stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    shutil.copy2(BASE/'selection.json',OUT/'selection.json');assert digest(BASE/'selection.json')==digest(OUT/'selection.json')
    samples=json.loads((OUT/'selection.json').read_text())['samples'];assert len(samples)==280 and len({r['id'] for r in samples})==280
    assert sum(e['split']=='val' for e in samples)==37
    save(OUT/'MODEL_BINDINGS.json',bindings);save(OUT/'RUN_BINDING.json',dict(selection_sha256=digest(OUT/'selection.json'),worker_sha256=digest(ROOT/'tools/evaluate_v3pre_three_mojito280.py'),post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'),raw_outputs=1680,post_outputs=1680,train_images=243,val_images=37))


def post(job):
    """@luna-comment Input:one audited raw output. Output:frozen post PNG/receipt. Key Steps:hash/decode,current zero-trial,softcap/geometry checks. Key Params:.75/2/24,no USM. State/Dependencies:CPU-only,raw unchanged."""
    import cv2,numpy as np
    from tools.native_y_curve_four_images import process_pair,CurveTrial
    cv2.setNumThreads(1);arm,e=job;identity=e['id'];raw=OUT/'results'/arm/'final'/(identity+'.png');source_sha=digest(raw)
    core=digest(ROOT/'tools/native_y_curve_four_images.py');path=OUT/'post'/arm/'final'/(identity+'.png');receipt=OUT/'post'/arm/'receipts'/(identity+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['source_sha256']==source_sha and r['core_sha256']==core and digest(path)==r['output_sha256'];return
    image=cv2.imread(str(raw));assert image.shape[:2]==(e['height'],e['width'])
    value,zero,stats=process_pair(image,CurveTrial(16.,48.,0.,0.));assert np.array_equal(value,zero) and np.abs(value.astype(np.int16)-image.astype(np.int16)).max()<=24
    path.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(path),value,[cv2.IMWRITE_PNG_COMPRESSION,3])
    save(receipt,dict(id=identity,source_sha256=source_sha,core_sha256=core,output_sha256=digest(path),shape=list(value.shape),stats=stats))


def worker(arm):
    """@luna-comment Input:one frozen arm. Output:280 raw andpost outputs. Key Steps:delegate accepted inference then2CPU post workers. Key Params:native256/224/crop8/smooth16. State/Dependencies:assigned CUDA device;cached V3 inputs."""
    import tools.evaluate_v3pre_three_mojito280 as accepted
    accepted.OUT=OUT;accepted.worker(arm)
    samples=json.loads((OUT/'selection.json').read_text())['samples']
    with ProcessPoolExecutor(max_workers=2) as pool:
        for n,_ in enumerate(pool.map(post,[(arm,e) for e in samples]),1):print(json.dumps(dict(arm=arm,post_done=n,total=280)),flush=True)


def progress(started,phase,error=None):
    """@luna-comment Input:phase/receipts. Output:Chinese progress androlling ETA. Key Steps:count completed raw/post,walltime extrapolation. Key Params:3360 stages;30s refresh. State/Dependencies:metadata only."""
    counts={k:dict(raw=len(list((OUT/'results'/k/'receipts').glob('*.json'))),post=len(list((OUT/'post'/k/'receipts').glob('*.json')))) for k in LABELS}
    done=sum(sum(v.values()) for v in counts.values());elapsed=time.time()-started;eta=elapsed/done*(3360-done) if done else None
    now=datetime.now(timezone(timedelta(hours=8)));state=dict(phase=phase,counts=counts,completed=done,total=3360,elapsed_seconds=elapsed,eta_seconds=eta,updated=now.isoformat(),error=error)
    save(OUT/'PROGRESS.json',state)
    body='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Mojito280 六模型测试</title><style>body{font:16px/1.7 system-ui;background:#eff3f8;color:#18314b;margin:28px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:15px}article{background:white;padding:20px;border-radius:12px}progress{width:100%}</style><h1>Mojito280 · 六模型最终500K测试</h1><p>复用 V3 输入缓存 · crop8 + smooth16 · 固化Y后处理 · 243训练样本 / 37验证样本</p><p>'+now.strftime('%Y-%m-%d %H:%M:%S')+' 北京时间 · '+html.escape(phase)+' · 预计剩余：'+(f'{eta/60:.1f}分钟' if eta is not None else '等待首张完成')+'</p><div class="grid">'
    for arm,label in LABELS.items():
        c=counts[arm];body+=f'<article><h2>{label}</h2><p>模型输出 {c["raw"]}/280 · 后处理 {c["post"]}/280</p><progress max="560" value="{sum(c.values())}"></progress></article>'
    body+='</div><p><a href="index.html">打开对比页</a>（全部校验完成后生成）</p>'
    if error:body+='<pre>'+html.escape(error)+'</pre>'
    tmp=OUT/'progress.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'progress.html')


def verify(entry):
    """@luna-comment Input:one selected identity. Output:fully verified viewer row/audit. Key Steps:hash all raw/post,strict binding/input/stitch/geometry. Key Params:no silent missing outputs. State/Dependencies:read-only finals."""
    from PIL import Image
    from tools.build_camera_2x2_viewer import file_url
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());run=json.loads((OUT/'RUN_BINDING.json').read_text());identity=entry['id'];images={};records={}
    assert digest(entry['input'])==entry['input_sha256'];old=json.loads((BASE/'manifest.json').read_text());previous=next(r for r in old['rows'] if r['id']==identity)
    images['v3_input']=previous['images']['v3_input']
    for arm in LABELS:
        raw=OUT/'results'/arm/'final'/(identity+'.png');receipt=OUT/'results'/arm/'receipts'/(identity+'.json');r=json.loads(receipt.read_text())
        assert r['binding']==bindings[arm] and r['input_sha256']==entry['input_sha256'] and r['selection_sha256']==run['selection_sha256'] and digest(raw)==r['final_sha256']
        d=r['restoration'];assert d['stitch_policy']=='crop8_smooth16' and d['crop_pixels']==8 and d['blend_pixels']==16 and d['canvas_hw']==[256,256] and d['stride_hw']==[224,224] and not d['ocr_cropping']
        postpath=OUT/'post'/arm/'final'/(identity+'.png');postreceipt=OUT/'post'/arm/'receipts'/(identity+'.json');p=json.loads(postreceipt.read_text())
        assert p['source_sha256']==r['final_sha256'] and p['core_sha256']==run['post_core_sha256'] and digest(postpath)==p['output_sha256']
        for path in [raw,postpath]:
            with Image.open(path) as im:assert im.size==(entry['width'],entry['height'])
        images[arm]=file_url(postpath);images[arm+'_raw']=file_url(raw);records[arm]=dict(raw_sha256=r['final_sha256'],post_sha256=p['output_sha256'])
    return dict(id=identity,name=entry['name']+('【验证37】' if entry['split']=='val' else '【训练243】'),width=entry['width'],height=entry['height'],lq=previous['lq'],images=images,split=entry['split']),dict(id=identity,outputs=records)


def publish():
    """@luna-comment Input:all3360 outputs. Output:Chinese native raw/post viewer andcomplete audit. Key Steps:exact ID inventory,model/source hashes,parallel perimage audit,shared renderer. Key Params:14choices;post defaults. State/Dependencies:mappedY assets,no quality ranking."""
    from tools.build_camera_2x2_viewer import render_html
    samples=json.loads((OUT/'selection.json').read_text())['samples'];ids={e['id'] for e in samples};bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    for arm,b in bindings.items():
        for k in ['config','checkpoint']:assert digest(b[k])==b[k+'_sha256']
        for root in ['results','post']:
            for folder,ext in [('final','png'),('receipts','json')]:assert {q.stem for q in (OUT/root/arm/folder).glob('*.'+ext)}==ids
    with ThreadPoolExecutor(max_workers=4) as pool:checked=list(pool.map(verify,samples))
    models=[dict(key='lq',label='原始 LQ',note='相机原始输入，按原EXIF方向显示'),dict(key='v3_input',label='V3 预处理输入',note='复用既有缓存，不重复处理')]
    for raw in [False,True]:
        models.extend(dict(key=k+('_raw' if raw else ''),label=v+' | 500K | '+('模型原始输出' if raw else '固化后处理'),note='256/224/batch8/crop8/smooth16'+('；未加后处理' if raw else '；Y后处理 .75 / 2 / 24')) for k,v in LABELS.items())
    manifest=dict(title='Mojito280 · 六模型最终500K对比',subtitle='V3M / V5 · guard / L1-only / 50% GT轻退化',notes='六模型均从头训练500K；相同固定数据快照、对比度对齐、有效文字patch采样、noCP和noSDF。\n混合组保留Full辅助loss，关闭可信边缘抵消guard。\n默认展示固化Y后处理结果；下拉菜单可切到模型原始输出、V3输入和LQ。\n243张属于本轮训练，37张是原划分验证；文件名旁有标记，搜索“验证37”可筛选。\n输入复用已完成V3预处理缓存，无新OCR、warp、预处理或推理时对比度匹配。\n图片保存在Y盘；效果由人工评审，不按总loss排名。',defaults='v3m_guard,v5_guard,v3m_mix50,v5_mix50',models=models,rows=[r for r,a in checked])
    save(OUT/'manifest.json',manifest);(OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    save(OUT/'AUDIT.json',dict(passed=True,images=280,raw_outputs=1680,post_outputs=1680,records=[a for r,a in checked]));save(OUT/'COMPLETE.json',dict(passed=True,outputs=3360,completed=datetime.now(timezone(timedelta(hours=8))).isoformat()))


def run(resume=False):
    """@luna-comment Input:authorized run/resume. Output:resident4GPU evaluation anddelivery. Key Steps:exclusive lock,freeze,capacity dispatch,fail reporting,full audit. Key Params:one worker/GPU>=6GiB free. State/Dependencies:owned workers only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not resume:prepare()
    else:assert (OUT/'MODEL_BINDINGS.json').exists()
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=time.time()));started=time.time();pending=list(LABELS);processes={};assigned={}
    try:
        while pending or any(p.poll() is None for p in processes.values()):
            for k,p in processes.items():assert p.poll() in (None,0),(k,p.returncode)
            free=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()];busy={assigned[k] for k,p in processes.items() if p.poll() is None}
            for gpu in range(4):
                if not pending or gpu in busy or free[gpu]<6144:continue
                arm=pending.pop(0);(OUT/'logs').mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                with (OUT/'logs'/(arm+'.log')).open('ab') as f:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--arm',arm],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                processes[arm]=p;assigned[arm]=gpu;save(OUT/'launch'/(arm+'.json'),dict(pid=p.pid,gpu=gpu))
            progress(started,'推理与后处理');time.sleep(15)
        progress(started,'完整性校验');publish();progress(started,'全部完成')
    except Exception as e:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        save(OUT/'FAILED.json',dict(error=repr(e)));progress(started,'异常停止',repr(e));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','resume','worker','publish']);parser.add_argument('--arm',choices=list(LABELS));args=parser.parse_args()
    if args.mode=='worker':assert args.arm;worker(args.arm)
    elif args.mode=='publish':publish()
    else:run(args.mode=='resume')
