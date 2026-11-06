"""Shared V3 preprocessing and four GPU shards for explicit BMP/NV12 sources."""
from pathlib import Path
import sys,os,json,re,copy,time,shutil,subprocess,argparse,fcntl,html
from datetime import datetime,timedelta,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'zoom_plate_two_models_20261105'
LABELS={'r1_width':'V3M · 原始GT+2px · L1+粗细R1 · 500K','mf_uv_mix50':'V3M · L1+Missing/False+UV+50%GT退化混合 · 500K'}


def write(path,image):
    """@luna-comment Input:owned PNG andBGR8. Output:atomic native PNG/hash. Key Steps:PNG3 temp thenrename. Key Params:no lossy export. State/Dependencies:OpenCV,owned files."""
    import cv2
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.tmp.png');assert cv2.imwrite(str(tmp),image,[cv2.IMWRITE_PNG_COMPRESSION,3]);tmp.replace(path);return digest(path)


def prepare():
    """@luna-comment Input:explicit source roots andexact models. Output:frozen28inputs/two500K modelbindings. Key Steps:globonly requested files;validateNV12stride;copyhashconfig/net;bind algorithmcode. Key Params:12+12+4,no inputNV12. State/Dependencies:sources read-only."""
    import yaml
    from PIL import Image
    roots={'zoom_bmp':DATA.parent/'Zoomsr传统结果_对比机','plate_261102':DATA.parent/'车牌相关的输入图/261102','plate_261102_kehu':DATA.parent/'车牌相关的输入图/261102-kehu'}
    samples=[]
    for group,folder in roots.items():
        files=sorted(folder.rglob('*_result.bmp')) if group=='zoom_bmp' else sorted(p for p in folder.rglob('*') if p.is_file() and 'output' in p.name.lower() and p.suffix.lower()=='.nv12')
        assert len(files)==(4 if group.endswith('kehu') else 12),(group,len(files))
        for source in files:
            fmt='BMP' if group=='zoom_bmp' else 'NV12';stride=None
            if fmt=='NV12':
                w,h,stride=map(int,re.search(r'_(\d+)x(\d+)_(\d+)_',source.name).groups());assert w%2==h%2==stride%2==0 and stride>=w and source.stat().st_size==stride*h*3//2
            else:
                with Image.open(source) as image:w,h=image.size
            key=group+'/'+source.stem
            samples.append(dict(id=key,group=group,stem=source.stem,source=str(source),source_sha256=digest(source),format=fmt,width=w,height=h,stride=stride,output=str(OUT/'output'/(key+'.png')),input=str(OUT/'preprocess'/(key+'.png'))))
    assert len(samples)==len({s['id'] for s in samples})==28
    definitions={'r1_width':('v3m_dilate2_l1_r1_width_500k_20261103','launch.yml'),'mf_uv_mix50':('v3m_loss_ablation_500k_20261103','configs/v3m_mf_uv_mix50.yml')};bindings={};(OUT/'models').mkdir(exist_ok=True)
    for arm,(folder,conf) in definitions.items():
        config=DATA/folder/conf;cfg=yaml.safe_load(config.read_text());checkpoint=DATA/folder/'experiments'/cfg['name']/'models/net_g_500000.pth';sha=digest(checkpoint)
        model=OUT/'models'/(arm+'.pth');target=OUT/'models'/(arm+'.yml');shutil.copy2(checkpoint,model);shutil.copy2(config,target)
        assert digest(model)==sha==digest(checkpoint) and digest(target)==digest(config)
        bindings[arm]=dict(config=str(target),config_sha256=digest(target),checkpoint=str(model),checkpoint_sha256=sha,source_checkpoint=str(checkpoint),iteration=500000,label=LABELS[arm])
    save(OUT/'selection.json',dict(samples=samples));save(OUT/'MODEL_BINDINGS.json',bindings)
    code={str(p):digest(ROOT/p) for p in ['tools/restore_crop8_block256.py','tools/native_y_curve_four_images.py','evaluation/document_box_neighbor_20261027/neighbor.py','evaluation/document_box_pre640_revision2_20261027/revision2.py','evaluation/document_box_pre640_astra_20261027/pre640.py']}
    save(OUT/'RUN_BINDING.json',dict(selection_sha256=digest(OUT/'selection.json'),code_sha256=code,nv12_conversion='NV12;explicit filename width/height/stride;OpenCV BT.601 limited-range;stored orientation',groups={k:sum(s['group']==k for s in samples) for k in roots},outputs=168))


def pre_one(entry):
    """@luna-comment Input:frozen source. Output:lossless original/output andV3 preinput withreceipt. Key Steps:verifieddecode,crop paddedNV12 planes,accepted fullV3,geometry/sharedluma checks. Key Params:strength1/chunk128,noOCR. State/Dependencies:CPU1thread,shared cache."""
    import cv2,numpy as np
    from evaluation.document_box_neighbor_20261027 import neighbor
    from evaluation.document_box_pre640_revision2_20261027 import revision2
    from evaluation.document_box_pre640_astra_20261027 import pre640
    cv2.setNumThreads(1);assert digest(entry['source'])==entry['source_sha256'];receipt=OUT/'pre_receipts'/(entry['id']+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['source_sha256']==entry['source_sha256'] and digest(entry['input'])==r['input_sha256'] and digest(entry['output'])==r['output_sha256'];return r
    if entry['format']=='NV12':
        w,h,s=entry['width'],entry['height'],entry['stride'];a=np.fromfile(entry['source'],np.uint8);assert a.size==s*h*3//2
        y=a[:s*h].reshape(h,s)[:,:w];uv=a[s*h:].reshape(h//2,s)[:,:w];image=cv2.cvtColor(np.concatenate([y,uv]),cv2.COLOR_YUV2BGR_NV12)
    else:image=cv2.imread(entry['source'],cv2.IMREAD_UNCHANGED)
    assert image is not None and image.dtype==np.uint8 and image.shape==(entry['height'],entry['width'],3)
    source_sha=write(entry['output'],image);estimate=revision2.estimate(image);q,tail=neighbor.descriptors(estimate['canvas'],estimate['nodes'])
    low,info=neighbor.repair(estimate['fields']['low'],estimate['fields']['high'],q,tail,estimate['nodes'])
    packed=dict(nodes=estimate['nodes'],tables=neighbor.tables(low,estimate['fields']['high']),native_shape=list(image.shape[:2]));value,telemetry=pre640.render(image,packed,strength=1.,chunk=128)
    assert value.shape==image.shape and value.dtype==np.uint8
    delta=value.astype(np.int16)-image.astype(np.int16);assert np.array_equal(delta[...,0],delta[...,1]) and np.array_equal(delta[...,1],delta[...,2])
    r=dict(id=entry['id'],source_sha256=entry['source_sha256'],output_sha256=source_sha,input_sha256=write(entry['input'],value),shape=list(image.shape),preprocessing='V3 full strength1 chunk128',no_ocr=True)
    save(receipt,r);return r


def preprocess():
    """@luna-comment Input:frozen28samples. Output:shared pre-cache. Key Steps:8independentCPUjobs;failvisible. Key Params:one cached result shared by both models. State/Dependencies:owned cache."""
    samples=json.loads((OUT/'selection.json').read_text())['samples']
    with ProcessPoolExecutor(max_workers=8) as pool:
        for n,r in enumerate(pool.map(pre_one,samples),1):print('preprocessed',n,r['id'],flush=True)


def worker(arm,rank):
    """@luna-comment Input:model andmodulo2shard. Output:rawmodel/post PNGs andboundreceipt. Key Steps:strict500K load,wait sharedV3,acceptedrestore,frozenpost,range/hashchecks. Key Params:256/224/batch8/crop8/smooth16,Y.75/2/24. State/Dependencies:one assignedGPU,2CPUthreads."""
    import torch,cv2,numpy as np,yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.restore_crop8_block256 import restore_block
    from tools.native_y_curve_four_images import process_pair,CurveTrial
    cv2.setNumThreads(1);torch.set_num_threads(2);b=json.loads((OUT/'MODEL_BINDINGS.json').read_text())[arm]
    assert digest(b['config'])==b['config_sha256'] and digest(b['checkpoint'])==b['checkpoint_sha256']
    model=build_network(yaml.safe_load(Path(b['config']).read_text())['network_g']).cuda().eval();model.load_state_dict(torch.load(b['checkpoint'],map_location='cpu')['params'],strict=True)
    for n,e in enumerate(json.loads((OUT/'selection.json').read_text())['samples']):
        if n%2!=rank:continue
        receipt=OUT/'receipts'/arm/(e['id']+'.json');raw=OUT/'model'/arm/(e['id']+'.png');post=OUT/'post'/arm/(e['id']+'.png')
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['binding']==b and digest(raw)==old['model_sha256'] and digest(post)==old['post_sha256'];continue
        cached=OUT/'pre_receipts'/(e['id']+'.json');started=time.time()
        while not cached.exists():
            assert not (OUT/'PRE_FAILED.json').exists(),'Preprocessing failed';assert time.time()-started<1800,'Preprocessing timeout';time.sleep(1)
        p=json.loads(cached.read_text());assert digest(e['input'])==p['input_sha256'];image=cv2.imread(e['input']);prediction,details=restore_block(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),model,'cuda');value=cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR)
        enhanced,zero,stats=process_pair(value,CurveTrial(16.,48.,0.,0.));assert np.array_equal(enhanced,zero) and np.abs(enhanced.astype(np.int16)-value.astype(np.int16)).max()<=24
        assert value.shape==enhanced.shape==(e['height'],e['width'],3)
        save(receipt,dict(id=e['id'],binding=b,input_sha256=p['input_sha256'],model_sha256=write(raw,value),post_sha256=write(post,enhanced),restoration=details,shape=list(value.shape),post_stats=stats,seconds=time.time()-started));print(arm,rank,e['id'],flush=True)


def progress(start,phase,error=None):
    """@luna-comment Input:controller state. Output:Chinese stage counters/ETA. Key Steps:count complete receipts only. Key Params:168PNGs,10srefresh. State/Dependencies:owned monitor."""
    pre=len(list((OUT/'pre_receipts').rglob('*.json')));counts={a:len(list((OUT/'receipts'/a).rglob('*.json'))) for a in LABELS};done=2*(pre+sum(counts.values()));elapsed=time.time()-start;remaining=elapsed/max(done,1)*(168-done) if done else None
    save(OUT/'PROGRESS.json',dict(phase=phase,preprocessed=pre,models=counts,completed_stages=done,total=168,elapsed_seconds=elapsed,eta_seconds=remaining,error=error))
    body='<meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>Zoom/车牌两模型进度</title><style>body{font:18px/1.8 system-ui;margin:35px;background:#eef3f8}article{background:white;padding:20px;margin:15px}progress{width:90%}</style><h1>Zoom BMP / 车牌 NV12 · 两模型500K</h1><p>'+html.escape(phase)+f' · 已完成 {done}/168 个阶段文件 · 预计剩余 '+(f'{remaining/60:.1f} 分钟' if remaining else '估算中')+'</p><article>原始 output / V3预处理：'+str(pre)+'/28</article>'
    for a,label in LABELS.items():body+='<article>'+label+f'<p>模型与后处理 {counts[a]}/28</p></article>'
    if error:body+='<pre>'+html.escape(error)+'</pre>'
    tmp=OUT/'progress.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'progress.html')


def audit():
    """@luna-comment Input:complete28x6stageoutputs. Output:exacthash/stitch/shape audit andfile manifest. Key Steps:verifybindings/code,allids,168PNGs andreceiptcoverage. Key Params:no silent missingfiles. State/Dependencies:source/config readonly."""
    import cv2
    selection=json.loads((OUT/'selection.json').read_text());bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());run=json.loads((OUT/'RUN_BINDING.json').read_text());assert digest(OUT/'selection.json')==run['selection_sha256']
    for p,sha in run['code_sha256'].items():assert digest(ROOT/p)==sha
    outputs=[]
    for e in selection['samples']:
        p=json.loads((OUT/'pre_receipts'/(e['id']+'.json')).read_text());assert digest(e['source'])==p['source_sha256'];paths={'output':(e['output'],p['output_sha256']),'preprocess':(e['input'],p['input_sha256'])}
        for a,b in bindings.items():
            assert digest(b['checkpoint'])==b['checkpoint_sha256'] and digest(b['config'])==b['config_sha256']
            r=json.loads((OUT/'receipts'/a/(e['id']+'.json')).read_text());assert r['binding']==b and r['input_sha256']==p['input_sha256'];d=r['restoration'];assert d['stitch_policy']=='crop8_smooth16' and d['weight_min']>0 and d['crop_pixels']==8 and d['blend_pixels']==16
            paths[a+'_model']=(str(OUT/'model'/a/(e['id']+'.png')),r['model_sha256']);paths[a+'_post']=(str(OUT/'post'/a/(e['id']+'.png')),r['post_sha256'])
        for key,(path,sha) in paths.items():
            assert digest(path)==sha and cv2.imread(path).shape==(e['height'],e['width'],3)
        outputs.append(dict(id=e['id'],source=e['source'],stages={k:dict(path=p,sha256=s) for k,(p,s) in paths.items()}))
    assert len(outputs)==28;save(OUT/'RESULTS.json',dict(passed=True,samples=outputs));save(OUT/'COMPLETE.json',dict(passed=True,samples=28,outputs=168,models=2))


def run():
    """@luna-comment Input:authorized28image run. Output:resident8CPU+4GPU pipeline. Key Steps:lock,freeze,dispatch2modelsx2shards,monitor exits,audit. Key Params:GPU0..3>=4GiB,no unrelatedprocesssignals. State/Dependencies:owned child jobs only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
    if not (OUT/'MODEL_BINDINGS.json').exists():prepare()
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=start));command=[sys.executable,'-u',__file__];children=[]
    for gpu,(arm,rank) in enumerate((a,r) for a in LABELS for r in [0,1]):
        free=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()];assert free[gpu]>=4096,(gpu,'insufficient memory')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        with (OUT/(arm+'_'+str(rank)+'.log')).open('ab') as f:p=subprocess.Popen(command+['worker','--arm',arm,'--rank',str(rank)],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        children.append(p)
    with (OUT/'preprocess.log').open('ab') as f:pre=subprocess.Popen(command+['preprocess'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    children.append(pre);save(OUT/'LAUNCHES.json',dict(pids=[p.pid for p in children],gpu_workers=4,cpu_workers=8))
    while any(p.poll() is None for p in children):
        if pre.poll() not in [None,0]:save(OUT/'PRE_FAILED.json',dict(exit_code=pre.returncode))
        failed=[(p.pid,p.poll()) for p in children if p.poll() not in [None,0]]
        if failed:save(OUT/'FAILED.json',dict(processes=failed));progress(start,'异常：查看任务日志',str(failed));raise RuntimeError(failed)
        progress(start,'四组并行处理中');time.sleep(10)
    assert all(p.returncode==0 for p in children);progress(start,'全量校验中');audit();progress(start,'全部完成')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','preprocess','worker']);p.add_argument('--arm',choices=list(LABELS));p.add_argument('--rank',type=int,default=0);args=p.parse_args()
    if args.mode=='worker':worker(args.arm,args.rank)
    elif args.mode=='preprocess':preprocess()
    else:run()
