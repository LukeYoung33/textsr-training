"""Four disjoint image shards, two frozen500K models, reused camera V3 inputs."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_uniform20_steep_scurve import save, digest

DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'titians_two_500k_20261105'
CACHE = DATA/'titians_camera_three_20261030'
ARMS = {
    'v3m_dilate2_r1_width': 'V3M · 原始GT+2px · L1+粗细R1 · 500K',
    'v3m_mf_uv_mix50': 'V3M · L1+Missing/False+UV+50%GT退化混合 · 500K',
}


def prepare_input(item):
    """@luna-comment Input:number/uploaded original. Output:canonical V3 cache row. Key Steps:delegate accepted camera V3 preparation with own output root. Key Params:no OCR,8 CPU workers. State/Dependencies:uploaded hash-verified originals;historical inputs read-only."""
    from tools import evaluate_titians_camera_three as camera
    camera.OUT = OUT
    return camera.prepare(item)


def prepare():
    """@luna-comment Input:W source hashes,cached camera selection,two final weights. Output:frozen bindings and four disjoint selections. Key Steps:exact source identity,copy/hash checkpoints,partition IDs. Key Params:500K only,4 shards. State/Dependencies:source/cache/training read-only;owned OUT."""
    import yaml
    inventory = json.loads((OUT/'source_inventory.json').read_text())
    files = inventory['files']
    cached = json.loads((CACHE/'selection.json').read_text())['samples']
    assert len(files) == 300 and len(cached) == 160
    assert len({f['name'] for f in files}) == len(files)
    byname = {f['name']: f for f in files}
    cache = {s['name']:s for s in cached}
    rows={};missing=[]
    for number,f in enumerate(files,1):
        ident=f'{number:03d}'
        if f['name'] in cache and cache[f['name']]['source_sha256']==f['sha256']:
            row=dict(cache[f['name']],id=ident)
            assert Path(row['input']).is_file() and Path(row['lq']).is_file()
            rows[ident]=row
        else:
            source=OUT/'originals'/f['name']
            assert digest(source)==f['sha256'],f['name']
            missing.append((number,source))
    save(OUT/'PREPROCESS_STATUS.json',dict(completed=len(rows),total=len(files),reused=len(files)-len(missing),new_done=0))
    prep_started=time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(prepare_input,missing):
            rows[row['id']]=row
            save(OUT/'PREPROCESS_STATUS.json',dict(completed=len(rows),total=len(files),reused=len(files)-len(missing),new_done=len(rows)-(len(files)-len(missing))))
            progress(prep_started,'补充新增图像的V3预处理')
    cached=[rows[f'{n:03d}'] for n in range(1,len(files)+1)]
    (OUT/'models').mkdir(exist_ok=True)
    bindings = {}
    for arm in ARMS:
        if arm == 'v3m_dilate2_r1_width':
            source = DATA/'v3m_dilate2_l1_r1_width_500k_20261103'
            config = source/'launch.yml'
        else:
            source = DATA/'v3m_loss_ablation_500k_20261103'
            declared = json.loads((source/'CONFIG_BINDINGS.json').read_text())['runs'][arm]
            config = Path(declared['config'])
            assert digest(config) == declared['config_sha256']
        cfg = yaml.safe_load(config.read_text())
        checkpoint = source/'experiments'/cfg['name']/'models/net_g_500000.pth'
        before = digest(checkpoint)
        model = OUT/'models'/(arm+'.pth')
        conf = OUT/'models'/(arm+'.yml')
        shutil.copy2(checkpoint, model)
        shutil.copy2(config, conf)
        assert digest(model) == before == digest(checkpoint)
        bindings[arm] = dict(config=str(conf),config_sha256=digest(conf),checkpoint=str(model),
            checkpoint_sha256=before,source_checkpoint=str(checkpoint),iteration=500000,
            label=ARMS[arm],stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
    save(OUT/'selection.json', dict(samples=cached))
    save(OUT/'MODEL_BINDINGS.json', bindings)
    save(OUT/'RUN_BINDING.json', dict(source_inventory_sha256=digest(OUT/'source_inventory.json'),
        selection_sha256=digest(OUT/'selection.json'),post_core_sha256=digest(ROOT/'tools/native_y_curve_four_images.py'),
        cache_reused=160,new_preprocessed=len(missing),ocr_run=False,shards=4,images=300,outputs=1200))
    for shard in range(4):
        job = OUT/'shards'/str(shard)
        job.mkdir(parents=True,exist_ok=True)
        samples = cached[shard::4]
        assert len(samples) == 75
        save(job/'selection.json', dict(samples=samples))
        save(job/'MODEL_BINDINGS.json', bindings)
        for folder in ['models','results']:
            (OUT/folder).mkdir(exist_ok=True)
            (job/folder).symlink_to(OUT/folder,target_is_directory=True)


def worker(shard):
    """@luna-comment Input:disjoint75 images and assigned GPU. Output:two models' raw/post predictions. Key Steps:accepted strict native worker then frozen CPU post. Key Params:256/224/batch8/crop8/smooth16,Y .75/2/24. State/Dependencies:one worker/GPU;shared output IDs disjoint."""
    from tools import evaluate_v3pre_three_mojito280 as accepted
    from tools import evaluate_latest_six_mojito280 as postcore
    job = OUT/'shards'/str(shard)
    accepted.OUT = job
    postcore.OUT = OUT
    samples = json.loads((job/'selection.json').read_text())['samples']
    for arm in ARMS:
        accepted.worker(arm)
        for entry in samples:
            postcore.post((arm,entry))
        print(json.dumps(dict(shard=shard,arm=arm,phase='complete',images=len(samples))),flush=True)


def progress(started,phase,error=None):
    """@luna-comment Input:controller wall time/phase. Output:Chinese progress and ETA. Key Steps:count only new receipts,observed rate,atomic metadata/page. Key Params:1200 total stages,15s refresh. State/Dependencies:owned evaluation metadata."""
    counts = {arm:{kind:len(list((OUT/folder/arm/'receipts').glob('*.json'))) for kind,folder in [('raw','results'),('post','post')]} for arm in ARMS}
    done = sum(sum(c.values()) for c in counts.values())
    elapsed = time.time()-started
    eta = elapsed/max(done,1)*(1200-done) if done else None
    now = datetime.now(timezone(timedelta(hours=8)))
    finish = (now+timedelta(seconds=eta)).strftime('%m-%d %H:%M') if eta is not None else '等待首批完成'
    save(OUT/'PROGRESS.json',dict(phase=phase,counts=counts,completed=done,total=1200,elapsed_seconds=elapsed,eta_seconds=eta,finish=finish,updated=now.isoformat(),error=error))
    body = '<meta charset="utf-8"><meta http-equiv="refresh" content="15"><style>body{font:18px system-ui;background:#eff3f8;color:#19324c;padding:30px}article{background:white;padding:20px;margin:15px 0;border-radius:12px}progress{width:100%}</style><h1>Titians camera · 两模型500K · 四进程</h1>'
    body += '<p>'+phase+' · 预计完成 '+finish+' · '+now.strftime('%H:%M:%S')+'</p><p>300张：复用160张V3缓存，补做140张V3预处理，不重做OCR。两个模型，均保留原始及固化后处理版本。</p>'
    if (OUT/'PREPROCESS_STATUS.json').exists():
        prep=json.loads((OUT/'PREPROCESS_STATUS.json').read_text())
        body += f'<p>输入准备 {prep["completed"]}/300 · 复用 {prep["reused"]} · 新增V3处理 {prep["new_done"]}/140</p>'
    for arm,label in ARMS.items():
        c=counts[arm]
        body += f'<article><b>{label}</b><p>模型输出 {c["raw"]}/300 · 后处理 {c["post"]}/300</p><progress value="{sum(c.values())}" max="600"></progress></article>'
    body += '<p><a href="index.html">打开完成后的对比页</a></p>'
    if error:
        import html
        body += '<pre>'+html.escape(error)+'</pre>'
    temp=OUT/'progress.tmp';temp.write_text(body,encoding='utf-8');temp.replace(OUT/'progress.html')


def audit_publish():
    """@luna-comment Input:1200 new outputs and frozen shard receipts. Output:complete audit and Chinese comparison. Key Steps:input/model/selection/post hashes,native geometry,render only complete results. Key Params:no old outputs rechecked. State/Dependencies:PIL/accepted viewer,owned page."""
    from PIL import Image
    from tools.build_camera_2x2_viewer import render_html,file_url
    samples=json.loads((OUT/'selection.json').read_text())['samples']
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    run=json.loads((OUT/'RUN_BINDING.json').read_text())
    for b in bindings.values():
        assert digest(b['checkpoint'])==digest(b['source_checkpoint'])==b['checkpoint_sha256']
        assert digest(b['config'])==b['config_sha256']
    ids={s['id'] for s in samples}
    for arm in ARMS:
        for folder in ['results','post']:
            assert {f.stem for f in (OUT/folder/arm/'final').glob('*.png')}==ids
            assert {f.stem for f in (OUT/folder/arm/'receipts').glob('*.json')}==ids
    rows=[];records=[]
    for n,s in enumerate(samples):
        assert digest(s['input'])==s['input_sha256']
        shard=n%4
        selection_hash=digest(OUT/'shards'/str(shard)/'selection.json')
        images={'v3_input':file_url(s['input'])}
        for arm,b in bindings.items():
            raw=OUT/'results'/arm/'final'/(s['id']+'.png')
            post=OUT/'post'/arm/'final'/(s['id']+'.png')
            r=json.loads((OUT/'results'/arm/'receipts'/(s['id']+'.json')).read_text())
            p=json.loads((OUT/'post'/arm/'receipts'/(s['id']+'.json')).read_text())
            assert r['binding']==b and r['selection_sha256']==selection_hash
            assert r['input_sha256']==s['input_sha256'] and r['restoration']['stitch_policy']=='crop8_smooth16'
            assert digest(raw)==r['final_sha256']==p['source_sha256']
            assert digest(post)==p['output_sha256'] and p['core_sha256']==run['post_core_sha256']
            for path in [raw,post]:
                with Image.open(path) as im:
                    assert im.size==(s['width'],s['height']);im.verify()
            images[arm+'_raw']=file_url(raw);images[arm]=file_url(post)
            records.append(dict(id=s['id'],arm=arm,raw=str(raw),post=str(post),raw_sha256=r['final_sha256'],post_sha256=p['output_sha256']))
        rows.append(dict(id=s['id'],name=s['name'],width=s['width'],height=s['height'],lq=file_url(s['lq']),images=images))
    models=[dict(key='lq',label='原始LQ',note='EXIF方向归一化'),dict(key='v3_input',label='V3预处理输入',note='复用已核验的历史缓存')]
    for arm,label in ARMS.items():
        for suffix,name in [('_raw','模型原始输出'),('','固化后处理')]:
            models.append(dict(key=arm+suffix,label=label+' | '+name,note='256/224/batch8/crop8/smooth16'+('；未加后处理' if suffix else '；Y .75/2/24')))
    manifest=dict(title='Titians camera · 两模型500K对比',subtitle='300张 · 原始GT+2px粗细R1 vs Missing/False+UV混合退化',models=models,rows=rows,defaults='v3_input,v3m_dilate2_r1_width_raw,v3m_mf_uv_mix50_raw,lq',notes='四进程按图分组；两个模型均最终500K。同一V3输入、crop8+smooth16。默认原始模型输出，后处理版本从下拉选择；未做额外OCR、warp或对比度对齐。')
    save(OUT/'OUTPUTS.json',dict(samples=records))
    save(OUT/'manifest.json',manifest)
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    save(OUT/'AUDIT.json',dict(passed=True,images=300,models=2,outputs=1200,old_results_rechecked=False,source_inventory_sha256=run['source_inventory_sha256']))
    save(OUT/'COMPLETE.json',dict(passed=True,outputs=1200,at=time.time()))


def run():
    """@luna-comment Input:authorized two-model camera batch. Output:four resident workers plus audited page. Key Steps:exclusive lock,freeze,one shard/GPU,check exits,final audit. Key Params:four GPUs,6GiB gate. State/Dependencies:existing training never signaled;failure explicit."""
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=started))
    try:
        assert not (OUT/'launch').exists(),'Existing run found; inspect before re-launching'
        progress(started,'核对输入并准备V3缓存')
        prepare()
        pending=list(range(4));active={}
        while pending or active:
            for shard,(p,gpu) in list(active.items()):
                if p.poll() is not None:
                    assert p.returncode==0,(shard,p.returncode)
                    del active[shard]
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            gpus=sorted([tuple(map(int,l.split(','))) for l in inventory.splitlines()],key=lambda x:x[2])
            busy={gpu for p,gpu in active.values()}
            for gpu,free,util in gpus:
                if not pending or gpu in busy or free<6144:continue
                shard=pending.pop(0);(OUT/'logs').mkdir(exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                with (OUT/'logs'/f'shard{shard}.log').open('ab') as log:
                    p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(shard)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                save(OUT/'launch'/f'{shard}.json',dict(pid=p.pid,gpu=gpu,images=75,models=2))
                active[shard]=(p,gpu)
            progress(started,'四进程推理及固化后处理')
            time.sleep(15)
        progress(started,'校验新增640张结果')
        audit_publish()
        progress(started,'全部完成')
    except Exception as error:
        save(OUT/'FAILED.json',dict(error=repr(error)))
        progress(started,'异常停止',repr(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('--shard',type=int,choices=range(4));args=parser.parse_args()
    worker(args.shard) if args.mode=='worker' else run()
