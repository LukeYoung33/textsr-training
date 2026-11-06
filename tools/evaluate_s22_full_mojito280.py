"""Frozen latest S2.2 Full four-strategy Mojito280 inference and audited viewer."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time

DATA=Path('/AI_DNS/guang.yang/TextSR/data')
BASE=DATA/'v3pre_three_mojito280_20261028'
TRAIN=DATA/'full_s22_training_matrix_20261028'
OUT=DATA/'s22_full_four_mojito280_20261029'
sys.path.insert(0,str(BASE/'_build'))
LABELS={'paired_full':'Real pairs Full','gt_degrade30_full':'30% GT degradation Full','identity_full':'GT identity Full','guard_full':'Trusted edge guard Full'}


def digest(path):
    """@luna-comment Input:read-only file. Output:SHA256. Key Steps:stream4MiB. Key Params:no source writes. State/Dependencies:filesystem."""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304),b''):value.update(chunk)
    return value.hexdigest()


def save(path,value):
    """@luna-comment Input:owned JSON path/value. Output:atomic receipt. Key Steps:sibling replace. Key Params:UTF8. State/Dependencies:owned output only."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def prepare():
    """@luna-comment Input:live declared configs/checkpoints and frozen prior280 selection. Output:four immutable model bindings. Key Steps:latest numeric scan,double hash copies,exact selection copy. Key Params:Full only,no100%arm. State/Dependencies:no training modifications."""
    import yaml
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'MODEL_BINDINGS.json').exists(),'Use resume for existing immutable run'
    source=json.loads((TRAIN/'CONFIG_BINDINGS.json').read_text())['runs'];bindings={}
    (OUT/'models').mkdir(exist_ok=True)
    stitcher=BASE/'_build/tools/restore_crop8_block256.py'
    for key in LABELS:
        declared=source[key];config=Path(declared['config']);assert digest(config)==declared['sha256']
        cfg=yaml.safe_load(config.read_text());assert declared['arm']=='full' and cfg['train']['sdf_boundary_weight']==0
        directory=TRAIN/'training/experiments'/declared['name']/'models'
        files=[p for p in directory.glob('net_g_*.pth') if p.stem[6:].isdigit()]
        checkpoint=max(files,key=lambda p:int(p.stem[6:]));expected=digest(checkpoint)
        target=OUT/'models'/(key+'.pth');shutil.copy2(checkpoint,target)
        assert digest(target)==expected==digest(checkpoint)
        target_config=OUT/'models'/(key+'.yml');shutil.copy2(config,target_config)
        bindings[key]=dict(config=str(target_config),config_sha256=digest(target_config),checkpoint=str(target),checkpoint_sha256=expected,source_checkpoint=str(checkpoint),iteration=int(checkpoint.stem[6:]),stitcher_sha256=digest(stitcher),label=LABELS[key],frozen_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    selection=BASE/'selection.json';shutil.copy2(selection,OUT/'selection.json');assert digest(selection)==digest(OUT/'selection.json')
    samples=json.loads((OUT/'selection.json').read_text())['samples'];assert len(samples)==280 and {e['id'] for e in samples}=={f'{n:03d}' for n in range(1,281)}
    save(OUT/'MODEL_BINDINGS.json',bindings)
    save(OUT/'RUN_BINDING.json',dict(selection_sha256=digest(OUT/'selection.json'),accepted_worker_sha256=digest(BASE/'_build/tools/evaluate_v3pre_three_mojito280.py'),runner_sha256=digest(__file__),total_images=280,total_model_outputs=1120))


def worker(arm):
    """@luna-comment Input:one immutable arm. Output:280 native finals and receipts. Key Steps:process-local output binding then unchanged accepted worker. Key Params:original256/224/batch8/crop8/smooth16. State/Dependencies:selected CUDA device,read-only inputs."""
    import tools.evaluate_v3pre_three_mojito280 as accepted
    accepted.OUT=OUT
    accepted.worker(arm)


def progress(started,phase,processes=None,error=None):
    """@luna-comment Input:controller state/start. Output:four counts,observed ETA and readable live page. Key Steps:receipt counts and elapsed rate. Key Params:1120 total,30s refresh. State/Dependencies:owned metadata only."""
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());counts={key:len(list((OUT/'results'/key/'receipts').glob('*.json'))) for key in LABELS}
    done=sum(counts.values());elapsed=time.time()-started
    eta=elapsed/done*(1120-done) if done else None
    state=dict(phase=phase,counts=counts,completed=done,total=1120,elapsed_seconds=elapsed,eta_seconds=eta,updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),error=error,process_exit_codes={k:p.poll() for k,p in (processes or {}).items()})
    save(OUT/'PROGRESS.json',state)
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>S2.2 Full Mojito280</title><style>body{font:16px/1.7 system-ui;margin:30px auto;max-width:1100px;background:#f4f7fa;color:#18283a}section{padding:18px;background:white;margin:15px;border-radius:10px}progress{width:100%;height:24px}</style><h1>S2.2 Full · Mojito280 四组验证</h1><p>复用原 V3 缓存输入 · 256/224/batch8 · crop8 + smooth16 · 100%退化组不参与</p>'
    for key in LABELS:body+=f'<section><h2>{html.escape(LABELS[key])} | {bindings[key]["iteration"]//1000}K</h2><progress value="{counts[key]}" max="280"></progress><p>{counts[key]} / 280</p></section>'
    body+=f'<p>阶段：{html.escape(phase)} · 总计 {done}/1120 · 预计剩余：'+(f'{eta/60:.1f} 分钟' if eta is not None else '统计中')+'</p><p>'+state['updated_at']+'</p>'
    if error:body+='<pre>'+html.escape(error)+'</pre>'
    if phase=='complete':body+='<p><a href="viewer.html">打开完整对比页</a></p>'
    body+='</html>';temporary=OUT/'progress.html.tmp';temporary.write_text(body,encoding='utf-8');temporary.replace(OUT/'progress.html')


def verify(entry,bindings,selection_hash,previous):
    """@luna-comment Input:one frozen sample/bindings. Output:audited native row and receipt hashes. Key Steps:all four output+receipt+selection/input identities and PIL geometry. Key Params:complete ID set checked by publisher. State/Dependencies:read-only NAS."""
    from PIL import Image
    from tools.build_camera_2x2_viewer import file_url
    old=previous[entry['id']]
    images={'v3_input':old['images']['v3_input']};audit={}
    assert digest(entry['input'])==entry['input_sha256']
    for key in LABELS:
        path=OUT/'results'/key/'final'/(entry['id']+'.png');receipt=OUT/'results'/key/'receipts'/(entry['id']+'.json');r=json.loads(receipt.read_text())
        assert r['binding']==bindings[key] and r['input_sha256']==entry['input_sha256'] and r['selection_sha256']==selection_hash
        assert digest(path)==r['final_sha256']
        with Image.open(path) as image:assert image.size==(entry['width'],entry['height'])
        images[key]=file_url(path);audit[key]=dict(final_sha256=r['final_sha256'],receipt_sha256=digest(receipt))
    return dict(id=entry['id'],name=entry['name'],width=entry['width'],height=entry['height'],lq=old['lq'],images=images),dict(id=entry['id'],outputs=audit)


def publish():
    """@luna-comment Input:1120 complete finals. Output:audited280 six-choice native viewer. Key Steps:full ID/hash/geometry gates,accepted shared renderer. Key Params:distinct actual checkpoint labels,no GT quality score. State/Dependencies:mappedY assets,owned metadata."""
    from tools.build_camera_2x2_viewer import render_html
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());samples=json.loads((OUT/'selection.json').read_text())['samples'];identities={e['id'] for e in samples}
    for key in LABELS:
        for folder,suffix in [('final','png'),('receipts','json')]:assert {p.stem for p in (OUT/'results'/key/folder).glob('*.'+suffix)}==identities
        for name in ('config','checkpoint'):assert digest(bindings[key][name])==bindings[key][name+'_sha256']
    previous={r['id']:r for r in json.loads((BASE/'manifest.json').read_text())['rows']};selection_hash=digest(OUT/'selection.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda e:verify(e,bindings,selection_hash,previous),samples))
    models=[dict(key='lq',label='Original LQ',note='Original camera input;display orientation only'),dict(key='v3_input',label='V3 preprocessed input',note='Exact reused completed V3 cache;no second preprocessing')]
    models.extend(dict(key=k,label=LABELS[k]+f' | {bindings[k]["iteration"]//1000}K',note='Full | new S2.2 training targets | native256/224/batch8/crop8/smooth16') for k in LABELS)
    manifest=dict(title='Mojito280 | S2.2 Full four strategies',subtitle='Latest frozen checkpoints | native synchronized comparison',models=models,rows=[r for r,a in rows],defaults=','.join(LABELS),notes='Four Full strategies only;100% GT degradation excluded. Latest checkpoints are frozen and actual iterations labeled;stages differ. Same original280 V3 input cache and native crop8/smooth16 inference. No new preprocessing,tone,warp or OCR. Full-resolution assets reside on mappedY NAS. Quality is user-reviewed;no unreliable pseudo-GT scoring.')
    save(OUT/'manifest.json',manifest);(OUT/'viewer.html').write_text(render_html(manifest),encoding='utf-8')
    save(OUT/'AUDIT.json',dict(passed=True,images=280,model_outputs=1120,selection_sha256=digest(OUT/'selection.json'),records=[a for r,a in rows]))


def run(resume=False):
    """@luna-comment Input:explicit run/resume command. Output:four detached GPU workers and complete audit. Key Steps:immutable prepare,one process perGPU,live progress,fail on worker error,publish only complete. Key Params:training never signaled;resume validates cached receipts. State/Dependencies:owned lock/output."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not resume:prepare()
    else:assert (OUT/'MODEL_BINDINGS.json').exists()
    started=time.time();processes={};streams=[]
    try:
        for gpu,key in enumerate(LABELS):
            folder=OUT/'logs';folder.mkdir(exist_ok=True);stream=(folder/(key+'.log')).open('ab');streams.append(stream)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--arm',key]
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            processes[key]=process;save(OUT/'launch'/(key+'.json'),dict(pid=process.pid,gpu=gpu,command=command))
        while any(p.poll() is None for p in processes.values()):
            for key,p in processes.items():assert p.poll() in (None,0),(key,p.returncode)
            progress(started,'running',processes);time.sleep(15)
        assert all(p.returncode==0 for p in processes.values())
        progress(started,'auditing',processes);publish();progress(started,'complete',processes)
        save(OUT/'COMPLETE.json',dict(passed=True,total_outputs=1120,completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    except Exception as error:
        for process in processes.values():
            if process.poll() is None:process.terminate()
        for process in processes.values():
            if process.poll() is None:process.wait(timeout=20)
        progress(started,'failed',processes,repr(error));save(OUT/'FAILED.json',dict(error=repr(error)));raise
    finally:
        for stream in streams:stream.close()


def main():
    """@luna-comment Input:explicit controller/worker/publish CLI. Output:owned evaluation. Key Steps:dispatch validated mode and arm. Key Params:resume never refreshes bindings. State/Dependencies:no production interface change."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','resume','worker','publish'));parser.add_argument('--arm',choices=tuple(LABELS));args=parser.parse_args()
    if args.mode=='worker':
        assert args.arm;worker(args.arm)
    elif args.mode=='publish':publish()
    else:run(args.mode=='resume')


if __name__=='__main__':main()
