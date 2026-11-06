"""Apply frozen V11 to three approved full-frame sets, isolated sequential jobs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCES=[
    ('ppt_paste_fixed',ROOT/'results/remote58_local_review/preprocess_ppt_paste_seam_fixed_v1/paste_seam_fixed_v1'),
    ('height128_paste_fixed',ROOT/'results/remote58_local_review/preprocess_height128_native96x480/height128_native96x480_paste_fixed_v1'),
    ('blackwhite',Path('C:/Users/Guang.Yang/Pictures/联想测试用例/黑白板')),
]
IMPLEMENTATIONS=['tools/postprocess_ppt_dehalo_20260924.py','tools/postprocess_ppt_normal_dehalo_v4.py','tools/postprocess_ppt_surface_v11.py']


def save_json(path,data):
    """@luna-comment Input: path/data. Output: atomic status or receipt JSON.
    Key Steps: write sibling temporary then replace. Key Params: UTF8.
    State/Dependencies: only new batch output files may be replaced.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def process_job(manifest,index):
    """@luna-comment Input: frozen manifest/index. Output: one full result/diagnostics.
    Key Steps: hash check, frozen V11 functions, decode verification and receipt.
    Key Params: bounded-detail final,4 OpenCV threads, no retuning/fallback.
    State/Dependencies: isolated child; original immutable; output must not exist.
    """
    import cv2
    import numpy as np
    from tools.postprocess_ppt_dehalo_20260924 import prepare,write_png
    from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
    from tools.postprocess_ppt_surface_v11 import reconstruct
    from tools.postprocess_ppt_contour_v6 import read_image
    cv2.setNumThreads(4)
    job=manifest['jobs'][index];source=Path(job['source']);dest=Path(job['output']);start=time.time()
    receipt=Path(job['receipt'])
    try:
        if dest.exists():raise FileExistsError(str(dest))
        for rel,digest in manifest['implementation_hashes'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,f'Frozen code changed: {rel}'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        src=read_image(source);prepared=prepare(src,robust_seed=True)
        _,maps,_=normal_correct(prepared)
        base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0])
        (_,result),debug,report=reconstruct(base,prepared,maps['correction_weight'])
        assert result.shape==src.shape
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        write_png(dest,result);assert np.array_equal(read_image(dest),result)
        d=receipt.parent/source.stem
        for key in ('weight','anchors','protected'):
            write_png(d/(key+'.png'),np.clip(np.rint(debug[key]*255),0,255).astype(np.uint8))
        write_png(d/'surface.png',np.clip(np.rint(debug['target']),0,255).astype(np.uint8))
        save_json(receipt,dict(status='complete',**job,source_sha256=digest,shape=list(result.shape),elapsed_seconds=time.time()-start,algorithm_report=report))
    except Exception:
        save_json(receipt,dict(status='failed',**job,elapsed_seconds=time.time()-start,error=traceback.format_exc()))
        raise
    finally:
        gc.collect()


def main():
    """@luna-comment Input: prepare/run/child CLI. Output:72-job manifest/status/log.
    Key Steps: root-only discovery, sequential subprocesses, incremental statuses.
    Key Params: exclusive prepare directory, one worker, continue with recorded errors.
    State/Dependencies: authorized local batch only; no external writes or model calls.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--index',type=int)
    args=p.parse_args();folder=args.output.resolve();mp=folder/'manifest.json'
    if args.prepare:
        folder.mkdir(parents=True,exist_ok=False);jobs=[]
        for group,source in SOURCES:
            paths=sorted(source.glob('*.png'))
            if len(paths)!=24:raise ValueError(f'Expected24 full frames in {source}, got{len(paths)}')
            for path in paths:
                jobs.append(dict(group=group,source=str(path),output=str(folder/group/path.name),receipt=str(folder/'diagnostics'/group/(path.stem+'.json'))))
        manifest=dict(algorithm='V11/P13 bounded-detail final',created=time.strftime('%Y-%m-%d %H:%M:%S'),jobs=jobs,implementation_hashes={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in IMPLEMENTATIONS})
        save_json(mp,manifest);save_json(folder/'status.json',dict(state='prepared',total=len(jobs),completed=0,failed=0))
        print(json.dumps(dict(manifest=str(mp),total=len(jobs))));return
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    if args.index is not None:process_job(manifest,args.index);return
    complete=0;failures=[];started=time.time()
    for index,job in enumerate(manifest['jobs']):
        save_json(folder/'status.json',dict(state='running',total=len(manifest['jobs']),completed=complete,failed=len(failures),current=index+1,current_source=job['source'],started_at=started,updated_at=time.time()))
        print(f"START {index+1}/{len(manifest['jobs'])} {job['source']}",flush=True)
        with (folder/'jobs.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(folder),'--index',str(index)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0:complete+=1
        else:failures.append(dict(index=index,source=job['source'],returncode=result.returncode))
        print(f"DONE {index+1}: exit={result.returncode}; successful={complete}; failed={len(failures)}",flush=True)
    save_json(folder/'status.json',dict(state='finished' if not failures else 'finished_with_errors',total=len(manifest['jobs']),completed=complete,failed=len(failures),failures=failures,elapsed_seconds=time.time()-started,updated_at=time.time()))


if __name__=='__main__':main()
