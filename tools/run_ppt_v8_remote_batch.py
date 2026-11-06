"""Frozen V8 multi-process application to user-provided shared PNG groups."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    """@luna-comment Input:file. Output:SHA256. Key Steps:stream1MiB chunks.
    Key Params:binary bytes. State/Dependencies:read-only filesystem.
    """
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def save(path,data):
    """@luna-comment Input:path/object. Output:atomic UTF8 JSON.
    Key Steps:write temporary then replace. Key Params:unique per-job filenames.
    State/Dependencies:parent owns shared status, worker owns receipt.
    """
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(path)


def process(folder,manifest,index):
    """@luna-comment Input:frozen manifest/job. Output:V8 PNG/maps and receipt.
    Key Steps:hash gate, prepare/V4/outward_feather, PNG identity, source rehash.
    Key Params:unchanged V8,OpenCV4 threads. State:one isolated child per image.
    """
    import cv2
    import numpy as np
    from tools.postprocess_ppt_dehalo_20260924 import prepare,write_png
    from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
    from tools.postprocess_ppt_outward_v8 import outward_feather
    from tools.postprocess_ppt_contour_v6 import read_image
    cv2.setNumThreads(4); job=manifest['jobs'][index]; source=Path(job['source']); start=time.time()
    receipt=folder/'diagnostics'/job['group']/(source.stem+'.json')
    try:
        for rel,value in manifest['code_hashes'].items(): assert digest(ROOT/rel)==value,rel
        assert digest(source)==job['sha256']; dest=folder/job['output']; assert not dest.exists()
        src=read_image(source); prepared=prepare(src,robust_seed=True); _,maps,_=normal_correct(prepared)
        base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0])
        result,debug,report=outward_feather(base,prepared,maps['correction_weight'])
        assert result.shape==src.shape
        write_png(dest,result); assert np.array_equal(read_image(dest),result)
        d=receipt.parent/source.stem
        for key,gain in [('patch',255),('stroke_protect',255),('rho',12),('q',255)]:
            write_png(d/(key+'.png'),np.clip(np.rint(debug[key]*gain),0,255).astype(np.uint8))
        for key in ('full_before','full_after'):
            write_png(d/(key+'.png'),np.clip(np.rint(128+debug[key]*4),0,255).astype(np.uint8))
        assert digest(source)==job['sha256']
        save(receipt,dict(status='complete',algorithm='V8/P10',pid=os.getpid(),**job,shape=list(result.shape),output_sha256=digest(dest),elapsed_seconds=time.time()-start,report=report))
    except Exception:
        save(receipt,dict(status='failed',**job,error=traceback.format_exc())); raise


def child(folder,index):
    """@luna-comment Input:output/index. Output:isolated child exit and log.
    Key Steps:subprocess per frame. Key Params:bounded parent threadpool.
    State/Dependencies:no shared image buffers; child exits release memory.
    """
    with (folder/'logs'/f'{index:03d}.log').open('w') as log:
        return subprocess.run([sys.executable,__file__,'--output',str(folder),'--job',str(index)],stdout=log,stderr=subprocess.STDOUT).returncode


def main():
    """@luna-comment Input:prepare/launch/run/job CLI. Output:72-file parallel batch.
    Key Steps:exclusive manifest,detached parent,bounded subprocesses,atomic counts.
    Key Params:6 workers,groups explicit. State:no overwrite or silent fallback.
    """
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--input',type=Path)
    p.add_argument('--prepare',action='store_true'); p.add_argument('--launch',action='store_true'); p.add_argument('--run',action='store_true'); p.add_argument('--job',type=int); p.add_argument('--workers',type=int,default=6)
    args=p.parse_args(); folder=args.output.resolve()
    if args.prepare:
        folder.mkdir(parents=True,exist_ok=False); (folder/'logs').mkdir(); jobs=[]
        for group in ('blackwhite','native96x480','paste_fixed'):
            paths=sorted((args.input/group).glob('*.png')); assert len(paths)==24,(group,len(paths))
            for path in paths: jobs.append(dict(group=group,source=str(path.resolve()),sha256=digest(path),output=f'{group}/{path.stem}_dehalo_v8.png'))
        hashes={str(path.relative_to(ROOT)):digest(path) for path in (ROOT/'tools').glob('*.py')}
        save(folder/'manifest.json',dict(algorithm='V8/P10',jobs=jobs,code_hashes=hashes)); print('PREPARED',len(jobs),flush=True); return
    manifest=json.loads((folder/'manifest.json').read_text())
    if args.job is not None: process(folder,manifest,args.job); return
    if args.launch:
        assert not (folder/'launch.json').exists()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with (folder/'batch.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,__file__,'--output',str(folder),'--run','--workers',str(args.workers)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        save(folder/'launch.json',dict(pid=proc.pid,workers=args.workers)); print('STARTED',proc.pid,flush=True); return
    if args.run:
        assert not (folder/'status.json').exists(); start=time.time(); complete=0; failed=[]
        save(folder/'status.json',dict(state='running',completed=0,failed=0,total=len(manifest['jobs']),workers=args.workers,pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(child,folder,i):i for i in range(len(manifest['jobs']))}
            for f in as_completed(futures):
                code=f.result(); index=futures[f]
                if code==0: complete+=1
                else: failed.append(dict(index=index,exit=code))
                save(folder/'status.json',dict(state='running',completed=complete,failed=len(failed),total=len(futures),workers=args.workers,elapsed_seconds=time.time()-start,errors=failed))
                print(f'DONE {complete}/{len(futures)} FAILED {len(failed)} JOB {index}',flush=True)
        save(folder/'status.json',dict(state='finished' if not failed else 'finished_with_errors',completed=complete,failed=len(failed),total=len(futures),workers=args.workers,elapsed_seconds=time.time()-start,errors=failed))


if __name__=='__main__': main()
