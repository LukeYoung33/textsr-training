"""Portable, hash-pinned V12a batch packaging and isolated parallel execution."""
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
import zipfile

CODE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE_ROOT))
SOURCES=[
    ('ppt_paste_fixed','results/remote58_local_review/preprocess_ppt_paste_seam_fixed_v1/paste_seam_fixed_v1'),
    ('height128_paste_fixed','results/remote58_local_review/preprocess_height128_native96x480/height128_native96x480_paste_fixed_v1'),
    ('blackwhite','C:/Users/Guang.Yang/Pictures/联想测试用例/黑白板'),
]
MODULES=['postprocess_ppt_dehalo_20260924.py','postprocess_ppt_normal_dehalo_v4.py',
         'postprocess_ppt_contour_v5.py','postprocess_ppt_contour_v6.py',
         'postprocess_ppt_crossseam_v9.py','postprocess_ppt_wide_v10.py',
         'postprocess_ppt_direct_rgb_v12.py','run_ppt_v12a_remote_batch.py']


def sha(path):
    """@luna-comment Input: file path. Output: SHA256; stream1MiB chunks.
    Key Params: binary bytes unchanged. State/Dependencies: read-only filesystem.
    """
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def save_json(path,data):
    """@luna-comment Input: path/data. Output: atomic JSON status/receipt.
    Key Steps: temporary sibling then replace. Key Params: UTF8.
    State/Dependencies: unique job or parent owns each output path.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8'); temp.replace(path)


def pack(root):
    """@luna-comment Input: exclusive local run directory. Output: portable ZIP/manifest.
    Key Steps: top-level PNG selection, hashes, fixed code and reference pair.
    Key Params:24 images/group, no nested debug. State: source inputs read-only.
    """
    root.mkdir(parents=True,exist_ok=False); jobs=[]
    with zipfile.ZipFile(root/'payload.zip','x',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for group,location in SOURCES:
            source=Path(location) if Path(location).is_absolute() else CODE_ROOT/location
            files=sorted(source.glob('*.png'))
            if len(files)!=24: raise ValueError(f'Expected24 images: {source}, got{len(files)}')
            for path in files:
                rel=f'inputs/{group}/{path.name}'; digest=sha(path)
                archive.write(path,rel)
                jobs.append(dict(group=group,source=rel,original=str(path.resolve()),source_sha256=digest,output=f'outputs/{group}/{path.stem}_dehalo_v12a.png'))
            print(f'PACKED {group}: {len(files)}',flush=True)
        hashes={}
        for name in MODULES:
            rel='tools/'+name; archive.write(CODE_ROOT/rel,'code/'+rel); hashes[rel]=sha(CODE_ROOT/rel)
        for name in ('test_ppt_direct_rgb_v12.py','test_ppt_wide_v10.py'):
            archive.write(CODE_ROOT/'tests'/name,'code/tests/'+name)
        source=Path('C:/Users/Guang.Yang/Pictures/联想测试用例/PPT/IMG_20260610_052927799_preprocessInput.png')
        expected=CODE_ROOT/'results/ppt_dehalo_20260924/sample_v12a_direct_rgb/IMG_20260610_052927799_preprocessInput_dehalo.png'
        archive.write(source,'reference/input.png'); archive.write(expected,'reference/expected.png')
        manifest=dict(algorithm='V12a/P14a',continuous=True,created=time.time(),jobs=jobs,implementation_hashes=hashes,
                      reference_hashes={'reference/input.png':sha(source),'reference/expected.png':sha(expected)})
        save_json(root/'manifest.json',manifest); archive.writestr('manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps(dict(jobs=len(jobs),bytes=(root/'payload.zip').stat().st_size,payload_sha256=sha(root/'payload.zip'))),flush=True)


def verify(root,manifest):
    """@luna-comment Input: deployed manifest. Output: fail on modified code/input.
    Key Steps: validate every frozen SHA. Key Params: no mismatch tolerance.
    State/Dependencies: read-only preflight; raises before processing.
    """
    for rel,digest in manifest['implementation_hashes'].items():
        assert sha(CODE_ROOT/rel)==digest,rel
    for job in manifest['jobs']: assert sha(root/job['source'])==job['source_sha256'],job['source']
    for rel,digest in manifest['reference_hashes'].items(): assert sha(root/rel)==digest,rel


def process(root,job,smoke=False):
    """@luna-comment Input: immutable PNG job. Output: V12a result/debug/receipt.
    Key Steps: hash, prepare/V4/direct_feather continuous, roundtrip, debug decode.
    Key Params: OpenCV4 threads, exact frozen algorithm. State: isolated subprocess.
    """
    import cv2
    import numpy as np
    import scipy
    from tools.postprocess_ppt_dehalo_20260924 import prepare,write_png
    from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
    from tools.postprocess_ppt_direct_rgb_v12 import direct_feather
    from tools.postprocess_ppt_contour_v6 import read_image
    cv2.setNumThreads(4); start=time.time(); source=root/job['source']; dest=root/job['output']
    receipt=root/'diagnostics'/job['group']/(source.stem+'.json')
    try:
        assert not dest.exists(),str(dest)
        digest=sha(source); assert digest==job['source_sha256']
        src=read_image(source); prepared=prepare(src,robust_seed=True); _,maps,_=normal_correct(prepared)
        base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0])
        ready=time.time(); result,debug,report=direct_feather(base,prepared,maps['correction_weight'],continuous=True)
        done=time.time(); assert result.shape==src.shape
        write_png(dest,result); assert np.array_equal(read_image(dest),result)
        assert sha(source)==digest
        folder=receipt.parent/source.stem
        if not smoke:
            for key,gain in [('weight',255),('raw_weight',255),('inner',15),('outer',7),('patch',255),('protected',255)]:
                write_png(folder/(key+'.png'),np.clip(np.rint(debug[key]*gain),0,255).astype(np.uint8))
            for key in ('raw_target','continuous_target'): write_png(folder/(key+'.png'),debug[key])
            write_png(folder/'signed_rgb_change.png',np.clip(128+(result.astype(np.float32)-base)*8,0,255).astype(np.uint8))
            for path in folder.glob('*.png'):
                assert cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_UNCHANGED) is not None,str(path)
        record=dict(status='complete',pid=os.getpid(),**job,shape=list(result.shape),output_sha256=sha(dest),algorithm='V12a/P14a',continuous=True,
                    elapsed_seconds=time.time()-start,prepare_seconds=ready-start,feather_seconds=done-ready,
                    versions=dict(python=sys.version,opencv=cv2.__version__,numpy=np.__version__,scipy=scipy.__version__),algorithm_report=report)
        if smoke:
            expected=read_image(root/'reference/expected.png'); diff=np.abs(result.astype(np.int16)-expected.astype(np.int16))
            record['reference_max_difference']=int(diff.max()); record['reference_changed_fraction']=float(np.mean(diff!=0))
            assert diff.max()<=1 and np.mean(diff!=0)<=.001,record
        save_json(receipt,record)
    except Exception:
        save_json(receipt,dict(status='failed',pid=os.getpid(),**job,error=traceback.format_exc(),elapsed_seconds=time.time()-start))
        raise


def run_child(root,index):
    """@luna-comment Input: root/index. Output: child exit code and isolated log.
    Key Steps: subprocess per image. Key Params: parent concurrency bound.
    State/Dependencies: environment inherits pinned thread limits; no shared buffers.
    """
    log=root/'logs'/f'{index:03d}.log'; log.parent.mkdir(exist_ok=True)
    with log.open('w') as f:
        result=subprocess.run([sys.executable,__file__,'--root',str(root),'--job',str(index)],stdout=f,stderr=subprocess.STDOUT)
    return result.returncode


def run(root,manifest,workers):
    """@luna-comment Input: root/manifest/worker count. Output: incremental batch status.
    Key Steps: hash gate, bounded parallel children, aggregate exit codes.
    Key Params:6 workers default; no overwrite/resume. State: parent owns status JSON.
    """
    verify(root,manifest); assert not (root/'status.json').exists()
    start=time.time(); completed=0; failures=[]
    save_json(root/'status.json',dict(state='running',total=len(manifest['jobs']),completed=0,failed=0,workers=workers,pid=os.getpid(),started_at=start))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_child,root,i):i for i in range(len(manifest['jobs']))}
        for future in as_completed(futures):
            index=futures[future]; code=future.result()
            if code==0: completed+=1
            else: failures.append(dict(index=index,exit_code=code))
            save_json(root/'status.json',dict(state='running',total=len(futures),completed=completed,failed=len(failures),workers=workers,pid=os.getpid(),started_at=start,elapsed_seconds=time.time()-start,failures=failures))
            print(f'COMPLETE {completed}/{len(futures)} FAILED {len(failures)} LAST {index} exit={code}',flush=True)
    save_json(root/'status.json',dict(state='finished' if not failures else 'finished_with_errors',total=len(futures),completed=completed,failed=len(failures),workers=workers,elapsed_seconds=time.time()-start,failures=failures))


def main():
    """@luna-comment Input: pack/smoke/launch/run/job/archive CLI. Output: batch artifacts.
    Key Steps: explicit modes; detached launcher and exclusive archive destinations.
    Key Params: pinned manifest,6 workers. State: no external model calls or deletion.
    """
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pack',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--launch',action='store_true')
    p.add_argument('--run',action='store_true'); p.add_argument('--archive',action='store_true'); p.add_argument('--job',type=int); p.add_argument('--workers',type=int,default=6)
    args=p.parse_args(); root=args.root.resolve()
    if args.pack: pack(root); return
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if args.smoke:
        verify(root,manifest); process(root,dict(group='reference',source='reference/input.png',output='reference/remote_v12a.png',source_sha256=manifest['reference_hashes']['reference/input.png']),True); return
    if args.job is not None:
        for rel,digest in manifest['implementation_hashes'].items(): assert sha(CODE_ROOT/rel)==digest,rel
        process(root,manifest['jobs'][args.job]); return
    if args.launch:
        assert not (root/'launch.json').exists()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with (root/'batch.log').open('w') as log:
            child=subprocess.Popen([sys.executable,__file__,'--root',str(root),'--run','--workers',str(args.workers)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        save_json(root/'launch.json',dict(pid=child.pid,workers=args.workers,started_at=time.time())); print(child.pid); return
    if args.run: run(root,manifest,args.workers); return
    if args.archive:
        status=json.loads((root/'status.json').read_text()); assert status['state']=='finished' and status['completed']==len(manifest['jobs'])
        with zipfile.ZipFile(root/'results.zip','x',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for directory in ('outputs','diagnostics','logs'):
                for path in sorted((root/directory).rglob('*')):
                    if path.is_file(): archive.write(path,str(path.relative_to(root)))
            for name in ('status.json','manifest.json','launch.json','batch.log'): archive.write(root/name,name)
        print(json.dumps(dict(archive=str(root/'results.zip'),sha256=sha(root/'results.zip')))); return
    p.error('Specify one execution mode')


if __name__=='__main__': main()
