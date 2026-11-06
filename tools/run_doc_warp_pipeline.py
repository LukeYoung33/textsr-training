"""Run source-bound warped OCR, shared preparation and four final-model pipelines."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_offline_camera_matrix import ARMS,save_json

OUT=ROOT.parent
REAL='/home/guang.yang/realSR/bin/python'
PADDLE='/home/guang.yang/deployTfLite/bin/python'


def launch(command,log,env=None):
    """@luna-comment
    Input: explicit stage command/log and optional environment. Output: owned child process.
    Key Steps: append isolated log, redirect stdin, spawn known Python entry.
    Key Params: bounded CPU thread environment inherited. State/Dependencies: own run only.
    """
    log.parent.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    environment.update(env or {})
    with log.open('ab') as stream:
        return subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment)


def wait_jobs(jobs,phase):
    """@luna-comment
    Input: owned stage children and phase label. Output: zero exits or explicit failure.
    Key Steps: await all children; reject any nonzero exit with phase/code evidence.
    Key Params: no kill/restart of other processes. State/Dependencies: task-owned children.
    """
    codes=[job.wait() for job in jobs]
    if any(codes):
        raise RuntimeError(phase+' child exit codes: '+str(codes))


def main():
    """@luna-comment
    Input: immutable staged package and running released document detector.
    Output: full warped pipeline results, preserved stages and side-by-side gallery.
    Key Steps: overlap16CPU OCR workers with doc detection; prepare once; real pilot
    on4models; then3 disjoint workers per GPU; aggregate counts; build/validate gallery.
    Key Params: same280/four250K checkpoints; no old results or release mutated.
    State/Dependencies: known Paddle/RealSR environments; fail-closed receipts.
    """
    entry=str(ROOT/'tools/evaluate_doc_warp_matrix.py')
    logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    try:
        jobs=[launch([PADDLE,entry,'ocr','--rank',str(rank),'--shards','16'],logs/f'ocr_rank{rank}.log') for rank in range(16)]
        save_json(OUT/'pipeline_status.json',dict(phase='documents_and_warped_ocr',pids=[job.pid for job in jobs]))
        wait_jobs(jobs,'warped OCR')
        detector=json.loads((OUT/'documents/run.json').read_text())
        if detector['status']!='completed' or detector['errors'] or len(detector['results'])!=280:
            raise ValueError('Document detector incomplete or has errors')
        save_json(OUT/'pipeline_status.json',dict(phase='shared_preprocessing'))
        wait_jobs([launch([REAL,entry,'prepare','--workers','8'],logs/'prepare.log')],'preparation')
        pilot=[]
        for gpu,arm in enumerate(ARMS):
            pilot.append(launch([REAL,entry,'infer','--arm',arm,'--limit','2'],logs/('pilot_'+arm+'.log'),dict(CUDA_VISIBLE_DEVICES=str(gpu))))
        save_json(OUT/'pipeline_status.json',dict(phase='four_model_pilot',pids=[job.pid for job in pilot]))
        wait_jobs(pilot,'four-model pilot')
        jobs=[]
        for gpu,arm in enumerate(ARMS):
            for rank in range(3):
                jobs.append(launch([REAL,entry,'infer','--arm',arm,'--rank',str(rank),'--shards','3'],logs/f'{arm}_rank{rank}.log',dict(CUDA_VISIBLE_DEVICES=str(gpu))))
        save_json(OUT/'pipeline_status.json',dict(phase='full_inference',pids=[job.pid for job in jobs],workers_per_arm=3))
        wait_jobs(jobs,'full corrected inference')
        for arm in ARMS:
            records=[json.loads((OUT/arm/f'COMPLETE.rank{rank}.json').read_text()) for rank in range(3)]
            if sum(r['images'] for r in records)!=280 or len({r['checkpoint_sha256'] for r in records})!=1:
                raise ValueError('Arm count/checkpoint mismatch: '+arm)
            save_json(OUT/arm/'COMPLETE.json',dict(records[0],images=280,workers=3))
        save_json(OUT/'pipeline_status.json',dict(phase='gallery_and_validation'))
        gallery=str(ROOT/'tools/doc_warp_gallery.py')
        wait_jobs([launch([REAL,gallery],logs/'gallery.log')],'gallery and validation')
        save_json(OUT/'pipeline_status.json',dict(phase='complete',images=280,new_outputs=1120,baseline_outputs=1120,gallery='index.html'))
    except Exception as error:
        save_json(OUT/'pipeline_status.json',dict(phase='failed',error=repr(error)))
        raise


if __name__=='__main__':
    main()
