"""Supervise the authorized final-model evaluation in an isolated output tree."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent
ARMS = ('01_v5_preprocessed','02_v3_preprocessed','03_v5_raw','04_v3_raw','05_v5_block256_raw','06_v3_block256_raw')


def execute(arm, suite, gpu, limit=0, rank=0, shards=1):
    """@luna-comment Input:owned job options. Output:child. Key Steps:redirect log and select GPU. Key Params:bounded threads. State/Dependencies:known Python only."""
    args = [sys.executable,'-u',str(ROOT/'tools/evaluate_vosr_six_models.py'),'run','--arm',arm,'--suite',suite,'--rank',str(rank),'--shards',str(shards)]
    if limit:
        args += ['--limit',str(limit)]
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with (OUT/'logs'/f'{suite}_{arm}_{limit}_{rank}.log').open('ab') as log:
        return subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=environment)


def main():
    """@luna-comment Input:staged sources. Output:all evaluations/report or explicit failure. Key Steps:pilot all strategies then four-GPU queue. Key Params:three workers/GPU. State/Dependencies:own children only."""
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    status = OUT/'STATUS.json'
    try:
        for suite in ('camera','validation'):
            for offset in (0,4):
                jobs = [execute(arm,suite,i,2) for i,arm in enumerate(ARMS[offset:offset+4])]
                codes = [job.wait() for job in jobs]
                if any(codes):
                    raise RuntimeError(f'Pilot failed {suite}: {codes}')
        queue = [(arm,suite,rank) for arm in ARMS for suite in ('camera','validation') for rank in range(3)]
        active = []
        done = []
        while queue or active:
            for slot in range(12):
                if queue and not any(item['slot']==slot for item in active):
                    arm,suite,rank = queue.pop(0)
                    active.append(dict(slot=slot,arm=arm,suite=suite,rank=rank,process=execute(arm,suite,slot%4,rank=rank,shards=3)))
            for item in active[:]:
                code = item['process'].poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'Job failed: {item["arm"]}/{item["suite"]}/{item["rank"]}: {code}')
                    done.append({k:v for k,v in item.items() if k!='process'})
                    active.remove(item)
            payload = dict(phase='evaluating',completed=done,queued=len(queue),active=[{**{k:v for k,v in item.items() if k!='process'},'pid':item['process'].pid} for item in active])
            temp = status.with_suffix('.tmp')
            temp.write_text(json.dumps(payload,indent=2));temp.replace(status)
            time.sleep(10)
        subprocess.run([sys.executable,str(ROOT/'tools/evaluate_vosr_six_models.py'),'report'],check=True)
        status.write_text(json.dumps(dict(phase='complete',gallery='index.html')))
    except Exception as error:
        status.write_text(json.dumps(dict(phase='failed',error=repr(error))))
        raise


if __name__=='__main__':
    main()
