"""Bounded task-owned inference queue for the six missing comparison columns."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(job):
    """@luna-comment Input:arm/rank/GPU. Output:exit code. Key Steps:run immutable adapter with private log. Key Params:two shards per arm. State/Dependencies:task-owned CUDA subprocess."""
    arm, rank, gpu = job
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
    log = root/'logs'/f'{arm}.{rank}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, 'tools/run_camera_compare_missing.py', '--arm', arm, '--rank', str(rank), '--shards', '2'], cwd=root/'_build', env=env, stdout=stream, stderr=subprocess.STDOUT)
    return dict(arm=arm, rank=rank, gpu=gpu, exit_code=result.returncode)


def main():
    """@luna-comment Input:fixed six-arm matrix. Output:queue completion receipt. Key Steps:bounded twelve workers, collect every status. Key Params:three processes/GPU maximum. State/Dependencies:isolated output tree; no existing process termination."""
    from tools.run_camera_compare_missing import ARMS
    jobs = [(arm, rank, (i*2+rank)%4) for i, arm in enumerate(ARMS) for rank in range(2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(worker, jobs))
    path = Path(__file__).resolve().parents[2]/'QUEUE_COMPLETE.json'
    path.write_text(json.dumps(rows, indent=2))
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
