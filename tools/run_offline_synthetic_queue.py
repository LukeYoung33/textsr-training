"""Detached approved synthetic build, acceptance and two-arm launch queue."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.manage_offline_comparison import DATA, RUN, REPO, write_json


def main():
    """@luna-comment
    Input: fixed approved matrix roots. Output: synthetic bank and two running arms.
    Key Steps: spawn bounded offline generators, await success, finalize complete bank,
    smoke each synthetic model, launch only if passed; fail closed on exceptions.
    Key Params: eight generators pinned to GPUs2/3; caller must ensure available GPU capacity.
    Source may be selected by OFFLINE_MATRIX_SOURCE; this historical queue is unused by preprocessing comparisons.
    State/Dependencies: own logs/receipts; outputs never overwrite real bank or old runs.
    """
    logdir = RUN / 'synthetic_build_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, str(ROOT / 'tools/build_offline_synthetic_comparison.py'),
              '--repo-root', REPO, '--source', os.environ.get('OFFLINE_MATRIX_SOURCE', '/AI_DNS/guang.yang/TextSR/data/displayfit_training_20261015/full_images'),
              '--base', str(DATA / 'bank_v1'), '--output', str(RUN / 'synthetic_bank_v1'),
              '--config', str(RUN / 'configs/02_v3_real.yml')]
    workers = []
    started = time.time()
    try:
        for rank in range(8):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(2 + rank % 2), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (logdir / f'worker{rank}.log').open('ab') as stream:
                proc = subprocess.Popen(common + ['--rank', str(rank), '--world', '8'], env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            workers.append(proc)
        write_json(RUN / 'synthetic_queue.json', dict(phase='building', pids=[p.pid for p in workers], started=started))
        codes = [proc.wait() for proc in workers]
        if any(codes):
            raise RuntimeError('Synthetic worker exit codes: ' + str(codes))
        subprocess.run(common + ['--finalize'], check=True)
        for arm, gpu in (('03_v5_synthetic', 2), ('04_v3_synthetic', 3)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            manager = [sys.executable, str(ROOT / 'tools/manage_offline_comparison.py')]
            with (logdir / (arm + '_smoke.log')).open('ab') as stream:
                subprocess.run(manager + ['smoke', '--arm', arm], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            subprocess.run(manager + ['launch', '--arm', arm], check=True)
        write_json(RUN / 'synthetic_queue.json', dict(phase='launched', started=started, seconds=time.time()-started))
    except Exception as error:
        write_json(RUN / 'synthetic_queue.json', dict(phase='failed', error=repr(error), started=started, seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
