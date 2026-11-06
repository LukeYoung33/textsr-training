"""Bounded two-phase orientation review and final-only-feather HX execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_hx_upright_revision import OUT,save
import tools.launch_hx_nv12_comparison as runner
REAL=runner.REAL;OCR=runner.OCR


def parallel(jobs,workers):
    """@luna-comment Input:owned stage jobs/concurrency. Output:all checked completions. Key Steps:bounded executor collects each subprocess result. Key Params:4 orientation/8 OCR/4 pre/8 inference. State/Dependencies:existing checked launcher; no other process control."""
    with ThreadPoolExecutor(max_workers=workers) as pool:return list(pool.map(runner.execute,jobs))


def main(mode):
    """@luna-comment Input:orientation, reviewed-run or resume phase. Output:orientation receipts or fully audited refreshed gallery. Key Steps:separate visual review from materialize/OCR/pre/infer/publish; resume retains materialized reviewed geometry. Key Params:local models/original checkpoints. State/Dependencies:task-owned output; old results retained."""
    runner.OUT=OUT;entry=str(ROOT/'tools/run_hx_upright_revision.py')
    if mode=='orient':
        parallel([([OCR,entry,'orient','--rank',str(rank),'--shards','4'],f'orientation.{rank}',0) for rank in range(4)],4)
        save(OUT/'ORIENTATION_COMPLETE.json',dict(status='complete'))
        return
    if mode!='resume':runner.execute(([REAL,entry,'materialize'],'materialize',0))
    for stage,python,workers in [('ocr',OCR,8),('prepare',REAL,4)]:
        save(OUT/'STATUS.json',dict(stage=stage))
        parallel([([python,entry,stage,'--rank',str(rank),'--shards',str(workers)],f'{stage}.{rank}',0) for rank in range(workers)],workers)
    save(OUT/'STATUS.json',dict(stage='inference'))
    jobs=[]
    for gpu,arm in enumerate(('original_v3_pre','vosr_v3_pre','vosr_v5_block','pre')):
        jobs.extend(([REAL,entry,'infer','--arm',arm,'--rank',str(rank),'--shards','2'],f'{arm}.{rank}',gpu) for rank in range(2))
    parallel(jobs,8)
    runner.execute(([REAL,entry,'publish'],'publish',0))
    save(OUT/'COMPLETE.json',dict(status='passed',images=61,revision='upright_postblend_v2'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('orient','run','resume'));main(parser.parse_args().mode)
