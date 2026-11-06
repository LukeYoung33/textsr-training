"""Run fresh HX document/OCR/preparation and the requested exact three model strategies."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.hx_nv12_comparison import OUT,BUNDLE,save
REAL='/home/guang.yang/realSR/bin/python'
OCR='/home/guang.yang/deployTfLite/bin/python'
DOC='/home/guang.yang/doc_yolo11_env/bin/python'


def execute(job):
    """@luna-comment Input:stage command/log/GPU. Output:checked subprocess result. Key Steps:bound CPU threads, redirect owned logs, await child. Key Params:no shell or downloads. State/Dependencies:approved installed interpreters and task outputs."""
    command,label,gpu=job
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True',CUDA_VISIBLE_DEVICES=str(gpu))
    log=OUT/'logs'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return label


def main():
    """@luna-comment Input:staged NV12 archive/code. Output:complete shared caches and four result columns. Key Steps:decode, fresh documents, parallel OCR/pre, requested inference. Key Params:8 OCR/4 pre/2 shards per result. State/Dependencies:read-only release and checkpoints; isolated61-pair run."""
    entry=str(ROOT/'tools/hx_nv12_comparison.py')
    execute(([REAL,entry,'decode','--archive',str(OUT.with_name('hx_nv12_upload_20261022')/'camera.tar.gz')],'decode',0))
    execute(([DOC,str(BUNDLE/'tools/doc_quad_robust.py'),'--weights',str(BUNDLE/'models/document/best.pt'),'--source',str(OUT/'lq'),'--output',str(OUT/'run/documents'),'--device','0','--max-side','8192','--max-pixels','33554432'],'documents',3))
    for stage,python,workers in [('ocr',OCR,8),('prepare',REAL,4)]:
        save(OUT/'STATUS.json',dict(stage=stage))
        jobs=[([python,entry,stage,'--rank',str(rank),'--shards',str(workers)],f'{stage}.{rank}',0) for rank in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(execute,jobs))
    save(OUT/'STATUS.json',dict(stage='inference'))
    jobs=[]
    for gpu,arm in enumerate(('original_v3_pre','vosr_v3_pre','vosr_v5_block','pre')):
        jobs.extend(([REAL,entry,'infer','--arm',arm,'--rank',str(rank),'--shards','2'],f'{arm}.{rank}',gpu) for rank in range(2))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(execute,jobs))
    save(OUT/'STATUS.json',dict(stage='complete'))


if __name__=='__main__':main()
