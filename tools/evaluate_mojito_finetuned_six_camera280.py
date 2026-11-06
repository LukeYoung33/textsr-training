"""Matched native Camera280 evaluation of six completed150K finetuned models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'mojito_finetuned_six_camera280_20261027'
TRAIN=DATA/'six_models_mojito_finetune_20261027/training'
BEFORE=DATA/'five_plus_v3_camera280_20261026'
LABELS={'full_v3':'V3 | Full','full_v3m':'V3M | Full','text15_v3':'V3 | Text x1.5',
        'text15_v3m':'V3M | Text x1.5','full_v5lite':'V5Lite | Full','text15_v5lite':'V5Lite | Text x1.5'}


def worker(arm,rank):
    """@luna-comment Input: arm/parity shard. Output:140 native finals with strict provenance.
    Key Steps: bind accepted locked extension worker to isolated root. Key Params:256/224/batch8, repaired017.
    State/Dependencies: clean-process bootstrap, immutable caches, CUDA; no old-result mutation.
    """
    from tools import evaluate_five_plus_v3_camera280 as accepted
    accepted.OUT=OUT
    accepted.worker(arm,rank)


def execute(job):
    """@luna-comment Input: arm/parity/GPU. Output: child exit receipt.
    Key Steps: independent log/environment and checked subprocess. Key Params: twelve disjoint jobs, four GPUs.
    State/Dependencies: own children only; never signals external jobs.
    """
    arm,rank,gpu=job
    log=OUT/'logs'/f'{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--rank',str(rank)],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,rank=rank,gpu=gpu,exit_code=result.returncode)


def verify_row(job):
    """@luna-comment Input: one old manifest row and before/after bindings. Output: verified twelve-image row/audit.
    Key Steps: exact checkpoint/config/source/final hashes, native size, common document/warp/OCR signatures.
    Key Params: before250K, after150K additional; same256/224 no OCR crop. State/Dependencies: read-only files/PIL.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import digest,file_url
    from tools.build_camera280_six_block_viewer import asset_path
    original,bindings,old=job
    identity=original['id'];row={k:original[k] for k in ('id','name','width','height','lq')}
    row['images']={};records={};signatures=[]
    source_hash=digest(asset_path(row['lq']))
    for arm in LABELS:
        for phase in ('before','after'):
            path=asset_path(original['images'][arm]) if phase=='before' else OUT/'results'/arm/'final'/(identity+'.png')
            receipt=json.loads((path.parent.parent/'receipts'/(identity+'.json')).read_text())
            expected=(old if phase=='before' else bindings)[arm]
            for key in ('checkpoint_sha256','config_sha256'):assert receipt['binding'][key]==expected[key],(identity,arm,phase,key)
            assert receipt['source_sha256']==source_hash and receipt['outside_owned_unchanged']
            assert digest(path)==receipt['final_sha256']
            with Image.open(path) as image:assert image.size==(row['width'],row['height'])
            for stage in receipt['stages']:
                detail=stage['restoration'];assert detail['canvas_hw']==[256,256] and detail['stride_hw']==[224,224] and detail['ocr_cropping'] is False
            signatures.append((receipt['document_record_sha256'],[(s['warp_sha256'],s['ocr_sha256']) for s in receipt['stages']]))
            key=phase+'_'+arm;row['images'][key]=file_url(path);records[key]=receipt['final_sha256']
    assert all(s==signatures[0] for s in signatures),(identity,'geometry mismatch')
    if identity=='017':assert signatures[0][0]=='11e239d64b7cac331eab0f9301717dcbb75170b011a219a9bf68215b173f72cd'
    return row,dict(id=identity,source_sha256=source_hash,finals=records,document_record_sha256=signatures[0][0])


def publish():
    """@luna-comment Input: complete six280-result sets plus old six-model gallery. Output: audited before/after viewer.
    Key Steps: exact coverage/binding checks, parallel per-image hashes/provenance, shared thirteen-choice renderer.
    Key Params:1680 new and1680 reused finals; no metrics or quality verdict. State/Dependencies: immutable old assets.
    """
    from tools.build_camera_2x2_viewer import digest,render_html
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    old=json.loads((BEFORE/'AUDIT.json').read_text())['bindings']
    original=json.loads((BEFORE/'manifest.json').read_text())
    identities={f'{i:03d}' for i in range(1,281)}
    assert len(original['rows'])==280 and {r['id'] for r in original['rows']}==identities
    for arm in LABELS:
        for folder,ext in [('final','png'),('receipts','json')]:
            assert {p.stem for p in (OUT/'results'/arm/folder).glob('*.'+ext)}==identities
        for field in ('checkpoint','config'):assert digest(bindings[arm][field])==bindings[arm][field+'_sha256']
        for rank in (0,1):assert json.loads((OUT/'results'/arm/f'COMPLETE.{rank}.json').read_text())['images']==140
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified=list(pool.map(verify_row,[(row,bindings,old) for row in original['rows']]))
    models=[dict(key='lq',label='LQ',note='Original Mojito camera input')]
    for arm,label in LABELS.items():
        for phase in ('before','after'):
            models.append(dict(key=phase+'_'+arm,label=label+(' | Before FT (250K)' if phase=='before' else ' | After FT (+150K)'),note='Same source/warp; native256 blocks, stride224; no tonal pre/post'))
    manifest=dict(title='Mojito280 | Before / after150K finetune',subtitle='Six models; synchronized native-resolution comparison',
        notes='After: six completed150K Mojito finetunes initialized from their respective250K checkpoints. Before: unchanged previous results. Same280 source images and document geometry, including repaired017. No tonal preprocessing/postprocessing. Image quality is for manual review. Native assets require mapped drive Y.',
        defaults='before_full_v3,after_full_v3,before_full_v3m,after_full_v3m',models=models,rows=[r for r,_ in verified])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    (OUT/'AUDIT.json').write_text(json.dumps(dict(passed=True,images=280,new_outputs=1680,reused_outputs=1680,bindings=bindings,before_bindings=old,records=[a for _,a in verified]),indent=2))
    print(json.dumps(dict(passed=True,images=280,new_outputs=1680,viewer=str(OUT/'index.html'))),flush=True)


def run():
    """@luna-comment Input: six final150K configs/checkpoints. Output: twelve successful workers and audited viewer.
    Key Steps: lock controller, freeze snapshots/hashes, parallel parity workers, require all exits0 then publish.
    Key Params: three workers per idle GPU0/1/2/3. State/Dependencies: isolated NAS outputs; old training untouched.
    """
    import fcntl,yaml
    from tools.build_camera_2x2_viewer import digest
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bindings={};(OUT/'models').mkdir(exist_ok=True)
    for arm in LABELS:
        config=TRAIN/'configs'/(arm+'.yml');cfg=yaml.safe_load(config.read_text())
        checkpoint=TRAIN/'experiments'/cfg['name']/'models/net_g_150000.pth'
        assert cfg['train']['total_iter']==150000
        log=(TRAIN/'logs'/(arm+'.log')).read_text(errors='replace');assert 'End of training' in log and 'iter 150000:' in log
        bindings[arm]=dict(config=str(config),checkpoint=str(checkpoint),config_sha256=digest(config),checkpoint_sha256=digest(checkpoint),finetune_iter=150000)
        for source,target in [(config,OUT/'models'/(arm+'.yml')),(checkpoint,OUT/'models'/(arm+'.pth'))]:
            if target.exists():assert digest(source)==digest(target)
            else:shutil.copy2(source,target)
    (OUT/'MODEL_BINDINGS.json').write_text(json.dumps(bindings,indent=2))
    jobs=[(arm,rank,i%4) for i,(arm,rank) in enumerate((a,r) for a in LABELS for r in (0,1))]
    with ThreadPoolExecutor(max_workers=12) as pool:results=list(pool.map(execute,jobs))
    (OUT/'QUEUE_COMPLETE.json').write_text(json.dumps(results,indent=2))
    assert all(r['exit_code']==0 for r in results),results
    publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','worker','publish'))
    parser.add_argument('--arm',choices=tuple(LABELS));parser.add_argument('--rank',type=int,choices=(0,1));args=parser.parse_args()
    if args.mode=='worker':
        if args.arm is None or args.rank is None:parser.error('worker requires --arm and --rank')
        worker(args.arm,args.rank)
    else:dict(run=run,publish=publish)[args.mode]()
