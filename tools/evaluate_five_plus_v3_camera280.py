"""Parallel, source-bound Camera280 comparison for five new arms and A V3 Full."""
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
OUT=DATA/'five_plus_v3_camera280_20261026'
ARMS={'full_v3m':'V3M | Full','text15_v3':'V3 | Text x1.5',
      'text15_v3m':'V3M | Text x1.5','full_v5lite':'V5Lite | Full',
      'text15_v5lite':'V5Lite | Text x1.5'}
REFERENCE=DATA/'v3_val583_abc_camera280_20261026/camera'


def worker(arm,rank):
    """@luna-comment Input: named arm and parity shard. Output:140 native images/receipts.
    Key Steps: lock owned shard, register trained extensions, bind accepted worker, execute unchanged inference.
    Key Params:256/224/batch8, repaired017; no tonal processing. State/Dependencies: CUDA, frozen source/cache.
    """
    import fcntl
    locks=OUT/'shard_locks';locks.mkdir(exist_ok=True)
    with (locks/f'{arm}.{rank}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from tools.train_v5lite_block256 import bootstrap
        bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
        from tools import evaluate_vosr2u_v5_ab as accepted
        accepted.OUT=OUT
        accepted.worker(arm,rank)


def queue(gpu,jobs):
    """@luna-comment Input: GPU and disjoint arm/parity jobs. Output: exit records.
    Key Steps: sequential subprocesses per GPU with separate logs. Key Params: three independent queues.
    State/Dependencies: own children only; no signals to preexisting processes.
    """
    results=[]
    for arm,rank in jobs:
        log=OUT/'logs'/f'{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('ab') as stream:
            result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--rank',str(rank)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        results.append(dict(arm=arm,rank=rank,gpu=gpu,exit_code=result.returncode))
        assert result.returncode==0,results[-1]
    return results


def publish():
    """@luna-comment Input: five complete result sets and existing A V3 Full. Output: verified seven-choice viewer.
    Key Steps: exact coverage, final/config/checkpoint/source hashes, dimensions and paired geometry; shared renderer.
    Key Params:280 IDs, final250K; no objective quality verdict. State/Dependencies: read-only old assets/NAS.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import render_html,file_url,digest
    from tools.build_camera280_six_block_viewer import asset_path
    original=json.loads((REFERENCE/'manifest.json').read_text())
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text())
    ref=json.loads((REFERENCE/'MODEL_BINDINGS.json').read_text())['a_v3']
    assert ref['checkpoint_sha256']=='ad6362d77c9895546f9c3430239cec0dc2a322d58eeed5057e4e07e0e513cc41'
    assert digest(ref['checkpoint'])==ref['checkpoint_sha256'] and digest(ref['config'])==ref['config_sha256']
    identities={f'{i:03d}' for i in range(1,281)}
    assert len(original['rows'])==280 and {r['id'] for r in original['rows']}==identities
    for arm in ARMS:
        assert {p.stem for p in (OUT/'results'/arm/'final').glob('*.png')}==identities
        assert {p.stem for p in (OUT/'results'/arm/'receipts').glob('*.json')}==identities
        assert digest(bindings[arm]['checkpoint'])==bindings[arm]['checkpoint_sha256']
        assert digest(bindings[arm]['config'])==bindings[arm]['config_sha256']
    rows=[];audit=[]
    for source in original['rows']:
        identity=source['id'];row={k:source[k] for k in ('id','name','width','height','lq')}
        row['images']={};signatures=[];records={}
        source_hash=digest(asset_path(row['lq']))
        for arm in (*ARMS,'full_v3'):
            image=asset_path(source['images']['a_v3']) if arm=='full_v3' else OUT/'results'/arm/'final'/(identity+'.png')
            receipt=json.loads((image.parent.parent/'receipts'/(identity+'.json')).read_text())
            expected=ref if arm=='full_v3' else bindings[arm]
            for key in ('checkpoint_sha256','config_sha256'):assert receipt['binding'][key]==expected[key],(identity,arm,key)
            assert receipt['source_sha256']==source_hash and receipt['outside_owned_unchanged']
            assert digest(image)==receipt['final_sha256']
            with Image.open(image) as decoded:assert decoded.size==(row['width'],row['height'])
            for stage in receipt['stages']:
                detail=stage['restoration']
                assert detail['canvas_hw']==[256,256] and detail['stride_hw']==[224,224] and detail['ocr_cropping'] is False
            signatures.append((receipt['document_record_sha256'],[(s['warp_sha256'],s['ocr_sha256']) for s in receipt['stages']]))
            row['images'][arm]=file_url(image);records[arm]=receipt['final_sha256']
        assert all(s==signatures[0] for s in signatures),(identity,'geometry mismatch')
        audit.append(dict(id=identity,source_sha256=source_hash,finals=records,document_record_sha256=signatures[0][0]))
        rows.append(row)
    models=[dict(key='lq',label='LQ',note='Original camera input')]
    for key,label in {**ARMS,'full_v3':'VOSR2u V3 | Full (A)'}.items():
        models.append(dict(key=key,label=label,note='250K final; native256 blocks; stride224; no tonal pre/post'))
    manifest=dict(title='Camera280 | V3 / V3M / V5Lite',subtitle='Six final250K models; synchronized comparison',
        notes='All six use the same source and document warp, including repaired017. Text x1.5 is the text-weighted training loss recipe. Full V3 is the completed original-start A reference. Image quality is for manual review. Images require mapped drive Y.',
        defaults='full_v3,full_v3m,full_v5lite,text15_v5lite',models=models,rows=rows)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    (OUT/'AUDIT.json').write_text(json.dumps(dict(passed=True,images=280,outputs=1680,bindings={**bindings,'full_v3':ref},records=audit),indent=2))
    print(json.dumps(dict(passed=True,images=280,outputs=1680,viewer=str(OUT/'index.html'))),flush=True)


def run():
    """@luna-comment Input: five completed250K training roots. Output: frozen bindings, parallel results and viewer.
    Key Steps: exclusive lock, snapshot config/weights, three GPU queues, require all success, publish full audit.
    Key Params: GPUs0/1/2 only; parity shards; reference A reused. State/Dependencies: isolated NAS writes/CUDA.
    """
    import fcntl
    import yaml
    from tools.build_camera_2x2_viewer import digest
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bindings={};(OUT/'models').mkdir(exist_ok=True)
        for arm in ARMS:
            family='v5lite' if arm.endswith('v5lite') else 'v3m'
            training=DATA/(family+'_vosr2u_20261026')/'training_20261026'
            config=training/'configs'/(arm+'.yml');options=yaml.safe_load(config.read_text())
            checkpoint=training/'experiments'/options['name']/'models/net_g_250000.pth'
            binding=dict(config=str(config),checkpoint=str(checkpoint),config_sha256=digest(config),checkpoint_sha256=digest(checkpoint))
            for source,target in [(config,OUT/'models'/(arm+'.yml')),(checkpoint,OUT/'models'/(arm+'.pth'))]:
                if target.exists():assert digest(source)==digest(target)
                else:shutil.copy2(source,target)
            bindings[arm]=binding
        (OUT/'MODEL_BINDINGS.json').write_text(json.dumps(bindings,indent=2))
        jobs=[(arm,rank) for arm in ARMS for rank in (0,1)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(lambda gpu:queue(gpu,jobs[gpu::3]),range(3)))
        (OUT/'QUEUE_COMPLETE.json').write_text(json.dumps(results,indent=2))
        publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker','publish'))
    parser.add_argument('--arm',choices=tuple(ARMS));parser.add_argument('--rank',type=int,choices=(0,1))
    args=parser.parse_args()
    if args.mode=='worker':
        if args.arm is None or args.rank is None:parser.error('worker requires --arm and --rank')
        worker(args.arm,args.rank)
    else:dict(run=run,publish=publish)[args.mode]()
