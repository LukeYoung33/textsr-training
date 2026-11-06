"""Three pretrain250K V3M pipelines and four immutable historical Mojito280 references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'v3pre_three_mojito280_20261028'
TRAIN = DATA/'v3pre_v3m_three_20261027'
VALIDATION = DATA/'v3pre_three_pretrain_val583_20261028'
HISTORY = DATA/'five_plus_v3_camera280_20261026'
EARLIEST = DATA/'camera_compare_2x2_20261022'
LABELS = {'full': 'V3M Full | V3 preprocessing | Pretrain250K',
          'text15': 'V3M Text x1.5 | V3 preprocessing | Pretrain250K',
          'adapt_l1': 'V3M adaptive-L1 cap4 | V3 preprocessing | Pretrain250K'}
OLD = {'original_v3_pre': 'Original V3 + V2_2 | historical pipeline',
       'full_v3': 'V3 Full | no preprocessing | Pretrain250K',
       'full_v3m': 'V3M Full | no preprocessing | Pretrain250K',
       'text15_v3m': 'V3M Text x1.5 | no preprocessing | Pretrain250K'}


def canonical(image, orientation):
    """@luna-comment Input:stored RGB8 raster/original EXIF orientation. Output:display-oriented RGB8.
    Key Steps:apply only the standard lossless EXIF permutation after inference.
    Key Params:1 or6 in this inventory. State/Dependencies:PIL; no resampling/tone.
    """
    import numpy as np
    from PIL import Image
    assert orientation in (1, 6)
    return image if orientation == 1 else np.asarray(Image.fromarray(image).transpose(Image.Transpose.ROTATE_270)).copy()


def worker(arm):
    """@luna-comment Input:frozen arm/280 selection. Output:native PNGs/provenance receipts.
    Key Steps:strict250K load; rehash completed finetune V3 input; infer stored pixels;
    EXIF display permutation only; save result. Key Params:256/224/batch8/crop8/smooth16.
    State/Dependencies:GPU3, immutable views; no preprocessing or training mutation.
    """
    import cv2
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.restore_crop8_block256 import restore_block
    from tools.evaluate_offline_camera_matrix import sha, save_json, image_write
    cv2.setNumThreads(1); torch.set_num_threads(2)
    binding=json.loads((OUT/'MODEL_BINDINGS.json').read_text())[arm]
    config=OUT/'models'/(arm+'.yml'); checkpoint=OUT/'models'/(arm+'.pth')
    assert sha(config)==binding['config_sha256'] and sha(checkpoint)==binding['checkpoint_sha256']
    assert sha(ROOT/'tools/restore_crop8_block256.py')==binding['stitcher_sha256']
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    selection_hash=sha(OUT/'selection.json')
    for number,entry in enumerate(json.loads((OUT/'selection.json').read_text())['samples'],1):
        identity=entry['id']; path=OUT/'results'/arm/'final'/(identity+'.png')
        receipt=OUT/'results'/arm/'receipts'/(identity+'.json')
        if receipt.exists():
            cached=json.loads(receipt.read_text())
            assert cached['binding']==binding and cached['selection_sha256']==selection_hash and sha(path)==cached['final_sha256']
            continue
        started=time.time()
        assert sha(entry['input'])==entry['input_sha256']
        source=cv2.imread(entry['input'],cv2.IMREAD_UNCHANGED)
        assert source is not None and list(source.shape)==entry['stored_shape']
        prediction,details=restore_block(cv2.cvtColor(source,cv2.COLOR_BGR2RGB),model,'cuda')
        prediction=canonical(prediction,entry['orientation'])
        assert list(prediction.shape[:2])==[entry['height'],entry['width']]
        image_write(path,cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR))
        save_json(receipt,dict(id=identity,binding=binding,selection_sha256=selection_hash,
            source_sha256=entry['source_sha256'],input_sha256=entry['input_sha256'],
            preprocess_version='document_preprocessing_V3',split=entry['split'],sample_id=entry['sample_id'],
            stored_shape=entry['stored_shape'],display_orientation=entry['orientation'],
            restoration=details,final=str(path),final_sha256=sha(path),seconds=time.time()-started))
        print(json.dumps(dict(arm=arm,done=number,total=280,id=identity,seconds=time.time()-started)),flush=True)


def execute(arm):
    """@luna-comment Input:arm key. Output:owned worker exit record.
    Key Steps:separate subprocess/log/CUDA environment. Key Params:GPU3 only/two CPU threads.
    State/Dependencies:does not signal existing training or preprocessing processes.
    """
    log=OUT/'logs'/(arm+'.log'); log.parent.mkdir(exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,gpu=3,exit_code=result.returncode)


def progress(started,phase='running',error=None):
    """@luna-comment Input:controller start/phase/error. Output:atomic counts/ETA and refreshing page.
    Key Steps:count completed receipts, derive observed aggregate rate; publish actual status.
    Key Params:840 results/30s page refresh. State/Dependencies:owned output only.
    """
    from tools.evaluate_offline_camera_matrix import save_json
    counts={arm:len(list((OUT/'results'/arm/'receipts').glob('*.json'))) for arm in LABELS}
    done=sum(counts.values()); elapsed=time.time()-started; now=datetime.now(timezone(timedelta(hours=8)))
    eta=(now+timedelta(seconds=elapsed/max(done,1)*(840-done))).isoformat() if done else None
    state=dict(phase=phase,counts=counts,completed=done,total=840,elapsed_seconds=elapsed,
               estimated_completion=eta,updated_at=now.isoformat(),error=error)
    save_json(OUT/'PROGRESS.json',state)
    body='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Mojito280 progress</title><style>body{font:16px system-ui;margin:30px}article{padding:20px;background:#eef3f8;margin:12px 0}progress{width:100%}</style><h1>Mojito280 | Three Pretrain250K models</h1><p>Reuse completed finetune V3 inputs | GPU3 | crop8 + smooth16</p>'
    for arm in LABELS:body+='<article>'+html.escape(LABELS[arm])+f': {counts[arm]}/280<progress value="{counts[arm]}" max="280"></progress></article>'
    (OUT/'progress.html').write_text(body+'<p>'+html.escape(phase)+' | ETA: '+html.escape(eta or 'warming up')+'</p><p>'+html.escape(error or '')+'</p><p>'+state['updated_at']+'</p>',encoding='utf-8')


def verify_row(entry):
    """@luna-comment Input:one frozen source/input/reference row. Output:audited viewer row and hashes.
    Key Steps:source/input/old/new receipt hashes and native geometry; optional lossless input-display transform.
    Key Params:seven model outputs; historical pipelines stay unchanged. State/Dependencies:PIL/OpenCV/NAS.
    """
    import cv2
    from PIL import Image
    from tools.build_camera_2x2_viewer import digest,file_url
    from tools.evaluate_offline_camera_matrix import image_write
    identity=entry['id']; assert digest(entry['source'])==entry['source_sha256']
    assert digest(entry['input'])==entry['input_sha256']
    processed=entry['input']
    if entry['orientation']!=1:
        source=cv2.imread(processed,cv2.IMREAD_UNCHANGED)
        processed=OUT/'reference/v3_input'/(identity+'.png')
        display=canonical(cv2.cvtColor(source,cv2.COLOR_BGR2RGB),entry['orientation'])
        image_write(processed,cv2.cvtColor(display,cv2.COLOR_RGB2BGR))
        import numpy as np
        assert np.array_equal(cv2.imread(str(processed),cv2.IMREAD_UNCHANGED),cv2.cvtColor(display,cv2.COLOR_RGB2BGR))
    row={k:entry[k] for k in ('id','name','width','height')}
    row.update(lq=file_url(entry['source']),images=dict(v3_input=file_url(processed)))
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text()); selection_hash=digest(OUT/'selection.json')
    audit=dict(id=identity,source_sha256=entry['source_sha256'],input_sha256=entry['input_sha256'],models={})
    old_geometry=[]
    for arm in (*OLD,*LABELS):
        if arm in OLD:
            ref=entry['references'][arm]; path=Path(ref['image']); receipt_path=Path(ref['receipt'])
            assert digest(receipt_path)==ref['receipt_sha256']
            r=json.loads(receipt_path.read_text()); assert r['binding']['checkpoint_sha256']==ref['checkpoint_sha256']
            assert r['document_record_sha256']==entry['historical_document_record_sha256']
            old_geometry.append([(s['warp_sha256'],s['ocr_sha256']) for s in r['stages']])
            if arm=='original_v3_pre':assert r['binding']['preprocessing']=='V2_2'
            else:assert r['outside_owned_unchanged']
        else:
            path=OUT/'results'/arm/'final'/(identity+'.png'); receipt_path=path.parent.parent/'receipts'/(identity+'.json')
            r=json.loads(receipt_path.read_text())
            assert r['binding']==bindings[arm] and r['selection_sha256']==selection_hash
            assert r['input_sha256']==entry['input_sha256'] and r['display_orientation']==entry['orientation']
            d=r['restoration']; assert d['stitch_policy']=='crop8_smooth16' and d['crop_pixels']==8 and d['blend_pixels']==16
            assert d['canvas_hw']==[256,256] and d['stride_hw']==[224,224] and not d['ocr_cropping'] and d['weight_min']>0
        assert r['id']==identity and r['source_sha256']==entry['source_sha256']
        assert digest(path)==r['final_sha256']
        with Image.open(path) as image:assert image.size==(entry['width'],entry['height'])
        row['images'][arm]=file_url(path)
        audit['models'][arm]=dict(sha256=r['final_sha256'],receipt_sha256=digest(receipt_path))
    assert all(g==old_geometry[0] for g in old_geometry)
    return row,audit


def publish():
    """@luna-comment Input:840 new/1120 historical complete finals. Output:nine-choice viewer and1960-result audit.
    Key Steps:exact280 IDs, immutable bindings/selection/native hash checks, render existing synchronized UI.
    Key Params:no GT metrics or subjective ranking. State/Dependencies:read-only source/history and owned metadata.
    """
    from tools.build_camera_2x2_viewer import render_html,digest
    from tools.evaluate_offline_camera_matrix import save_json
    selection=json.loads((OUT/'selection.json').read_text()); samples=selection['samples']
    identities={e['id'] for e in samples}; assert identities=={f'{i:03d}' for i in range(1,281)}
    for arm in LABELS:
        for folder,suffix in [('final','png'),('receipts','json')]:assert {p.stem for p in (OUT/'results'/arm/folder).glob('*.'+suffix)}==identities
    for b in json.loads((OUT/'MODEL_BINDINGS.json').read_text()).values():
        for field in ('config','checkpoint'):assert digest(b[field])==b[field+'_sha256']
    with ThreadPoolExecutor(max_workers=3) as pool:verified=list(pool.map(verify_row,samples))
    models=[dict(key='lq',label='LQ | Original',note='Original camera source; native display orientation'),
            dict(key='v3_input',label='V3 preprocessing only | reused finetune LQ',note='Existing complete V3 input; display orientation only; no second preprocessing')]
    for key,label in OLD.items():models.append(dict(key=key,label=label,note='Historical V2_2 + tone postprocess / 96x480 text regions' if key=='original_v3_pre' else 'Historical no-tone document warp; 256/224 blocks and original feathering'))
    for key,label in LABELS.items():models.append(dict(key=key,label=label,note='Exact250K; reused V3 full native input; 256/224/batch8/crop8/smooth16; no additional tone/warp/OCR'))
    manifest=dict(title='Mojito280 | Three V3-preprocessed pretrains + four historical references',
        subtitle='Seven model pipelines | Native synchronized comparison',models=models,rows=[r for r,_ in verified],
        defaults='full_v3m,full,text15,adapt_l1',notes='All280 sources matched by original LQ SHA256 to completed finetune V3 views:243 train +37 val. Model weights here are pretrain250K, not finetuned. New models use the entire V3-preprocessed stored raster and crop8/smooth16; display applies original EXIF permutation only. Historical references retain original document warp/text-region/tonal-postprocess/feathering strategies. This is a full-pipeline comparison, not a strict preprocessing-only ablation. No GT metrics or automated quality verdict. Native assets require mapped drive Y.')
    save_json(OUT/'manifest.json',manifest); (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    save_json(OUT/'AUDIT.json',dict(passed=True,images=280,new_outputs=840,reused_outputs=1120,total_model_outputs=1960,
        selection_sha256=digest(OUT/'selection.json'),records=[a for _,a in verified]))


def run():
    """@luna-comment Input:completed finetune V3 manifests and exact previous250K bindings. Output:audited280 comparison.
    Key Steps:lock; SHA-match all280/447 parents; freeze inputs/references/models; three GPU3 workers; audit/publish.
    Key Params:243 train/37 val inputs; weights remain pretrain250K. State/Dependencies:immutable NAS sources.
    """
    import fcntl
    from PIL import Image
    from tools.build_camera_2x2_viewer import digest
    from tools.build_camera280_six_block_viewer import asset_path
    from tools.evaluate_offline_camera_matrix import save_json
    OUT.mkdir(exist_ok=True); lock=(OUT/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bindings=json.loads((VALIDATION/'MODEL_BINDINGS.json').read_text()); (OUT/'models').mkdir(exist_ok=True)
    for arm,b in bindings.items():
        assert b['pretrain_iter']==250000 and b['stitcher_sha256']==digest(ROOT/'tools/restore_crop8_block256.py')
        for field,suffix in [('config','.yml'),('checkpoint','.pth')]:
            assert digest(b[field])==b[field+'_sha256']; destination=OUT/'models'/(arm+suffix)
            if destination.exists():assert digest(destination)==b[field+'_sha256']
            else:shutil.copy2(b[field],destination)
    save_json(OUT/'MODEL_BINDINGS.json',bindings)
    lookup={};manifest_hashes={}
    for split,count in [('train',380),('val',67)]:
        path=TRAIN/'finetune/source_view/manifests'/(split+'.jsonl'); manifest_hashes[str(path)]=digest(path)
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]; assert len(rows)==count
        for r in rows:
            key=r['preprocess']['source_hashes']['lq']; assert key not in lookup
            lookup[key]=(split,r)
    history=json.loads((HISTORY/'manifest.json').read_text()); earliest=json.loads((EARLIEST/'manifest.json').read_text())
    prior_bindings=json.loads((HISTORY/'AUDIT.json').read_text())['bindings']
    original={r['id']:r for r in earliest['rows']}; assert len(history['rows'])==len(original)==280
    samples=[]
    for base in history['rows']:
        identity=base['id']; source=asset_path(base['lq']); source_hash=digest(source)
        split,r=lookup[source_hash]; assert r['preprocess']['version']=='document_preprocessing_V3'
        assert r['preprocess']['full_repair']==1 and r['preprocess']['independent_gt_lq']
        assert digest(r['files']['lq'])==r['output_sha256']['lq']
        assert digest(r['preprocess']['source_files']['lq'])==source_hash
        with Image.open(r['preprocess']['source_files']['lq']) as image:orientation=image.getexif().get(274,1)
        with Image.open(r['files']['lq']) as image:w,h=image.size
        assert ((w,h) if orientation==1 else (h,w))==(base['width'],base['height'])
        refs={};record_hashes=[]
        for arm in OLD:
            old=original[identity] if arm=='original_v3_pre' else base
            assert old['lq']==base['lq'] and old['name']==base['name']
            image=asset_path(old['images'][arm]); receipt=image.parent.parent/'receipts'/(identity+'.json')
            detail=json.loads(receipt.read_text()); record_hashes.append(detail['document_record_sha256'])
            checkpoint=detail['binding']['checkpoint_sha256']
            if arm=='original_v3_pre':assert checkpoint=='6d07811a0964ee60828d541b0a961be833ca39fb1776015f50ca41ccb01cb1a3'
            else:assert checkpoint==prior_bindings[arm]['checkpoint_sha256']
            refs[arm]=dict(image=str(image),receipt=str(receipt),receipt_sha256=digest(receipt),checkpoint_sha256=checkpoint)
        assert len(set(record_hashes))==1
        samples.append(dict(id=identity,name=base['name'],width=base['width'],height=base['height'],source=str(source),
            source_sha256=source_hash,input=r['files']['lq'],input_sha256=r['output_sha256']['lq'],
            stored_shape=[h,w,3],orientation=orientation,split=split,sample_id=r['sample_id'],references=refs,
            preprocess=r['preprocess'],historical_document_record_sha256=record_hashes[0]))
    assert len(samples)==280 and len({e['id'] for e in samples})==280
    assert {s:sum(e['split']==s for e in samples) for s in ('train','val')}=={'train':243,'val':37}
    save_json(OUT/'selection.json',dict(samples=samples,finetune_manifest_hashes=manifest_hashes,
        history_manifest_sha256=digest(HISTORY/'manifest.json'),earliest_manifest_sha256=digest(EARLIEST/'manifest.json')))
    started=time.time()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(execute,arm) for arm in LABELS]
            while not all(f.done() for f in futures):
                progress(started)
                for f in futures:
                    if f.done():assert f.result()['exit_code']==0,f.result()
                time.sleep(15)
            queue=[f.result() for f in futures]
        save_json(OUT/'QUEUE_COMPLETE.json',queue); assert all(r['exit_code']==0 for r in queue)
        progress(started,'auditing'); publish(); progress(started,'complete')
    except Exception as exc:
        progress(started,'failed',repr(exc)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker','publish')); parser.add_argument('--arm',choices=tuple(LABELS)); args=parser.parse_args()
    if args.mode=='worker':
        if args.arm is None:parser.error('worker requires --arm')
        worker(args.arm)
    else:dict(run=run,publish=publish)[args.mode]()
