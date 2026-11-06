"""Three exact V3-preprocessed V3M pretrain250K checkpoints on583 full validation parents."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'v3pre_three_pretrain_val583_20261028'
TRAINING=DATA/'v3pre_v3m_three_20261027'
BEFORE=DATA/'pretrained_four_val583_20261027'
VAL=TRAINING/'pretrain/source_view/manifests/val.jsonl'
LABELS={'full':'V3M | Full | Pretrain250K', 'text15':'V3M | Text x1.5 | Pretrain250K', 'adapt_l1':'V3M | adaptive-L1 cap4 | Pretrain250K'}


def worker(arm,rank):
    """@luna-comment Input: exact arm/rank0 and frozen583 selection. Output: native predictions/common metrics.
    Key Steps: strict extension load; verify frozen crop8 stitcher; training-matched stored-pixel decode; LQ-only restore; separate GT/mask scoring.
    Key Params:256/224/batch8/crop8/smooth16; common unweighted valid RGB PSNR/Y SSIM. State/Dependencies: CUDA, immutable V3-processed older GT_VOSR.
    """
    import cv2,numpy as np,torch,yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    from tools import evaluate_vosr_six_models as shared
    from tools import restore_crop8_block256 as stitcher
    cv2.setNumThreads(1);torch.set_num_threads(2)
    binding=json.loads((OUT/'MODEL_BINDINGS.json').read_text())[arm]
    config=OUT/'models'/(arm+'.yml');checkpoint=OUT/'models'/(arm+'.pth')
    assert sha(config)==binding['config_sha256'] and sha(checkpoint)==binding['checkpoint_sha256']
    assert sha(shared.__file__)==binding['metric_helper_sha256']
    assert sha(stitcher.__file__)==binding['stitcher_sha256']
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    selection=OUT/'selection.json';selection_hash=sha(selection);entries=json.loads(selection.read_text())['samples'];assert len(entries)==583
    for done,entry in enumerate(entries[rank::1],1):
        identity=entry['id'];row=entry['row'];receipt=OUT/'results'/arm/'receipts'/(identity+'.json')
        if receipt.exists():
            cached=json.loads(receipt.read_text());assert cached['binding']==binding and cached['selection_sha256']==selection_hash and sha(cached['final'])==cached['final_sha256'];continue
        hashes={k:sha(v) for k,v in row['files'].items()};assert hashes==row['output_sha256']
        source=cv2.imread(row['files']['lq'],cv2.IMREAD_UNCHANGED);target=cv2.imread(row['files']['gt'],cv2.IMREAD_UNCHANGED)
        valid=cv2.imread(row['files']['valid_mask'],0);text=cv2.imread(row['files']['text_mask'],0)
        assert source is not None and target is not None and valid is not None and text is not None
        assert source.shape==target.shape and source.shape[:2]==valid.shape==text.shape and source.dtype==target.dtype==np.uint8
        prediction,details=stitcher.restore_block(cv2.cvtColor(source,cv2.COLOR_BGR2RGB),model,'cuda')
        prediction=cv2.cvtColor(prediction,cv2.COLOR_RGB2BGR);assert prediction.shape==source.shape
        measured=shared.metrics(prediction,target,valid>0,text>0)
        baseline=shared.metrics(source,target,valid>0,text>0) if arm=='full' else None
        path=OUT/'results'/arm/'final'/(identity+'.png');image_write(path,prediction)
        save_json(receipt,dict(id=identity,sample_id=row['sample_id'],dataset=row['dataset'],binding=binding,selection_sha256=selection_hash,source_hashes=hashes,source_hw=list(source.shape[:2]),final=str(path),final_sha256=sha(path),restoration=details,metrics=measured,baseline=baseline))
        print(json.dumps(dict(arm=arm,rank=rank,done=done,total=len(entries[rank::1]),id=identity)),flush=True)
    save_json(OUT/'results'/arm/f'COMPLETE.{rank}.json',dict(images=len(entries[rank::1]),binding=binding))


def execute(job):
    """@luna-comment Input: arm/parity/GPU. Output: child exit receipt.
    Key Steps: separate subprocess/environment/log. Key Params:3 jobs/GPU3, bounded threads.
    State/Dependencies: owned processes only; no old jobs or results mutated.
    """
    arm,rank,gpu=job;log=OUT/'logs'/f'{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--rank',str(rank)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,rank=rank,gpu=gpu,exit_code=result.returncode)


def verify_row(job):
    """@luna-comment Input: selection row, three bindings and selection hash. Output: audited native viewer row/metrics.
    Key Steps: rehash assets/finals; pixel-verify lossless EXIF LQ display; check native support/model identities.
    Key Params: GT display/scoring only; identical metric denominators per arm. State/Dependencies: PIL, read-only files.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import digest,file_url
    entry,bindings,selection_hash=job;row=entry['row'];identity=entry['id']
    hashes={k:digest(v) for k,v in row['files'].items()};assert hashes==row['output_sha256']
    with Image.open(row['files']['lq']) as image:w,h=image.size;orientation=image.getexif().get(274,1)
    with Image.open(row['files']['gt']) as image:assert image.size==(w,h)
    item=dict(id=identity,name=row['dataset']+' | '+row['sample_id'],width=w,height=h,lq=file_url(row['files']['lq']),images=dict(gt=file_url(row['files']['gt'])))
    item['source_exif_orientation']=orientation
    raw=row['preprocess']['source_files']['lq'];assert digest(raw)==row['preprocess']['source_hashes']['lq']
    item['images']['raw_lq']=file_url(raw)
    if orientation!=1:
        import cv2,numpy as np
        from tools.evaluate_offline_camera_matrix import image_write
        raw=cv2.imread(row['files']['lq'],cv2.IMREAD_UNCHANGED);assert raw.shape[:2]==(h,w)
        display=OUT/'reference/lq'/(identity+'.png');image_write(display,raw)
        assert np.array_equal(cv2.imread(str(display),cv2.IMREAD_UNCHANGED),raw)
        item['lq']=file_url(display);item['display_lq_sha256']=digest(display)
    receipts={};support=[]
    for arm in LABELS:
        r=json.loads((OUT/'results'/arm/'receipts'/(identity+'.json')).read_text())
        assert r['id']==identity and r['sample_id']==row['sample_id'] and r['source_hashes']==hashes and r['binding']==bindings[arm] and r['selection_sha256']==selection_hash and r['source_hw']==[h,w]
        details=r['restoration'];assert details['stitch_policy']=='crop8_smooth16' and details['crop_pixels']==8 and details['blend_pixels']==16 and details['weight_min']>0
        assert digest(ROOT/'tools/restore_crop8_block256.py')==bindings[arm]['stitcher_sha256']
        assert details['canvas_hw']==[256,256] and details['stride_hw']==[224,224] and details['ocr_cropping'] is False
        assert digest(r['final'])==r['final_sha256']
        with Image.open(r['final']) as image:assert image.size==(w,h)
        support.append([r['metrics'][k]['count'] for k in ('all','text','background','ssim_y')])
        item['images'][arm]=file_url(r['final']);receipts[arm]=r
    assert all(s==support[0] for s in support) and support[0][0]==support[0][1]+support[0][2]
    assert [receipts['full']['baseline'][k]['count'] for k in ('all','text','background','ssim_y')]==support[0]
    return item,receipts


def publish():
    """@luna-comment Input: complete three-model583 finals/metric receipts. Output: audited viewer/objective tables/CSV.
    Key Steps: exact coverage and hashes, pooled SSE/count/SSIM sums, LQ baseline, six-choice synchronized renderer.
    Key Params: no training-loss weights or qualitative ranking; V3-processed older GT_VOSR. State/Dependencies: accepted metric aggregator.
    """
    from tools.build_camera_2x2_viewer import digest,render_html
    from tools.summarize_vosr_validation import aggregate
    selection=OUT/'selection.json';selection_hash=digest(selection);entries=json.loads(selection.read_text())['samples'];assert len(entries)==583
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());identities={e['id'] for e in entries};assert len(identities)==583
    for arm in LABELS:
        for folder,suffix in [('final','png'),('receipts','json')]:assert {p.stem for p in (OUT/'results'/arm/folder).glob('*.'+suffix)}==identities
        for field,suffix in [('config','.yml'),('checkpoint','.pth')]:
            assert digest(bindings[arm][field])==bindings[arm][field+'_sha256'] and digest(OUT/'models'/(arm+suffix))==bindings[arm][field+'_sha256']
    with ThreadPoolExecutor(max_workers=3) as pool:verified=list(pool.map(verify_row,[(e,bindings,selection_hash) for e in entries]))
    summary={arm:aggregate([r[arm]['metrics'] for _,r in verified]) for arm in LABELS}
    summary['lq']=aggregate([r['full']['baseline'] for _,r in verified])
    (OUT/'VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    models=[dict(key='raw_lq',label='Original LQ',note='Visual-only raw input'),dict(key='lq',label='LQ | V3 preprocessed',note='Exact model input; not preprocessed twice'),dict(key='gt',label='GT | V3-processed older GT_VOSR',note='Scoring/display only')]
    models.extend(dict(key=k,label=v,note='Exact pretrained250K; native256/224/batch8/crop8/smooth16; no tone/warp') for k,v in LABELS.items())
    manifest=dict(title='Validation583 | V3-preprocessed V3M pretrain250K',subtitle='Full / Text x1.5 / adaptive-L1 cap4',notes='Exact583 historical parent IDs, V3-preprocessed LQ and older GT_VOSR targets from current training view. Original LQ is visual-only. No new preprocessing, finetune weights, tone/warp or OCR-gated inference. Same256/224/batch8 and crop8/smooth16 for all three arms. PSNR/SSIM are unweighted common-mask metrics; training loss weighting is not applied. Not directly comparable to raw-target historical scores. Native images require mapped Y.',defaults='lq,full,text15,adapt_l1',models=models,rows=[r for r,_ in verified])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    report='<!doctype html><meta charset="utf-8"><title>Pretrained583 validation metrics</title><style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:10px}</style><h1>583-parent V3-preprocessed pretrain validation | Crop8 + smooth16</h1><p><a href="index.html">Native image comparison</a> | <a href="per_image.csv">Per-image CSV</a> | <a href="VALIDATION_SUMMARY.json">Full statistics</a></p><p>Same held-out V3-processed older GT_VOSR and valid masks for all models. RGB PSNR uses pooled pixel SSE/channel count; text/background use the same masks. Y SSIM uses Gaussian11/sigma1.5 windows wholly inside valid support. Macro finite-image PSNR is separate. No training-loss weighting or visual-quality verdict. Exact/empty values are explicit in JSON.</p><table><tr><th>Model</th><th>Images</th><th>RGB pooled PSNR</th><th>Text pooled PSNR</th><th>Background pooled PSNR</th><th>Y pooled SSIM</th><th>RGB macro PSNR</th></tr>'
    for arm in ('lq',*LABELS):
        s=summary[arm];values=[s[k]['pooled_psnr'] for k in ('all','text','background')]+[s['ssim_y']['pooled'],s['all']['macro_finite_psnr']]
        report+='<tr><td>'+html.escape(LABELS.get(arm,'LQ baseline'))+'</td><td>583</td>'+''.join('<td>'+('exact/empty' if v is None else f'{v:.6f}')+'</td>' for v in values)+'</tr>'
    (OUT/'metrics.html').write_text(report+'</table>',encoding='utf-8')
    with (OUT/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['id','sample_id','dataset','model','rgb_psnr','text_psnr','background_psnr','ssim_y'])
        for entry,(_,receipts) in zip(entries,verified):
            for arm in ('lq',*LABELS):
                m=receipts['full']['baseline'] if arm=='lq' else receipts[arm]['metrics'];ssim=m['ssim_y']
                writer.writerow([entry['id'],entry['row']['sample_id'],entry['row']['dataset'],arm,*[m[k]['psnr'] for k in ('all','text','background')],ssim['sum']/ssim['count'] if ssim['count'] else None])
    records=[dict(id=item['id'],source_hashes=r['full']['source_hashes'],display_lq=item['lq'],source_exif_orientation=item['source_exif_orientation'],display_lq_sha256=item.get('display_lq_sha256'),finals={a:r[a]['final_sha256'] for a in LABELS}) for item,r in verified]
    (OUT/'AUDIT.json').write_text(json.dumps(dict(passed=True,images=583,outputs=1749,bindings=bindings,selection_sha256=selection_hash,records=records),indent=2))
    print(json.dumps(dict(passed=True,images=583,outputs=1749,summary=summary,viewer=str(OUT/'index.html'))),flush=True)


def progress(started):
    """@luna-comment Input:controller start time. Output:actual per-arm counts and observed ETA page.
    Key Steps:count completed receipts; report source timestamps and GPU3 restriction.
    Key Params:1749 outputs; final viewer appears only after audit. State/Dependencies:owned output.
    """
    import time
    from datetime import datetime,timedelta,timezone
    counts={a:len(list((OUT/'results'/a/'receipts').glob('*.json'))) for a in LABELS}
    n=sum(counts.values());elapsed=time.time()-started
    eta=(datetime.now(timezone(timedelta(hours=8)))+timedelta(seconds=elapsed/max(n,1)*(1749-n))).isoformat() if n else None
    state=dict(phase='inference',counts=counts,total=1749,completed=n,elapsed_seconds=elapsed,eta=eta,gpu=3,updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    (OUT/'PROGRESS.json').write_text(json.dumps(state,indent=2))
    body='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Three pretrain583 validation progress</title><style>body{font:16px system-ui;margin:30px}article{padding:20px;margin:12px 0;background:#f1f5f9;border-radius:12px}progress{width:100%}</style><h1>V3M Pretrain250K validation583</h1><p>GPU3 only | Full / Text x1.5 / adaptive-L1 cap4 | Crop8 + smooth16</p>'
    for arm in LABELS:body+='<article>'+html.escape(LABELS[arm])+f' : {counts[arm]}/583<progress value="{counts[arm]}" max="583"></progress></article>'
    (OUT/'progress.html').write_text(body+'<p>Observed ETA: '+html.escape(eta or 'warming up')+'</p><p>Source updated: '+state['updated_at']+'</p>',encoding='utf-8')


def run():
    """@luna-comment Input:three clean completed250K models and processed583 view. Output:1749 finals/report.
    Key Steps:lock, freeze exact source/config/weight/helper hashes, original ID match,
    three GPU3 workers, actual progress, require exits0 then full audit/publish.
    Key Params:finetune GPUs untouched; no double preprocessing or target substitution.
    State/Dependencies:owned output; read-only current training and historical identity map.
    """
    import fcntl,time
    from tools.build_camera_2x2_viewer import digest
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((TRAINING/'TRAINING_STATUS.json').read_text());bindings={};(OUT/'models').mkdir(exist_ok=True)
    cfgbind=json.loads((TRAINING/'CONFIG_BINDINGS.json').read_text())
    for arm in LABELS:
        assert state['children']['pretrain_'+arm]['exit_code']==0
        config=TRAINING/'pretrain/training/configs'/(arm+'.yml')
        checkpoint=TRAINING/'pretrain/training/experiments'/('v3pre_pretrain_'+arm+'_20261027')/'models/net_g_250000.pth'
        assert digest(config)==cfgbind['configs']['pretrain_'+arm]['sha256'] and checkpoint.is_file()
        bindings[arm]=dict(config=str(config),checkpoint=str(checkpoint),config_sha256=digest(config),checkpoint_sha256=digest(checkpoint),pretrain_iter=250000,metric_helper_sha256=digest(ROOT/'tools/evaluate_vosr_six_models.py'),stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
        for field,suffix in [('config','.yml'),('checkpoint','.pth')]:
            source=Path(bindings[arm][field]);target=OUT/'models'/(arm+suffix)
            if target.exists():assert digest(target)==bindings[arm][field+'_sha256']
            else:shutil.copy2(source,target)
    (OUT/'MODEL_BINDINGS.json').write_text(json.dumps(bindings,indent=2))
    rows=[json.loads(line) for line in VAL.read_text().splitlines() if line.strip()];assert len(rows)==583
    lookup={(r['dataset'],r['sample_id']):r for r in rows};assert len(lookup)==583
    original=json.loads((BEFORE/'selection.json').read_text())['samples'];assert len(original)==583
    assert set(lookup)=={(e['row']['dataset'],e['row']['sample_id']) for e in original}
    samples=[dict(id=e['id'],row=lookup[e['row']['dataset'],e['row']['sample_id']]) for e in original]
    assert all(e['row']['preprocess']['version']=='document_preprocessing_V3' for e in samples)
    (OUT/'selection.json').write_text(json.dumps(dict(source_manifest=str(VAL),source_manifest_sha256=digest(VAL),historical_identity_sha256=digest(BEFORE/'selection.json'),samples=samples),indent=2))
    started=time.time();jobs=[(arm,0,3) for arm in LABELS]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(execute,job) for job in jobs]
        while not all(f.done() for f in futures):
            progress(started)
            for f in futures:
                if f.done():assert f.result()['exit_code']==0,f.result()
            time.sleep(15)
        queue=[f.result() for f in futures]
    (OUT/'QUEUE_COMPLETE.json').write_text(json.dumps(queue,indent=2));assert all(r['exit_code']==0 for r in queue),queue
    progress(started);publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','worker','publish'));parser.add_argument('--arm',choices=tuple(LABELS));parser.add_argument('--rank',type=int,choices=(0,1));args=parser.parse_args()
    if args.mode=='worker':
        if args.arm is None or args.rank is None:parser.error('worker requires --arm and --rank')
        worker(args.arm,args.rank)
    else:dict(run=run,publish=publish)[args.mode]()
