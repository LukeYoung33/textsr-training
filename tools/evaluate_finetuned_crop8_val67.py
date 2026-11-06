"""Four exact finetuned150K checkpoints on the fixed67-parent validation set."""
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
OUT=DATA/'finetuned_four_crop8_val67_20261027'
BEFORE=DATA/'mojito_finetuned_six_camera280_20261027'
VAL=DATA/'six_models_mojito_finetune_20261027/source_view/manifests/val.jsonl'
LABELS={'full_v3':'V3 | Full | Finetuned150K','full_v3m':'V3M | Full | Finetuned150K',
        'text15_v3':'V3 | Text x1.5 | Finetuned150K','text15_v3m':'V3M | Text x1.5 | Finetuned150K'}


def worker(arm,rank):
    """@luna-comment Input: exact arm/parity and frozen67 selection. Output: native predictions/common metrics.
    Key Steps: strict extension load; verify frozen crop8 stitcher; training-matched stored-pixel decode; LQ-only restore; separate GT/mask scoring.
    Key Params:256/224/batch8/crop8/smooth16; common unweighted valid RGB PSNR/Y SSIM. State/Dependencies: CUDA, immutable Mojito pseudo-GT.
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
    selection=OUT/'selection.json';selection_hash=sha(selection);entries=json.loads(selection.read_text())['samples'];assert len(entries)==67
    for done,entry in enumerate(entries[rank::2],1):
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
        if arm=='text15_v3m' and identity=='030':
            accepted=cv2.imread(str(DATA/'finetuned030_feather_ab_20261027/crop8.png'),cv2.IMREAD_UNCHANGED)
            assert np.array_equal(prediction,accepted), 'Approved030 crop8 pixels differ'
        measured=shared.metrics(prediction,target,valid>0,text>0)
        baseline=shared.metrics(source,target,valid>0,text>0) if arm=='full_v3' else None
        path=OUT/'results'/arm/'final'/(identity+'.png');image_write(path,prediction)
        save_json(receipt,dict(id=identity,sample_id=row['sample_id'],dataset=row['dataset'],binding=binding,selection_sha256=selection_hash,source_hashes=hashes,source_hw=list(source.shape[:2]),final=str(path),final_sha256=sha(path),restoration=details,metrics=measured,baseline=baseline))
        print(json.dumps(dict(arm=arm,rank=rank,done=done,total=len(entries[rank::2]),id=identity)),flush=True)
    save_json(OUT/'results'/arm/f'COMPLETE.{rank}.json',dict(images=len(entries[rank::2]),binding=binding))


def execute(job):
    """@luna-comment Input: arm/parity/GPU. Output: child exit receipt.
    Key Steps: separate subprocess/environment/log. Key Params:8 jobs/four GPUs, bounded threads.
    State/Dependencies: owned processes only; no old jobs or results mutated.
    """
    arm,rank,gpu=job;log=OUT/'logs'/f'{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--rank',str(rank)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,rank=rank,gpu=gpu,exit_code=result.returncode)


def verify_row(job):
    """@luna-comment Input: selection row, four bindings and selection hash. Output: audited native viewer row/metrics.
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
    assert [receipts['full_v3']['baseline'][k]['count'] for k in ('all','text','background','ssim_y')]==support[0]
    return item,receipts


def publish():
    """@luna-comment Input: complete four-model67 finals/metric receipts. Output: audited viewer/objective tables/CSV.
    Key Steps: exact coverage and hashes, pooled SSE/count/SSIM sums, LQ baseline, six-choice synchronized renderer.
    Key Params: no training-loss weights or qualitative ranking; Mojito teacher pseudo-GT. State/Dependencies: accepted metric aggregator.
    """
    from tools.build_camera_2x2_viewer import digest,render_html
    from tools.summarize_vosr_validation import aggregate
    selection=OUT/'selection.json';selection_hash=digest(selection);entries=json.loads(selection.read_text())['samples'];assert len(entries)==67
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());identities={e['id'] for e in entries};assert len(identities)==67
    for arm in LABELS:
        for folder,suffix in [('final','png'),('receipts','json')]:assert {p.stem for p in (OUT/'results'/arm/folder).glob('*.'+suffix)}==identities
        for field,suffix in [('config','.yml'),('checkpoint','.pth')]:
            assert digest(bindings[arm][field])==bindings[arm][field+'_sha256'] and digest(OUT/'models'/(arm+suffix))==bindings[arm][field+'_sha256']
    with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify_row,[(e,bindings,selection_hash) for e in entries]))
    summary={arm:aggregate([r[arm]['metrics'] for _,r in verified]) for arm in LABELS}
    summary['lq']=aggregate([r['full_v3']['baseline'] for _,r in verified])
    (OUT/'VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    models=[dict(key='lq',label='LQ',note='Original aligned input'),dict(key='gt',label='GT | Mojito finetune validation pseudo-GT',note='Display/scoring only; not model input')]
    models.extend(dict(key=k,label=v,note='Exact finetuned150K; native256/224/batch8/crop8/smooth16; no tone/warp') for k,v in LABELS.items())
    manifest=dict(title='Validation67 | Crop8 + smooth16 | Finetuned V3 / V3M',subtitle='Full / Text x1.5 | identical native inference and target support',notes='All67 held-out Mojito finetune validation parents; scene groups disjoint from380 training parents. VOSR teacher pseudo-GT targets, not the old583 validation targets. Four final150K finetuned weights. Full-frame256/224 with discarded outer8 and remaining16px smoothstep overlap feather; GT never enters model inference. Objective tables: metrics.html. Text x1.5 describes training only; no training-loss weights enter PSNR/SSIM. Scores cannot be directly compared with old583 validation scores because images and targets differ. Visual quality is for user review. Native images require mapped Y.',defaults='lq,gt,full_v3,full_v3m',models=models,rows=[r for r,_ in verified])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    report='<!doctype html><meta charset="utf-8"><title>Finetuned67 validation metrics</title><style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:10px}</style><h1>67-parent finetuned validation | Crop8 + smooth16</h1><p><a href="index.html">Native image comparison</a> | <a href="per_image.csv">Per-image CSV</a> | <a href="VALIDATION_SUMMARY.json">Full statistics</a></p><p>Same held-out Mojito VOSR teacher pseudo-GT and valid masks for all models. RGB PSNR uses pooled pixel SSE/channel count; text/background use the same masks. Y SSIM uses Gaussian11/sigma1.5 windows wholly inside valid support. Macro finite-image PSNR is separate. No training-loss weighting or visual-quality verdict. Exact/empty values are explicit in JSON.</p><table><tr><th>Model</th><th>Images</th><th>RGB pooled PSNR</th><th>Text pooled PSNR</th><th>Background pooled PSNR</th><th>Y pooled SSIM</th><th>RGB macro PSNR</th></tr>'
    for arm in ('lq',*LABELS):
        s=summary[arm];values=[s[k]['pooled_psnr'] for k in ('all','text','background')]+[s['ssim_y']['pooled'],s['all']['macro_finite_psnr']]
        report+='<tr><td>'+html.escape(LABELS.get(arm,'LQ baseline'))+'</td><td>67</td>'+''.join('<td>'+('exact/empty' if v is None else f'{v:.6f}')+'</td>' for v in values)+'</tr>'
    (OUT/'metrics.html').write_text(report+'</table>',encoding='utf-8')
    with (OUT/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['id','sample_id','dataset','model','rgb_psnr','text_psnr','background_psnr','ssim_y'])
        for entry,(_,receipts) in zip(entries,verified):
            for arm in ('lq',*LABELS):
                m=receipts['full_v3']['baseline'] if arm=='lq' else receipts[arm]['metrics'];ssim=m['ssim_y']
                writer.writerow([entry['id'],entry['row']['sample_id'],entry['row']['dataset'],arm,*[m[k]['psnr'] for k in ('all','text','background')],ssim['sum']/ssim['count'] if ssim['count'] else None])
    records=[dict(id=item['id'],source_hashes=r['full_v3']['source_hashes'],display_lq=item['lq'],source_exif_orientation=item['source_exif_orientation'],display_lq_sha256=item.get('display_lq_sha256'),finals={a:r[a]['final_sha256'] for a in LABELS}) for item,r in verified]
    (OUT/'AUDIT.json').write_text(json.dumps(dict(passed=True,images=67,outputs=268,bindings=bindings,selection_sha256=selection_hash,records=records),indent=2))
    print(json.dumps(dict(passed=True,images=67,outputs=268,summary=summary,viewer=str(OUT/'index.html'))),flush=True)


def run():
    """@luna-comment Input: approved Mojito67 finetune validation and four finetuned150K crop8 bindings. Output: rerun finals/report.
    Key Steps: exclusive lock, freeze original selection/models/helper hashes, eight GPU workers, require exits0, audit/publish.
    Key Params: all67, no sampling/pretrained substitution. State/Dependencies: isolated NAS outputs and read-only old sources.
    """
    import fcntl
    from tools.build_camera_2x2_viewer import digest
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((BEFORE/'AUDIT.json').read_text())['bindings'];bindings={};(OUT/'models').mkdir(exist_ok=True)
    for arm in LABELS:
        bindings[arm]=dict(old[arm],metric_helper_sha256=digest(ROOT/'tools/evaluate_vosr_six_models.py'),stitcher_sha256=digest(ROOT/'tools/restore_crop8_block256.py'))
        assert Path(old[arm]['checkpoint']).name=='net_g_150000.pth'
        for field,suffix in [('config','.yml'),('checkpoint','.pth')]:
            source=Path(old[arm][field]);target=OUT/'models'/(arm+suffix);assert digest(source)==old[arm][field+'_sha256']
            if target.exists():assert digest(target)==digest(source)
            else:shutil.copy2(source,target)
    (OUT/'MODEL_BINDINGS.json').write_text(json.dumps(bindings,indent=2))
    rows=[json.loads(line) for line in VAL.read_text().splitlines() if line.strip()];assert len(rows)==67 and all(r.get('bucket','val')=='val' for r in rows)
    delivery=DATA/'lenovo_mojito_vosr_finetune_20261026'
    assert (delivery/'DELIVERY_READY.json').exists()
    report=json.loads((delivery/'delivery_report.json').read_text())
    assert report['status']=='complete' and report['val_pairs']==67 and report['train_pairs']==380 and report['scene_groups_disjoint']
    raw={split:[json.loads(line) for line in (delivery/'manifests'/(split+'.jsonl')).read_text().splitlines() if line.strip()] for split in ('train','val')}
    assert len(raw['val'])==67 and len(raw['train'])==380
    assert {r['sample_id'] for r in rows}=={r['sample_id'] for r in raw['val']}
    assert not {r['scene_group'] for r in raw['train']} & {r['scene_group'] for r in raw['val']}
    for row in rows:
        original=next(r for r in raw['val'] if r['sample_id']==row['sample_id'])
        assert row['output_sha256']['lq']==original['sha256'] and row['output_sha256']['gt']==original['gt_sha256']
    (OUT/'selection.json').write_text(json.dumps(dict(source_manifest=str(VAL),source_manifest_sha256=digest(VAL),samples=[dict(id=f'{i:03d}',row=r) for i,r in enumerate(rows,1)]),indent=2))
    jobs=[(arm,rank,i%4) for i,(arm,rank) in enumerate((a,r) for a in LABELS for r in (0,1))]
    with ThreadPoolExecutor(max_workers=8) as pool:queue=list(pool.map(execute,jobs))
    (OUT/'QUEUE_COMPLETE.json').write_text(json.dumps(queue,indent=2));assert all(r['exit_code']==0 for r in queue),queue
    publish()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','worker','publish'));parser.add_argument('--arm',choices=tuple(LABELS));parser.add_argument('--rank',type=int,choices=(0,1));args=parser.parse_args()
    if args.mode=='worker':
        if args.arm is None or args.rank is None:parser.error('worker requires --arm and --rank')
        worker(args.arm,args.rank)
    else:dict(run=run,publish=publish)[args.mode]()
