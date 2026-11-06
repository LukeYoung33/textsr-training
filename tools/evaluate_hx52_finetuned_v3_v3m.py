"""Native HX52 four-model150K finetune evaluation with explicit0921 references."""
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
SOURCE=DATA/'HX_testset'
OUT=DATA/'hx52_finetuned_v3_v3m_0921_20261027'
MODELS=DATA/'mojito_finetuned_six_camera280_20261027'
HISTORY=Path('/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation')
TRACES=HISTORY/'v4_formal_hx_20260903/c0_cp45/traces'
LABELS={'full_v3':'V3 | Full | FT150K','full_v3m':'V3M | Full | FT150K',
        'text15_v3':'V3 | Text x1.5 | FT150K','text15_v3m':'V3M | Text x1.5 | FT150K'}
REFERENCES={
 'old0921':('0921 original V5','v3_newdeg_vs_0921_original_20261016/0921_original/summary.json','617dac8daa0e608b5df91300fa56646883f7dafb749d36ece4b01cd74ef6b575'),
 'old0921_v3':('0921-data V3 (historical reference)','v3_native_four_hx_20261015/01_0921data_degraded/summary.json','3d48a3c76c3c017f551b5a25708b5c607ed194fddbaaa118549a6b2844068a1a')}


def worker(arm,rank):
    """@luna-comment Input: named final150K arm/parity. Output:26 full-resolution HX predictions and receipts.
    Key Steps: clean extension bootstrap, hash/strict load, full-frame block restore and native PNG publication.
    Key Params:256/224/batch8; no OCR restriction, warp or tone. State/Dependencies: CUDA, frozen sources, owned outputs.
    """
    import cv2,torch,yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from tools.evaluate_vosr_six_models import restore_block
    from tools.evaluate_offline_camera_matrix import sha,save_json,image_write
    cv2.setNumThreads(1);torch.set_num_threads(2)
    binding=json.loads((OUT/'MODEL_BINDINGS.json').read_text())[arm]
    config=OUT/'models'/(arm+'.yml');checkpoint=OUT/'models'/(arm+'.pth')
    assert sha(config)==binding['config_sha256'] and sha(checkpoint)==binding['checkpoint_sha256']
    model=build_network(yaml.safe_load(config.read_text())['network_g']).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    rows=json.loads((OUT/'SOURCE_INVENTORY.json').read_text());assert len(rows)==52
    for row in rows[rank::2]:
        identity=row['id'];path=Path(row['source']);assert sha(path)==row['source_sha256']
        receipt=OUT/'results'/arm/'receipts'/(identity+'.json')
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['binding']==binding and old['source_sha256']==row['source_sha256'] and sha(old['final'])==old['final_sha256'];continue
        image=cv2.imread(str(path));assert image is not None
        result,details=restore_block(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),model,'cuda');assert result.shape==image.shape
        target=OUT/'results'/arm/'final'/(identity+'.png');image_write(target,cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
        save_json(receipt,dict(id=identity,source=str(path),source_sha256=row['source_sha256'],source_hw=list(image.shape[:2]),binding=binding,restoration=details,final=str(target),final_sha256=sha(target),adapter_sha256=sha(__file__)))
        print(json.dumps(dict(arm=arm,rank=rank,id=identity)),flush=True)
    save_json(OUT/'results'/arm/f'COMPLETE.{rank}.json',dict(images=26,binding=binding))


def execute(job):
    """@luna-comment Input: arm/parity/GPU. Output: child exit record.
    Key Steps: separate process/environment/log. Key Params:8 workers across4 GPUs.
    State/Dependencies: own children only; no existing job signals.
    """
    arm,rank,gpu=job;log=OUT/'logs'/f'{arm}.{rank}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        p=subprocess.run([sys.executable,'-u',__file__,'worker','--arm',arm,'--rank',str(rank)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    return dict(arm=arm,rank=rank,gpu=gpu,exit_code=p.returncode)


def publish():
    """@luna-comment Input:208 new and104 historical HX native predictions. Output: checked seven-choice viewer/audit.
    Key Steps: exact source names/hashes/trace/weights, final hashes and native geometry; explicit strategy labels.
    Key Params:52 original HX photos, original0921 primary baseline. State/Dependencies: immutable old results/PIL/shared renderer.
    """
    from PIL import Image
    from tools.build_camera_2x2_viewer import digest,file_url,render_html
    bindings=json.loads((OUT/'MODEL_BINDINGS.json').read_text());inventory=json.loads((OUT/'SOURCE_INVENTORY.json').read_text())
    identities={r['id'] for r in inventory};names={Path(r['source']).name for r in inventory};assert len(identities)==len(names)==52
    groups={};summaries={}
    for key,(_,relative,expected) in REFERENCES.items():
        summary=json.loads((HISTORY/relative).read_text());assert summary['status']=='passed' and summary['images']==52 and summary['checkpoint_sha256']==expected
        assert digest(summary['checkpoint'])==expected
        groups[key]={Path(r['source']).name:r for r in summary['results']};assert set(groups[key])==names
        summaries[key]=dict(checkpoint=summary['checkpoint'],checkpoint_sha256=expected,summary_sha256=digest(HISTORY/relative))
    for arm in LABELS:
        for folder,suffix in [('final','png'),('receipts','json')]:assert {p.stem for p in (OUT/'results'/arm/folder).glob('*.'+suffix)}==identities
        for field in ('config','checkpoint'):
            assert digest(bindings[arm][field])==bindings[arm][field+'_sha256']
            assert digest(OUT/'models'/(arm+('.yml' if field=='config' else '.pth')))==bindings[arm][field+'_sha256']
    rows=[];audit=[]
    for entry in inventory:
        source=Path(entry['source']);identity=entry['id'];source_hash=digest(source);assert source_hash==entry['source_sha256']
        with Image.open(source) as image:
            orientation=image.getexif().get(274,1);size=image.size;w,h=size[::-1] if orientation in (5,6,7,8) else size
        row=dict(id=identity,name=source.name,width=w,height=h,lq=file_url(source),images={});record=dict(id=identity,source_sha256=source_hash,finals={});traces=[]
        for key,group in groups.items():
            old=group[source.name];assert old['source_sha256']==source_hash and old['outside_unchanged'] and old['canvas_hw']==[96,480]
            assert digest(TRACES/(source.stem+'.json'))==old['trace_sha256'];traces.append(old['trace_sha256'])
            path=Path(old['output']);assert digest(path)==old['output_sha256']
            with Image.open(path) as image:assert image.size==(w,h)
            row['images'][key]=file_url(path);record['finals'][key]=old['output_sha256']
        assert len(set(traces))==1;record['trace_sha256']=traces[0]
        for arm in LABELS:
            receipt=json.loads((OUT/'results'/arm/'receipts'/(identity+'.json')).read_text());path=Path(receipt['final'])
            assert receipt['id']==identity and receipt['binding']==bindings[arm] and receipt['source_sha256']==source_hash and receipt['source_hw']==[h,w]
            details=receipt['restoration'];assert details['canvas_hw']==[256,256] and details['stride_hw']==[224,224] and details['ocr_cropping'] is False
            assert digest(path)==receipt['final_sha256']
            with Image.open(path) as image:assert image.size==(w,h)
            row['images'][arm]=file_url(path);record['finals'][arm]=receipt['final_sha256']
        rows.append(row);audit.append(record)
    models=[dict(key='lq',label='LQ | HX_testset',note='52 original HX photos; not NV12')]
    models.extend(dict(key=key,label=label,note='Historical96x480 OCR-region restoration; fixed traces and16px feather') for key,(label,_,_) in REFERENCES.items())
    models.extend(dict(key=key,label=label,note='Mojito finetuned150K final; whole-frame256/224, no OCR restriction or warp') for key,label in LABELS.items())
    manifest=dict(title='HX52 | Finetuned V3 / V3M versus0921',subtitle='Full / Text x1.5; synchronized native-resolution comparison',
        notes='Four models use their final150K Mojito finetune weights. Primary baseline:0921 original V5. A separate0921-data V3 historical reference is explicitly labeled. Historical baselines use native96x480 OCR-region inference; new models use whole-frame256/224 blocks. No tonal processing. Training and inference coverage differ; this is not a single-factor ablation. Native files require mapped drive Y.',defaults='old0921,full_v3,full_v3m,text15_v3m',models=models,rows=rows)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'index.html').write_text(render_html(manifest),encoding='utf-8')
    (OUT/'AUDIT.json').write_text(json.dumps(dict(passed=True,images=52,new_outputs=208,historical_outputs=104,bindings=bindings,references=summaries,records=audit),indent=2))
    print(json.dumps(dict(passed=True,images=52,new_outputs=208,viewer=str(OUT/'index.html'))),flush=True)


def run():
    """@luna-comment Input: four accepted150K snapshots and52 original HX sources. Output: checked new predictions/viewer.
    Key Steps: exclusive lock, snapshot/hash model/source identities, eight GPU workers, require all exits0, publish audit.
    Key Params:4 requested V3/V3M arms only; no V5Lite or training. State/Dependencies: own NAS outputs and readonly previous models.
    """
    import fcntl
    from tools.build_camera_2x2_viewer import digest
    OUT.mkdir(exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=json.loads((MODELS/'MODEL_BINDINGS.json').read_text());bindings={};(OUT/'models').mkdir(exist_ok=True)
    for arm in LABELS:
        bindings[arm]=baseline[arm]
        for suffix,field in [('.yml','config'),('.pth','checkpoint')]:
            source=MODELS/'models'/(arm+suffix);target=OUT/'models'/(arm+suffix);assert digest(source)==baseline[arm][field+'_sha256']
            if target.exists():assert digest(target)==digest(source)
            else:shutil.copy2(source,target)
    (OUT/'MODEL_BINDINGS.json').write_text(json.dumps(bindings,indent=2))
    files=sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'));assert len(files)==52
    inventory=[dict(id=f'{i:03d}',source=str(p),source_sha256=digest(p)) for i,p in enumerate(files,1)]
    (OUT/'SOURCE_INVENTORY.json').write_text(json.dumps(inventory,indent=2))
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
