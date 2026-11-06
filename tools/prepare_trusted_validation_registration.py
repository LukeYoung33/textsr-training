"""Replay accepted teacher, screening and trusted repair on historical validation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import argparse, copy, hashlib, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'trusted_gt_validation_20261106'
OLD=DATA/'mask_gt_filtered_trusted_20261105'
TRAIN=DATA/'filtered_trusted_no_mojito_three_500k_20261105/source_view/manifests/train.jsonl'
VAL=DATA/'full_s22_training_matrix_20261028/source_view/manifests/val.jsonl'
PYTHON='/data/guang.yang/vosr2_u1_tile512/orientation_env/bin/python'


def digest(path):
    """@luna-comment Input:file. Output:SHA256. Key Steps:stream1MiB. Key Params:read-only. State/Dependencies:hashlib/filesystem."""
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):result.update(chunk)
    return result.hexdigest()


def save(path,value):
    """@luna-comment Input:owned path/JSON. Output:atomic receipt. Key Steps:sibling temp/replace. Key Params:UTF8. State/Dependencies:isolated directory."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def rows(path):
    """@luna-comment Input:JSONL. Output:records. Key Steps:parse nonblank lines. Key Params:no edits. State/Dependencies:filesystem."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def consensus(metrics,thresholds,rules):
    """@luna-comment Input:original measurements/frozen rules. Output:historical consensus decision. Key Steps:teacher/LQ corroboration then existing decide. Key Params:.45NCC,1.75px,.04gain,2px agreement. State/Dependencies:pure copy/numpy/screener."""
    import numpy as np
    from tools.screen_mask_gt_quality import decide
    result=copy.deepcopy(metrics);patches=result['patches']
    confirmed=[p for p in patches if p['gt']['peak']>=.45 and np.hypot(p['gt']['dx'],p['gt']['dy'])>=1.75 and p['gt']['peak']-p['gt']['zero']>=.04 and p['lq']['peak']>=rules['minimum_lq_correlation'] and np.hypot(p['gt']['dx']-p['lq']['dx'],p['gt']['dy']-p['lq']['dy'])<=rules['maximum_gt_lq_displacement_difference']]
    result['misaligned_fraction']=len(confirmed)/max(len(patches),1)
    result['misaligned_text_fraction']=sum(p['text_pixels'] for p in confirmed)/max(sum(p['text_pixels'] for p in patches),1)
    return result,decide(result,thresholds)


def verify_sources(row):
    """@luna-comment Input:manifest row. Output:native shape. Key Steps:hash all assets/decode pair and masks. Key Params:equal geometry. State/Dependencies:read-only OpenCV/files."""
    import cv2
    for key,path in row['files'].items():assert digest(path)==row['output_sha256'][key],(row['sample_id'],key)
    images=[cv2.imread(row['files'][k],cv2.IMREAD_UNCHANGED) for k in ['lq','gt','text_mask','valid_mask']]
    assert all(image is not None for image in images) and len({image.shape[:2] for image in images})==1
    return list(images[0].shape[:2])


def prepare():
    """@luna-comment Input:583val/5284train and frozen adapter. Output:bound isolated teacher inputs. Key Steps:hash/disjointness checks;restore original detector;redirect paths. Key Params:unchanged route/model/partition. State/Dependencies:NAS/deployed environment;old artifacts immutable."""
    validation,training=rows(VAL),rows(TRAIN);assert len(validation)==583 and len(training)==5284
    assert len({(r['dataset'],r['sample_id']) for r in validation})==583 and all(r['dataset']!='lenovo_mojito' for r in validation)
    assert not ({r['source_identity'] for r in validation}&{r['source_identity'] for r in training})
    assert not ({r['output_sha256']['lq'] for r in validation}&{r['output_sha256']['lq'] for r in training})
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:shapes=list(pool.map(verify_sources,validation))
    for split in ['train','val']:(OUT/'teacher_source'/split/'LQ').mkdir(parents=True,exist_ok=True)
    bindings=[]
    for index,(row,shape) in enumerate(zip(validation,shapes)):
        name=f'{index:05d}__'+row['sample_id']+'.png';link=OUT/'teacher_source/val/LQ'/name
        if not link.exists():link.symlink_to(row['files']['lq'])
        assert link.resolve()==Path(row['files']['lq']).resolve();bindings.append(dict(name=name,shape_hw=shape,row=row))
    save(OUT/'SOURCE_ROWS.json',bindings);binding=json.loads((OLD/'BINDING.json').read_text())
    for path,expected in binding['code_sha256'].items():assert digest(ROOT/path)==expected,('Frozen code changed',path)
    save(OUT/'SCREENING_RULES.json',dict(thresholds=binding['thresholds'],final_rules=json.loads((OLD/'FINAL_RULES.json').read_text())))
    snapshot=ROOT/'evaluation/trusted_validation_teacher_snapshot_20261106.py';text=snapshot.read_text()
    replacements={"DATA=Path('/AI_DNS/guang.yang/TextSR/data/full_s22_training_matrix_20261028/teacher_source')":'DATA=Path('+repr(str(OUT/'teacher_source'))+')',"RUN=Path('/AI_DNS/guang.yang/TextSR/data/full_s22_training_matrix_20261028/teacher_run')":'RUN=Path('+repr(str(OUT/'teacher_run'))+')','# Detector results are reused from the source-bound historical cache.':'from paddleocr import TextDetection',"OCR_CACHE=json.loads((DATA/'OCR_CACHE.json').read_text());print('CACHED_OCR_ONLY',len(OCR_CACHE),flush=True)":"det=TextDetection(model_name='PP-OCRv5_mobile_det',model_dir=str(BASE/'ocr_eval_models/PP-OCRv5_mobile_det_infer'),device='cpu',enable_mkldnn=False,cpu_threads=2)","prediction=OCR_CACHE[path.name];assert prediction['size']==[w,h];lines=[];patches=[]":'prediction=next(iter(det.predict(src[:,:,::-1].copy())));lines=[];patches=[]'}
    for old,new in replacements.items():assert text.count(old)==1,old;text=text.replace(old,new)
    adapter=OUT/'teacher_adapter.py'
    if adapter.exists():assert adapter.read_text()==text
    else:adapter.write_text(text)
    save(OUT/'BINDING.json',dict(passed=True,validation_count=583,training_count=5284,validation_manifest_sha256=digest(VAL),training_manifest_sha256=digest(TRAIN),source_rows_sha256=digest(OUT/'SOURCE_ROWS.json'),adapter_sha256=digest(adapter),replacements=replacements,rules_sha256=digest(OUT/'SCREENING_RULES.json'),training_binding_sha256=digest(OLD/'BINDING.json'),identities_disjoint=True,lq_hashes_disjoint=True,source_code_sha256=digest(__file__)))
    print('PREPARED',len(bindings),flush=True)


def classify_smoke():
    """@luna-comment Input:8352 original/final historical receipts. Output:exact consensus parity. Key Steps:replay measurements/compare all decisions. Key Params:no new images. State/Dependencies:immutable JSON."""
    source=json.loads((OLD/'SCREEN_RECORDS.json').read_text());expected={r['id']:r['decision'] for r in json.loads((OLD/'FINAL_DECISIONS.json').read_text())['records']};frozen=json.loads((OUT/'SCREENING_RULES.json').read_text())
    for record in source:
        _,actual=consensus(record['measurements'],frozen['thresholds'],frozen['final_rules']);assert actual==expected[record['id']],record['id']
    save(OUT/'CONSENSUS_PARITY.json',dict(passed=True,historical_cases=len(source),different_decisions=0));print('CONSENSUS_PARITY',len(source),flush=True)


def process(job):
    """@luna-comment Input:bound validation/completed teacher route. Output:trusted row or explicit rejection. Key Steps:S2.2,original-mask screening,unchanged M2/full repair,trusted pockets. Key Params:frozen functions/rules. State/Dependencies:isolated stage outputs/source hash checks."""
    import cv2
    from tools.run_uniform20_steep_scurve import curve_pair
    from tools.screen_mask_gt_quality import measure
    from tools import prepare_mask_repaired_vosr_mix50 as strong
    from tools import run_filtered_trusted_gt as trusted
    cv2.setNumThreads(1);item,route=job;row=copy.deepcopy(item['row']);key=row['dataset']+'/'+row['sample_id'];receipt=OUT/'receipts'/(key+'.json')
    signature=dict(source_hashes=copy.deepcopy(row['output_sha256']),teacher_sha256=digest(route['output']),binding_sha256=digest(OUT/'BINDING.json'))
    if receipt.exists():
        previous=json.loads(receipt.read_text());assert previous['signature']==signature
        if previous['decision']['keep']:assert digest(previous['row']['files']['gt'])==previous['row']['output_sha256']['gt']
        return previous
    verify_sources(row);image=cv2.imread(route['output']);assert list(image.shape[:2])==item['shape_hw']
    folder=OUT/'s22_targets'/key;folder.mkdir(parents=True,exist_ok=True);curves=curve_pair(image,folder)
    row['files']['gt']=str(folder/'steep_s.png');row['output_sha256']['gt']=curves['arms']['steep_s']['sha256']
    row['target_refresh']=dict(version='V3LQ_full_documented_VOSR_S2.2',teacher_output=route['output'],teacher_sha256=signature['teacher_sha256'],alpha=.75,gamma=2.6,midpoint_derivative=2.2,exterior_slope=.1,no_second_v3=True);row['repair_mode']='stroke_repair'
    frozen=json.loads((OUT/'SCREENING_RULES.json').read_text());metrics,decision=consensus(measure(row),frozen['thresholds'],frozen['final_rules'])
    quality=dict(id=item['name'].split('__')[0],dataset=row['dataset'],sample_id=row['sample_id'],measurements=metrics,decision=decision)
    save(OUT/'trusted/screen'/(key+'.json'),quality);result=dict(passed=True,signature=signature,dataset=row['dataset'],sample_id=row['sample_id'],source_identity=row['source_identity'],decision=decision,quality=quality)
    if decision['keep']:
        strong.OUT=OUT/'strong_repair';repaired=strong.process(row);trusted.OUT=OUT/'trusted';completed=trusted.repair_one((repaired['row'],quality,signature['binding_sha256']))
        result.update(row=completed['row'],strong_audit=repaired['audit'],trusted_audit=completed['audit']);verify_sources(completed['row'])
    save(receipt,result);return result


def status(phase,completed,total,started,**extra):
    """@luna-comment Input:measured phase/counts/time. Output:progress. Key Steps:elapsed/rate ETA. Key Params:stage-only. State/Dependencies:owned STATUS.json."""
    elapsed=time.time()-started;save(OUT/'STATUS.json',dict(phase=phase,completed=completed,total=total,elapsed_seconds=elapsed,eta_seconds=(total-completed)*elapsed/completed if completed else None,updated=time.time(),pid=os.getpid(),**extra))


def run():
    """@luna-comment Input:authorized583 replay. Output:retained/rejected validation manifests. Key Steps:exclusive lock,4GPU teacher,8CPU exact processing,full hash checks. Key Params:no training changes. State/Dependencies:owned root/existing deployment."""
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'COMPLETE.json').exists():return
    prepare();classify_smoke();items=json.loads((OUT/'SOURCE_ROWS.json').read_text())
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');children=[];streams=[];started=time.time()
    for gpu in range(4):
        stream=(OUT/f'teacher{gpu}.log').open('ab');streams.append(stream);children.append(subprocess.Popen([PYTHON,'-u',str(OUT/'teacher_adapter.py'),str(gpu)],stdout=stream,stderr=subprocess.STDOUT,env=environment))
    save(OUT/'TEACHER_LAUNCH.json',dict(pids=[p.pid for p in children],started=started))
    while any(child.poll() is None for child in children):
        assert all(child.poll() in (None,0) for child in children),'Teacher failed; inspect teacher logs'
        status('teacher',len(list((OUT/'teacher_run').glob('worker*/batch*/*_done.json'))),len(items),started);time.sleep(10)
    assert all(child.returncode==0 for child in children)
    for stream in streams:stream.close()
    routes={}
    for routing in sorted((OUT/'teacher_run').glob('worker*/batch*/routing.json')):
        for route in json.loads(routing.read_text()):
            done=json.loads((routing.parent/(str(route['index'])+'_done.json')).read_text());assert done['verified'] and not done['quality_gate'];assert len(route['patches'])==sum(line['large'] for line in route['lines'])
            name=Path(route['input']).name;assert name not in routes;routes[name]=route
    assert set(routes)=={item['name'] for item in items};started=time.time();results=[];status('screen_repair',0,len(items),started)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(process,(item,routes[item['name']])) for item in items]
        for future in as_completed(futures):
            results.append(future.result());status('screen_repair',len(results),len(items),started,retained=sum(r['decision']['keep'] for r in results))
    results.sort(key=lambda r:(r['dataset'],r['sample_id']));retained=[r['row'] for r in results if r['decision']['keep']];rejected=[r for r in results if not r['decision']['keep']]
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify_sources,[item['row'] for item in items]))
    binding=json.loads((OUT/'BINDING.json').read_text());assert digest(VAL)==binding['validation_manifest_sha256'] and digest(TRAIN)==binding['training_manifest_sha256']
    manifest=OUT/'source_view/manifests/val.jsonl';manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(''.join(json.dumps(row)+'\n' for row in retained))
    save(OUT/'REJECTED.json',rejected);save(OUT/'FINAL_DECISIONS.json',results)
    save(OUT/'COMPLETE.json',dict(passed=True,total=583,retained=len(retained),rejected=len(rejected),manifest_sha256=digest(manifest),binding_sha256=digest(OUT/'BINDING.json'),original_sources_unchanged=True,training_unchanged=True,identities_disjoint=True));status('complete',len(results),len(items),started,retained=len(retained),rejected=len(rejected))


def materialize(split):
    """@luna-comment Input:complete trusted train/val manifest. Output:independent paired registry directory. Key Steps:8thread copy,source/destination hashes,resumable identity checks,full provenance receipt. Key Params:only LQ/GT images copied;mask/OCR references preserved. State/Dependencies:owned registry output,immutable originals."""
    import shutil
    source=TRAIN if split=='train' else OUT/'source_view/manifests/val.jsonl'
    if split=='val':assert json.loads((OUT/'COMPLETE.json').read_text())['passed']
    source_sha=digest(source);records=rows(source);target=DATA/'registry/vosr_gt_repaired_20261106'/split
    target.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for row in records:
        for key,folder in [('lq','LQ'),('gt','GT')]:
            destination=target/folder/row['dataset']/(row['sample_id']+'.png')
            jobs.append((row['files'][key],row['output_sha256'][key],destination))
    assert len({str(j[2]) for j in jobs})==len(jobs)
    def copy_image(job):
        """@luna-comment Input:source/expected SHA/destination. Output:verified byte count. Key Steps:hash source;independent atomic copy;rehash both. Key Params:no links/overwrite of untracked mismatches. State/Dependencies:owned per-image file only."""
        original,expected,destination=job;assert digest(original)==expected
        if destination.exists():assert not destination.is_symlink() and digest(destination)==expected
        else:
            destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.copying');shutil.copyfile(original,temporary);assert digest(temporary)==expected;temporary.replace(destination)
        assert digest(original)==expected
        return destination.stat().st_size
    started=time.time();completed=0;total_bytes=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(copy_image,job) for job in jobs]
        for future in as_completed(futures):
            total_bytes+=future.result();completed+=1
            if completed%100==0 or completed==len(jobs):save(target/'COPY_STATUS.json',dict(phase='copying',completed=completed,total=len(jobs),bytes=total_bytes,elapsed_seconds=time.time()-started))
    assert digest(source)==source_sha
    snapshot=target/'SOURCE_MANIFEST.jsonl';snapshot.write_bytes(source.read_bytes());assert digest(snapshot)==source_sha
    receipt=dict(passed=True,split=split,pairs=len(records),images=len(jobs),image_bytes=total_bytes,source_manifest=str(source),source_manifest_sha256=source_sha,pair_rule=dict(mode='dirs',input='LQ',target='GT'),independent_copies=True,all_sha256_verified=True,masks_and_ocr='References and hashes in SOURCE_MANIFEST.jsonl; not copied into image roots',seconds=time.time()-started)
    save(target/'COMPLETE.json',receipt);save(target/'COPY_STATUS.json',dict(phase='complete',**receipt));print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--materialize',choices=['train','val']);args=parser.parse_args()
    try:
        if args.materialize:materialize(args.materialize)
        elif args.prepare_only:prepare()
        else:run()
    except Exception as error:
        import traceback
        save(OUT/'LAST_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),updated=time.time()));raise
