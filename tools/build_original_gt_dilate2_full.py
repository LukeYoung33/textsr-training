"""Immutable full original-aligned GT version using accepted stroke width+2px."""
from pathlib import Path
import sys,json,copy,time,os,fcntl,html
from datetime import datetime,timedelta,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.original_gt_dilation30 import thicken
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');TRAIN=DATA/'latest_vosr_four_500k_20261102'
OUT=DATA/'original_aligned_gt_dilate2_20261103'
VERSION='original_aligned_GT_stroke_width_plus2_radius1_v1_20261103'


def rows(path):
    """@luna-comment Input:published JSONL. Output:nonblank records. Key Steps:parse source only. Key Params:UTF8. State/Dependencies:immutable source."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inventory():
    """@luna-comment Input:frozen current training population and original lineages. Output:7385 hash-bound jobs. Key Steps:base original lineage;Lenovo pre-teacher croppedGT;current matching masks/LQ;exclude pseudo-only Mojito. Key Params:4873+2512. State/Dependencies:source metadata only."""
    audit=json.loads((TRAIN/'SOURCE_ELIGIBILITY.json').read_text());jobs=[]
    base=DATA/'full_s22_training_matrix_20261028/source_view/manifests/train.jsonl'
    old=DATA/'Lenovo_data_260912/training_20260927_validcrop_v4'
    original={r['sample_id']:r for r in rows(old/'manifests/train.jsonl')}
    for group,path in [('base',base),('lenovo',TRAIN/'lenovo_view/manifests/train.jsonl')]:
        allowed=set(audit['allowed'][group]);selected=[r for r in rows(path) if r['sample_id'] in allowed];assert len(selected)==len(allowed)
        for row in selected:
            if group=='base':gt=row['original_gt_lineage'];dataset=row['dataset']
            else:
                src=original[row['sample_id']];gt=dict(path=str(old/src['files']['gt']),sha256=src['output_sha256']['gt']);dataset='lenovo0912'
            files=copy.deepcopy(row['files']);files['gt']=gt['path'];hashes=copy.deepcopy(row['output_sha256']);hashes['gt']=gt['sha256']
            jobs.append(dict(key=dataset+'/'+row['sample_id'],dataset=dataset,sample_id=row['sample_id'],source_identity=row['source_identity'],files=files,source_sha256=hashes,source_group=group,crop_xywh=row.get('crop_xywh'),previous_target=dict(path=row['files']['gt'],sha256=row['output_sha256']['gt'])))
    assert len(jobs)==7385 and len({r['key'] for r in jobs})==7385
    plan=dict(version=VERSION,radius=1.,nominal_stroke_width_increase=2.,count=len(jobs),excluded_mojito=243,source_eligibility_sha256=digest(TRAIN/'SOURCE_ELIGIBILITY.json'),jobs=jobs)
    path=OUT/'SOURCE_PLAN.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Source population changed;refuse mutable version'
    else:save(path,plan)
    return jobs


def verify_reference():
    """@luna-comment Input:accepted30 width2 outputs. Output:exact replay receipt. Key Steps:hash sources,morphology,image equality. Key Params:radius1,one OpenCV thread. State/Dependencies:accepted diagnostic immutable."""
    import cv2,numpy as np
    cv2.setNumThreads(1);reference=DATA/'original_gt_dilation30_20261102/RESULTS.json';result=json.loads(reference.read_text());assert result['passed'] and len(result['rows'])==30
    verified=[]
    for row in result['rows']:
        for k in ['original','mask','valid']:assert digest(row['source'][k])==row['source_sha256'][k]
        gt=cv2.imread(row['source']['original']);mask=cv2.imread(row['source']['mask'],0);valid=cv2.imread(row['source']['valid'],0)
        value,_=thicken(gt,mask,valid,1.);path=reference.parent/row['images']['width2'];assert digest(path)==row['output_sha256']['width2']
        assert np.array_equal(value,cv2.imread(str(path))),row['id'];verified.append(row['id'])
    save(OUT/'REFERENCE_REPLAY.json',dict(passed=True,count=30,ids=verified,algorithm_sha256=digest(ROOT/'tools/original_gt_dilation30.py')))


def process(job):
    """@luna-comment Input:one hash-bound source job. Output:atomic native GT and receipt. Key Steps:verify source RGB/masks;exact morphology;outside support equality;encoded output hash;resume validation. Key Params:radius1,PNG3. State/Dependencies:owned per-key files only."""
    import cv2,numpy as np
    cv2.setNumThreads(1);receipt=OUT/'receipts'/(job['key']+'.json');target=OUT/'GT'/(job['key']+'.png')
    for k in ['gt','text_mask','valid_mask']:assert digest(job['files'][k])==job['source_sha256'][k],(job['key'],k,'hash mismatch')
    if receipt.exists():
        record=json.loads(receipt.read_text());assert record['passed'] and record['version']==VERSION and record['source_sha256']==job['source_sha256']
        assert digest(target)==record['output_sha256'];return record
    gt=cv2.imread(job['files']['gt'],cv2.IMREAD_UNCHANGED);mask=cv2.imread(job['files']['text_mask'],0);valid=cv2.imread(job['files']['valid_mask'],0)
    assert gt is not None and gt.dtype==np.uint8 and gt.ndim==3 and gt.shape[2]==3,(job['key'],'RGB format')
    assert mask is not None and valid is not None and gt.shape[:2]==mask.shape==valid.shape,(job['key'],'mask geometry')
    value,support=thicken(gt,mask,valid,1.);assert np.array_equal(value[~support],gt[~support]) and value.shape==gt.shape
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.stem+'.tmp.png');assert cv2.imwrite(str(tmp),value,[cv2.IMWRITE_PNG_COMPRESSION,3]);tmp.replace(target)
    record=dict(passed=True,version=VERSION,key=job['key'],output=str(target),output_sha256=digest(target),source_sha256=job['source_sha256'],shape_hw=list(gt.shape[:2]),changed_pixels=int(np.any(value!=gt,axis=2).sum()),outside_support_unchanged=True,radius=1.,nominal_width_increase=2.)
    save(receipt,record);return record


def publish(completed,total,start,phase,failed=0):
    """@luna-comment Input:completed counts/runtime. Output:Chinese live progress and ETA. Key Steps:rolling run average,atomic JSON/HTML. Key Params:Beijing time,15second refresh. State/Dependencies:owned output only."""
    now=datetime.now(timezone(timedelta(hours=8)));elapsed=time.time()-start;remaining=elapsed/max(completed,1)*(total-completed)
    eta=(now+timedelta(seconds=remaining)).strftime('%m-%d %H:%M') if completed>=8 else '估算中'
    state=dict(phase=phase,completed=completed,total=total,failed=failed,elapsed_seconds=elapsed,eta_seconds=remaining if completed>=8 else None,estimated_finish=eta,updated=now.isoformat(),pid=os.getpid())
    save(OUT/'STATUS.json',state)
    body=f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>原始对齐GT · 笔画+2px</title><style>body{{font:18px/1.8 system-ui;background:#eef3f8;color:#18314b;max-width:1000px;margin:40px auto}}article{{background:white;padding:28px;border-radius:15px}}progress{{width:100%}}</style><h1>原始对齐 GT · 笔画约 +2 px</h1><article><b>{html.escape(phase)}</b><h2>{completed:,} / {total:,}</h2><progress max="{total}" value="{completed}"></progress><p>预计完成：{eta}（北京时间） · 失败 {failed}</p><p>8 个 CPU 进程；复用 TextMask / 有效区域。4873 张基础集 + Lenovo 2512 张。Mojito 243 张按要求排除。</p><p>单侧半径 1 px；使用原始对齐 GT。逐张校验来源、尺寸、输出哈希及区域外不变。</p><p>更新：{now:%Y-%m-%d %H:%M:%S}</p><a href="SOURCE_PLAN.json">来源清单</a> · <a href="REFERENCE_REPLAY.json">30 张精确复现检查</a></article>'
    tmp=OUT/'index.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(OUT/'index.html')


def main():
    """@luna-comment Input:explicit7385 version update. Output:all-or-failed completion and new source manifest. Key Steps:exclusive lock;freeze inventory;reference gate;8workers;receipt coverage;publish manifest. Key Params:no in-place GT/bank edits. State/Dependencies:server-resident CPU task."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();publish(0,7385,start,'核对来源与30张样本复现')
    jobs=inventory();verify_reference();records={};errors=[];publish(0,len(jobs),start,'全量处理')
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(process,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:records[job['key']]=future.result()
            except Exception as error:
                errors.append(dict(key=job['key'],error=repr(error)));save(OUT/'ERRORS.json',dict(errors=errors))
            publish(len(records),len(jobs),start,'全量处理',len(errors))
    assert not errors,(len(errors),'see ERRORS.json');assert len(records)==len(jobs)
    manifest=[]
    for job in jobs:
        r=records[job['key']];files=copy.deepcopy(job['files']);files['gt']=r['output'];sha=copy.deepcopy(job['source_sha256']);sha['gt']=r['output_sha256']
        manifest.append(dict(bucket='train',dataset=job['dataset'],sample_id=job['sample_id'],source_identity=job['source_identity'],crop_xywh=job['crop_xywh'],files=files,output_sha256=sha,original_gt_lineage=dict(path=job['files']['gt'],sha256=job['source_sha256']['gt']),previous_training_target=job['previous_target'],target_version=VERSION,target_transform=dict(type='masked_polarity_aware_foreground_dilation',radius=1.,nominal_stroke_width_increase=2.,extra_preprocessing=False),state='published'))
    folder=OUT/'manifests';folder.mkdir(exist_ok=True);tmp=folder/'train.tmp';tmp.write_text(''.join(json.dumps(r)+'\n' for r in manifest));tmp.replace(folder/'train.jsonl')
    save(OUT/'COMPLETE.json',dict(passed=True,count=len(records),version=VERSION,source_plan_sha256=digest(OUT/'SOURCE_PLAN.json'),manifest_sha256=digest(folder/'train.jsonl'),changed_images=sum(r['changed_pixels']>0 for r in records.values()),excluded_mojito=243,validation='not processed;training targets only',finished=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    publish(len(records),len(jobs),start,'已完成')


if __name__=='__main__':
    try:main()
    except Exception as error:
        save(OUT/'FAILED.json',dict(error=repr(error)));raise
