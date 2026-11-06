"""Frozen original GT nominal stroke+1px with synchronized masks and no contrast."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone,timedelta
import copy
import fcntl
import json
import math
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.original_gt_dilation30 import thicken
from tools.run_uniform20_steep_scurve import save,digest
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'original_aligned_gt_dilate1_20261104'
VERSION = 'original_aligned_GT_stroke_plus1_radius0p5_syncedmask_no_contrast_20261104'


def expand_mask(mask,valid):
    """@luna-comment Input:uint8 softtext/valid masks. Output:radius.5expandedsoftmask/support. Key Steps:8bilinearhalf-pixelshifts,maxcoverage,validgates. Key Params:supportm>.05,valid>=254;monotonic. State/Dependencies:OpenCV;inputimmutable."""
    import cv2
    import numpy as np
    m = mask.astype(np.float32)/255
    support = (cv2.dilate((m>.05).astype(np.uint8),np.ones((3,3),np.uint8))>0)&(valid>=254)
    result = m.copy()
    for angle in np.arange(8)*math.pi/4:
        shift = np.array([[1,0,.5*math.cos(angle)],[0,1,.5*math.sin(angle)]],np.float32)
        candidate = cv2.warpAffine(m,shift,(mask.shape[1],mask.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        source_valid = cv2.warpAffine(valid,shift,(mask.shape[1],mask.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        selected = support&(source_valid>=254)
        result[selected] = np.maximum(result[selected],candidate[selected])
    result = np.clip(np.rint(result*255),0,255).astype(np.uint8)
    assert np.all(result>=mask) and np.array_equal(result[~support],mask[~support])
    return result,support


def process(job):
    """@luna-comment Input:onefrozenoriginalsourceGT/masks. Output:newGT/maskPNGs andreceipt. Key Steps:sourcehashgate,resumehashcheck,acceptedradius.5GT,matchingmask,locality/coverage,atomicsave. Key Params:nocontrastorpreprocessing. State/Dependencies:ownednewversiononly,OpenCV1thread."""
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    target = OUT/'GT'/(job['key']+'.png')
    mask_target = OUT/'text_mask'/(job['key']+'.png')
    receipt = OUT/'receipts'/(job['key']+'.json')
    for key in ['gt','text_mask','valid_mask']:
        assert digest(job['files'][key])==job['source_sha256'][key],(job['key'],key)
    if receipt.exists():
        r = json.loads(receipt.read_text())
        assert r['passed'] and r['version']==VERSION and r['source_sha256']==job['source_sha256']
        assert digest(target)==r['gt_sha256'] and digest(mask_target)==r['mask_sha256']
        return r
    gt = cv2.imread(job['files']['gt'],cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(job['files']['text_mask'],0)
    valid = cv2.imread(job['files']['valid_mask'],0)
    assert gt is not None and gt.ndim==3 and gt.shape[2]==3 and gt.dtype==np.uint8
    assert gt.shape[:2]==mask.shape==valid.shape
    value,support = thicken(gt,mask,valid,.5)
    new_mask,mask_support = expand_mask(mask,valid)
    changed = np.any(value!=gt,axis=2)
    assert np.array_equal(support,mask_support)
    assert np.array_equal(value[~support],gt[~support])
    assert np.array_equal(value[valid<254],gt[valid<254])
    assert np.array_equal(new_mask[valid<254],mask[valid<254])
    assert np.all(new_mask[changed]>0)
    for p,image in [(target,value),(mask_target,new_mask)]:
        p.parent.mkdir(parents=True,exist_ok=True)
        tmp = p.with_name(p.stem+'.tmp.png')
        assert cv2.imwrite(str(tmp),image,[cv2.IMWRITE_PNG_COMPRESSION,3])
        tmp.replace(p)
    record = dict(passed=True,version=VERSION,key=job['key'],gt=str(target),text_mask=str(mask_target),
                  gt_sha256=digest(target),mask_sha256=digest(mask_target),source_sha256=job['source_sha256'],
                  shape_hw=list(gt.shape[:2]),changed_gt_pixels=int(changed.sum()),grown_mask_pixels=int((new_mask>mask).sum()),
                  radius=.5,nominal_width_increase=1.,contrast_alignment_applied=False,
                  outside_support_unchanged=True,invalid_unchanged=True,gt_changes_covered=True,mask_monotonic=True)
    save(receipt,record)
    return record


def reference():
    """@luna-comment Input:previous30width1evidence. Output:exactGTreplay plusnewmaskcoveragegate. Key Steps:rehashoriginal/reference,acceptedGTreplay,syncmaskasserts. Key Params:radius.5,all30. State/Dependencies:readonlyoldapprovedoutputs."""
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    path = DATA/'original_gt_dilation30_20261102/RESULTS.json'
    d = json.loads(path.read_text())
    assert d['passed'] and len(d['rows'])==30
    for row in d['rows']:
        for key in ['original','mask','valid']:
            assert digest(row['source'][key])==row['source_sha256'][key]
        gt = cv2.imread(row['source']['original'])
        mask = cv2.imread(row['source']['mask'],0)
        valid = cv2.imread(row['source']['valid'],0)
        output,support = thicken(gt,mask,valid,.5)
        expected = path.parent/row['images']['width1']
        assert digest(expected)==row['output_sha256']['width1'] and np.array_equal(output,cv2.imread(str(expected)))
        grown,_ = expand_mask(mask,valid)
        assert np.all(grown[np.any(output!=gt,axis=2)]>0)
    save(OUT/'REFERENCE_REPLAY.json',dict(passed=True,count=30,exact_width1_replay=True,masks_synced=True,radius=.5))


def publish(done,total,started,phase,failed=0):
    """@luna-comment Input:progress/runtime. Output:atomicChineseETApage/JSON. Key Steps:observedthroughputafter8completions,Beijingtime. Key Params:15srefresh,CPU8. State/Dependencies:ownedmetadataonly."""
    now = datetime.now(timezone(timedelta(hours=8)))
    elapsed = time.time()-started
    remaining = elapsed/max(done,1)*(total-done) if done>=8 else None
    finish = (now+timedelta(seconds=remaining)).strftime('%m-%d %H:%M') if remaining is not None else '采样中'
    save(OUT/'STATUS.json',dict(phase=phase,completed=done,total=total,failed=failed,elapsed_seconds=elapsed,
         eta_seconds=remaining,estimated_finish=finish,updated=now.isoformat(),pid=os.getpid()))
    page = f'<meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>原始GT＋1px</title><style>body{{font:18px/1.8 system-ui;margin:40px;background:#eef3f8}}progress{{width:700px}}</style><h1>原始对齐GT · 笔画总宽约＋1px</h1><p>{phase}</p><h2>{done} / {total}</h2><progress value="{done}" max="{total}"></progress><p>预计完成：{finish} 北京时间；失败 {failed}</p><p>单侧半径0.5px；同步扩展TextMask；不做对比度对齐、V3、S-curve或VOSR处理。</p><p>使用相同7385张原始GT；LQ和有效mask不变；不启动训练。增粗作用范围外GT逐像素不变。</p><p>8个CPU进程；更新时间 {now:%m-%d %H:%M:%S}</p>'
    tmp = OUT/'index.tmp'
    tmp.write_text(page,encoding='utf-8')
    tmp.replace(OUT/'index.html')


def main():
    """@luna-comment Input:userauthorized+1/no-contrastversion. Output:resident7385GT/masks/fullmanifest orfailedreceipt. Key Steps:freezepriorjobs,30referencegate,CPU8workers,exactcoverage/publication. Key Params:separateimmutableversion,nocontrast/training. State/Dependencies:NAS/sourcehashes/exclusivelock."""
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'build.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started = time.time()
    save(OUT/'CONTROLLER.json',dict(pid=os.getpid(),started=started))
    path = DATA/'original_aligned_gt_dilate2_20261103/SOURCE_PLAN.json'
    prior = json.loads(path.read_text())
    jobs = prior['jobs']
    assert len(jobs)==len({j['key'] for j in jobs})==7385
    plan = dict(version=VERSION,radius=.5,nominal_stroke_width_increase=1.,contrast_alignment_applied=False,
         count=7385,excluded_mojito=243,parent_plan=str(path),parent_plan_sha256=digest(path),jobs=jobs)
    if (OUT/'SOURCE_PLAN.json').exists():
        assert json.loads((OUT/'SOURCE_PLAN.json').read_text())==plan
    else:
        save(OUT/'SOURCE_PLAN.json',plan)
    publish(0,7385,started,'核对30张＋1px精确复现与TextMask覆盖')
    reference()
    started = time.time()
    publish(0,7385,started,'生成＋1px GT及同步TextMask')
    records = {}
    errors = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process,job):job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                records[job['key']] = future.result()
            except Exception as error:
                errors.append(dict(key=job['key'],error=repr(error)))
                save(OUT/'ERRORS.json',dict(errors=errors))
            publish(len(records),7385,started,'生成＋1px GT及同步TextMask',len(errors))
    assert not errors,errors[:3]
    manifest = []
    for job in jobs:
        r = records[job['key']]
        files = copy.deepcopy(job['files'])
        hashes = copy.deepcopy(job['source_sha256'])
        files.update(gt=r['gt'],text_mask=r['text_mask'])
        hashes.update(gt=r['gt_sha256'],text_mask=r['mask_sha256'])
        manifest.append(dict(dataset=job['dataset'],sample_id=job['sample_id'],source_identity=job['source_identity'],
             files=files,output_sha256=hashes,crop_xywh=job['crop_xywh'],
             original_gt_lineage=dict(path=job['files']['gt'],sha256=job['source_sha256']['gt']),
             original_text_mask_lineage=dict(path=job['files']['text_mask'],sha256=job['source_sha256']['text_mask']),
             target_version=VERSION,target_transform=dict(radius=.5,nominal_width_increase=1.,text_mask_synchronized=True,contrast_alignment=False)))
    folder = OUT/'manifests'
    folder.mkdir(exist_ok=True)
    tmp = folder/'train.tmp'
    tmp.write_text(''.join(json.dumps(row)+'\n' for row in manifest))
    tmp.replace(folder/'train.jsonl')
    save(OUT/'COMPLETE.json',dict(passed=True,count=7385,version=VERSION,manifest_sha256=digest(folder/'train.jsonl'),
         source_plan_sha256=digest(OUT/'SOURCE_PLAN.json'),radius=.5,nominal_width_increase=1.,
         contrast_alignment_applied=False,all_masks_synced=True,all_outside_support_unchanged=True,training_started=False))
    publish(7385,7385,started,'已完成：GT＋1px，无对比度对齐，TextMask已同步')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        save(OUT/'FAILED.json',dict(error=repr(error)))
        raise
