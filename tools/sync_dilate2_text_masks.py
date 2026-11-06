"""Synchronize soft foreground masks with the completed width+2px GT version."""
from pathlib import Path
import sys,json,math,copy,time,fcntl
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from tools import build_original_gt_dilate2_full as monitor
from tools.run_uniform20_steep_scurve import save,digest
SOURCE=monitor.OUT;OUT=SOURCE.parent/'original_aligned_gt_dilate2_mask_sync_20261103'
VERSION='original_aligned_GT_width_plus2_with_synced_text_mask_v2_20261103'
monitor.OUT=OUT


def expand(mask,valid):
    """@luna-comment Input:uint8 soft foreground and validity masks. Output:monotonic expanded soft mask,support. Key Steps:same8radius1 bilinear shifted samples as GT,maximum coverage. Key Params:valid>=254,source foreground>0.05. State/Dependencies:OpenCV;geometric validity unchanged."""
    assert mask.dtype==valid.dtype==np.uint8 and mask.shape==valid.shape
    m=mask.astype(np.float32)/255
    support=(cv2.dilate((m>.05).astype(np.uint8),np.ones((3,3),np.uint8))>0)&(valid>=254)
    result=m.copy()
    for angle in np.arange(8)*math.pi/4:
        shift=np.array([[1,0,math.cos(angle)],[0,1,math.sin(angle)]],np.float32)
        candidate=cv2.warpAffine(m,shift,(mask.shape[1],mask.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        source_valid=cv2.warpAffine(valid,shift,(mask.shape[1],mask.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        choose=support&(source_valid>=254)
        result[choose]=np.maximum(result[choose],candidate[choose])
    result=np.clip(np.rint(result*255),0,255).astype(np.uint8)
    assert np.all(result>=mask) and np.array_equal(result[~support],mask[~support])
    return result,support


def worker(row):
    """@luna-comment Input:completed GT manifest row. Output:new soft-mask PNG and integrity receipt. Key Steps:hash existingGT/original/masks,expand,verify changedGT support,atomic image. Key Params:8CPU,one CVthread,no GT modification. State/Dependencies:owned output only."""
    cv2.setNumThreads(1);key=row['dataset']+'/'+row['sample_id'];target=OUT/'text_mask'/(key+'.png');receipt=OUT/'receipts'/(key+'.json')
    for k in ['gt','text_mask','valid_mask']:assert digest(row['files'][k])==row['output_sha256'][k],(key,k)
    lineage=row['original_gt_lineage'];assert digest(lineage['path'])==lineage['sha256']
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['passed'] and digest(target)==r['mask_sha256'] and r['source_mask_sha256']==row['output_sha256']['text_mask'];return r
    mask=cv2.imread(row['files']['text_mask'],0);valid=cv2.imread(row['files']['valid_mask'],0);result,support=expand(mask,valid)
    gt=cv2.imread(row['files']['gt']);original=cv2.imread(lineage['path']);assert gt.shape==original.shape and mask.shape==gt.shape[:2]
    changed=np.any(gt!=original,axis=2);assert np.all(result[changed]>0),'New GT stroke support uncovered: '+key
    assert np.array_equal(result[valid<254],mask[valid<254])
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.stem+'.tmp.png');assert cv2.imwrite(str(tmp),result,[cv2.IMWRITE_PNG_COMPRESSION,3]);tmp.replace(target)
    r=dict(passed=True,key=key,mask_path=str(target),mask_sha256=digest(target),source_mask_sha256=row['output_sha256']['text_mask'],gt_sha256=row['output_sha256']['gt'],changed_gt_pixels=int(changed.sum()),grown_mask_pixels=int((result>mask).sum()),gt_changes_covered=True,monotonic=True,invalid_and_outside_support_unchanged=True)
    save(receipt,r);return r


def publish(n,total,start,phase,failed=0):
    """@luna-comment Input:work counters. Output:mask-specific status/ETA page. Key Steps:reuse owned atomic monitor,correct task wording. Key Params:15second refresh. State/Dependencies:isolated new version."""
    monitor.publish(n,total,start,phase,failed)
    p=OUT/'index.html';s=p.read_text().replace('原始对齐 GT · 笔画约 +2 px','GT +2 px · TextMask 同步').replace('复用 TextMask / 有效区域','同步扩展软 TextMask；保留有效区域').replace('逐张校验来源、尺寸、输出哈希及区域外不变','逐张校验新旧GT及mask哈希、笔画覆盖、软mask单调扩展及有效区域外不变')
    tmp=OUT/'mask_index.tmp';tmp.write_text(s);tmp.replace(p)


def main():
    """@luna-comment Input:completed7385 GT version. Output:new complete GT/mask manifest. Key Steps:lock source binding,8worker masks,coverage gate,atomic manifest andcompletion. Key Params:GT reused exactly,validity unchanged. State/Dependencies:new version only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
    complete=json.loads((SOURCE/'COMPLETE.json').read_text());assert complete['passed'] and complete['count']==7385
    path=SOURCE/'manifests/train.jsonl';assert digest(path)==complete['manifest_sha256'];rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];assert len(rows)==7385
    save(OUT/'SOURCE_PLAN.json',dict(version=VERSION,source_manifest=str(path),source_manifest_sha256=digest(path),count=len(rows),gt_version=complete['version']))
    link=OUT/'GT'
    if not link.exists():link.symlink_to(SOURCE/'GT',target_is_directory=True)
    assert link.resolve()==(SOURCE/'GT').resolve();records={};errors=[];publish(0,len(rows),start,'TextMask同步处理中')
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(worker,row):row for row in rows}
        for f in as_completed(futures):
            row=futures[f];key=row['dataset']+'/'+row['sample_id']
            try:records[key]=f.result()
            except Exception as e:errors.append(dict(key=key,error=repr(e)));save(OUT/'ERRORS.json',dict(errors=errors))
            publish(len(records),len(rows),start,'TextMask同步处理中',len(errors))
    assert not errors,errors[:3];manifest=[]
    for row in rows:
        r=records[row['dataset']+'/'+row['sample_id']];new=copy.deepcopy(row);new['original_text_mask_lineage']=dict(path=row['files']['text_mask'],sha256=row['output_sha256']['text_mask'])
        new['files']['text_mask']=r['mask_path'];new['output_sha256']['text_mask']=r['mask_sha256'];new['target_version']=VERSION
        new['target_transform']['text_mask_transform']='same8direction radius1 soft dilation;max coverage;valid support constrained';manifest.append(new)
    folder=OUT/'manifests';folder.mkdir(exist_ok=True);tmp=folder/'train.tmp';tmp.write_text(''.join(json.dumps(row)+'\n' for row in manifest));tmp.replace(folder/'train.jsonl')
    save(OUT/'COMPLETE.json',dict(passed=True,count=len(manifest),version=VERSION,manifest_sha256=digest(folder/'train.jsonl'),source_manifest_sha256=digest(path),all_gt_changes_covered=True,all_masks_monotonic=True,all_valid_regions_preserved=True,gt_unchanged=True));publish(len(rows),len(rows),start,'已完成：GT与TextMask已同步')


if __name__=='__main__':
    try:main()
    except Exception as e:save(OUT/'FAILED.json',dict(error=repr(e)));raise
