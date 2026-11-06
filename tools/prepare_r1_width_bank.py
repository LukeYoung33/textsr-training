"""Resume-safe offline synchronized-mask and width sidecars for one L1/R1 run."""
from pathlib import Path
import sys,json,os,time,hashlib,fcntl
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data')
OUT=DATA/'v3m_dilate2_l1_r1_width_500k_20261103'
SOURCE=DATA/'original_aligned_gt_dilate2_mask_sync_20261103'
OLD=DATA/'v3m_original_dilate2_l1_500k_20261103/bank'
LOCAL=Path('/data/guang.yang/TextSR/r1_width_20261103')
VERSION='synced-mask-full-augmented-skeleton-width-v1'


def digest(path):
    """@luna-comment Input:file path. Output:SHA256. Key Steps:stream8MiB. Key Params:read-only. State/Dependencies:filesystem."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    """@luna-comment Input:owned path/JSON data. Output:atomic durable document. Key Steps:tempwrite/fsync/rename. Key Params:UTF8. State/Dependencies:own parent created."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def width_codes(mask):
    """@luna-comment Input:full augmented valid-intersected bool mask. Output:uint8 nearest-skeleton bins0/1/2. Key Steps:2xnearest EDT,max2x2,skeleton nearest propagation. Key Params:<=2/<=4/>4;emptylabel0. State/Dependencies:scipy/skimage CPU;no OCR."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    if not mask.any():return np.zeros(mask.shape,np.uint8)
    high=np.repeat(np.repeat(mask,2,0),2,1)
    distance=ndi.distance_transform_edt(high).reshape(mask.shape[0],2,mask.shape[1],2).max((1,3))
    skeleton=skeletonize(mask);nearest=ndi.distance_transform_edt(~skeleton,return_distances=False,return_indices=True)
    widths=distance[tuple(nearest)]
    return np.where(widths<=2.00001,0,np.where(widths<=4.00001,1,2)).astype(np.uint8)


def process(job):
    """@luna-comment Input:oldshard/syncedsource job. Output:verified code receipt. Key Steps:hash binding;nearest mask homographies;fullframe grouping thenexactcrop;atomic uint8 sidecar. Key Params:bit2text,low2bitswidth,originalvalidunchanged. State/Dependencies:ownedlocalfiles,8CPU pool."""
    shard,row=job;cv2.setNumThreads(1);path=LOCAL/'width_codes'/(shard['array']+'.codes.npy');receipt=path.with_suffix('.json')
    binding=dict(version=VERSION,metadata_sha256=shard['metadata_sha256'],source_mask_sha256=row['output_sha256']['text_mask'],array_sha256=shard['array_sha256'])
    if receipt.exists():
        record=json.loads(receipt.read_text());assert record['binding']==binding
        if path.exists() and digest(path)==record['sha256']:return record
        # Owned damaged/incomplete sidecars can be rebuilt from immutable sources.
    for key in ['gt','lq','valid_mask','ocr']:assert shard['source_hashes'][key]==row['output_sha256'][key],(shard['sample_id'],key)
    for key in ['text_mask','valid_mask']:assert digest(row['files'][key])==row['output_sha256'][key]
    meta_path=OLD/shard['metadata'];assert digest(meta_path)==shard['metadata_sha256'];metas=json.loads(meta_path.read_text());assert len(metas)==shard['count']
    mask=cv2.imread(row['files']['text_mask'],0);valid=cv2.imread(row['files']['valid_mask'],0);assert mask is not None and mask.shape==valid.shape
    result=np.zeros((len(metas),256,256),np.uint8);cache={};hist=np.zeros(3,np.int64);text_pixels=0;foreground=[]
    for i,meta in enumerate(metas):
        matrix=np.asarray(meta['matrix'],np.float64);key=matrix.tobytes()
        if key not in cache:
            h,w=mask.shape
            m=mask if np.array_equal(matrix,np.eye(3)) else cv2.warpPerspective(mask,matrix,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            v=valid if np.array_equal(matrix,np.eye(3)) else cv2.warpPerspective(valid,matrix,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            text=(m>127)&(v>127);codes=width_codes(text)|(text.astype(np.uint8)<<2);cache[key]=(codes,v)
        codes,v=cache[key];x,y=map(int,meta['xy']);h,w=map(int,meta['real_hw']);result[i,:h,:w]=codes[y:y+h,x:x+w]
        assert h<=256 and w<=256 and ((result[i]&3)<=2).all()
        pixels=result[i];selected=(pixels&4)>0;count=int(selected.sum());text_pixels+=count;foreground.append(count)
        hist+=np.bincount((pixels[selected]&3).ravel(),minlength=3)
    # Exact final geometry replay against existing valid planes, without rereading RGB values.
    old=np.load(OLD/shard['array'],mmap_mode='r');assert old.shape==(len(metas),256,256,8)
    for i,meta in enumerate(metas):
        x,y=meta['xy'];h,w=meta['real_hw'];v=cache[np.asarray(meta['matrix'],np.float64).tobytes()][1]
        assert np.array_equal(old[i,:h,:w,7],v[y:y+h,x:x+w])
        assert not (((result[i]&4)>0)&(old[i,...,7]<=127)).any()
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,result,allow_pickle=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path);record=dict(passed=True,binding=binding,path=str(path),sha256=digest(path),count=len(metas),sample_id=shard['sample_id'],dataset=shard['dataset'],width_hist=hist.tolist(),text_pixels=text_pixels,foreground=foreground)
    save(receipt,record);return record


def setup():
    """@luna-comment Input:complete synced source andpriorbank. Output:read-only linked bank and frozen jobs. Key Steps:identity/hash/split checks;reuse sampling andline regions;publishlinks. Key Params:7385/620;no old mutations. State/Dependencies:own metadata."""
    OUT.mkdir(exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True);complete=json.loads((SOURCE/'COMPLETE.json').read_text());assert complete['passed'] and complete['count']==7385 and complete['all_gt_changes_covered']
    assert digest(SOURCE/'manifests/train.jsonl')==complete['manifest_sha256']
    rows=[json.loads(l) for l in (SOURCE/'manifests/train.jsonl').read_text().splitlines() if l];lookup={(r['dataset'],r['sample_id']):r for r in rows}
    train=json.loads((OLD/'train.json').read_text());val=json.loads((OLD/'val.json').read_text());assert len(train['shards'])==7385 and len(val['shards'])==620
    assert not ({s['source_identity'] for s in train['shards']}&{s['source_identity'] for s in val['shards']})
    binding=dict(version=VERSION,synced_manifest_sha256=complete['manifest_sha256'],train_sha256=digest(OLD/'train.json'),val_sha256=digest(OLD/'val.json'),initial_source=str(SOURCE),base_bank=str(OLD),sidecar_root=str(LOCAL/'width_codes'))
    if (OUT/'SOURCE_BINDING.json').exists():assert json.loads((OUT/'SOURCE_BINDING.json').read_text())==binding
    else:save(OUT/'SOURCE_BINDING.json',binding)
    bank=OUT/'bank';bank.mkdir(exist_ok=True)
    for name,dest in [('sources',OLD/'sources'),('line_regions',OLD/'line_regions'),('width_codes',LOCAL/'width_codes')]:
        dest.mkdir(exist_ok=True) if name=='width_codes' else None
        link=bank/name
        if not link.is_symlink():link.symlink_to(dest,target_is_directory=True)
        assert link.resolve()==dest.resolve()
    for split,m in [('train',train),('val',val)]:
        for key in ['index','unique_index']:
            path=bank/m[key]
            if not path.exists():path.symlink_to(OLD/m[key])
        save(bank/(split+'.json'),m)
    save(bank/'COMPLETE.json',json.loads((OLD/'COMPLETE.json').read_text()))
    return [(s,lookup[s['dataset'],s['sample_id']]) for s in train['shards']]


def main():
    """@luna-comment Input:authoritative7385jobs. Output:resumable sidecars/ETA/completion. Key Steps:exclusiveprocesslock,widthunitchecks,8workers,allreceipt gate. Key Params:atomicprogress,failclosed. State/Dependencies:childprocesses;owneddata only."""
    OUT.mkdir(exist_ok=True);lock=(OUT/'width_processing.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=setup();start=time.time();records=[]
    for width in [1,2,3,4,6,8]:
        m=np.zeros((41,41),bool);m[4:37,16:16+width]=True;expected=0 if width<=2 else 1 if width<=4 else 2;assert (width_codes(m)[15:25,16:16+width]==expected).all()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(process,j) for j in jobs]):
            records.append(future.result());n=len(records);elapsed=time.time()-start
            if n%10==0 or n==len(jobs):
                eta=(len(jobs)-n)*elapsed/n;save(OUT/'DATA_STATUS.json',dict(phase='离线粗细分组',completed=n,total=len(jobs),elapsed_seconds=elapsed,eta_seconds=eta,estimated_finish=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()+eta)),updated=time.time(),workers=8,pid=os.getpid()));print(n,flush=True)
    assert len(records)==7385 and all(r['passed'] for r in records)
    save(OUT/'WIDTH_RECORDS.json',records);save(OUT/'DATA_READY.json',dict(passed=True,count=len(records),patches=sum(r['count'] for r in records),width_hist=np.sum([r['width_hist'] for r in records],0).tolist(),source_binding_sha256=digest(OUT/'SOURCE_BINDING.json'),width_records_sha256=digest(OUT/'WIDTH_RECORDS.json'),elapsed_seconds=time.time()-start))
    save(OUT/'DATA_STATUS.json',dict(phase='离线粗细分组完成',completed=7385,total=7385,eta_seconds=0,updated=time.time()))


if __name__=='__main__':main()
