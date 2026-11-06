"""Shared no-Mojito trusted-target bank and accepted full-frame R1 width sidecars."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import sys,json,time,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_r1_width_bank import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'filtered_trusted_no_mojito_three_500k_20261105'


def width_worker_init(bank,local):
    """@luna-comment Input:owned bank/sidecar root. Output:accepted worker binding and immutable reuse inventory. Key Steps:bind worker only;load old cache metadata. Key Params:no global parent-process mutation. State/Dependencies:private child globals."""
    from tools.prepare_four_r1_sources import width_init
    from tools.reuse_filtered_width_codes import inventory
    global WIDTH_CACHE,WIDTH_BANK,WIDTH_LOCAL
    width_init(bank,local);WIDTH_CACHE=inventory();WIDTH_BANK=Path(bank);WIDTH_LOCAL=Path(local)


def width_worker(job):
    """@luna-comment Input:new source/shard. Output:verified reused or newly computed accepted codes. Key Steps:try exact geometry/mask reuse;otherwise original processor. Key Params:same normalization labels,not approximate patch-local widths. State/Dependencies:owned code receipts only."""
    from tools.reuse_filtered_width_codes import try_reuse
    from tools.prepare_four_r1_sources import width_job
    result=try_reuse(job,WIDTH_CACHE,WIDTH_BANK,WIDTH_LOCAL)
    return result if result is not None else width_job(job)


def finalize(out=OUT,count=5284):
    """@luna-comment Input:complete native train bank/source view. Output:shared positive-text train and583 non-Mojito val references. Key Steps:reuse OCR sidecars;filter draw indices;remap validation after Mojito exclusion;unify build ID. Key Params:no RGB/GT transformations. State/Dependencies:owned metadata,NAS immutable arrays."""
    from tools.run_v3pre_training import sidecars
    from tools.run_s22_nocp_text_guard import read_support,filter_sampling_index
    out=Path(out);train=out/'train_bank';old=DATA/'latest_vosr_four_500k_20261102/bank'
    if not (train/'LINE_REGIONS_COMPLETE.json').exists():sidecars(train,out/'source_view',8)
    bank=out/'bank';bank.mkdir(exist_ok=True)
    for name,target in [('train',train),('validation',old)]:
        for parent,destination in [(bank/'sources',target),(bank/'line_regions/sources',target/'line_regions')]:
            parent.mkdir(parents=True,exist_ok=True);link=parent/name
            if not link.exists():link.symlink_to(destination,target_is_directory=True)
            assert link.resolve()==destination.resolve()
    summaries={}
    for split,source,name,label in [('train',train,'train_balanced.json','train'),('val',old,'val.json','validation')]:
        m=json.loads((source/name).read_text());index=np.load(source/m['index']);unique=np.load(source/m['unique_index'])
        if split=='val':
            retained=[i for i,s in enumerate(m['shards']) if s['dataset']!='lenovo_mojito'];assert len(retained)==583
            mapping=np.full(len(m['shards']),-1,np.int64);mapping[retained]=np.arange(len(retained));arrays=[]
            for a in [index,unique]:
                a=a[np.isin(a[:,0],retained)].copy();a[:,0]=mapping[a[:,0]];arrays.append(a)
            index,unique=arrays;m['shards']=[m['shards'][i] for i in retained]
        else:assert len(m['shards'])==count
        for s in m['shards']:
            assert s['dataset']!='lenovo_mojito'
            for key in ['array','metadata']:s[key]='sources/'+label+'/'+s[key]
        if split=='train':
            with ThreadPoolExecutor(max_workers=8) as pool:flags=list(pool.map(read_support,[(bank,s) for s in m['shards']]))
            index,_=filter_sampling_index(index,flags);unique,_=filter_sampling_index(unique,flags)
        np.save(bank/(split+'.index.npy'),index);np.save(bank/(split+'.unique.npy'),unique)
        m.update(build_id='filtered_trusted_no_mojito_20261105_'+str(count),index=split+'.index.npy',unique_index=split+'.unique.npy')
        save(bank/(split+'.json'),m);summaries[split]=dict(parents=len(m['shards']),draws=len(index),unique=len(unique))
    a=json.loads((bank/'train.json').read_text());b=json.loads((bank/'val.json').read_text());assert not ({s['source_identity'] for s in a['shards']}&{s['source_identity'] for s in b['shards']})
    save(bank/'COMPLETE.json',dict(complete=True,build_id=a['build_id'],splits=summaries))
    save(out/'DATA_READY.json',dict(passed=True,counts=summaries,train_sha256=digest(bank/'train.json'),val_sha256=digest(bank/'val.json'),source_manifest_sha256=digest(out/'source_view/manifests/train.jsonl'),no_mojito_in_train_or_val=True,contrast_alignment=False,gt_mixture_probability=0))


def widths(out=OUT,count=5284):
    """@luna-comment Input:complete shared bank and adopted M2 source rows. Output:accepted augmented skeleton-width codes bound to new GT arrays. Key Steps:unchanged width processor under private worker binding;all receipts gate. Key Params:bins2/4,8CPU,NAS sidecars. State/Dependencies:source immutable,own code plane only."""
    out=Path(out);bank=out/'bank';assert json.loads((out/'DATA_READY.json').read_text())['passed']
    rows=[json.loads(l) for l in (out/'source_view/manifests/train.jsonl').read_text().splitlines() if l];lookup={(r['dataset'],r['sample_id']):r for r in rows}
    shards=json.loads((bank/'train.json').read_text())['shards'];assert len(shards)==len(lookup)==count
    local=out/'r1';(local/'width_codes').mkdir(parents=True,exist_ok=True);link=bank/'width_codes'
    if not link.exists():link.symlink_to(local/'width_codes',target_is_directory=True)
    assert link.resolve()==(local/'width_codes').resolve();records=[];started=time.time()
    with ProcessPoolExecutor(max_workers=8,initializer=width_worker_init,initargs=(str(bank),str(local))) as pool:
        for f in as_completed([pool.submit(width_worker,(s,lookup[s['dataset'],s['sample_id']])) for s in shards]):
            records.append(f.result());n=len(records)
            if n%10==0 or n==count:
                elapsed=time.time()-started;save(out/'WIDTH_STATUS.json',dict(completed=n,total=count,seconds=elapsed,eta_seconds=(count-n)*elapsed/n));print('width',n,count,flush=True)
    assert len(records)==count and all(r['passed'] for r in records)
    save(out/'WIDTH_RECORDS.json',records);save(out/'WIDTH_READY.json',dict(passed=True,count=count,records_sha256=digest(out/'WIDTH_RECORDS.json'),bank_sha256=digest(bank/'train.json')))


if __name__=='__main__':
    if sys.argv[1]=='finalize':finalize()
    elif sys.argv[1]=='widths':widths()
    else:raise ValueError('Unknown preparation stage')
