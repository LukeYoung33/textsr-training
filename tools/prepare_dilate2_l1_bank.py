"""Build one independent7385-parent dilated-original-GT bank with unchanged620 validation."""
from pathlib import Path
import sys,json,copy,subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');SOURCE=DATA/'original_aligned_gt_dilate2_20261103';BASE=DATA/'latest_vosr_four_500k_20261102';OUT=DATA/'v3m_original_dilate2_l1_500k_20261103'


def main():
    """@luna-comment Input:immutable7385GTversionandpriorvalidationbank. Output:hash-boundfilteredtrain/valmanifests. Key Steps:strictsourcepopulation,8CPUnewbank/sidecars,referenceoldvalwithoutmodification. Key Params:nonepreprocessing,validtextonly. State/Dependencies:owneddatasets,allsourcefilesread-only."""
    import numpy as np
    from tools.run_v3pre_training import sidecars
    from tools.run_s22_nocp_text_guard import read_support,filter_sampling_index
    OUT.mkdir(exist_ok=True)
    if (OUT/'DATA_READY.json').exists():return
    complete=json.loads((SOURCE/'COMPLETE.json').read_text());assert complete['passed'] and complete['count']==7385 and digest(SOURCE/'manifests/train.jsonl')==complete['manifest_sha256']
    rows=[json.loads(l) for l in (SOURCE/'manifests/train.jsonl').read_text().splitlines() if l.strip()];assert len(rows)==7385 and all(r['dataset']!='lenovo_mojito' for r in rows)
    allowed=json.loads((BASE/'SOURCE_ELIGIBILITY.json').read_text())['allowed'];assert {r['sample_id'] for r in rows}==set(allowed['base'])|set(allowed['lenovo'])
    val=json.loads((BASE/'bank/val.json').read_text());assert len(val['shards'])==620
    assert not ({r['source_identity'] for r in rows}&{r['source_identity'] for r in val['shards']})
    view=OUT/'source_view';(view/'manifests').mkdir(parents=True,exist_ok=True)
    target=view/'manifests/train.jsonl'
    if target.exists():assert digest(target)==complete['manifest_sha256']
    else:target.write_bytes((SOURCE/'manifests/train.jsonl').read_bytes())
    (view/'manifests/val.jsonl').write_text('');save(OUT/'SOURCE_BINDING.json',dict(complete=complete,source_complete_sha256=digest(SOURCE/'COMPLETE.json'),train_manifest_sha256=digest(target),train_count=7385,val_count=620,excluded_mojito=243,lq='existing V3 cache',gt='original aligned plus nominal2px;no extra preprocessing'))
    train=OUT/'train_bank'
    if not (train/'COMPLETE.json').exists():
        save(OUT/'DATA_STATUS.json',dict(phase='构建7385张新GT训练patch',total=7385))
        with (OUT/'bank.log').open('ab') as f:subprocess.run([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(view),'--output',str(train),'--workers','8','--preprocess-mode','none'],stdout=f,stderr=subprocess.STDOUT,check=True)
    if not (train/'LINE_REGIONS_COMPLETE.json').exists():
        save(OUT/'DATA_STATUS.json',dict(phase='复用OCR生成文字几何sidecar'));sidecars(train,view,8)
    old=json.loads((BASE/'bank/train.json').read_text());lookup={(s['dataset'],s['sample_id']):s for s in old['shards']}
    new=json.loads((train/'train_balanced.json').read_text());replay=[]
    for dataset in sorted({s['dataset'] for s in new['shards']}):
        group=[s for s in new['shards'] if s['dataset']==dataset]
        for s in [group[0],group[-1]]:
            prior=lookup[s['dataset'],s['sample_id']];a=np.load(train/s['array'],mmap_mode='r');b=np.load(BASE/'bank'/prior['array'],mmap_mode='r')
            assert a.shape==b.shape and np.array_equal(a[...,:3],b[...,:3]) and np.array_equal(a[...,6:],b[...,6:]),(dataset,s['sample_id'],'non-GT channels differ')
            assert json.loads((train/s['metadata']).read_text())==json.loads((BASE/'bank'/prior['metadata']).read_text())
            replay.append(dict(dataset=dataset,sample_id=s['sample_id'],lq_masks_geometry_equal=True))
    save(OUT/'NON_GT_REPLAY.json',dict(passed=True,records=replay))
    bank=OUT/'bank';bank.mkdir(exist_ok=True)
    for name,source in [('train',train),('validation',BASE/'bank')]:
        for parent,destination in [(bank/'sources',source),(bank/'line_regions/sources',source/'line_regions')]:
            parent.mkdir(parents=True,exist_ok=True);link=parent/name
            if not link.exists():link.symlink_to(destination,target_is_directory=True)
            assert link.resolve()==destination.resolve()
    counts={};save(OUT/'DATA_STATUS.json',dict(phase='筛选有效文字patch并绑定原验证集'))
    for split,source,label,name in [('train',train,'train','train_balanced.json'),('val',BASE/'bank','validation','val.json')]:
        m=copy.deepcopy(json.loads((source/name).read_text()));index=np.load(source/m['index']);unique=np.load(source/m['unique_index'])
        for shard in m['shards']:
            for key in ['array','metadata']:shard[key]='sources/'+label+'/'+shard[key]
        if split=='train':
            with ThreadPoolExecutor(max_workers=8) as pool:flags=list(pool.map(read_support,[(bank,s) for s in m['shards']]))
            index,_=filter_sampling_index(index,flags);unique,_=filter_sampling_index(unique,flags)
        assert len(index)>0;np.save(bank/(split+'.index.npy'),index);np.save(bank/(split+'.unique.npy'),unique)
        m.update(build_id='original_dilate2_l1_20261103',index=split+'.index.npy',unique_index=split+'.unique.npy',paired_only=True,runtime_preprocessing=False,runtime_geometry=False,preprocess_mode='none')
        save(bank/(split+'.json'),m);counts[split]=dict(parents=len(m['shards']),draws=len(index),unique=len(unique))
    assert counts['train']['parents']==7385 and counts['val']['parents']==620
    save(bank/'COMPLETE.json',dict(complete=True,build_id='original_dilate2_l1_20261103',splits=counts))
    save(OUT/'DATA_READY.json',dict(passed=True,counts=counts,train_manifest_sha256=digest(bank/'train.json'),val_manifest_sha256=digest(bank/'val.json'),source_binding_sha256=digest(OUT/'SOURCE_BINDING.json')));save(OUT/'DATA_STATUS.json',dict(phase='complete'))


if __name__=='__main__':main()
