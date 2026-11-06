"""Independent source-bound GT+1 bank for the requested four500K recipes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'gt1_r1_mixture_four_500k_20261104/gt1'
SOURCE = DATA/'original_aligned_gt_dilate1_20261104'
BASE = DATA/'latest_vosr_four_500k_20261102'


def main():
    """@luna-comment Input:complete7385GT1/syncedmaskversion. Output:filteredblockbankwithold620validation. Key Steps:sourcepopulationhashgates,CPU8build/cachedOCRgeometry,LQvalidreplay,ownindexmerge. Key Params:nocontrast/preprocess/OCRteacherredo,oldtextmaskneednotmatch. State/Dependencies:immutablepreviousbanks/ownednewdata."""
    import numpy as np
    from tools.run_v3pre_training import sidecars
    from tools.run_s22_nocp_text_guard import read_support,filter_sampling_index
    OUT.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'prepare.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'DATA_READY.json').exists():
        return
    complete = json.loads((SOURCE/'COMPLETE.json').read_text())
    assert complete['passed'] and complete['count']==7385 and complete['radius']==.5
    assert complete['all_masks_synced'] and not complete['contrast_alignment_applied']
    manifest = SOURCE/'manifests/train.jsonl'
    assert digest(manifest)==complete['manifest_sha256']
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    allowed = json.loads((BASE/'SOURCE_ELIGIBILITY.json').read_text())['allowed']
    assert len(rows)==7385 and {r['sample_id'] for r in rows}==set(allowed['base'])|set(allowed['lenovo'])
    val = json.loads((BASE/'bank/val.json').read_text())
    assert len(val['shards'])==620 and not ({r['source_identity'] for r in rows}&{s['source_identity'] for s in val['shards']})
    view = OUT/'source_view'
    (view/'manifests').mkdir(parents=True,exist_ok=True)
    target = view/'manifests/train.jsonl'
    if target.exists():
        assert digest(target)==complete['manifest_sha256']
    else:
        target.write_bytes(manifest.read_bytes())
    (view/'manifests/val.jsonl').write_text('')
    save(OUT/'SOURCE_BINDING.json',dict(source=str(SOURCE),complete_sha256=digest(SOURCE/'COMPLETE.json'),
         manifest_sha256=complete['manifest_sha256'],train_count=7385,val_count=620,gt_radius=.5,
         masks_synced=True,contrast_alignment=False,preprocessing='none',ocr='reuse_existing_annotations'))
    train = OUT/'train_bank'
    if not (train/'COMPLETE.json').exists():
        save(OUT/'DATA_STATUS.json',dict(phase='building_gt1_bank',total=7385))
        with (OUT/'bank.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(view),
                 '--output',str(train),'--workers','8','--preprocess-mode','none'],stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (train/'LINE_REGIONS_COMPLETE.json').exists():
        save(OUT/'DATA_STATUS.json',dict(phase='cached_ocr_geometry_sidecars',total=7385))
        sidecars(train,view,8)
    old = json.loads((BASE/'bank/train.json').read_text())
    lookup = {(s['dataset'],s['sample_id']):s for s in old['shards']}
    new = json.loads((train/'train_balanced.json').read_text())
    checks = []
    for dataset in sorted({s['dataset'] for s in new['shards']}):
        group = [s for s in new['shards'] if s['dataset']==dataset]
        for shard in [group[0],group[-1]]:
            previous = lookup[shard['dataset'],shard['sample_id']]
            a = np.load(train/shard['array'],mmap_mode='r')
            b = np.load(BASE/'bank'/previous['array'],mmap_mode='r')
            assert a.shape==b.shape and np.array_equal(a[...,:3],b[...,:3]) and np.array_equal(a[...,7],b[...,7])
            ma = json.loads((train/shard['metadata']).read_text())
            mb = json.loads((BASE/'bank'/previous['metadata']).read_text())
            for left,right in zip(ma,mb):
                for key in ['xy','real_hw','matrix','category','recipe']:
                    assert left[key]==right[key],(dataset,key)
            checks.append(dict(dataset=dataset,sample_id=shard['sample_id'],lq_valid_geometry_equal=True,text_mask_changed_intentionally=True))
    save(OUT/'LQ_VALID_REPLAY.json',dict(passed=True,records=checks))
    bank = OUT/'bank'
    bank.mkdir(exist_ok=True)
    for name,source in [('train',train),('validation',BASE/'bank')]:
        for parent,destination in [(bank/'sources',source),(bank/'line_regions/sources',source/'line_regions')]:
            parent.mkdir(parents=True,exist_ok=True)
            link = parent/name
            if not link.exists():
                link.symlink_to(destination,target_is_directory=True)
            assert link.resolve()==destination.resolve()
    counts = {}
    save(OUT/'DATA_STATUS.json',dict(phase='filtering_effective_text'))
    for split,source,label,name in [('train',train,'train','train_balanced.json'),('val',BASE/'bank','validation','val.json')]:
        m = copy.deepcopy(json.loads((source/name).read_text()))
        index,unique = np.load(source/m['index']),np.load(source/m['unique_index'])
        for shard in m['shards']:
            for key in ['array','metadata']:
                shard[key] = 'sources/'+label+'/'+shard[key]
        if split=='train':
            with ThreadPoolExecutor(max_workers=8) as pool:
                flags = list(pool.map(read_support,[(bank,s) for s in m['shards']]))
            index,_ = filter_sampling_index(index,flags)
            unique,_ = filter_sampling_index(unique,flags)
        np.save(bank/(split+'.index.npy'),index)
        np.save(bank/(split+'.unique.npy'),unique)
        m.update(build_id='gt1_r1_four_20261104',index=split+'.index.npy',unique_index=split+'.unique.npy',
                 paired_only=True,runtime_preprocessing=False,runtime_geometry=False,preprocess_mode='none')
        save(bank/(split+'.json'),m)
        counts[split] = dict(parents=len(m['shards']),draws=len(index),unique=len(unique))
    save(bank/'COMPLETE.json',dict(complete=True,build_id='gt1_r1_four_20261104',splits=counts))
    save(OUT/'DATA_READY.json',dict(passed=True,counts=counts,train_manifest_sha256=digest(bank/'train.json'),
         val_manifest_sha256=digest(bank/'val.json'),source_binding_sha256=digest(OUT/'SOURCE_BINDING.json')))
    save(OUT/'DATA_STATUS.json',dict(phase='complete'))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        save(OUT/'FAILED.json',dict(error=repr(error)))
        raise
