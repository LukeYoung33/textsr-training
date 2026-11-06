"""Frozen-partial-data-gated reused-bank assembly for four scratch controls."""
from pathlib import Path
import sys,json,copy,time,os,subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'latest_vosr_four_500k_20261102'
LENOVO=DATA/'lenovo_validcrop_full_gt_s22_20261029'
OLD=DATA/'Lenovo_data_260912/training_20260927_validcrop_v4'
BASE=DATA/'full_s22_training_matrix_20261028'
MOJITO=DATA/'mojito280_guard_finetune_20261030'


def rows(path):
    """@luna-comment Input:JSONL manifest. Output:rows. Key Steps:skip blanks. Key Params:read-only. State/Dependencies:immutable source."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def eligibility():
    """@luna-comment Input:three historical source manifests. Output:train parent allowlist and exclusion audit. Key Steps:holdout identities/raw hashes across sources;dedup training raw hashes;validation always wins. Key Params:preserve583+37 validation,exclude Lenovo oldval identities. State/Dependencies:metadata-only."""
    sources={'base':BASE/'source_view','mojito':MOJITO/'source_view','lenovo':OLD}
    sets={k:{s:rows(v/'manifests'/(s+'.jsonl')) for s in ['train','val']} for k,v in sources.items()}
    val_ids={r['source_identity'] for data in sets.values() for r in data['val']}
    def rawhash(r):
        """@luna-comment Input:source row. Output:original LQ hash. Key Steps:prefer preprocessing lineage. Key Params:no hash substitution. State/Dependencies:published manifests."""
        return r.get('preprocess',{}).get('source_hashes',{}).get('lq',r['output_sha256']['lq'])
    val_hashes={rawhash(r) for data in sets.values() for r in data['val']};seen=set();allowed={};excluded=[]
    for key in sources:
        allowed[key]=[]
        for r in sets[key]['train']:
            sha=rawhash(r);reason='heldout_identity' if r['source_identity'] in val_ids else 'heldout_hash' if sha in val_hashes else 'duplicate_train_hash' if sha in seen else None
            if reason:excluded.append(dict(source=key,id=r['sample_id'],reason=reason));continue
            allowed[key].append(r['sample_id']);seen.add(sha)
    result=dict(allowed=allowed,excluded=excluded,train_counts={k:len(v) for k,v in allowed.items()},validation_counts={'base':len(sets['base']['val']),'mojito':len(sets['mojito']['val'])},source_manifest_hashes={k:{s:digest(v/'manifests'/(s+'.jsonl')) for s in ['train','val']} for k,v in sources.items()},scope='4873 refreshed S2.2 + Mojito train243/val37 + Lenovo full3236; diagnostic IQ35 excluded; legacy583 GT validation retained')
    save(OUT/'SOURCE_ELIGIBILITY.json',result);return sets,result


def prepare_lenovo(sets,audit):
    """@luna-comment Input:completed teacher chunk receipts. Output:eligible original-mask/V3LQ/S2.2GT source view. Key Steps:map complete chunks;bind hashes;reuse masks/OCR;no transformations. Key Params:existing source split. State/Dependencies:immutable partial snapshot mandatory."""
    snapshot=OUT/'LENOVO_PARTIAL_SNAPSHOT.json'
    found={}
    for f in LENOVO.glob('*/*/COMPLETE.json'):
        for r in json.loads(f.read_text())['records']:found[r['id']]=(r,f.parent/'images'/r['id'])
    for r in json.loads((LENOVO/'RECOVERED_TEACHER.json').read_text())['records']:found.setdefault(r['id'],(r,Path(r['stage_folder'])))
    if snapshot.exists():
        found={k:(v[0],Path(v[1])) for k,v in json.loads(snapshot.read_text())['records'].items()}
    else:
        assert found
        save(snapshot,dict(created=time.time(),records={k:[r,str(folder)] for k,(r,folder) in found.items()}))
    byname={Path(r['filename']).stem:(r,folder) for r,folder in found.values()};view=OUT/'lenovo_view';(view/'manifests').mkdir(parents=True,exist_ok=True)
    audit['allowed']['lenovo']=[k for k in audit['allowed']['lenovo'] if k in byname]
    audit['train_counts']['lenovo']=len(audit['allowed']['lenovo'])
    audit['scope']='Frozen completed Lenovo subset + base4873 + Mojito243; unchanged validation583+37'
    save(OUT/'SOURCE_ELIGIBILITY.json',audit)
    eligible=set(audit['allowed']['lenovo']);output=[]
    for original in sets['lenovo']['train']:
        if original['sample_id'] not in eligible:continue
        record,folder=byname[original['sample_id']];assert record['source_sha256']==original['output_sha256']['lq']
        assert digest(folder/'pre_lq.png')==record['pre_lq_sha256']
        assert digest(Path(record['candidate_gt']))==record['arms']['steep_s']['sha256']
        row=copy.deepcopy(original);row['dataset']='lenovo0912';row['files']={k:str(OLD/v) for k,v in row['files'].items()}
        row['files'].update(lq=str(folder/'pre_lq.png'),gt=record['candidate_gt']);row['output_sha256'].update(lq=record['pre_lq_sha256'],gt=record['arms']['steep_s']['sha256']);row['target_version']='V3LQ_fullVOSR_S2.2';output.append(row)
    (view/'manifests/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output));(view/'manifests/val.jsonl').write_text('')
    save(OUT/'LENOVO_SOURCE_READY.json',dict(passed=True,count=len(output),snapshot_sha256=digest(snapshot),manifest_sha256=digest(view/'manifests/train.jsonl')))
    return view


def merge(audit):
    """@luna-comment Input:three complete banks with sidecars. Output:symlink-backed merged manifests/filtered indices. Key Steps:retain parent allowlist;remap shardIDs;preserve draw multiplicity;positive-valid-text filter. Key Params:unchanged arrays and validation. State/Dependencies:owned symlinks/index only."""
    import numpy as np
    from tools.run_s22_nocp_text_guard import read_support,filter_sampling_index
    bank=OUT/'bank';bank.mkdir(exist_ok=True);sources={'base':BASE/'bank256','mojito':MOJITO/'bank256','lenovo':OUT/'lenovo_bank'};counts={}
    for name,source in sources.items():
        assert json.loads((source/'COMPLETE.json').read_text())['complete'] and json.loads((source/'LINE_REGIONS_COMPLETE.json').read_text())['passed']
        for parent,target in [(bank/'sources',source),(bank/'line_regions/sources',source/'line_regions')]:
            parent.mkdir(parents=True,exist_ok=True);link=parent/name
            if not link.exists():link.symlink_to(target,target_is_directory=True)
            assert link.resolve()==target.resolve()
    for split in ['train','val']:
        shards=[];draws=[];unique=[]
        for name,source in sources.items():
            m=json.loads((source/('train_balanced.json' if split=='train' else 'val.json')).read_text());mapping={}
            for old,s in enumerate(m['shards']):
                if split=='train' and s['sample_id'] not in set(audit['allowed'][name]):continue
                mapping[old]=len(shards);s=copy.deepcopy(s)
                for key in ['array','metadata']:s[key]='sources/'+name+'/'+s[key]
                shards.append(s)
            for key,dest in [('index',draws),('unique_index',unique)]:
                a=np.load(source/m[key],allow_pickle=False);keep=np.isin(a[:,0],list(mapping));a=a[keep].copy()
                if len(a):a[:,0]=np.asarray([mapping[int(i)] for i in a[:,0]],np.uint32)
                dest.append(a)
        index=np.concatenate(draws);uni=np.concatenate(unique)
        if split=='train':
            with ThreadPoolExecutor(max_workers=8) as pool:flags=list(pool.map(read_support,[(bank,s) for s in shards]))
            index,_=filter_sampling_index(index,flags);uni,_=filter_sampling_index(uni,flags)
        assert len(index)>0;np.save(bank/(split+'.index.npy'),index);np.save(bank/(split+'.unique.npy'),uni)
        m=dict(build_id='latest_vosr_20261102',shape_hw=[256,256],paired_only=True,runtime_preprocessing=False,runtime_geometry=False,preprocess_mode='none',index=split+'.index.npy',unique_index=split+'.unique.npy',shards=shards)
        save(bank/(split+'.json'),m);counts[split]=dict(parents=len(shards),draws=len(index),unique=len(uni))
    assert not ({s['source_identity'] for s in json.loads((bank/'train.json').read_text())['shards']}&{s['source_identity'] for s in json.loads((bank/'val.json').read_text())['shards']})
    save(bank/'COMPLETE.json',dict(complete=True,build_id='latest_vosr_20261102',splits=counts,source_banks={k:str(v) for k,v in sources.items()}))
    save(OUT/'DATA_READY.json',dict(passed=True,counts=counts,train_manifest_sha256=digest(bank/'train.json'),val_manifest_sha256=digest(bank/'val.json'),eligibility_sha256=digest(OUT/'SOURCE_ELIGIBILITY.json')))


def main():
    """@luna-comment Input:GT completion and approved data scope. Output:ready merged bank. Key Steps:freeze completed population;build only new Lenovo shards;reuse base/Mojito;filter. Key Params:8CPU,identical frozen subset. State/Dependencies:independent teacher continues."""
    OUT.mkdir(exist_ok=True)
    if (OUT/'DATA_READY.json').exists():return
    sets,audit=eligibility();save(OUT/'DATA_STATUS.json',dict(phase='freezing_completed_lenovo',counts=audit['train_counts']))
    view=prepare_lenovo(sets,audit);bank=OUT/'lenovo_bank'
    if not (bank/'COMPLETE.json').exists():
        save(OUT/'DATA_STATUS.json',dict(phase='building_lenovo_bank'))
        with (OUT/'bank.log').open('ab') as f:subprocess.run([sys.executable,'-u',str(ROOT/'tools/build_vosr_block256_dataset.py'),'--source',str(view),'--output',str(bank),'--workers','8','--preprocess-mode','none'],stdout=f,stderr=subprocess.STDOUT,check=True)
    if not (bank/'LINE_REGIONS_COMPLETE.json').exists():
        save(OUT/'DATA_STATUS.json',dict(phase='building_lenovo_text_sidecars'));from tools.run_v3pre_training import sidecars;sidecars(bank,view,8)
    save(OUT/'DATA_STATUS.json',dict(phase='merging_reused_banks'));merge(audit);save(OUT/'DATA_STATUS.json',dict(phase='complete'))


if __name__=='__main__':main()
