"""Additional no-CP/effective-text-only Full SDF/noSDF scratch comparison."""
from pathlib import Path
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_s22_patch_contrast_guard as base
from tools.run_uniform20_steep_scurve import save,digest
SOURCE=base.OUT
OUT=SOURCE.parent/'s22_nocp_text_guard_20261029'
ORIGINAL_PUBLISH=base.publish


def filter_sampling_index(index,nonempty):
    """@luna-comment Input:uint32 Nx2 source draw index and per-shard positive-valid-text booleans. Output:stable filtered index and selection mask. Key Steps:flatten shard flags;map indexed rows;keep duplicates/order. Key Params:strict positive existing coverage,no CP threshold. State/Dependencies:pure NumPy;source arrays unchanged."""
    assert index.ndim==2 and index.shape[1]==2 and index.dtype==np.uint32
    counts=np.array([len(row) for row in nonempty],np.int64);offsets=np.r_[0,np.cumsum(counts)]
    assert np.all(index[:,0]<len(counts)) and np.all(index[:,1]<counts[index[:,0]])
    flags=np.concatenate(nonempty).astype(bool);selected=flags[offsets[index[:,0]]+index[:,1]]
    return index[selected].copy(),selected


def read_support(job):
    """@luna-comment Input:original bank root/shard binding. Output:per-patch positive effective OCR support flags. Key Steps:verify existing receipt source hashes/count/pass;use coverage from valid-intersected region sidecar. Key Params:coverage>0,not CP eligibility. State/Dependencies:read-only immutable sidecar receipts;runtime separately checks actual masks."""
    bank,shard=job;target=bank/'line_regions'/(shard['array']+'.bits.npy')
    receipt=json.loads(target.with_suffix('.receipt.json').read_text())
    assert receipt['passed'] and receipt['count']==shard['count']
    assert receipt['source_shard_sha256']==shard['array_sha256'] and receipt['source_metadata_sha256']==shard['metadata_sha256']
    coverage=np.asarray(receipt['coverage'],np.float64);assert len(coverage)==shard['count'] and np.isfinite(coverage).all() and np.all((coverage>=0)&(coverage<=1))
    return coverage>0


def build_filter():
    """@luna-comment Input:ready parent bank and verified coverage receipts. Output:training-only manifest/index/retained-count binding. Key Steps:parallel metadata reads;stable balanced/unique filtering;hash new index and original manifest. Key Params:8read threads,no data/OCR/validation rewrite. State/Dependencies:owned filter artifacts;existing immutable bank."""
    from concurrent.futures import ThreadPoolExecutor
    bank=base.SOURCE/'bank256';path=bank/'train_balanced.json';manifest=json.loads(path.read_text());start=time.time()
    assert base.ready(base.SOURCE)
    target=OUT/'filter';target.mkdir(exist_ok=True);nonempty=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count,flags in enumerate(pool.map(read_support,[(bank,s) for s in manifest['shards']]),1):
            nonempty.append(flags)
            if count%100==0:save(OUT/'FILTER_PROGRESS.json',dict(shards_done=count,shards_total=len(manifest['shards']),elapsed_seconds=time.time()-start))
    original=np.load(bank/manifest['index'],allow_pickle=False);unique=np.load(bank/manifest['unique_index'],allow_pickle=False)
    selected,mask=filter_sampling_index(original,nonempty);selected_unique,_=filter_sampling_index(unique,nonempty)
    assert len(selected)>0 and len(selected_unique)>0
    np.save(target/'train_balanced.index.npy',selected,allow_pickle=False);np.save(target/'train.index.npy',selected_unique,allow_pickle=False)
    manifest['index']=str(target/'train_balanced.index.npy');manifest['unique_index']=str(target/'train.index.npy')
    manifest['sampling_filter']='existing OCR polygon union intersected with native valid support has positive area;independent of CP eligibility'
    save(target/'train_balanced.json',manifest)
    receipt=dict(passed=True,source_manifest_sha256=digest(path),bank_build_id=manifest['build_id'],source_draws=len(original),kept_draws=len(selected),removed_draws=int((~mask).sum()),source_unique=len(unique),kept_unique=len(selected_unique),removed_unique=len(unique)-len(selected_unique),parents_with_text=sum(bool(x.any()) for x in nonempty),parents_total=len(nonempty),index_sha256=digest(target/'train_balanced.index.npy'),unique_index_sha256=digest(target/'train.index.npy'),manifest=str(target/'train_balanced.json'),seconds=time.time()-start,criterion=manifest['sampling_filter'])
    save(OUT/'FILTER_COMPLETE.json',receipt);return receipt


def prepare():
    """@luna-comment Input:existing two contrast recipes and completed effective-text filter. Output:two new immutable noCP configs and smoke configs. Key Steps:clone corresponding SDF arm;disable CP criterion/weight;bind filtered training index and owned paths. Key Params:same fresh init250K/guard/contrast;validation unchanged. State/Dependencies:no changes to original jobs/configs/bank."""
    import yaml
    filtered=build_filter();records={};(OUT/'configs').mkdir(exist_ok=True)
    for arm,(gpu,label) in base.ARMS.items():
        source=SOURCE/'configs'/(arm+'.yml');cfg=yaml.safe_load(source.read_text())
        cfg['name']='s22_nocp_text_guard_'+arm+'_20261029';cfg['train']['cp_weight']=0.;cfg['train']['cp_opt']=None
        cfg['datasets']['train']['manifest']=filtered['manifest'];cfg['datasets']['train']['require_effective_text']=True
        cfg['path']['experiments_root']=str(OUT/'experiments');cfg['s22_strategy']['audit_root']=str(OUT/'audits'/arm)
        path=OUT/'configs'/(arm+'.yml');assert not path.exists();path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        smoke=copy.deepcopy(cfg);smoke['name']+='_smoke';smoke['s22_strategy']['audit_root']=str(OUT/'smoke/audits'/arm)
        sp=OUT/'configs'/(arm+'_smoke.yml');sp.write_text(yaml.safe_dump(smoke,sort_keys=False))
        records[arm]=dict(config=str(path),sha256=digest(path),smoke_config=str(sp),gpu=gpu,label=label,name=cfg['name'],initial=cfg['path']['pretrain_network_g'],initial_sha256=digest(cfg['path']['pretrain_network_g']),source_config_sha256=digest(source))
    save(OUT/'CONFIG_BINDINGS.json',dict(runs=records,filter=filtered,trainer_sha256=digest(ROOT/'tools/train_full_s22_strategy.py'),validation='Original583 unchanged;contrast correction and filtering training only'))
    return records


def publish(records,processes,phase):
    """@luna-comment Input:isolated run/process records. Output:shared monitor plus explicit noCP/text-filter counts and baseline link. Key Steps:reuse accepted observer;insert filter summary;atomic replace. Key Params:only owned new dashboard changes. State/Dependencies:existing chart/ETA helpers and immutable filter receipt."""
    states=ORIGINAL_PUBLISH(records,processes,phase);r=json.loads((OUT/'FILTER_COMPLETE.json').read_text())
    page=OUT/'dashboard.html';body=page.read_text(encoding='utf-8')
    info='<section><h2>\u65e0CP + \u4ec5\u6709\u6548\u6587\u672cpatch</h2><p>\u8bad\u7ec3\u91c7\u6837\u7d22\u5f15: '+str(r['kept_draws'])+' / '+str(r['source_draws'])+'; \u5254\u9664 '+str(r['removed_draws'])+'; \u552f\u4e00patch: '+str(r['kept_unique'])+' / '+str(r['source_unique'])+'</p><p><a href="../s22_patch_contrast_guard_20261029/dashboard.html">\u67e5\u770b\u539f\u6709CP\u5bf9\u7167\u7ec4</a></p></section>'
    body=body.replace('</h1>','</h1>'+info,1);temp=OUT/'dashboard.extra.tmp';temp.write_text(body,encoding='utf-8');temp.replace(page)
    return states


if __name__=='__main__':
    base.OUT=OUT;base.ARMS={'full_sdf':(2,'\u53ef\u4fe1\u8fb9\u7f18\u4fdd\u62a4 | Full\u542bSDF | \u65e0CP + \u6709\u6548\u6587\u672c'),'full_nosdf':(0,'\u53ef\u4fe1\u8fb9\u7f18\u4fdd\u62a4 | Full\u65e0SDF | \u65e0CP + \u6709\u6548\u6587\u672c')};base.prepare=prepare;base.publish=publish
    try:base.main()
    except Exception as error:save(OUT/'FAILED.json',dict(error=repr(error),epoch=time.time()));raise
