"""Reuse accepted width labels only after exact source-mask and geometry proof."""
from pathlib import Path
import sys,json,copy,shutil,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_r1_width_bank import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data')


def inventory():
    """@luna-comment Input:completed accepted+2px width cache. Output:source bank and per-parent receipt maps. Key Steps:read immutable source provenance once per worker. Key Params:only identical expanded masks eligible. State/Dependencies:old artifacts read-only."""
    source=DATA/'v3m_original_dilate2_l1_500k_20261103/bank';records=json.loads((DATA/'v3m_dilate2_l1_r1_width_500k_20261103/WIDTH_RECORDS.json').read_text());shards=json.loads((source/'train.json').read_text())['shards']
    return source,{(r['dataset'],r['sample_id']):r for r in records},{(s['dataset'],s['sample_id']):s for s in shards}


def try_reuse(job,cache,bank,local):
    """@luna-comment Input:new shard/row and accepted cache. Output:rebound verified byte-identical receipt or None. Key Steps:exact mask/valid hashes;all augmented matrix/crop/order checks;copy bytes/hash;publish new-array binding. Key Params:GT may differ,geometric labels cannot. State/Dependencies:old cache immutable,owned NAS output only."""
    from tools.prepare_r1_width_bank import VERSION
    shard,row=job;oldbank,records,shards=cache;key=(row['dataset'],row['sample_id']);record=records.get(key);old=shards.get(key)
    if record is None or old is None or record['binding']['source_mask_sha256']!=row['output_sha256']['text_mask'] or old['source_hashes']['valid_mask']!=row['output_sha256']['valid_mask']:return None
    source=Path(record['path'])
    if not source.exists() or record['count']!=shard['count']:return None
    old_meta=Path(oldbank)/old['metadata'];new_meta=Path(bank)/shard['metadata']
    assert digest(old_meta)==old['metadata_sha256'] and digest(new_meta)==shard['metadata_sha256']
    a=json.loads(old_meta.read_text());b=json.loads(new_meta.read_text())
    if len(a)!=len(b) or any(any(x[k]!=y[k] for k in ['xy','real_hw','matrix','category']) for x,y in zip(a,b)):return None
    assert all(shard['source_hashes'][k]==row['output_sha256'][k] for k in ['gt','lq','text_mask','valid_mask','ocr'])
    path=Path(local)/'width_codes'/(shard['array']+'.codes.npy');receipt=path.with_suffix('.json');binding=dict(version=VERSION,metadata_sha256=shard['metadata_sha256'],source_mask_sha256=row['output_sha256']['text_mask'],array_sha256=shard['array_sha256'])
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['binding']==binding
        if path.exists() and digest(path)==r['sha256']:return r
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');shutil.copyfile(source,tmp)
    if digest(tmp)!=record['sha256']:
        tmp.unlink();return None
    tmp.replace(path);r=copy.deepcopy(record);r.update(binding=binding,path=str(path),reused_from=str(source),geometry_replay_verified=True,source_valid_sha256=row['output_sha256']['valid_mask'])
    save(receipt,r);return r
