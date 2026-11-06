"""Matched teacher subset and accepted full-augmented width labels for four runs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,copy,json,sys,time,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'gt1_r1_mixture_four_500k_20261104';BASE=DATA/'latest_vosr_four_500k_20261102'


def teacher_subset():
    """@luna-comment Input:frozenold7628teacherbank andmatched7385sourceplan. Output:readonlylinkedteacher7385bank/manifest. Key Steps:excludeMojito,remapdrawIDs,retain620val,bindoriginalmasks. Key Params:noarrayrewrite/contrast. State/Dependencies:ownedindices/links only."""
    import numpy as np
    out=OUT/'vosr';out.mkdir(exist_ok=True);bank=out/'bank';bank.mkdir(exist_ok=True)
    if (out/'DATA_READY.json').exists():return
    source=json.loads((DATA/'original_aligned_gt_dilate2_20261103/SOURCE_PLAN.json').read_text());jobs={(j['dataset'],j['sample_id']):j for j in source['jobs']};assert len(jobs)==7385
    for name,dest in [('sources',BASE/'bank/sources'),('line_regions',BASE/'bank/line_regions')]:
        link=bank/name
        if not link.exists():link.symlink_to(dest,target_is_directory=True)
        assert link.resolve()==dest.resolve()
    counts={};rows=[]
    for split in ['train','val']:
        m=copy.deepcopy(json.loads((BASE/'bank'/(split+'.json')).read_text()));index=np.load(BASE/'bank'/m['index']);unique=np.load(BASE/'bank'/m['unique_index'])
        if split=='train':
            retained=[i for i,s in enumerate(m['shards']) if (s['dataset'],s['sample_id']) in jobs];assert len(retained)==7385
            mapping=np.zeros(len(m['shards']),np.uint32);mapping[retained]=np.arange(len(retained),dtype=np.uint32)
            arrays=[]
            for a in [index,unique]:
                a=a[np.isin(a[:,0],retained)].copy();a[:,0]=mapping[a[:,0]];arrays.append(a)
            index,unique=arrays;m['shards']=[m['shards'][i] for i in retained]
            for s in m['shards']:
                j=jobs[s['dataset'],s['sample_id']];files=copy.deepcopy(j['files']);hashes=copy.deepcopy(j['source_sha256']);files['gt']=j['previous_target']['path'];hashes['gt']=j['previous_target']['sha256']
                assert all(s['source_hashes'][k]==hashes[k] for k in ['lq','gt','text_mask','valid_mask','ocr'])
                rows.append(dict(dataset=j['dataset'],sample_id=j['sample_id'],source_identity=j['source_identity'],files=files,output_sha256=hashes))
        else:assert len(m['shards'])==620
        np.save(bank/(split+'.index.npy'),index);np.save(bank/(split+'.unique.npy'),unique)
        m.update(build_id='vosr_matched7385_20261104',index=split+'.index.npy',unique_index=split+'.unique.npy');save(bank/(split+'.json'),m)
        counts[split]=dict(parents=len(m['shards']),draws=len(index),unique=len(unique))
    save(out/'SOURCE_ROWS.json',dict(rows=rows));save(bank/'COMPLETE.json',dict(complete=True,build_id='vosr_matched7385_20261104',splits=counts))
    save(out/'DATA_READY.json',dict(passed=True,counts=counts,excluded_mojito=243,contrast_alignment=False,train_sha256=digest(bank/'train.json')))


def width_init(bank,local):
    """@luna-comment Input:ownedbank/sidecarroot. Output:processlocalacceptedworkerbinding. Key Steps:rebindonlyworkerOLD/LOCALconstants. Key Params:unchangedwidthalgorithm. State/Dependencies:isolatedCPUchild;oldprocessesunaffected."""
    from tools import prepare_r1_width_bank as core
    core.OLD=Path(bank);core.LOCAL=Path(local)


def width_job(job):
    """@luna-comment Input:oneboundsource/shard. Output:acceptedwidthcodereceipt. Key Steps:delegateunchangedfullaugmentedmaskEDT/skeletonprocessor. Key Params:bins<=2/<=4/>4,validgated. State/Dependencies:childwidth_initbinding."""
    from tools.prepare_r1_width_bank import process
    return process(job)


def widths(domain):
    """@luna-comment Input:readygt1orvosr7385bank. Output:7385exactacceptedwidthsidecars/completion. Key Steps:source/hashbinding,CPU8work,receiptcoverage. Key Params:NASonly,fullframeaugmentationbeforecrop. State/Dependencies:ownsidecars;noOCR/GT/LQrewrites."""
    out=OUT/domain;bank=out/'bank';assert json.loads((out/'DATA_READY.json').read_text())['passed']
    if (out/'WIDTH_READY.json').exists():return
    lock=(out/'width.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if domain=='gt1':rows=[json.loads(line) for line in (DATA/'original_aligned_gt_dilate1_20261104/manifests/train.jsonl').read_text().splitlines() if line.strip()]
    else:rows=json.loads((out/'SOURCE_ROWS.json').read_text())['rows']
    lookup={(r['dataset'],r['sample_id']):r for r in rows};shards=json.loads((bank/'train.json').read_text())['shards'];assert len(shards)==len(lookup)==7385
    local=out/'r1';(local/'width_codes').mkdir(parents=True,exist_ok=True);link=bank/'width_codes'
    if not link.exists():link.symlink_to(local/'width_codes',target_is_directory=True)
    assert link.resolve()==(local/'width_codes').resolve()
    jobs=[(s,lookup[s['dataset'],s['sample_id']]) for s in shards];records=[];started=time.time()
    with ProcessPoolExecutor(max_workers=8,initializer=width_init,initargs=(str(bank),str(local))) as pool:
        for future in as_completed([pool.submit(width_job,j) for j in jobs]):
            records.append(future.result());n=len(records)
            if n%10==0 or n==7385:
                elapsed=time.time()-started;save(out/'WIDTH_STATUS.json',dict(phase='width_groups',completed=n,total=7385,elapsed_seconds=elapsed,eta_seconds=elapsed/n*(7385-n)))
    assert len(records)==7385 and all(r['passed'] for r in records)
    save(out/'WIDTH_RECORDS.json',records);save(out/'WIDTH_READY.json',dict(passed=True,parents=7385,records_sha256=digest(out/'WIDTH_RECORDS.json'),bank_sha256=digest(bank/'train.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['teacher_subset','width_gt1','width_vosr']);a=p.parse_args()
    if a.mode=='teacher_subset':teacher_subset()
    else:widths(a.mode[6:])
