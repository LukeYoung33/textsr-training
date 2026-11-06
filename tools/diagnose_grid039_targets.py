"""Matched non-Mojito training-target audit; no training or model changes."""
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_uniform20_steep_scurve import save, digest
from tools.patch_contrast_alignment import match_patch_contrast
DATA = Path('/AI_DNS/guang.yang/TextSR/data')
OUT = DATA/'grid039_diagnostic_20261104'


def measure(target, lq, mask):
    """@luna-comment Input:RGBCHWfloat target/LQ andnontextvalid support. Output:luma/chroma/backgroundtransferDN statistics. Key Steps:commonpixelresidual andchromaticcomponent. Key Params:0..255DN;noalignmentestimation. State/Dependencies:NumPy read-only."""
    import numpy as np
    delta = (target-lq)*255
    y = (delta*np.array([.299,.587,.114],np.float32)[:,None,None]).sum(0)
    chroma = delta-y[None]
    return dict(mean_y_dn=float(y[mask].mean()), mae_y_dn=float(np.abs(y[mask]).mean()),
                rms_y_dn=float(np.sqrt((y[mask]**2).mean())),
                mae_chroma_dn=float(np.abs(chroma[:,mask]).mean()))


def main():
    """@luna-comment Input:shared7385old/newbanks andoriginalGTlineage. Output:20parent pairedtarget audit excludingMojito. Key Steps:stratifyactualeligiblebaselinepatches,identitygates,exactcontrastmatch,nontextstatistics. Key Params:4parents/dataset,up8patches/parent,bgerosion9px. State/Dependencies:immutablebanks,CPUonly;nomodelupdates."""
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    start = time.time()
    oldbank = DATA/'latest_vosr_four_500k_20261102/bank'
    newbank = DATA/'v3m_original_dilate2_l1_500k_20261103/bank'
    new = json.loads((newbank/'train.json').read_text())
    old = json.loads((oldbank/'train.json').read_text())
    prior = {(s['dataset'],s['sample_id']):s for s in old['shards']}
    source = json.loads((DATA/'original_aligned_gt_dilate2_20261103/SOURCE_PLAN.json').read_text())
    originals = {(j['dataset'],j['sample_id']):j for j in source['jobs']}
    index = np.load(newbank/new['unique_index'])
    selected = []
    for dataset in sorted({s['dataset'] for s in new['shards']}):
        group = [(i,s) for i,s in enumerate(new['shards']) if s['dataset']==dataset]
        group.sort(key=lambda row:row[1]['sample_id'])
        selected.extend(group[i] for i in np.linspace(0,len(group)-1,4,dtype=int))
    assert len(selected)==20
    records = []
    skipped = []
    for shardid, shard in selected:
        identity = shard['dataset'],shard['sample_id']
        previous = prior[identity]
        metadata = json.loads((newbank/shard['metadata']).read_text())
        assert metadata == json.loads((oldbank/previous['metadata']).read_text())
        eligible = set(map(int,index[index[:,0]==shardid,1]))
        candidates = [i for i,r in enumerate(metadata) if i in eligible and r['category']=='baseline' and r['real_hw']==[256,256]]
        if not candidates:
            skipped.append(dict(dataset=identity[0],sample_id=identity[1],reason='No eligible full256 baseline patch in frozen sampling index'))
            continue
        candidates = [candidates[i] for i in sorted(set(np.linspace(0,len(candidates)-1,min(8,len(candidates)),dtype=int)))]
        a = np.load(newbank/shard['array'],mmap_mode='r')
        b = np.load(oldbank/previous['array'],mmap_mode='r')
        original = originals[identity]
        assert digest(original['files']['gt']) == original['source_sha256']['gt']
        raw_gt = cv2.imread(original['files']['gt'])
        assert raw_gt is not None
        assert digest(original['files']['text_mask']) == original['source_sha256']['text_mask']
        soft_text_mask = cv2.imread(original['files']['text_mask'],0)
        assert soft_text_mask.shape == raw_gt.shape[:2]
        for rowid in candidates:
            av = np.array(a[rowid]);bv = np.array(b[rowid])
            assert np.array_equal(av[...,:3],bv[...,:3]) and np.array_equal(av[...,6:],bv[...,6:]),identity
            x,y = metadata[rowid]['xy']
            raw = raw_gt[y:y+256,x:x+256]
            assert raw.shape == (256,256,3)
            lq = av[...,:3][...,::-1].transpose(2,0,1).astype(np.float32)/255
            valid = (av[...,7].astype(np.float32)/255)[None]
            support = ((soft_text_mask[y:y+256,x:x+256]<=12)&(av[...,7]>=254)).astype(np.uint8)
            support = cv2.erode(support,np.ones((19,19),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            if support.sum()<256:
                continue
            targets = dict(vosr=bv[...,3:6],original=raw,dilate2=av[...,3:6])
            row = dict(dataset=identity[0],sample_id=identity[1],row=rowid,xy=[x,y],background_pixels=int(support.sum()),variants={})
            adjusted = {}
            for label, value in targets.items():
                gt = value[...,::-1].transpose(2,0,1).astype(np.float32)/255
                aligned,stats = match_patch_contrast(gt,lq,valid)
                adjusted[label] = aligned
                row['variants'][label] = dict(raw=measure(gt,lq,support),aligned=measure(aligned,lq,support),contrast=stats)
            rawdiff = np.abs(targets['dilate2'].astype(np.int16)-raw.astype(np.int16))
            row['raw_dilation_bg_max_dn'] = int(rawdiff[support].max())
            assert row['raw_dilation_bg_max_dn']==0,(identity,rowid,'Background support touched by dilation')
            row['aligned_dilation_bg_mae_dn'] = float(np.abs(adjusted['dilate2']-adjusted['original'])[:,support].mean()*255)
            row['aligned_dilation_bg_max_dn'] = float(np.abs(adjusted['dilate2']-adjusted['original'])[:,support].max()*255)
            ly = (lq*np.array([.299,.587,.114],np.float32)[:,None,None]).sum(0)*255
            dark = support&(ly<100)
            row['dark_background_pixels'] = int(dark.sum())
            if dark.sum()>=256:
                row['dark_aligned'] = {label:measure(value,lq,dark) for label,value in adjusted.items()}
            records.append(row)
        print('PARENT',identity,len(records),flush=True)
    assert len(records)>=80
    summaries = {}
    for group in ['all']+sorted({r['dataset'] for r in records}):
        rows = records if group=='all' else [r for r in records if r['dataset']==group]
        result = dict(patches=len(rows),parents=len({r['sample_id'] for r in rows}),
                      dilation_raw_background_changed_patches=sum(r['raw_dilation_bg_max_dn']>0 for r in rows),
                      dilation_alignment_spillover_mean_dn=float(np.mean([r['aligned_dilation_bg_mae_dn'] for r in rows])),variants={})
        for label in ['vosr','original','dilate2']:
            result['variants'][label] = {}
            for stage in ['raw','aligned']:
                result['variants'][label][stage] = {k:float(np.mean([r['variants'][label][stage][k] for r in rows])) for k in ['mean_y_dn','mae_y_dn','rms_y_dn','mae_chroma_dn']}
            result['variants'][label]['contrast_skipped'] = sum(not r['variants'][label]['contrast']['applied'] for r in rows)
        darkrows = [r for r in rows if 'dark_aligned' in r]
        result['dark_patches'] = len(darkrows)
        if darkrows:
            result['dark_aligned'] = {label:{k:float(np.mean([r['dark_aligned'][label][k] for r in darkrows])) for k in ['mean_y_dn','mae_y_dn','mae_chroma_dn']} for label in ['vosr','original','dilate2']}
        summaries[group] = result
    save(OUT/'TARGET_AUDIT.json',dict(passed=True,planned_parents=20,parents=len({(r['dataset'],r['sample_id']) for r in records}),skipped=skipped,patches=len(records),mojito_included=False,
         all_non_gt_channels_equal=True,source_plan_sha256=digest(DATA/'original_aligned_gt_dilate2_20261103/SOURCE_PLAN.json'),
         summaries=summaries,records=records,seconds=time.time()-start))
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    main()
