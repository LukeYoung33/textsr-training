"""Read-only paired-data edge audit: lineage, exact bank bytes, GT targets and local alignment."""
from pathlib import Path
import hashlib
import json
import sys
import time
import argparse

DATA=Path('/AI_DNS/guang.yang/TextSR/data')
SOURCE=DATA/'v3pre_v3m_three_20261027'
BASE=DATA/'v3pre_three_mojito280_20261028'
OUT=DATA/'v3_paired_edge_audit_20261028'
sys.path.insert(0,str(BASE/'_build'))


def digest(path):
    """@luna-comment Input:read-only file. Output:SHA256. Key Steps:stream4MiB. Key Params:no source writes. State/Dependencies:filesystem."""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4194304),b''):value.update(chunk)
    return value.hexdigest()


def load(path):
    """@luna-comment Input:declared image path. Output:native BGR8. Key Steps:unchanged decoding and shape check. Key Params:EXIF rotation disabled. State/Dependencies:OpenCV, immutable source."""
    import cv2
    import numpy as np
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    assert image is not None and image.ndim==3 and image.shape[2]==3 and image.dtype==np.uint8,str(path)
    return image


def luma(image):
    """@luna-comment Input:BGR8. Output:float Rec601 Y in code units. Key Steps:fixed weighted channels. Key Params:no quantization/clamp. State/Dependencies:numpy."""
    return image[...,0]*.114+image[...,1]*.587+image[...,2]*.299


def gradients(value):
    """@luna-comment Input:float native Y. Output:Sobel dx/dy. Key Steps:3x3 filtering divided by8. Key Params:fixed operator. State/Dependencies:OpenCV."""
    import cv2
    return cv2.Sobel(value,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(value,cv2.CV_32F,0,1,ksize=3)/8


def support(x,valid):
    """@luna-comment Input:LQ Y/valid mask. Output:fixed interior/core/exterior/edge supports. Key Steps:input-only morphology. Key Params:16px border,45code ink contrast,15code edge floor. State/Dependencies:OpenCV/numpy; no prediction gates."""
    import cv2
    import numpy as np
    valid=valid.copy()>127;valid[:16]=False;valid[-16:]=False;valid[:,:16]=False;valid[:,-16:]=False
    ink=cv2.dilate(x,np.ones((15,15),np.uint8))-x>45
    core=cv2.erode(ink.astype(np.uint8),np.ones((3,3),np.uint8))>0
    ring=(cv2.dilate(ink.astype(np.uint8),np.ones((7,7),np.uint8))>0)&~ink
    gx,gy=gradients(x)
    return dict(valid=valid,core=core&valid,ring=ring&valid,edge=(gx*gx+gy*gy>225)&valid)


def measure(x,y,masks):
    """@luna-comment Input:reference/compared native Y/fixed supports. Output:separate gradient norm,direction,projection and signed changes. Key Steps:orthogonal edge decomposition. Key Params:empty supports explicit. State/Dependencies:numpy, no quality score."""
    import numpy as np
    x=x.astype(np.float32);y=y.astype(np.float32)
    gx,gy=gradients(x);tx,ty=gradients(y);edge=masks['edge']
    ex=float((gx*gx+gy*gy)[edge].sum());ey=float((tx*tx+ty*ty)[edge].sum());cross=float((gx*tx+gy*ty)[edge].sum())
    result=dict(edge_pixels=int(edge.sum()),gradient_gain=cross/max(ex,1e-12),gradient_norm_ratio=(ey/max(ex,1e-12))**.5,gradient_cosine=cross/max((ex*ey)**.5,1e-12))
    for key in ('core','ring'):
        mask=masks[key];result[key]=dict(count=int(mask.sum()),mean=float((y-x)[mask].mean()) if mask.any() else None)
    for sigma in (.7,1.4,2.8):
        hx=x-cv2_gaussian(x,sigma);hy=y-cv2_gaussian(y,sigma);mask=masks['valid']
        a=hx[mask];b=hy[mask];xx=float(a@a);yy=float(b@b);xy=float(a@b)
        result['highpass_'+str(sigma)]=dict(norm_ratio=(yy/max(xx,1e-12))**.5,cosine=xy/max((xx*yy)**.5,1e-12),projection=xy/max(xx,1e-12))
    return result


def cv2_gaussian(value,sigma):
    """@luna-comment Input:Y/sigma. Output:Gaussian lowpass. Key Steps:OpenCV kernel auto support. Key Params:native pixels. State/Dependencies:OpenCV."""
    import cv2
    return cv2.GaussianBlur(value,(0,0),sigma)


def align(x,y,masks):
    """@luna-comment Input:LQ/GT/fixed supports. Output:diagnostic GT shift and correlation. Key Steps:bounded integer gradient-cosine search then quarter-pixel refinement. Key Params:±3px,INTER_CUBIC,16px border. State/Dependencies:does not alter training pairs; weak peak gaps remain ambiguous."""
    import cv2
    import numpy as np
    gx,gy=gradients(x);tx,ty=gradients(y);mask=masks['edge'];ex=float((gx*gx+gy*gy)[mask].sum())
    if mask.sum()<30:return dict(available=False),y
    values=[]
    for dy in range(-3,4):
        for dx in range(-3,4):
            matrix=np.array([[1,0,dx],[0,1,dy]],np.float32)
            ux=cv2.warpAffine(tx,matrix,(x.shape[1],x.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
            uy=cv2.warpAffine(ty,matrix,(x.shape[1],x.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
            score=float((gx*ux+gy*uy)[mask].sum())/max((ex*float((ux*ux+uy*uy)[mask].sum()))**.5,1e-12)
            values.append((score,dx,dy))
    ordered=sorted(values,reverse=True);best=ordered[0];initial=next(v[0] for v in values if v[1:]==(0,0))
    for dy in np.arange(best[2]-.5,best[2]+.501,.25):
        for dx in np.arange(best[1]-.5,best[1]+.501,.25):
            matrix=np.array([[1,0,dx],[0,1,dy]],np.float32)
            moved=cv2.warpAffine(y,matrix,(x.shape[1],x.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
            mx,my=gradients(moved);ey=float((mx*mx+my*my)[mask].sum())
            score=float((gx*mx+gy*my)[mask].sum())/max((ex*ey)**.5,1e-12)
            if score>best[0]+1e-7:best=(score,float(dx),float(dy))
    matrix=np.array([[1,0,best[1]],[0,1,best[2]]],np.float32)
    moved=cv2.warpAffine(y,matrix,(x.shape[1],x.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    return dict(available=True,shift_applied_to_GT_xy=[best[1],best[2]],zero_shift_cosine=initial,best_cosine=best[0],improvement=best[0]-initial,integer_peak_gap=ordered[0][0]-ordered[1][0],metrics=measure(x,moved,masks)),moved


def colorize(value):
    """@luna-comment Input:signed Y codes. Output:fixed±32code red/blue BGR. Key Steps:neutral interpolation. Key Params:no per-image normalization. State/Dependencies:numpy."""
    import numpy as np
    t=np.clip(value/32,-1,1)[...,None];n=np.array([245,245,245],np.float32)
    return np.rint(n+np.maximum(t,0)*(np.array([20,20,190])-n)+np.maximum(-t,0)*(np.array([210,70,20])-n)).clip(0,255).astype(np.uint8)


def inspect_pair(key,pixels,metadata,raw,pre,model_output,extra=None):
    """@luna-comment Input:one exact paired crop and model prediction. Output:measurements and images. Key Steps:raw/V3 decomposition,shift check,masked residuals. Key Params:fixed LQ supports. State/Dependencies:owned artifacts only."""
    import cv2
    import numpy as np
    x=luma(pre['lq']).astype(np.float32);y=luma(pre['gt']).astype(np.float32)
    rx=luma(raw['lq']).astype(np.float32);ry=luma(raw['gt']).astype(np.float32)
    valid=pixels[...,7] if pixels is not None else np.full(x.shape,255,np.uint8)
    masks=support(x,valid);rawmasks=support(rx,valid)
    alignment,aligned=align(x,y,masks)
    raw_alignment,_=align(rx,ry,rawmasks)
    prediction=luma(model_output).astype(np.float32)
    record=dict(key=key,metadata=metadata,raw_pair=measure(rx,ry,rawmasks),v3_pair=measure(x,y,masks),lq_preprocessing=measure(rx,x,rawmasks),gt_preprocessing=measure(ry,y,support(ry,valid)),alignment=alignment,raw_alignment=raw_alignment,model=measure(x,prediction,masks))
    target_residual=y-x;model_residual=prediction-x
    ht=target_residual-cv2_gaussian(target_residual,1.4);hm=model_residual-cv2_gaussian(model_residual,1.4);mask=masks['valid']
    record['target_model_highpass_residual_cosine']=float((ht*hm)[mask].sum())/max(float((ht*ht)[mask].sum()*(hm*hm)[mask].sum())**.5,1e-12)
    images=dict(raw_lq=raw['lq'],raw_gt=raw['gt'],lq=pre['lq'],gt=pre['gt'],model=model_output,target_residual=colorize(target_residual),raw_target_residual=colorize(ry-rx),model_residual=colorize(model_residual),aligned_gt=np.repeat(np.rint(aligned).clip(0,255).astype(np.uint8)[...,None],3,axis=2))
    if extra is not None:
        record['original_gt']=measure(rx,luma(extra).astype(np.float32),rawmasks);images['original_gt']=extra
    else:images['original_gt']=raw['gt']
    for name,image in images.items():assert cv2.imwrite(str(OUT/f'{key}_{name}.png'),image)
    record['shape_hw']=list(x.shape)
    return record


def audit_scales():
    """@luna-comment Input:all source-view manifests/completed sampled parents. Output:global target-scale counts and exact sampled resize reproduction. Key Steps:inspect refresh lineage,verify physical teacher dimensions/hashes,replay declared cubic resize. Key Params:read-only sources;scale-only mode. State/Dependencies:owned receipt and OpenCV."""
    from collections import Counter
    import cv2
    import numpy as np
    counts={};summaries={}
    for split in ('train','val'):
        counter=Counter();summary=Counter()
        manifest=SOURCE/f'pretrain/source_view/manifests/{split}.jsonl'
        for line in manifest.read_text().splitlines():
            row=json.loads(line);target=row.get('target_refresh')
            key=(row['dataset'],target['interpolation'],round(target['output_hw'][0]/target['raw_hw'][0],4),round(target['output_hw'][1]/target['raw_hw'][1],4)) if target else (row['dataset'],'old_GT_VOSR_no_refresh',1,1)
            counter[key]+=1;summary[(key[0],key[1])]+=1
        counts[split]=[dict(dataset=k[0],interpolation=k[1],scale_hw=list(k[2:]),parents=v) for k,v in sorted(counter.items())]
        summaries[split]=[dict(dataset=k[0],interpolation=k[1],parents=v) for k,v in sorted(summary.items())]
    data=json.loads((OUT/'RESULT.json').read_text());verified=[]
    for parent in data['parents']:
        target=parent['target_refresh']
        if target['interpolation']!='INTER_CUBIC':continue
        assert digest(target['raw_path'])==target['raw_sha256']
        raw=load(target['raw_path']);native=load(parent['source_files']['gt'])
        rebuilt=cv2.resize(raw,(native.shape[1],native.shape[0]),interpolation=cv2.INTER_CUBIC)
        error=int(np.abs(rebuilt.astype(np.int16)-native.astype(np.int16)).max());assert error==0
        verified.append(dict(sample_id=parent['sample_id'],raw_hw=list(raw.shape[:2]),target_hw=list(native.shape[:2]),resize_reconstruction_max_error=error))
    result=dict(global_counts=counts,summary=summaries,verified=verified,source='tools/manage_vosr_2u_block256.py::source_parent target-only resize before current V3',runner_sha256=digest(__file__))
    (OUT/'TARGET_SCALE_AUDIT.json').write_text(json.dumps(result,indent=2))
    original=DATA/'lenovo_mojito_vosr_finetune_20261026'
    forensic={}
    for name in ('processing_config.json','delivery_report.json','annotation_audit_20261027.json'):
        path=original/name;forensic[name]=dict(path=str(path),sha256=digest(path),content=json.loads(path.read_text()))
    (OUT/'FINETUNE_LINEAGE.json').write_text(json.dumps(forensic,indent=2))
    print(json.dumps(dict(summary=summaries,exact_resize_replays=len(verified))))


def main():
    """@luna-comment Input:immutable actual V3 bank/manifests and frozen no-SDF75K Full. Output:stratified paired-data audit plus049-052. Key Steps:64 seeded parents/two patches,byte/hash checks,raw/pre/target/model comparisons and synthetic alignment control. Key Params:no source edits or new training. State/Dependencies:read-only NAS/CUDA and owned receipt images."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scale-only',action='store_true');args=parser.parse_args()
    if args.scale_only:
        audit_scales();return
    import cv2
    import numpy as np
    import torch
    import yaml
    from tools.train_v3m_block256 import bootstrap
    bootstrap('/home/guang.yang/workspace/Real-ESRGAN')
    from basicsr.archs import build_network
    from realesrgan.data.basic_text_line_y4uv_v3 import rgb_to_y4uv,y4uv_to_yuv444
    from realesrgan.utils import YUV2RGB
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'RESULT.json').exists()
    cv2.setNumThreads(1);torch.set_num_threads(2)
    # Validate translation direction on a known, sharp, nonperiodic pattern.
    rng=np.random.default_rng(20261028);test=np.full((256,256),205,np.float32)
    for _ in range(20):
        a,b=map(int,rng.integers(25,220,size=2));cv2.rectangle(test,(a,b),(min(a+int(rng.integers(4,20)),230),min(b+int(rng.integers(8,25)),230)),30,-1)
    shifted=cv2.warpAffine(test,np.array([[1,0,1],[0,1,-2]],np.float32),(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)
    control,_=align(test,shifted,support(test,np.full(test.shape,255,np.uint8)))
    assert control['shift_applied_to_GT_xy']==[-1,2],control
    cfg=SOURCE/'pretrain/training/configs/full.yml';checkpoint=DATA/'nosdf_75k_residual_049052_20261028/checkpoints/nosdf_full_75000.pth'
    model=build_network(yaml.safe_load(cfg.read_text())['network_g']).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu')['params'],strict=True)
    manifest=SOURCE/'pretrain/source_view/manifests/train.jsonl';rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    bank=SOURCE/'pretrain/bank256';records=[];parents=[];started=time.time()
    for group in ('HXA','HXB','lenovo0926','lenovo1011'):
        choices=[r for r in rows if r['dataset']==group];order=rng.permutation(len(choices));accepted=0
        for ordinal in order:
            row=choices[int(ordinal)];stem=bank/'shards/train'/group/row['sample_id']
            receipt=json.loads(Path(str(stem)+'.receipt.json').read_text());meta=json.loads((bank/receipt['metadata']).read_text());array=np.load(bank/receipt['array'],mmap_mode='r')
            candidates=[i for i,m in enumerate(meta) if m['category']=='baseline' and m['real_hw']==[256,256] and .02<(array[i,:,:,6]>127).mean()<.4 and (array[i,:,:,7]==255).all()]
            if len(candidates)<2:continue
            indices=[candidates[len(candidates)//3],candidates[2*len(candidates)//3]]
            assert digest(bank/receipt['array'])==receipt['array_sha256'] and digest(bank/receipt['metadata'])==receipt['metadata_sha256']
            raw={k:load(row['preprocess']['source_files'][k]) for k in ('lq','gt')};pre={k:load(row['files'][k]) for k in ('lq','gt')}
            for k in ('lq','gt'):
                assert digest(row['files'][k])==row['output_sha256'][k]
                assert digest(row['preprocess']['source_files'][k])==row['preprocess']['source_hashes'][k]
                assert raw[k].shape==pre[k].shape==pre['lq'].shape
            original=row.get('original_gt_lineage');extra=load(original['path']) if original else None
            if original:assert digest(original['path'])==original['sha256']
            if extra is not None and extra.shape!=pre['lq'].shape:extra=None
            for index in indices:
                pixels=np.array(array[index]);x,y=meta[index]['xy'];sl=np.s_[y:y+256,x:x+256]
                cropped={k:v[sl] for k,v in pre.items()};rawcrop={k:v[sl] for k,v in raw.items()}
                assert np.array_equal(pixels[:,:,:3],cropped['lq']) and np.array_equal(pixels[:,:,3:6],cropped['gt'])
                tensor=torch.from_numpy(cropped['lq'][...,::-1].transpose(2,0,1).copy()).cuda().float().unsqueeze(0)/255
                with torch.inference_mode():prediction=YUV2RGB(y4uv_to_yuv444(model(rgb_to_y4uv(tensor))),is_train=True).clamp(0,1)
                output=np.rint(prediction[0].cpu().numpy().transpose(1,2,0)[...,::-1]*255).clip(0,255).astype(np.uint8)
                key=f'train_{len(records):03d}'
                record=inspect_pair(key,pixels,meta[index],rawcrop,cropped,output,extra[sl] if extra is not None else None)
                records.append(dict(dataset=group,sample_id=row['sample_id'],patch_index=index,**record))
            parents.append(dict(dataset=group,sample_id=row['sample_id'],indices=indices,target_refresh=row.get('target_refresh'),previous_gt=row.get('previous_gt'),source_files=row['preprocess']['source_files'],preprocessed_files=row['files'],verified_hashes=row['output_sha256'],bank_array_sha256=receipt['array_sha256'],exact_baseline_pixels=True))
            accepted+=1
            state=dict(complete=False,records=records,parents=parents,elapsed_seconds=time.time()-started,synthetic_alignment_control=control)
            (OUT/'RESULT.json').write_text(json.dumps(state,indent=2));print(json.dumps(dict(dataset=group,parents=accepted,total_parents=len(parents),patches=len(records),seconds=state['elapsed_seconds'])),flush=True)
            if accepted==16:break
        assert accepted==16,(group,accepted)
    finetune=[json.loads(x) for x in (SOURCE/'finetune/source_view/manifests/train.jsonl').read_text().splitlines()]
    selected=json.loads((BASE/'selection.json').read_text())['samples']
    rois={'049':[850,1780,2050,2130],'050':[850,1750,2050,2100],'051':[850,1820,2050,2170],'052':[850,1900,2050,2250]}
    for identity,roi in rois.items():
        entry=next(e for e in selected if e['id']==identity);row=next(r for r in finetune if r['sample_id']==entry['sample_id'])
        raw={k:load(row['preprocess']['source_files'][k]) for k in ('lq','gt')};pre={k:load(row['files'][k]) for k in ('lq','gt')}
        for k in ('lq','gt'):
            assert digest(row['files'][k])==row['output_sha256'][k] and digest(row['preprocess']['source_files'][k])==row['preprocess']['source_hashes'][k]
        xa,ya,xb,yb=roi
        raw={k:cv2.rotate(v,cv2.ROTATE_90_CLOCKWISE)[ya:yb,xa:xb] for k,v in raw.items()}
        pre={k:cv2.rotate(v,cv2.ROTATE_90_CLOCKWISE)[ya:yb,xa:xb] for k,v in pre.items()}
        output=load(DATA/f'nosdf_75k_residual_049052_20261028/{identity}_nosdf_full_output.png')
        record=inspect_pair('mojito_'+identity,None,dict(display_roi=roi),raw,pre,output)
        records.append(dict(dataset='mojito_finetune_anchor',sample_id=row['sample_id'],**record))
    result=dict(complete=True,records=records,parents=parents,checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest),runner_sha256=digest(__file__),elapsed_seconds=time.time()-started,synthetic_alignment_control=control,scope='64 stratified seeded parents,128 baseline text patches plus4 finetune anchors; not population quality. Alignment is diagnostic only. No training/data writes.')
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2));print('COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
