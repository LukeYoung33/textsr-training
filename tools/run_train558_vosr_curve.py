"""Bounded source-bound teacher inference and independent tonal-curve diagnostics."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=Path('/AI_DNS/guang.yang/TextSR/data');OUT=DATA/'train558_pre_vosr_scurve_20261028'
DEPLOY=Path('/data/guang.yang/vosr2_u1_tile512')


def status(phase,**extra):
    """@luna-comment Input:stage/details. Output:atomic progress receipt. Key Steps:UTC timestamp and owned write. Key Params:no inferred completion. State/Dependencies:isolated run output."""
    from datetime import datetime,timezone
    from tools.train_v3_edge_pilot import save
    save(OUT/'STATUS.json',dict(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**extra))


def infer(image,arm):
    """@luna-comment Input:native BGR8/arm. Output:native teacher BGR and execution receipt. Key Steps:edge-pad right/bottom;invoke documented unmodified shell onGPU3;strict output shape;crop padding only. Key Params:u1,step1,seed42,tile512/32. State/Dependencies:external deployed VOSR runtime, no source edits/downloads."""
    import cv2
    import numpy as np
    from tools.train_v3_edge_pilot import digest,save
    h,w=image.shape[:2];ph=max(512,((h+15)//16)*16);pw=max(512,((w+15)//16)*16)
    folder=OUT/'teacher'/arm;inputs=folder/'input';outputs=folder/'output';inputs.mkdir(parents=True,exist_ok=True)
    padded=np.pad(image,((0,ph-h),(0,pw-w),(0,0)),mode='edge')
    source=inputs/'train558.png';assert cv2.imwrite(str(source),padded)
    command=['bash',str(DEPLOY/'src/run_remote58_vosr2.sh'),str(inputs),str(outputs)]
    started=time.time();status('teacher_inference',arm=arm,padded_hw=[ph,pw])
    with (folder/'inference.log').open('wb') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,VOSR_GPU='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
    candidates=list(outputs.rglob('*.png'));assert len(candidates)==1,[str(p) for p in candidates]
    result=cv2.imread(str(candidates[0]),cv2.IMREAD_UNCHANGED)
    assert result is not None and result.shape==padded.shape and result.dtype==np.uint8
    result=result[:h,:w].copy();path=OUT/('vosr_'+arm+'.png');assert cv2.imwrite(str(path),result)
    receipt=dict(command=command,seconds=time.time()-started,input_sha256=digest(source),native_hw=[h,w],padded_hw=[ph,pw],generated=str(candidates[0]),generated_sha256=digest(candidates[0]),final_sha256=digest(path),u=1,infer_steps=1,seed=42,align_method='wavelet',tile_size=512,tile_overlap=32,vae_tile_size=512,vae_tile_overlap=32,geometry='replicate right/bottom; crop only')
    save(folder/'receipt.json',receipt)
    return result,receipt


def profiles(images,text_mask):
    """@luna-comment Input:matched BGR arms and evaluation-only text mask. Output:fixed-normal edge profiles/conditional10-90 widths. Key Steps:select baseline strong source edges;common positions/directions;per-arm plateau normalization;reject ambiguous crossings. Key Params:up to8 probes,normal +/-5px,delta.125. State/Dependencies:OpenCV/NumPy,no quality ranking or GT fitting."""
    import cv2
    import numpy as np
    gray={key:cv2.cvtColor(value,cv2.COLOR_BGR2GRAY).astype(np.float32) for key,value in images.items()}
    base=gray['vosr_pre'];smooth=cv2.GaussianBlur(base,(0,0),.7)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)/8
    score=gx*gx+gy*gy;score[text_mask<=0]=0;score[:24]=0;score[-24:]=0;score[:,:24]=0;score[:,-24:]=0
    distance=np.linspace(-5,5,81,dtype=np.float32);results=[]
    for _ in range(120):
        y,x=np.unravel_index(np.argmax(score),score.shape)
        if score[y,x]<100 or len(results)>=8:break
        norm=float(np.hypot(gx[y,x],gy[y,x]));dx=float(gx[y,x]/norm);dy=float(gy[y,x]/norm)
        score[max(0,y-24):y+25,max(0,x-24):x+25]=0
        mx=(x+dx*distance)[None,:];my=(y+dy*distance)[None,:]
        sampled={key:cv2.remap(value,mx,my,cv2.INTER_LINEAR)[0] for key,value in gray.items()}
        baseline=sampled['vosr_pre'];lo=float(np.median(baseline[:8]));hi=float(np.median(baseline[-8:]))
        if hi-lo<35:continue
        measurements={}
        for key,value in sampled.items():
            low=float(np.median(value[:8]));high=float(np.median(value[-8:]));span=high-low
            normalized=(value-low)/max(span,1e-6);crossings=[]
            for threshold in [.1,.9]:
                where=np.flatnonzero((normalized[:-1]<threshold)&(normalized[1:]>=threshold))
                crossing=[float(distance[i]+(threshold-normalized[i])/(normalized[i+1]-normalized[i])*(distance[i+1]-distance[i])) for i in where]
                crossings.append(crossing)
            valid=span>=25 and all(len(c)==1 for c in crossings) and crossings[1][0]>crossings[0][0]
            measurements[key]=dict(values=value.tolist(),plateaus=[low,high],contrast=span,width_10_90=crossings[1][0]-crossings[0][0] if valid else None,unambiguous=bool(valid),overshoot_bright=max(0,float(value.max()-high)),overshoot_dark=max(0,float(low-value.min())))
        results.append(dict(x=int(x),y=int(y),normal=[dx,dy],distance=distance.tolist(),arms=measurements))
    return dict(probes=results,policy='Positions/normals selected on unmodified pre-input teacher output. Text mask only selects diagnostic probes, never inference or curves. Width valid only for unique rising crossings and adequate local plateau contrast; not a population quality score.')


def main():
    """@luna-comment Input:known train558 source/cache and documented teacher. Output:new raw/pre teacher results,weak/full curve arms and source/parameter/probe receipts. Key Steps:verify all source identities;two independently seeded executions;same output-derived rails for four post curves;monotonicity/common delta audits. Key Params:exterior.4/.1 and midpoint-S gamma1.8 alpha.25/.5. State/Dependencies:isolated outputs,no deployed source/training mutation."""
    import cv2
    import numpy as np
    from tools.train_v3_edge_pilot import save,digest
    from tools.prepare_v3pre_training import REFERENCES
    from evaluation.document_box_neighbor_20261027 import neighbor
    from evaluation.document_box_pre640_revision2_20261027 import revision2
    from evaluation.document_box_pre640_astra_20261027 import pre640
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'COMPLETE.json').exists(),'Refusing to overwrite completed output'
    assert {key:digest(ROOT/key) for key in REFERENCES}==REFERENCES
    rows=[json.loads(line) for line in (DATA/'vosr_2u_block256_v1_20261023/source_view/manifests/train.jsonl').read_text().splitlines() if line.strip()]
    row=next(r for r in rows if r['sample_id']=='batch_3_043-128_000558')
    current=next(json.loads(line) for line in (DATA/'v3pre_v3m_three_20261027/pretrain/source_view/manifests/train.jsonl').read_text().splitlines() if line.strip() and json.loads(line)['sample_id']==row['sample_id'])
    assert current['preprocess']['source_hashes']['lq']==row['output_sha256']['lq']
    for record in [row,current]:assert digest(record['files']['lq'])==record['output_sha256']['lq']
    raw=cv2.imread(row['files']['lq'],cv2.IMREAD_UNCHANGED);pre=cv2.imread(current['files']['lq'],cv2.IMREAD_UNCHANGED)
    assert raw.shape==pre.shape==(1255,1287,3)
    shutil.copyfile(row['files']['lq'],OUT/'lq.png');shutil.copyfile(current['files']['lq'],OUT/'pre_lq.png')
    status('hashing_teacher')
    teacher={str(p):digest(p) for p in [DEPLOY/'src/run_remote58_vosr2.sh',DEPLOY/'src/run_remote58_vosr2.py',DEPLOY/'src/preset/ckpts/VOSR2/args.json',DEPLOY/'src/preset/ckpts/VOSR2/checkpoints/ema_model.safetensors']}
    raw_output,raw_receipt=infer(raw,'raw');output,pre_receipt=infer(pre,'pre')
    assert raw_receipt['padded_hw']==pre_receipt['padded_hw']
    status('post_curves')
    estimate=revision2.estimate(output);q,tail=neighbor.descriptors(estimate['canvas'],estimate['nodes'])
    low,repair=neighbor.repair(estimate['fields']['low'],estimate['fields']['high'],q,tail,estimate['nodes']);high=estimate['fields']['high']
    np.savez_compressed(OUT/'fixed_output_rails.npz',low=low,high=high,nodes=estimate['nodes'])
    records={};images={'vosr_pre':output}
    for key,slope,alpha in [('tail_r04',.4,0),('tail_r01',.1,0),('scurve_a025',.1,.25),('scurve_a05',.1,.5)]:
        tables=np.empty((*low.shape,256),np.float32)
        for iy,ix in np.ndindex(low.shape):
            l,h=float(low[iy,ix]),float(high[iy,ix]);values=np.arange(256,dtype=np.float64)
            interior=pre640.curve(values,l,(l+h)/2,h,alpha,gamma=1.8)
            tables[iy,ix]=revision2.curve(interior,l,h,slope)
        assert np.isfinite(tables).all() and np.diff(tables,axis=2).min()>=-1e-5
        context=dict(nodes=estimate['nodes'],tables=tables,native_shape=list(output.shape[:2]))
        result,telemetry=pre640.render(output,context,strength=1.,chunk=128)
        delta=result.astype(np.int16)-output.astype(np.int16)
        assert np.array_equal(delta[...,0],delta[...,1]) and np.array_equal(delta[...,1],delta[...,2])
        path=OUT/(key+'.png');assert cv2.imwrite(str(path),result)
        records[key]=dict(slope=slope,interior_alpha=alpha,gamma=1.8,midpoint='(L+H)/2',output_sha256=digest(path),finite_monotone_lut=True,shared_channel_delta=True,telemetry=telemetry)
        images[key]=result
    mask=cv2.imread(row['files']['text_mask'],cv2.IMREAD_GRAYSCALE)
    save(OUT/'EDGE_PROBES.json',profiles(images,mask))
    receipt=dict(passed=True,sample_id=row['sample_id'],split='train',shape_hw=list(raw.shape[:2]),sources=dict(lq=row['files']['lq'],pre_lq=current['files']['lq'],lq_sha256=row['output_sha256']['lq'],pre_lq_sha256=current['output_sha256']['lq']),teacher_files=teacher,teacher_raw=raw_receipt,teacher_pre=pre_receipt,curves=records,fixed_rails_sha256=digest(OUT/'fixed_output_rails.npz'),preprocessing_references=REFERENCES,limitations=['One training image only','Teacher u1,not refreshed-u2 target replay','VOSR direct padded whole-image;no historical large-line OCR rerouting','Postcurve statistics estimated once from teacher output,not GT','Visual quality and content fidelity require user review'])
    save(OUT/'COMPLETE.json',receipt);status('complete')


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',error=repr(error));raise
