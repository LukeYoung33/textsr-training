"""Isolated stronger interior curves and luma USM on a verified teacher output."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.document_box_pre640_astra_20261027 import pre640
from evaluation.document_box_pre640_revision2_20261027 import revision2

SOURCE=ROOT/'evaluation/train558_full_vosr_scurve_20261028'
OUT=ROOT/'evaluation/train558_steep_usm_20261028'


def usm(image,amount,sigma=.8):
    """@luna-comment Input:native uint8 BGR,nonnegative amount,positive sigma in px. Output:uint8 BGR and clipping telemetry. Key Steps:float Y Gaussian residual;round;gamut-bound shared integer delta. Key Params:reflect101,no threshold or halo limiter. State/Dependencies:OpenCV/NumPy only;source unmodified."""
    assert image.dtype==np.uint8 and amount>=0 and sigma>0
    value=image.astype(np.float32)
    y=.114*value[...,0]+.587*value[...,1]+.299*value[...,2]
    residual=y-cv2.GaussianBlur(y,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
    wanted=np.rint(amount*residual).astype(np.int16)
    delta=np.clip(wanted,-image.min(axis=2).astype(np.int16),255-image.max(axis=2).astype(np.int16))
    result=(image.astype(np.int16)+delta[...,None]).astype(np.uint8)
    return result,dict(sigma=sigma,amount=amount,threshold=0,border='reflect101',mean_abs_delta=float(np.abs(delta).mean()),gamut_limited_pixels=int(np.count_nonzero(wanted!=delta)))


def fixed_profiles(images,old_probes):
    """@luna-comment Input:native BGR arms and frozen prior probes. Output:identical-position/normal profiles and per-arm widths/excursions. Key Steps:explicit float32 remap;local endpoint median normalization;unique10/90 rising crossing measurement. Key Params:contrast25,previous distances unchanged. State/Dependencies:read-only frozen geometry,OpenCV interpolation;no edge reselection."""
    gray={k:cv2.cvtColor(v,cv2.COLOR_BGR2GRAY).astype(np.float32) for k,v in images.items()}
    results=[]
    for p in old_probes['probes']:
        distance=np.asarray(p['distance'],np.float32);dx,dy=p['normal']
        mx=(float(p['x'])+dx*distance).astype(np.float32)[None,:]
        my=(float(p['y'])+dy*distance).astype(np.float32)[None,:]
        measurements={}
        for key,image in gray.items():
            value=cv2.remap(image,mx,my,cv2.INTER_LINEAR)[0]
            low=float(np.median(value[:8]));high=float(np.median(value[-8:]));span=high-low
            normalized=(value-low)/max(span,1e-6);crossings=[]
            for threshold in [.1,.9]:
                indices=np.flatnonzero((normalized[:-1]<threshold)&(normalized[1:]>=threshold))
                crossings.append([float(distance[i]+(threshold-normalized[i])/(normalized[i+1]-normalized[i])*(distance[i+1]-distance[i])) for i in indices])
            valid=span>=25 and all(len(c)==1 for c in crossings) and crossings[1][0]>crossings[0][0]
            measurements[key]=dict(values=value.tolist(),plateaus=[low,high],contrast=span,width_10_90=crossings[1][0]-crossings[0][0] if valid else None,unambiguous=bool(valid),overshoot_bright=max(0,float(value.max()-high)),overshoot_dark=max(0,float(low-value.min())))
        np.testing.assert_allclose(measurements['vosr_pre']['values'],p['arms']['vosr_pre']['values'],rtol=0,atol=1e-5)
        results.append(dict(x=p['x'],y=p['y'],normal=p['normal'],distance=p['distance'],arms=measurements))
    return dict(probes=results,policy='Exact prior positions/normals/distances;baseline sampled profile equals prior receipt within1e-5;float32 remap coordinates;unique rising10/90 crossings and contrast>=25 only.')


def main():
    """@luna-comment Input:frozen full teacher outputs/rails. Output:new native arms and provenance/fixed-edge JSON. Key Steps:verify hashes;copy original stages;render two stronger curves;weak-to-strong USM and combined arm;measure identical probes. Key Params:alpha.75 gamma1.8/2.6,sigma.8 amount.4/.8/1.2. State/Dependencies:owned separate output;no teacher or GT changes."""
    cv2.setNumThreads(1)
    assert not (OUT/'EXPERIMENT.json').exists(),'Refusing completed experiment overwrite'
    OUT.mkdir(parents=True,exist_ok=True)
    parent=json.loads((SOURCE/'COMPLETE.json').read_text(encoding='utf-8'))
    assert parent['passed'] and parent['full_documented_pipeline']
    expected=json.loads((SOURCE/'DELIVERY_HASHES.json').read_text(encoding='utf-8'))
    copied=['lq','pre_lq','vosr_raw','vosr_pre','tail_r04','tail_r01','scurve_a025','scurve_a05','training_gt_vosr2u_v3']
    source_hashes={}
    for name in [k+'.png' for k in copied]+['fixed_output_rails.npz']:
        digest=hashlib.sha256((SOURCE/name).read_bytes()).hexdigest();assert digest==expected[name]
        source_hashes[name]=digest
        if name.endswith('.png'):shutil.copyfile(SOURCE/name,OUT/name)
    image=cv2.imread(str(OUT/'vosr_pre.png'));assert image.shape==(1255,1287,3)
    rails=np.load(SOURCE/'fixed_output_rails.npz');low,high,nodes=[rails[k] for k in ['low','high','nodes']]
    records={}
    for key,alpha,gamma in [('scurve_a075_g18',.75,1.8),('scurve_a075_g26',.75,2.6)]:
        tables=np.empty((*low.shape,256),np.float32)
        for iy,ix in np.ndindex(low.shape):
            l,h=float(low[iy,ix]),float(high[iy,ix])
            tables[iy,ix]=revision2.curve(pre640.curve(np.arange(256),l,(l+h)/2,h,alpha,gamma),l,h,.1)
        assert np.isfinite(tables).all() and np.diff(tables,axis=2).min()>=-1e-5
        result,telemetry=pre640.render(image,dict(nodes=nodes,tables=tables,native_shape=list(image.shape[:2])),strength=1.,chunk=128)
        assert cv2.imwrite(str(OUT/(key+'.png')),result)
        records[key]=dict(method='interior_S_then_exterior',alpha=alpha,gamma=gamma,midpoint_derivative=1+alpha*(gamma-1),exterior_slope=.1,finite_monotone_lut=True,telemetry=telemetry)
    for key,amount,input_key in [('usm_a04',.4,'vosr_pre'),('usm_a08',.8,'vosr_pre'),('usm_a12',1.2,'vosr_pre'),('scurve_a05_usm_a04',.4,'scurve_a05')]:
        result,telemetry=usm(cv2.imread(str(OUT/(input_key+'.png'))),amount)
        assert cv2.imwrite(str(OUT/(key+'.png')),result)
        records[key]=dict(method='luma_USM',input=input_key,**telemetry)
    for key,record in records.items():
        result=cv2.imread(str(OUT/(key+'.png')))
        parent_image=cv2.imread(str(OUT/(record.get('input','vosr_pre')+'.png')))
        delta=result.astype(np.int16)-parent_image.astype(np.int16)
        assert result.shape==image.shape and np.array_equal(delta[...,0],delta[...,1]) and np.array_equal(delta[...,1],delta[...,2])
        record.update(shared_channel_delta=True,changed_pixels=int(np.any(delta!=0,axis=2).sum()),sha256=hashlib.sha256((OUT/(key+'.png')).read_bytes()).hexdigest())
        print(key,json.dumps(record,ensure_ascii=True),flush=True)
    old_probes=json.loads((SOURCE/'EDGE_PROBES.json').read_text(encoding='utf-8'))
    images={k:cv2.imread(str(OUT/(k+'.png'))) for k in ['vosr_pre','scurve_a05',*records]}
    measured=fixed_profiles(images,old_probes)
    assert [(p['x'],p['y']) for p in measured['probes']]==[(p['x'],p['y']) for p in old_probes['probes']]
    (OUT/'EDGE_PROBES.json').write_text(json.dumps(measured,indent=2),encoding='utf-8')
    receipt=dict(passed=True,sample_id=parent['sample_id'],shape_hw=[1255,1287],source=str(SOURCE),source_hashes=source_hashes,arms=records,full_teacher_reused=True,original_probe_positions_equal=True,limitations=['One training image','Per-arm local plateau normalization;not population MTF','No automatic visual quality ranking','USM intentionally has no halo suppression'])
    (OUT/'EXPERIMENT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__':main()
