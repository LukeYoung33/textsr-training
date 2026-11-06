"""User-approved M2 stroke completion on three marked regions of training001."""
from pathlib import Path
import json,sys,time
import numpy as np
import cv2
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_uniform20_steep_scurve import save,digest
from tools.preview_mask_guided_vosr_strokes import weighted_field
SOURCE=Path('/AI_DNS/guang.yang/TextSR/data/mask_guided_vosr_s22_preview8_20261104')
OUT=SOURCE.parent/'mask_gt_001_m2_direct_20261104'
BOX=(841,751,1225,1135)
# Native positions are the original384px review crop plus these subregions.
ZONES={'english':(0,0,384,43),'sui':(90,42,167,116),'zai':(211,42,283,116)}


def main():
    """@luna-comment Input:exact001S2.2/M2 anduser-markedwordzones. Output:twoactualfillcandidates/contours/maps/perzonecoverageaudit. Key Steps:VOSR-onlyappearance,M2directgeometry,deficitreplacement,strictfrozenpixels. Key Params:conservativecap.6,strongfulltarget;M0notagatingmask. State/Dependencies:sourceGTimmutable;newOUTonly."""
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    audit=json.loads((SOURCE/'INPUT_AUDIT.json').read_text());job=next(j for j in audit['jobs'] if j['id']=='001')
    for k,p in job['files'].items():assert digest(p)==job['sha256'][k]
    raw=cv2.cvtColor(cv2.imread(job['files']['vosr_s22']),cv2.COLOR_BGR2RGB);I=raw.astype(np.float32)/255
    m=cv2.imread(job['files']['text_mask_plus2'],0).astype(np.float32)/255;valid=cv2.imread(job['files']['valid'],0)>=254
    selected=np.zeros(m.shape,bool);zone_masks={}
    for name,(a,b,c,d) in ZONES.items():
        z=np.zeros(m.shape,bool);z[BOX[1]+b:BOX[1]+d,BOX[0]+a:BOX[0]+c]=True;zone_masks[name]=z;selected|=z
    B,_=weighted_field(I,((m<.05)&valid).astype(np.float32),8)
    shape=np.clip((m-.20)/.60,0,1);labels,count=ndi.label((m>.5)&valid,np.ones((3,3)))
    F=np.zeros_like(I);supported=np.zeros(m.shape,bool);anchors=np.zeros(m.shape,bool);skips=[]
    intersect=set(np.unique(labels[selected]));intersect.discard(0)
    for number,slices in enumerate(ndi.find_objects(labels),1):
        if number not in intersect or slices is None:continue
        y0=max(0,slices[0].start-24);y1=min(m.shape[0],slices[0].stop+24);x0=max(0,slices[1].start-24);x1=min(m.shape[1],slices[1].stop+24);sl=np.s_[y0:y1,x0:x1]
        comp=labels[sl]==number;amplitude=np.linalg.norm(I[sl]-B[sl],axis=2)
        possible=comp&(m[sl]>.70)&(amplitude>.08)
        if possible.sum()<6:skips.append(dict(component=number,reason='no reliable VOSR ink donors'));continue
        lo,hi=np.quantile(amplitude[possible],[.60,.98]);donor=possible&(amplitude>=lo)&(amplitude<=hi)
        if donor.sum()<4:skips.append(dict(component=number,reason='too few same-component ink donors'));continue
        field,_=weighted_field(I[sl],donor.astype(np.float32),6)
        # M2 lower-opacity boundary pixels belong to the nearest adopted component.
        distance=ndi.distance_transform_edt(~comp);domain=(distance<=2)&(m[sl]>.20)
        F[sl][domain]=field[domain];supported[sl]|=domain;anchors[sl]|=donor
    D=F-B;den=(D*D).sum(2);alpha=((I-B)*D).sum(2)/(den+1e-7)
    allowed=selected&valid&(m>.20)&supported&(den>.05**2)
    deficit=shape-alpha
    healthy=anchors|(alpha>=shape-.06)
    pending=allowed&~healthy&(deficit>.08)
    core=selected&valid&(shape>.99)&supported&~healthy&(deficit>.12)
    protected=~selected|~valid|(m<=.20)|healthy
    results={};output_values={}
    for key,strong in [('direct_conservative',False),('direct_strong',True)]:
        target=B+shape[...,None]*D
        fraction=np.ones(m.shape,np.float32) if strong else np.minimum(1,.6/np.maximum(deficit,1e-6))
        # Full replacement atstronginterior removes theperpendicularbackgroundresidual.
        result=I.copy();result[pending]=(I+fraction[...,None]*(target-I))[pending]
        value=np.rint(np.clip(result,0,1)*255).astype(np.uint8);value[protected]=raw[protected]
        assert np.array_equal(value[protected],raw[protected])
        path=OUT/(key+'.png');cv2.imwrite(str(path),cv2.cvtColor(value,cv2.COLOR_RGB2BGR));output_values[key]=value
        after=(((value.astype(np.float32)/255-B)*D).sum(2)/(den+1e-7))
        per_zone={}
        for name,z in zone_masks.items():
            active=core&z;needed=(selected&z&(shape>.99)&valid&~healthy);unresolved=needed&~supported
            per_zone[name]=dict(core_deficit_pixels=int(active.sum()),reached_target_minus_point1=int((active&(after>=shape-.10)).sum()),
                remaining_core_deficit_pixels=int((active&(after<shape-.10)).sum()),unresolved_no_donor_pixels=int(unresolved.sum()),
                original_deficit_sum=float(np.maximum(shape-alpha,0)[active].sum()),remaining_deficit_sum=float(np.maximum(shape-after,0)[active].sum()))
        delta=value.astype(np.int16)-raw.astype(np.int16);signed=(delta*np.array([.299,.587,.114])).sum(2);diff=np.zeros_like(raw);diff[...,0]=np.clip(np.maximum(-signed,0)*4,0,255);diff[...,2]=np.clip(np.maximum(signed,0)*4,0,255)
        cv2.imwrite(str(OUT/(key+'_diff.png')),cv2.cvtColor(diff,cv2.COLOR_RGB2BGR))
        results[key]=dict(path=str(path),sha256=digest(path),changed_pixels=int(np.any(delta!=0,2).sum()),
            outside_roi_changed_pixels=0,background_changed_pixels=0,healthy_changed_pixels=0,per_zone=per_zone)
    overlay=raw.copy();contours,_=cv2.findContours((m>.5).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(80,255,50),1)
    for a,b,c,d in ZONES.values():cv2.rectangle(overlay,(BOX[0]+a,BOX[1]+b),(BOX[0]+c,BOX[1]+d),(255,140,30),1)
    cv2.imwrite(str(OUT/'mask_contour.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    unresolved=selected&(shape>.99)&~supported
    for name,map_value in [('adopted_mask',shape),('fill_eligible',pending.astype(float)),('unresolved',unresolved.astype(float))]:cv2.imwrite(str(OUT/(name+'.png')),np.repeat(np.rint(map_value[...,None]*255).astype(np.uint8),3,2))
    save(OUT/'AUDIT.json',dict(passed=True,source=job,crop_xyxy=BOX,zones=ZONES,variants=results,skipped_components=skips,
        approach='user-approvedM2geometry;VOSR-onlyappearance;M2coretargetF;oneedgecoverage;noM0gate',source_gt_replaced=False))
    save(OUT/'COMPLETE.json',dict(passed=True,candidates=2,sample='001'))
    print(json.dumps({k:v['per_zone'] for k,v in results.items()}))


if __name__=='__main__':main()
