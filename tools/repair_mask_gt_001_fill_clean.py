"""Fill approved M2 strokes and remove near-boundary false pixels on001."""
from pathlib import Path
import json,sys
import numpy as np
import cv2
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import repair_mask_gt_001_direct as fill
from tools.preview_mask_guided_vosr_strokes import weighted_field
from tools.run_uniform20_steep_scurve import save,digest
OUT=fill.SOURCE.parent/'mask_gt_001_fill_clean_20261104'
ZONES={'english':(0,0,384,43),'sui':(90,42,167,120),'yue':(163,42,223,125),'zai':(211,42,283,120)}


def main():
    """@luna-comment Input:original001/approvedM2geometry andnewYuezone. Output:fill-clean2variants/maps/audit. Key Steps:directfill,far-backgroundfield,inkdirectionexcessdetector,localbackgroundreconstruction. Key Params:8pxmaxband,deepM2ink/farbackgroundfrozen;strongalsooppositepolarityshadow. State/Dependencies:originalGTimmutable;isolatedOUT."""
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    fill.OUT=OUT/'fill_stage';fill.ZONES=ZONES;fill.main()
    audit=json.loads((fill.OUT/'AUDIT.json').read_text());job=audit['source']
    raw=cv2.cvtColor(cv2.imread(job['files']['vosr_s22']),cv2.COLOR_BGR2RGB);I=raw.astype(np.float32)/255
    m=cv2.imread(job['files']['text_mask_plus2'],0).astype(np.float32)/255;valid=cv2.imread(job['files']['valid'],0)>=254;shape=np.clip((m-.2)/.6,0,1)
    current=cv2.cvtColor(cv2.imread(str(fill.OUT/'direct_strong.png')),cv2.COLOR_BGR2RGB);value=current.astype(np.float32)/255
    selected=np.zeros(m.shape,np.uint8)
    for a,b,c,d in ZONES.values():selected[fill.BOX[1]+b:fill.BOX[1]+d,fill.BOX[0]+a:fill.BOX[0]+c]=1
    selected=ndi.binary_dilation(selected,iterations=5)&valid
    distance=ndi.distance_transform_edt(m<=.2)
    bgdonor=(distance>=8)&valid
    B,_=weighted_field(I,bgdonor.astype(np.float32),12)
    labels,count=ndi.label(m>.5,np.ones((3,3)));nearest=ndi.distance_transform_edt(labels==0,return_distances=False,return_indices=True);owners=labels[tuple(nearest)]
    F=np.zeros_like(I);supported=np.zeros(m.shape,bool)
    chosen=set(np.unique(owners[selected]));chosen.discard(0)
    for number,slices in enumerate(ndi.find_objects(labels),1):
        if number not in chosen or slices is None:continue
        y0=max(0,slices[0].start-28);y1=min(m.shape[0],slices[0].stop+28);x0=max(0,slices[1].start-28);x1=min(m.shape[1],slices[1].stop+28);sl=np.s_[y0:y1,x0:x1]
        comp=labels[sl]==number;amp=np.linalg.norm(I[sl]-B[sl],axis=2);possible=comp&(m[sl]>.7)&(amp>.08)
        if possible.sum()<6:continue
        lo,hi=np.quantile(amp[possible],[.6,.98]);donor=possible&(amp>=lo)&(amp<=hi)
        if donor.sum()<4:continue
        field,_=weighted_field(I[sl],donor.astype(np.float32),6);domain=owners[sl]==number
        F[sl][domain]=field[domain];supported[sl]|=domain
    D=F-B;den=(D*D).sum(2);res=value-B;alpha=(res*D).sum(2)/(den+1e-7)
    perpendicular=np.linalg.norm(res-alpha[...,None]*D,axis=2);alignment=np.abs((res*D).sum(2))/(np.linalg.norm(res,axis=2)*np.sqrt(den)+1e-7)
    band=selected&supported&(distance<=8)&(shape<.99)&(den>.05**2)
    results={}
    for key,strong in [('fill_clean_conservative',False),('fill_clean_strong',True)]:
        excess=alpha-shape;candidate=band&(excess>(.015 if strong else .04))&(alignment>(.75 if strong else .85))
        if strong:candidate|=band&(shape==0)&(alpha<-.03)&(alignment>.8)&(distance<=4)
        # Same appearance endpoints for core/edge; no push beyond local B.
        target=B+shape[...,None]*D
        taper=np.clip((8-distance)/3,0,1);taper=taper*taper*(3-2*taper)
        fraction=(1. if strong else .65)*taper
        result=value.copy();result[candidate]=(value+fraction[...,None]*(target-value))[candidate]
        image=np.rint(np.clip(result,0,1)*255).astype(np.uint8)
        frozen=~selected|~valid|(distance>8)|(shape>=.99)
        image[frozen]=current[frozen];assert np.array_equal(image[frozen],current[frozen])
        p=OUT/(key+'.png');cv2.imwrite(str(p),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        changed=np.any(image!=current,2);after=((image.astype(np.float32)/255-B)*D).sum(2)/(den+1e-7)
        exterior=candidate&(shape==0)&(alpha>.015)&(distance<=5)
        results[key]=dict(path=str(p),sha256=digest(p),cleanup_changed_pixels=int(changed.sum()),
            exterior_ink_candidates=int(exterior.sum()),remaining_exterior_alpha_above_point02=int((exterior&(after>.02)).sum()),
            deep_ink_changed_pixels=int(np.any(image!=current,2)[shape>=.99].sum()),far_background_changed_pixels=int(changed[distance>8].sum()),outside_roi_changed_pixels=int(changed[~selected].sum()))
        cv2.imwrite(str(OUT/(key+'_map.png')),np.repeat(changed[...,None].astype(np.uint8)*255,3,2))
        difference=image.astype(np.int16)-current.astype(np.int16);y=(difference*np.array([.299,.587,.114])).sum(2);diff=np.zeros_like(image);diff[...,0]=np.clip(np.maximum(-y,0)*4,0,255);diff[...,2]=np.clip(np.maximum(y,0)*4,0,255);cv2.imwrite(str(OUT/(key+'_diff.png')),cv2.cvtColor(diff,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/'background_estimate.png'),cv2.cvtColor(np.rint(np.clip(B,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    save(OUT/'AUDIT.json',dict(passed=True,sample='001',crop_xyxy=fill.BOX,zones=ZONES,band_pixels=8,full_cleanup_distance=5,zero_ended_transition=[5,8],background_donor_minimum_distance=8,variants=results,source_gt_replaced=False))
    save(OUT/'COMPLETE.json',dict(passed=True,samples=1,candidates=2));print(json.dumps(results))


if __name__=='__main__':main()
