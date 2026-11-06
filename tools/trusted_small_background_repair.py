"""Bounded-box equivalent of the user-approved07693 trusted-pocket repair."""
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.preview_07693_trusted_background import fit_background
VERSION='trusted-small-pockets-original-plane-v1-20261105'


def repair(raw,previous,m2,validity,return_maps=False):
    """@luna-comment Input:uint8 original RGB/published repaired RGB/M2/valid. Output:equivalent trusted-pocket candidate and factual invariants. Key Steps:approved geometry/global distances;bounded donor and edge ownership boxes;original local plane;single soft adjustment. Key Params:8px donor exclusion,radius6,area1600;exact07693 replay required. State/Dependencies:source immutable,no OCR/model calls."""
    assert raw.shape==previous.shape and raw.shape[:2]==m2.shape==validity.shape
    I=raw.astype(np.float32)/255;old=previous.astype(np.float32)/255;m=m2.astype(np.float32)/255;valid=validity>=254
    shape=np.clip((m-.2)/.6,0,1);bg=(shape==0)&valid;distance=ndi.distance_transform_edt(bg);ink=m>.2
    labels,_=ndi.label(bg);selected=np.zeros(bg.shape,bool)
    for number,box in enumerate(ndi.find_objects(labels),1):
        if box is None:continue
        region=labels[box]==number;area=int(region.sum())
        if 4<=area<=1600 and distance[box][region].max()<=6 and box[0].start>0 and box[1].start>0 and box[0].stop<m.shape[0] and box[1].stop<m.shape[1]:selected[box]|=region
    closed=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
    selected|=closed&bg&(distance<=6)
    labels,_=ndi.label(selected);all_scope=np.zeros(bg.shape,bool);donors=np.zeros(bg.shape,bool);result=old.copy();records=[];background_count=0
    if not selected.any():return previous.copy(),dict(version=VERSION,changed_pixels=0,pockets=0,unresolved_pockets=0,deep_stroke_changed_pixels=0,invalid_changed_pixels=0,outside_scope_changed_pixels=0)
    stroke_distance=ndi.distance_transform_edt(~ink)
    y=cv2.GaussianBlur((I@np.array([.299,.587,.114])).astype(np.float32),(0,0),.8)
    gradient=np.hypot(cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)/8)
    nearest_bg=ndi.distance_transform_edt(~bg,return_distances=False,return_indices=True);old_background=old[tuple(nearest_bg)]
    owner_near=ndi.distance_transform_edt(labels==0,return_distances=False,return_indices=True);owners=labels[tuple(owner_near)];edge_distance=ndi.distance_transform_edt(labels==0)
    for number,box in enumerate(ndi.find_objects(labels),1):
        if box is None or (labels[box]==number).sum()<4:continue
        y0=max(0,box[0].start-80);y1=min(m.shape[0],box[0].stop+80);x0=max(0,box[1].start-80);x1=min(m.shape[1],box[1].stop+80);sl=np.s_[y0:y1,x0:x1]
        region=labels[sl]==number
        fit,audit,reference=fit_background(I[sl],m[sl],valid[sl],stroke_distance[sl],gradient[sl],region)
        audit['component']=number;audit['pixels']=int(region.sum());records.append(audit)
        if fit is None:continue
        coef,cx,cy=fit
        if return_maps:donors[sl][reference]=True
        edge=(owners[sl]==number)&(edge_distance[sl]<=1.5)&(shape[sl]>0)&(shape[sl]<.99)&valid[sl]
        scope=region|edge;yy,xx=np.where(scope);A=np.stack([np.ones(len(xx)),(xx-cx)/32,(yy-cy)/32],1);target=np.clip(A@coef,0,1).astype(np.float32)
        values=old[sl][yy,xx].copy();is_bg=region[yy,xx];values[is_bg]=target[is_bg]
        is_edge=~is_bg;opacity=shape[sl][yy,xx]
        values[is_edge]+=(1-opacity[is_edge,None])*(target[is_edge]-old_background[sl][yy[is_edge],xx[is_edge]])
        result[sl][yy,xx]=values;all_scope[sl]|=scope;background_count+=int(region.sum())
    value=np.rint(np.clip(result,0,1)*255).astype(np.uint8);value[~all_scope]=previous[~all_scope]
    assert np.array_equal(value[~valid],previous[~valid]) and np.array_equal(value[shape>=.99],previous[shape>=.99]) and np.array_equal(value[~all_scope],previous[~all_scope])
    audit=dict(version=VERSION,changed_pixels=int(np.any(value!=previous,axis=2).sum()),background_pixels=background_count,scope_pixels=int(all_scope.sum()),pockets=len(records),unresolved_pockets=sum(not r['passed'] for r in records),max_fit_rmse_dn=max((r.get('fit_rmse_dn',0) for r in records),default=0),deep_stroke_changed_pixels=0,invalid_changed_pixels=0,outside_scope_changed_pixels=0)
    if return_maps:audit['scope']=all_scope;audit['donors']=donors
    return value,audit


def replay():
    """@luna-comment Input:accepted07693 source/candidate receipts. Output:pixel parity evidence and runtime. Key Steps:strict source hashes;bounded equivalent full image;exact compare. Key Params:zero differing pixels required before bulk. State/Dependencies:read-only approved candidate."""
    import json,time
    from tools.preview_07693_trusted_background import DATA,OUT
    from tools.prepare_r1_width_bank import digest,save
    r=json.loads((DATA/'targets/lenovo0912/407-520_001479/receipt.json').read_text())['row'];a=json.loads((OUT/'AUDIT.json').read_text());assert digest(OUT/'trusted_small.png')==a['candidate_sha256']
    original=cv2.cvtColor(cv2.imread(r['previous_training_target']['path']),cv2.COLOR_BGR2RGB);previous=cv2.cvtColor(cv2.imread(r['files']['gt']),cv2.COLOR_BGR2RGB)
    start=time.time();result,audit=repair(original,previous,cv2.imread(r['files']['text_mask'],0),cv2.imread(r['files']['valid_mask'],0))
    accepted=cv2.cvtColor(cv2.imread(str(OUT/'trusted_small.png')),cv2.COLOR_BGR2RGB);different=int(np.any(result!=accepted,axis=2).sum());assert different==0,different
    record=dict(passed=True,different_pixels=different,seconds=time.time()-start,audit=audit)
    target=DATA.parent/'mask_gt_filtered_trusted_20261105/PARITY_07693.json';save(target,record);print(json.dumps(record),flush=True)


if __name__=='__main__':cv2.setNumThreads(1);replay()
