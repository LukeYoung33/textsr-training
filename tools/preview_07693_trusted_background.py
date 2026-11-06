"""Bounded trusted-background alternative for small counters in training07693."""
from pathlib import Path
import sys,json
import cv2
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_r1_width_bank import save,digest
DATA=Path('/AI_DNS/guang.yang/TextSR/data/vosr_s22_mask_repair_mix50_20261104')
OUT=DATA/'preview_07693_trusted_background_20261105'


def fit_background(image,mask,valid,stroke_distance,gradient,region):
    """@luna-comment Input:original RGB/M2/valid/stroke distance/gradient/pocket. Output:local RGB plane or explicit rejection and donor coordinates. Key Steps:expand local search;exclude8px ink band;polarity tail filtering;robust IRLS plane fit. Key Params:24/48/80px search,96 donors,4DN residual gate. State/Dependencies:original VOSR pixels only."""
    ys,xs=np.where(region);cy=float(ys.mean());cx=float(xs.mean());h,w=mask.shape
    for pad in [24,48,80]:
        y0=max(0,int(ys.min())-pad);y1=min(h,int(ys.max())+pad+1);x0=max(0,int(xs.min())-pad);x1=min(w,int(xs.max())+pad+1)
        sl=np.s_[y0:y1,x0:x1];candidate=valid[sl]&(mask[sl]<.05)&(stroke_distance[sl]>=8)&(gradient[sl]<.02)
        yy,xx=np.where(candidate)
        if len(xx)<96:continue
        pixels=image[sl][candidate];luma=pixels@np.array([.299,.587,.114])
        neighborhood=ndi.binary_dilation(region,iterations=5)&(mask>.8)&valid
        ink=image[neighborhood]
        if not len(ink):continue
        dark=np.median(ink@np.array([.299,.587,.114]))<np.median(luma)
        low,high=np.quantile(luma,[.4,.98] if dark else [.02,.6]);keep=(luma>=low)&(luma<=high)
        xx=xx[keep]+x0;yy=yy[keep]+y0;pixels=pixels[keep]
        if len(xx)<48:continue
        A=np.stack([np.ones(len(xx)),(xx-cx)/32,(yy-cy)/32],1);weights=np.ones(len(xx));coef=None
        for _ in range(5):
            scale=np.sqrt(weights);coef=np.linalg.lstsq(A*scale[:,None],pixels*scale[:,None],rcond=None)[0]
            residual=np.linalg.norm(pixels-A@coef,axis=1);mad=max(float(np.median(residual))*1.4826,.002)
            weights=np.minimum(1.,2.5*mad/np.maximum(residual,1e-6))
        reliable=np.linalg.norm(pixels-A@coef,axis=1)<max(.012,3*mad)
        if reliable.sum()<48:continue
        error=float(np.sqrt(np.mean((pixels[reliable]-A[reliable]@coef)**2)))
        if error>4/255:continue
        return (coef,cx,cy),dict(passed=True,donors=int(reliable.sum()),search_pad=pad,fit_rmse_dn=error*255), (yy[reliable],xx[reliable])
    return None,dict(passed=False,reason='insufficient coherent original background at least8px from strokes'),None


def main():
    """@luna-comment Input:exact07693 published source binding. Output:one independent candidate plus scope/donors/audit. Key Steps:small-counter and gap detection;trusted plane reconstruction;single soft coverage adjustment;frozen-region assertions. Key Params:radius<=6px,area<=1600,closing radius6. State/Dependencies:production GT/model/training untouched."""
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    source=DATA/'targets/lenovo0912/407-520_001479/receipt.json';r=json.loads(source.read_text());row=r['row']
    original=row['previous_training_target'];assert digest(original['path'])==original['sha256']
    for key in ['gt','text_mask','valid_mask']:assert digest(row['files'][key])==row['output_sha256'][key]
    raw=cv2.cvtColor(cv2.imread(original['path']),cv2.COLOR_BGR2RGB);previous=cv2.cvtColor(cv2.imread(row['files']['gt']),cv2.COLOR_BGR2RGB)
    I=raw.astype(np.float32)/255;old=previous.astype(np.float32)/255;m=cv2.imread(row['files']['text_mask'],0).astype(np.float32)/255;valid=cv2.imread(row['files']['valid_mask'],0)>=254
    shape=np.clip((m-.2)/.6,0,1);bg=(shape==0)&valid;distance=ndi.distance_transform_edt(bg);ink=m>.2
    labels,_=ndi.label(bg);selected=np.zeros(bg.shape,bool)
    for number,box in enumerate(ndi.find_objects(labels),1):
        if box is None:continue
        region=labels[box]==number;area=int(region.sum())
        if area>=4 and area<=1600 and distance[box][region].max()<=6 and box[0].start>0 and box[1].start>0 and box[0].stop<m.shape[0] and box[1].stop<m.shape[1]:selected[box]|=region
    closed=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
    selected|=closed&bg&(distance<=6)
    labels,_=ndi.label(selected);all_scope=np.zeros(bg.shape,bool);donors=np.zeros(bg.shape,bool);changed_background=np.zeros(bg.shape,bool);result=old.copy();records=[]
    stroke_distance=ndi.distance_transform_edt(~ink)
    y=cv2.GaussianBlur((I@np.array([.299,.587,.114])).astype(np.float32),(0,0),.8)
    gradient=np.hypot(cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)/8)
    nearest_bg=ndi.distance_transform_edt(~bg,return_distances=False,return_indices=True);old_background=old[tuple(nearest_bg)]
    # Soft-edge ownership follows the nearest selected small pocket, never deep ink.
    owner_near=ndi.distance_transform_edt(labels==0,return_distances=False,return_indices=True);owners=labels[tuple(owner_near)];edge_distance=ndi.distance_transform_edt(labels==0)
    for number,box in enumerate(ndi.find_objects(labels),1):
        if box is None:continue
        region=labels==number
        if region.sum()<4:continue
        fit,audit,reference=fit_background(I,m,valid,stroke_distance,gradient,region);audit['component']=number;audit['pixels']=int(region.sum());records.append(audit)
        if fit is None:continue
        coef,cx,cy=fit;donors[reference]=True
        edge=(owners==number)&(edge_distance<=1.5)&(shape>0)&(shape<.99)&valid
        scope=region|edge;yy,xx=np.where(scope);A=np.stack([np.ones(len(xx)),(xx-cx)/32,(yy-cy)/32],1);target=np.clip(A@coef,0,1).astype(np.float32)
        plane=np.zeros_like(I);plane[yy,xx]=target
        result[region]=plane[region]
        result[edge]=(old+(1-shape[...,None])*(plane-old_background))[edge]
        all_scope|=scope;changed_background|=region
    value=np.rint(np.clip(result,0,1)*255).astype(np.uint8);value[~all_scope]=previous[~all_scope]
    assert np.array_equal(value[~valid],previous[~valid]) and np.array_equal(value[shape>=.99],previous[shape>=.99]) and np.array_equal(value[~all_scope],previous[~all_scope])
    cv2.imwrite(str(OUT/'trusted_small.png'),cv2.cvtColor(value,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/'scope.png'),all_scope.astype(np.uint8)*255)
    overlay=previous.copy();contours,_=cv2.findContours(changed_background.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(255,145,25),1)
    yy,xx=np.where(donors);overlay[yy[::12],xx[::12]]=(0,230,255);cv2.imwrite(str(OUT/'donors.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    audit=dict(passed=True,sample='07693',source_receipt_sha256=digest(source),previous_gt_sha256=row['output_sha256']['gt'],candidate_sha256=digest(OUT/'trusted_small.png'),background_pixels=int(changed_background.sum()),scope_pixels=int(all_scope.sum()),changed_pixels=int(np.any(value!=previous,axis=2).sum()),donor_pixels=int(donors.sum()),components=records,deep_stroke_changed_pixels=0,invalid_changed_pixels=0,outside_scope_changed_pixels=0,production_gt_changed=False,training_changed=False)
    save(OUT/'AUDIT.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='components'}),flush=True)


if __name__=='__main__':main()
