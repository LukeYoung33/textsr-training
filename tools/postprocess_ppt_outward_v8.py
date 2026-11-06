"""P10 sample experiment: outward nearest-normal patch feather, all-stroke guard."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare,smoothstep,write_png
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
from tools.postprocess_ppt_contour_v6 import read_image


def outward_feather(base, prepared, alpha):
    """@luna-comment Input: float V4, original buffers, correction alpha. Output: V8/maps.
    Key Steps: plateau envelope; outward nearest-boundary extension; all-stroke guard.
    Key Params: alpha.5/close2, deficit12/protect6/full10, collar16/collision3..6.
    State/Dependencies: immutable buffers; scipy EDT; no smoothing of image or delta.
    """
    src,y,roi,core,bg,_,_=prepared
    by=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    all_strokes=roi & (by-y>12)
    clearance=distance_transform_edt(~all_strokes).astype(np.float32)
    seed=(roi & (core | (alpha>=.5))).astype(np.uint8)
    patch=binary_fill_holes(cv2.morphologyEx(seed,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))) & roi
    full=src-base
    if patch.any():
        rho,indices=distance_transform_edt(~patch,return_indices=True)
        rho=rho.astype(np.float32)
        boundary=full[indices[0],indices[1]]
        _,labels=cv2.connectedComponents(patch.astype(np.uint8))
        owner=labels[indices[0],indices[1]]
    else:
        rho=np.full(y.shape,1e6,np.float32); boundary=np.zeros_like(full)
        owner=np.zeros(y.shape,np.int32)
    q=roi*(~patch)*(1-smoothstep(rho/16))*smoothstep((clearance-6)/4)*smoothstep((clearance-rho-3)/3)
    change=-q[...,None]*(boundary-full)
    dy=cv2.cvtColor(change,cv2.COLOR_BGR2GRAY)
    y4=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    allowed=np.where(dy>=0,np.maximum(.5,by+2-y4),np.maximum(.5,y4-by+2))
    z=dy/allowed; t=np.ones_like(z)
    np.divide(np.tanh(z),z,out=t,where=np.abs(z)>=1e-6)
    change*=t[...,None]
    floating=base+change
    result=np.clip(np.rint(floating),0,255).astype(np.uint8)
    reference=np.clip(np.rint(base),0,255).astype(np.uint8)
    report=dict(max_change=float(np.abs(change).max()),active=int(np.count_nonzero(q)),
                clipping=int(np.count_nonzero((floating<0)|(floating>255))),protected={})
    for name,mask in dict(patch=patch,strokes=clearance<=6,outside=~roi,far=rho>=16).items():
        count=int(np.count_nonzero(np.any(result!=reference,axis=2)&mask))
        report['protected'][name]=count
        assert count==0
    return result,dict(patch=patch,stroke_protect=clearance<=6,rho=rho,q=q,owner=owner,full_before=full,full_after=src-floating),report


def main():
    """@luna-comment Input: sample CLI directories. Output: two results and diagnostics.
    Key Steps: reproduce V4, outward feather, roundtrip and protection checks.
    Key Params: fixed sample IDs; no batch; explicit common visualization gains.
    State/Dependencies: originals and prior versions read-only, new directory only.
    """
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False); cv2.setNumThreads(4)
    reports=[]
    for ident in ('044920599','052927799'):
        path=next(args.input.glob(f'*{ident}_preprocessInput.png'))
        src=read_image(path); prepared=prepare(src,robust_seed=True)
        v4,maps,_=normal_correct(prepared)
        base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0])
        name=path.stem+'_dehalo.png'
        assert np.array_equal(v4,read_image(Path('results/ppt_dehalo_20260924/sample_v4_normal')/name))
        result,debug,report=outward_feather(base,prepared,maps['correction_weight'])
        write_png(args.output/name,result); assert np.array_equal(read_image(args.output/name),result)
        folder=args.output/'debug'/path.stem
        for key,gain in [('patch',255),('stroke_protect',255),('rho',12),('q',255)]:
            write_png(folder/(key+'.png'),np.clip(np.rint(debug[key]*gain),0,255).astype(np.uint8))
        np.save(folder/'nearest_patch_owner.npy',debug['owner'])
        write_png(folder/'nearest_patch_owner.png',(debug['owner']%65535).astype(np.uint16))
        for key in ('full_before','full_after'):
            write_png(folder/(key+'.png'),np.clip(np.rint(128+debug[key]*4),0,255).astype(np.uint8))
        boundary=debug['patch'] & ~cv2.erode(debug['patch'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        overlay=v4.copy(); overlay[boundary]=(0,0,255)
        protect=v4.copy(); protect[debug['stroke_protect']]=(0,180,0)
        write_png(folder/'boundary_overlay.png',overlay); write_png(folder/'protection_overlay.png',protect)
        for label,(x0,y0,x1,y1) in dict(title=(950,450,2750,760),practice=(950,740,2150,990),degrees=(1750,1260,2950,1480)).items():
            panels=[]
            for title,im in [('V4',v4),('V8 outward feather',result),('patch boundary',overlay)]:
                panel=cv2.copyMakeBorder(im[y0:y1,x0:x1],26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
                cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1); panels.append(panel)
            write_png(args.output/'comparisons'/(path.stem+'_'+label+'.png'),np.concatenate(panels))
        report.update(image=name,alpha_threshold=.5,all_stroke_threshold=12,protected_radius=6,collar=16)
        reports.append(report); (args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
