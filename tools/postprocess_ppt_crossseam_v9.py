"""P11: two-sided normal interpolation across the pale background seam."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes,distance_transform_edt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare,smoothstep,write_png
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
from tools.postprocess_ppt_contour_v6 import read_image


def sample(array,x,y):
    """@luna-comment Input: image and <=20000 coordinates. Output: linear samples.
    Key Steps: compact remap; constant invalid border. Key Params: float coordinates.
    State/Dependencies: OpenCV; caller validates image bounds.
    """
    return cv2.remap(array,x.astype(np.float32)[None],y.astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]


def cross_seam(base,prepared,alpha):
    """@luna-comment Input: V4 float and buffers. Output: cross-seam result/maps/report.
    Key Steps: signed band; adaptive inner endpoint; ray collision; RGB interpolation.
    Key Params: inner8/outer12, stroke6/full10, five ray samples, normal sigma1.
    State/Dependencies: immutable inputs, scipy geometry; no image or delta blur.
    """
    src,y,roi,core,bg,_,_=prepared
    by=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    clear=distance_transform_edt(~(roi & (by-y>12))).astype(np.float32)
    seed=(roi & (core | (alpha>=.5))).astype(np.uint8)
    patch=binary_fill_holes(cv2.morphologyEx(seed,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))) & roi
    s=(distance_transform_edt(~patch)-distance_transform_edt(patch)).astype(np.float32)
    gy,gx=np.gradient(cv2.GaussianBlur(s,(0,0),1)); norm=np.hypot(gx,gy)
    gx/=np.maximum(norm,1e-6); gy/=np.maximum(norm,1e-6)
    full=src-base; change=np.zeros_like(base); weights=np.zeros_like(y); lengths=np.zeros_like(y)
    yy,xx=np.nonzero(roi & ~core & (s>-8)&(s<12)&(clear>6))
    roi_float=roi.astype(np.float32); h,w=y.shape; suppressed=0; profiles=[]
    for start in range(0,len(xx),20000):
        ax=xx[start:start+20000]; ay=yy[start:start+20000]
        ss=s[ay,ax]; nx=gx[ay,ax]; ny=gy[ay,ax]
        length=np.clip(sample(clear,ax-ss*nx,ay-ss*ny)-7,0,8)
        ix=ax-(ss+length)*nx; iy=ay-(ss+length)*ny
        ox=ax+(12-ss)*nx; oy=ay+(12-ss)*ny
        valid=(length>=1)&(ss>-length)
        ray=clear[ay,ax].copy()
        for fraction in (0,.25,.5,.75,1):
            rx=ix+fraction*(ox-ix); ry=iy+fraction*(oy-iy)
            valid &= (rx>=0)&(rx<=w-1)&(ry>=0)&(ry<=h-1)&(sample(roi_float,rx,ry)>.999)
            ray=np.minimum(ray,sample(clear,rx,ry))
        g=valid*smoothstep((clear[ay,ax]-6)/4)*smoothstep((norm[ay,ax]-.25)/.5)*smoothstep(ray-6)
        t=smoothstep((ss+length)/(length+12))
        inner=sample(full,ix,iy); outer=sample(full,ox,oy)
        target=(1-t[:,None])*inner+t[:,None]*outer
        change[ay,ax]=-g[:,None]*(target-full[ay,ax])
        weights[ay,ax]=g; lengths[ay,ax]=length
        suppressed+=int(np.count_nonzero(g==0))
        picks=np.flatnonzero((g>.5)&(np.abs(ss)<2)&(ay>450)&(ay<760)&(ax>1500)&(ax<2500))[:2]
        for j in picks:
            profiles.append(dict(pixel=[int(ax[j]),int(ay[j])],inner=[float(ix[j]),float(iy[j])],outer=[float(ox[j]),float(oy[j])],inner_correction=inner[j].tolist(),outer_correction=outer[j].tolist()))
    dy=cv2.cvtColor(change,cv2.COLOR_BGR2GRAY); y4=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    a=np.where(dy>=0,np.maximum(.5,by+2-y4),np.maximum(.5,y4-by+2)); z=dy/a
    atten=np.ones_like(z); np.divide(np.tanh(z),z,out=atten,where=np.abs(z)>=1e-6)
    change*=atten[...,None]; floating=base+change
    out=np.clip(np.rint(floating),0,255).astype(np.uint8); old=np.clip(np.rint(base),0,255).astype(np.uint8)
    changed=np.any(out!=old,axis=2)
    protected=(clear<=6)|core|~roi|(weights==0)
    assert not changed[protected].any()
    report=dict(inside_changed=int((changed&patch).sum()),outside_changed=int((changed&~patch).sum()),protected_changed=0,candidates=len(xx),suppressed=suppressed,max_change=float(np.abs(change).max()),clipping=int(np.count_nonzero((floating<0)|(floating>255))),endpoint_examples=profiles[:20])
    return out,dict(patch=patch,weight=weights,inner_length=lengths,signed_distance=s,protection=clear<=6),report


def main():
    """@luna-comment Input: two-sample paths. Output: results/debug/receipt.
    Key Steps: verify V4 reproduction, run P11, decode checks and A/B previews.
    Key Params: fixed IDs and title/Practice crops; no batch.
    State/Dependencies: read-only prior images; exclusive output folder.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);reports=[]
    for ident in ('044920599','052927799'):
        path=next(args.input.glob(f'*{ident}_preprocessInput.png'));src=read_image(path);prepared=prepare(src,robust_seed=True)
        v4,maps,_=normal_correct(prepared);base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0])
        name=path.stem+'_dehalo.png';root=Path('results/ppt_dehalo_20260924')
        assert np.array_equal(v4,read_image(root/'sample_v4_normal'/name))
        v8=read_image(root/'sample_v8_outward'/name)
        result,debug,report=cross_seam(base,prepared,maps['correction_weight'])
        write_png(args.output/name,result);assert np.array_equal(result,read_image(args.output/name))
        folder=args.output/'debug'/path.stem
        for key,gain,offset in [('patch',255,0),('weight',255,0),('inner_length',30,0),('signed_distance',5,128),('protection',255,0)]:
            write_png(folder/(key+'.png'),np.clip(np.rint(debug[key]*gain+offset),0,255).astype(np.uint8))
        overlay=v4.copy();active=debug['weight']>0
        overlay[active]=(.5*overlay[active]+.5*np.array([0,255,255])).astype(np.uint8)
        write_png(folder/'cross_seam_overlay.png',overlay)
        for label,(x0,y0,x1,y1) in dict(title=(950,450,2750,760),practice=(950,740,2150,990),degrees=(1750,1260,2950,1480)).items():
            panels=[]
            for title,im in [('V4',v4),('V8',v8),('V9 cross seam',result),('active corridor',overlay)]:
                panel=cv2.copyMakeBorder(im[y0:y1,x0:x1],26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30));cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
            write_png(args.output/'comparisons'/(path.stem+'_'+label+'.png'),np.concatenate(panels))
        report['image']=name;reports.append(report);(args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='endpoint_examples'}),flush=True)


if __name__=='__main__':main()
