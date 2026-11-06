"""P12 wider cross-seam interpolation with collision-truncated normal rays."""
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
from tools.postprocess_ppt_crossseam_v9 import sample


def safe_span(clear,roi,fx,fy,nx,ny,maximum):
    """@luna-comment Input: seam feet/directions and clearance. Output: safe ray spans.
    Key Steps: stop at FIRST obstacle; interpolate7 crossing with .1px margin.
    Key Params: step2, clearance7; never search through an intervening stroke.
    State/Dependencies: sampled arrays only, caller uses compact batches.
    """
    h,w=clear.shape
    previous=sample(clear,fx,fy)
    active=(fx>=0)&(fx<=w-1)&(fy>=0)&(fy<=h-1)&(previous>=7)&(sample(roi,fx,fy)>.999)
    lengths=np.zeros_like(fx,dtype=np.float32)
    for offset in range(2,maximum+1,2):
        x=fx+offset*nx;y=fy+offset*ny
        c=sample(clear,x,y)
        inside=(x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)&(sample(roi,x,y)>.999)
        blocked=active&(~inside|(c<7))
        crossing=(offset-2)+2*np.clip((previous-7)/np.maximum(previous-c,1e-6),0,1)
        lengths[blocked]=np.where(inside[blocked],np.maximum(0,crossing[blocked]-.1),offset-2)
        # Refine only failed actual samples within the last safe marching cell.
        ids=np.flatnonzero(blocked&inside)
        if len(ids):
            proposed=lengths[ids]
            check=sample(clear,fx[ids]+proposed*nx[ids],fy[ids]+proposed*ny[ids])
            bad=ids[check<7]
            if len(bad):
                low=np.full(len(bad),offset-2,np.float32);high=lengths[bad].copy()
                for _ in range(8):
                    mid=(low+high)*.5
                    good=sample(clear,fx[bad]+mid*nx[bad],fy[bad]+mid*ny[bad])>=7
                    low=np.where(good,mid,low);high=np.where(good,high,mid)
                lengths[bad]=low
        active &= ~blocked
        lengths[active]=offset
        previous=c
    return lengths


def wide_cross_seam(base,prepared,alpha):
    """@luna-comment Input: float V4/buffers. Output: P12 and geometry diagnostics.
    Key Steps: signed seam, contiguous collision-free rays, endpoint interpolation.
    Key Params: max16/32; pointwise guard6..10; unchanged scalar luma limiter.
    State/Dependencies: original and V4 immutable; no global image/correction blur.
    """
    src,y,roi,core,bg,_,_=prepared;by=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    clear=distance_transform_edt(~(roi&(by-y>12))).astype(np.float32)
    seed=(roi&(core|(alpha>=.5))).astype(np.uint8)
    patch=binary_fill_holes(cv2.morphologyEx(seed,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))))&roi
    s=(distance_transform_edt(~patch)-distance_transform_edt(patch)).astype(np.float32)
    gy,gx=np.gradient(cv2.GaussianBlur(s,(0,0),1));norm=np.hypot(gx,gy)
    gx/=np.maximum(norm,1e-6);gy/=np.maximum(norm,1e-6)
    change=np.zeros_like(base);full=src-base;weights=np.zeros_like(y);inner_map=np.zeros_like(y);outer_map=np.zeros_like(y)
    yy,xx=np.nonzero(roi&~core&(s>-16)&(s<32)&(clear>6));rf=roi.astype(np.float32)
    unavailable=0;endpoint_bad=0;truncated=0;profiles=[]
    for start in range(0,len(xx),20000):
        ax=xx[start:start+20000];ay=yy[start:start+20000];ss=s[ay,ax];nx=gx[ay,ax];ny=gy[ay,ax]
        fx=ax-ss*nx;fy=ay-ss*ny
        L=safe_span(clear,rf,fx,fy,-nx,-ny,16);R=safe_span(clear,rf,fx,fy,nx,ny,32)
        valid=(L>=1)&(R>=1)&(ss>-L)&(ss<R)
        ix=fx-L*nx;iy=fy-L*ny;ox=fx+R*nx;oy=fy+R*ny
        endpoint_bad+=int(np.count_nonzero(valid&((sample(clear,ix,iy)<6.9)|(sample(clear,ox,oy)<6.9))))
        unavailable+=int(np.count_nonzero((L<1)|(R<1)));truncated+=int(np.count_nonzero((L<16)|(R<32)))
        g=valid*smoothstep((clear[ay,ax]-6)/4)*smoothstep((norm[ay,ax]-.25)/.5)
        t=smoothstep((ss+L)/np.maximum(L+R,1e-6));di=sample(full,ix,iy);do=sample(full,ox,oy)
        target=(1-t[:,None])*di+t[:,None]*do
        change[ay,ax]=-g[:,None]*(target-full[ay,ax]);weights[ay,ax]=g;inner_map[ay,ax]=L;outer_map[ay,ax]=R
        picks=np.flatnonzero((g>.7)&(np.abs(ss)<2)&(ay>450)&(ay<760)&(ax>1500)&(ax<2500))[:1]
        for j in picks:
            positions=np.linspace(-float(L[j]),float(R[j]),33,dtype=np.float32)
            px=fx[j]+positions*nx[j];py=fy[j]+positions*ny[j]
            profiles.append(dict(pixel=[int(ax[j]),int(ay[j])],inner=float(L[j]),outer=float(R[j]),x=px.tolist(),y=py.tolist(),input_rgb=sample(src,px,py).tolist(),v4_rgb=sample(base,px,py).tolist()))
    dy=cv2.cvtColor(change,cv2.COLOR_BGR2GRAY);y4=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    a=np.where(dy>=0,np.maximum(.5,by+2-y4),np.maximum(.5,y4-by+2));z=dy/a;atten=np.ones_like(z)
    np.divide(np.tanh(z),z,out=atten,where=np.abs(z)>=1e-6);change*=atten[...,None]
    floating=base+change;out=np.clip(np.rint(floating),0,255).astype(np.uint8);old=np.clip(np.rint(base),0,255).astype(np.uint8)
    changed=np.any(out!=old,axis=2);protected=(clear<=6)|core|~roi|(weights==0)
    assert not changed[protected].any()
    for profile in profiles[:20]:
        profile['v10_rgb']=sample(floating,np.array(profile['x']),np.array(profile['y'])).tolist()
    report=dict(candidates=len(xx),unavailable=unavailable,truncated=truncated,endpoint_below6_9=endpoint_bad,protected_changed=0,max_change=float(np.abs(change).max()),inside_changed=int((changed&patch).sum()),outside_changed=int((changed&~patch).sum()),profiles=profiles[:20])
    return out,dict(weight=weights,inner=inner_map,outer=outer_map,patch=patch,protected=protected),report


def main():
    """@luna-comment Input: source/output paths. Output: two V10 images/debug/coverage.
    Key Steps: verify V4, apply P12, compare V9, save same-ROI counts and profiles.
    Key Params: fixed two samples; no inference/batch or overwriting results.
    State/Dependencies: local read-only inputs, exclusive output directory.
    """
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);reports=[]
    root=Path('results/ppt_dehalo_20260924')
    for ident in ('044920599','052927799'):
        path=next(args.input.glob(f'*{ident}_preprocessInput.png'));src=read_image(path);prepared=prepare(src,robust_seed=True)
        v4,maps,_=normal_correct(prepared);base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0]);name=path.stem+'_dehalo.png'
        assert np.array_equal(v4,read_image(root/'sample_v4_normal'/name))
        v9=read_image(root/'sample_v9_cross_seam'/name)
        result,debug,report=wide_cross_seam(base,prepared,maps['correction_weight']);write_png(args.output/name,result)
        assert np.array_equal(result,read_image(args.output/name))
        folder=args.output/'debug'/path.stem
        for key,gain in [('weight',255),('inner',15),('outer',7),('patch',255),('protected',255)]:
            write_png(folder/(key+'.png'),np.clip(np.rint(debug[key]*gain),0,255).astype(np.uint8))
        overlay=v4.copy();active=debug['weight']>0;overlay[active]=(.5*overlay[active]+.5*np.array([0,255,255])).astype(np.uint8);write_png(folder/'coverage.png',overlay)
        report['coverage']={}
        for label,(x0,y0,x1,y1) in dict(title=(950,450,2750,760),practice=(950,740,2150,990),degrees=(1750,1260,2950,1480)).items():
            sl=np.s_[y0:y1,x0:x1];panels=[]
            for title,im in [('V9',v9),('V10 wider collision-aware blend',result),('V10 coverage',overlay)]:
                panel=cv2.copyMakeBorder(im[sl],26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30));cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
            write_png(args.output/'comparisons'/(path.stem+'_'+label+'.png'),np.concatenate(panels))
            c9=np.any(v9[sl]!=v4[sl],axis=2);c10=np.any(result[sl]!=v4[sl],axis=2)
            report['coverage'][label]=dict(v9_changed=int(c9.sum()),v10_changed=int(c10.sum()),v10_inside=int((c10&debug['patch'][sl]).sum()),v10_outside=int((c10&~debug['patch'][sl]).sum()))
        report['image']=name;reports.append(report);(args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='profiles'}),flush=True)


if __name__=='__main__':main()
