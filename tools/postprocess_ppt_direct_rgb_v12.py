"""P14: direct V4 background endpoint feather, without source residual injection."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare, smoothstep, write_png
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
from tools.postprocess_ppt_contour_v6 import read_image
from tools.postprocess_ppt_crossseam_v9 import sample
from tools.postprocess_ppt_wide_v10 import safe_span


def continuous_join(base, target, weight):
    """@luna-comment Input: direct RGB targets/weights. Output: continuous P14a join.
    Key Steps: normalized finite background-only target filter; support fade.
    Key Params: kernel9 sigma1.5, support6. State: target updated in place.
    """
    den=cv2.GaussianBlur(weight,(9,9),1.5)
    for channel in range(3):
        target[:,:,channel]=cv2.GaussianBlur(weight*target[:,:,channel],(9,9),1.5)/np.maximum(den,1e-6)
    effective=weight*smoothstep(distance_transform_edt(weight>0).astype(np.float32)/6)
    return base+effective[...,None]*(target-base),effective,target


def direct_feather(base, prepared, alpha, continuous=False):
    """@luna-comment Input: float V4 and geometry buffers. Output: direct feather/debug.
    Key Steps: P14 signed seam, first-blocker endpoints, actual BGR interpolation.
    Key Params: inner16/outer32, stroke guard6..10; no luminance limiter.
    State/Dependencies: immutable source/base; scipy EDT and existing safe_span.
    """
    src, y, roi, core, bg, _, _ = prepared
    clear = distance_transform_edt(~(roi & (cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)-y > 12))).astype(np.float32)
    seed = (roi & (core | (alpha >= .5))).astype(np.uint8)
    patch = binary_fill_holes(cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))) & roi
    s = (distance_transform_edt(~patch)-distance_transform_edt(patch)).astype(np.float32)
    gy, gx = np.gradient(cv2.GaussianBlur(s, (0,0), 1)); norm = np.hypot(gx,gy)
    gx /= np.maximum(norm,1e-6); gy /= np.maximum(norm,1e-6)
    floating = base.copy(); weight = np.zeros_like(y); inner = np.zeros_like(y); outer = np.zeros_like(y)
    target_map=np.zeros_like(base) if continuous else None
    yy, xx = np.nonzero(roi & ~core & (s > -16) & (s < 32) & (clear > 6))
    rf = roi.astype(np.float32); bad = 0; unavailable = 0
    for start in range(0,len(xx),20000):
        ax=xx[start:start+20000]; ay=yy[start:start+20000]; ss=s[ay,ax]; nx=gx[ay,ax]; ny=gy[ay,ax]
        fx=ax-ss*nx; fy=ay-ss*ny
        L=safe_span(clear,rf,fx,fy,-nx,-ny,16); R=safe_span(clear,rf,fx,fy,nx,ny,32)
        valid=(L>=1)&(R>=1)&(ss>-L)&(ss<R)
        ix=fx-L*nx; iy=fy-L*ny; ox=fx+R*nx; oy=fy+R*ny
        bad+=int(np.count_nonzero(valid & ((sample(clear,ix,iy)<6.9)|(sample(clear,ox,oy)<6.9))))
        unavailable+=int(np.count_nonzero((L<1)|(R<1)))
        g=valid*smoothstep((clear[ay,ax]-6)/4)*smoothstep((norm[ay,ax]-.25)/.5)
        t=smoothstep((ss+L)/np.maximum(L+R,1e-6))
        target=(1-t[:,None])*sample(base,ix,iy)+t[:,None]*sample(base,ox,oy)
        floating[ay,ax]=base[ay,ax]+g[:,None]*(target-base[ay,ax])
        if continuous: target_map[ay,ax]=np.where(g[:,None]>0,target,0)
        weight[ay,ax]=g; inner[ay,ax]=L; outer[ay,ax]=R
    extra={}
    if continuous:
        extra['raw_target']=np.clip(np.rint(target_map),0,255).astype(np.uint8)
        extra['raw_weight']=weight
        floating,weight,target_map=continuous_join(base,target_map,weight)
        extra['continuous_target']=np.clip(np.rint(target_map),0,255).astype(np.uint8)
        del target_map
    out=np.clip(np.rint(floating),0,255).astype(np.uint8); old=np.clip(np.rint(base),0,255).astype(np.uint8)
    protected=(clear<=6)|core|~roi|(weight==0); changed=np.any(out!=old,axis=2)
    assert not changed[protected].any() and bad==0
    report=dict(candidates=len(xx),unavailable=unavailable,endpoint_violations=bad,protected_changed=0,
                clipping=int(np.count_nonzero((floating<0)|(floating>255))),inside_changed=int((changed&patch).sum()),outside_changed=int((changed&~patch).sum()))
    profiles=[]
    for x,y0 in ((1579,489),(1849,515)):
        if x>=s.shape[1] or y0>=s.shape[0]: continue
        nx=gx[y0,x]; ny=gy[y0,x]; fx=x-s[y0,x]*nx; fy=y0-s[y0,x]*ny
        L=float(safe_span(clear,rf,np.array([fx]),np.array([fy]),np.array([-nx]),np.array([-ny]),16)[0])
        R=float(safe_span(clear,rf,np.array([fx]),np.array([fy]),np.array([nx]),np.array([ny]),32)[0])
        pos=np.linspace(-L,R,max(2,int(L+R)+1),dtype=np.float32); px=fx+pos*nx; py=fy+pos*ny
        profiles.append(dict(pixel=[x,y0],weight=float(weight[y0,x]),clearance=float(clear[y0,x]),inner=L,outer=R,
                             x=px.tolist(),y=py.tolist(),v4_bgr=sample(base,px,py).tolist(),v12_bgr=sample(floating,px,py).tolist()))
    report['profiles']=profiles
    return out,dict(weight=weight,inner=inner,outer=outer,patch=patch,protected=protected,**extra),report


def main():
    """@luna-comment Input: two-sample directory and new output. Output: PNG/debug/receipt.
    Key Steps: reproduce V4, run direct feather, export matched crops and endpoint traces.
    Key Params: fixed sample IDs; no batch/inference; 100/200 percent diagnostics.
    State/Dependencies: original hashes verified; output directory must not exist.
    """
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--continuous',action='store_true',help='Recorded P14a continuity amendment')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False); cv2.setNumThreads(4)
    root=Path('results/ppt_dehalo_20260924'); reports=[]
    for ident in ('044920599','052927799'):
        start=time.perf_counter(); path=next(args.input.glob(f'*{ident}_preprocessInput.png'))
        digest=hashlib.sha256(path.read_bytes()).hexdigest(); src=read_image(path); prepared=prepare(src,robust_seed=True)
        v4,maps,_=normal_correct(prepared); base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0]); name=path.stem+'_dehalo.png'
        assert np.array_equal(v4,read_image(root/'sample_v4_normal'/name))
        ready=time.perf_counter(); result,debug,report=direct_feather(base,prepared,maps['correction_weight'],args.continuous); done=time.perf_counter()
        report.update(image=name,source_sha256=digest,prepare_seconds=ready-start,feather_seconds=done-ready)
        write_png(args.output/name,result); assert np.array_equal(result,read_image(args.output/name))
        v10=read_image(root/'sample_v10_wide_final'/name); folder=args.output/'debug'/path.stem
        for key,gain in [('weight',255),('inner',15),('outer',7),('patch',255),('protected',255)]:
            write_png(folder/(key+'.png'),np.clip(np.rint(debug[key]*gain),0,255).astype(np.uint8))
        if args.continuous:
            for key in ('raw_target','continuous_target'): write_png(folder/(key+'.png'),debug[key])
            write_png(folder/'raw_weight.png',np.rint(debug['raw_weight']*255).astype(np.uint8))
        write_png(folder/'signed_rgb_change.png',np.clip(128+(result.astype(np.float32)-base)*8,0,255).astype(np.uint8))
        overlay=v4.copy(); contours,_=cv2.findContours(debug['patch'].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(0,0,255),1)
        for profile in report['profiles']:
            px=np.array(profile['x']); py=np.array(profile['y']); profile['v10_bgr']=sample(v10.astype(np.float32),px,py).tolist()
            for version in ('v4','v10','v12'):
                colors=np.array(profile[version+'_bgr']); luma=colors@np.array([.114,.587,.299]); profile[version+'_y']=luma.tolist()
                profile[version+'_peak_step']=float(np.max(np.abs(np.diff(luma))))
            a=(int(round(px[0])),int(round(py[0]))); b=(int(round(px[-1])),int(round(py[-1])))
            cv2.line(overlay,a,b,(0,255,255),1); cv2.circle(overlay,a,3,(0,255,0),-1); cv2.circle(overlay,b,3,(255,0,0),-1)
        write_png(folder/'seam_endpoints.png',overlay)
        crops=dict(title=(950,450,2750,760),practice=(950,740,2150,990),degrees=(1750,1260,2950,1480),color=(600,1690,2700,1890),marked_seam=(1780,495,2110,665))
        for label,(x0,y0,x1,y1) in crops.items():
            sl=np.s_[y0:y1,x0:x1]
            for scale in (1,2):
                panels=[]
                for title,im in [('V4',v4),('V10',v10),('V12 direct RGB',result)]:
                    crop=cv2.resize(im[sl],None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
                    panel=cv2.copyMakeBorder(crop,26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30)); cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1); panels.append(panel)
                write_png(args.output/'comparisons'/f'{path.stem}_{label}_{scale}x.png',np.concatenate(panels))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        report['total_seconds']=time.perf_counter()-start; reports.append(report)
        (args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='profiles'}),flush=True)
        del src,prepared,v4,maps,base,result,debug,v10,overlay; gc.collect()


if __name__=='__main__': main()
