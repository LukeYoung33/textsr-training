"""P16: local polarity-adaptive V8 with shared ink-excluding background."""
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import dilate,smoothstep
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct


def prepare_adaptive(image):
    """@luna-comment Input:uint8 BGR. Output:common background,polarity cores/distances.
    Key Steps:paired median seed,dual cores,exclude both inks from normalized filters.
    Key Params:inset6,median31 at1/16,exclusion8,sigma22/48; no bright-ROI assumption.
    State/Dependencies:local float buffers,complement-equivariant preparation.
    """
    src=image.astype(np.float32); y=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY); h,w=y.shape
    roi=np.zeros((h,w),bool); roi[6:h-6,6:w-6]=True
    small=cv2.resize(y,(max(1,w//16),max(1,h//16)),interpolation=cv2.INTER_AREA)
    first=cv2.medianBlur(np.clip(np.rint(small),0,255).astype(np.uint8),31).astype(np.float32)
    second=cv2.medianBlur(np.clip(np.rint(255-small),0,255).astype(np.uint8),31).astype(np.float32)
    seed=cv2.resize(cv2.GaussianBlur(.5*(first+255-second),(0,0),2),(w,h),interpolation=cv2.INTER_LINEAR)
    dark=roi&(seed-y>np.maximum(24,.18*seed)); light=roi&(y-seed>np.maximum(24,.18*(255-seed)))
    provisional_dark=dark.copy(); provisional_light=light.copy()
    amplitude=np.abs(y-seed); kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    peak_dark=cv2.dilate(np.where(dark,amplitude,0),kernel)
    peak_light=cv2.dilate(np.where(light,amplitude,0),kernel)
    removed_dark=dark&(amplitude<.60*peak_light); removed_light=light&(amplitude<.60*peak_dark)
    dark&=~removed_dark; light&=~removed_light
    union=dark|light; weight=(roi&~dilate(union,8)).astype(np.float32)
    den=cv2.GaussianBlur(weight,(0,0),22); den_large=cv2.GaussianBlur(weight,(0,0),48)
    near=cv2.GaussianBlur(src*weight[...,None],(0,0),22)/np.maximum(den[...,None],1e-6)
    far=cv2.GaussianBlur(src*weight[...,None],(0,0),48)/np.maximum(den_large[...,None],1e-6)
    blend=smoothstep((den-.02)/.10)[...,None]; bg=near*blend+far*(1-blend)
    support=smoothstep((den_large-.02)/.06)
    distances=[distance_transform_edt(~mask).astype(np.float32) if mask.any() else np.full(y.shape,1e6,np.float32) for mask in (dark,light)]
    seeds=dict(provisional_dark=provisional_dark,provisional_light=provisional_light,removed_dark=removed_dark,removed_light=removed_light)
    return src,y,roi,bg,support,dark,light,distances,seeds


def feather_branch(base,src,y,roi,bg,own_core,protect,alpha,owner):
    """@luna-comment Input:normalized dark-ink branch. Output:float V8 exterior extension.
    Key Steps:own patch,clearance to both protected inks,owned outward correction.
    Key Params:unchanged V8 collar16,guard6..10,collision3..6 and tanh limiter.
    State/Dependencies:does not seed patch from opposite cores; no quantization.
    """
    clear=distance_transform_edt(~protect).astype(np.float32)
    seed=(roi&(own_core|(alpha>=.5))).astype(np.uint8)
    patch=binary_fill_holes(cv2.morphologyEx(seed,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))))&roi
    if not patch.any(): return base,np.zeros_like(y)
    rho,indices=distance_transform_edt(~patch,return_indices=True); rho=rho.astype(np.float32)
    full=src-base; boundary=full[indices[0],indices[1]]
    q=roi*(~patch)*(1-smoothstep(rho/16))*smoothstep((clear-6)/4)*smoothstep((clear-rho-3)/3)*owner
    change=-q[...,None]*(boundary-full); dy=cv2.cvtColor(change,cv2.COLOR_BGR2GRAY)
    by=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY); y4=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    allowed=np.where(dy>=0,np.maximum(.5,by+2-y4),np.maximum(.5,y4-by+2))
    z=dy/allowed; attenuation=np.ones_like(z)
    np.divide(np.tanh(z),z,out=attenuation,where=np.abs(z)>=1e-6); change*=attenuation[...,None]
    assert not np.any(change[(clear<=6)|patch|~roi])
    return base+change,q


def adaptive_correct(image):
    """@luna-comment Input:uint8 BGR scene. Output:uint8 result,debug,contract report.
    Key Steps:shared background,two complemented branches,disjoint ownership,float sum.
    Key Params:normal V4 then V8; both cores protected; no intermediate rounding.
    State/Dependencies:original unchanged; uniform/no-core branches explicitly inactive.
    """
    src,y,roi,bg,support,dark,light,distances,seeds=prepare_adaptive(image); union=dark|light
    out=src.copy(); debug=dict(dark_core=dark,light_core=light,background=np.clip(np.rint(bg),0,255).astype(np.uint8),**seeds)
    reports={}; written=np.zeros(y.shape,bool)
    for index,(name,core) in enumerate((('dark',dark),('light',light))):
        if not core.any():
            reports[name]=dict(core_pixels=0,changed=0); continue
        own=distances[index]; other=distances[1-index]
        owner=smoothstep((other-own)/4)*roi
        branch_src=src if index==0 else 255-src; branch_y=y if index==0 else 255-y; branch_bg=bg if index==0 else 255-bg
        branch_roi=roi&(owner>0); weak=branch_roi&(cv2.cvtColor(branch_bg,cv2.COLOR_BGR2GRAY)-branch_y>12)
        protect=union|weak
        prepared=(branch_src,branch_y,branch_roi,protect,branch_bg,support*owner,own)
        _,maps,_=normal_correct(prepared)
        alpha=maps['correction_weight']; base=branch_src+alpha[...,None]*(branch_bg-branch_src)
        final,q=feather_branch(base,branch_src,branch_y,branch_roi,branch_bg,core,protect,alpha,owner)
        delta=final-branch_src
        if index: delta=-delta
        delta[union|~roi|(support==0)]=0
        changed=np.any(delta!=0,axis=2)
        assert not np.any(written&changed),'Polarity branches overlap'
        written|=changed; out+=delta
        reports[name]=dict(core_pixels=int(core.sum()),changed=int(changed.sum()),normal_active=int((alpha>0).sum()),outward_active=int((q>0).sum()))
        debug[name+'_owner']=owner; debug[name+'_alpha']=alpha; debug[name+'_outward']=q
    result=np.clip(np.rint(out),0,255).astype(np.uint8)
    assert np.array_equal(result[union|~roi],image[union|~roi])
    reports.update(protected_changed=0,overlap=0,clipping=int(np.count_nonzero((out<0)|(out>255))))
    debug['signed_change']=np.clip(np.rint(128+(out-src)*4),0,255).astype(np.uint8)
    return result,debug,reports
