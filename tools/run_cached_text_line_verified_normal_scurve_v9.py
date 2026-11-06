"""Run V9 verified-normal cached-line S-curve experiment."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import run_cached_text_line_bimodal_scurve as base

MODE="cached_maskraw_verified_normal_scurve_outward_feather_v9"
_p10_segment_text=base._p10_segment_text
K=np.ones((3,3),np.uint8)

def _sample(y:np.ndarray,x:np.ndarray,yy:np.ndarray)->np.ndarray:
    """@luna-comment Bilinearly sample normalized luminance at normal-profile coordinates.

    Input: One finite luminance plane and equally shaped float x/y coordinate arrays.
    Output: Float samples; coordinates outside the crop are NaN and rejected by callers.
    Key Steps: Use OpenCV linear remapping with an explicit NaN border value.
    Key Params: Interpolation is fixed to linear for the frozen V9 profile definition.
    State/Dependencies: Read-only NumPy/OpenCV operation; no cache geometry is modified.
    """
    return cv2.remap(y.astype(np.float32),x.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))

def _p10(prediction:np.ndarray,valid:np.ndarray)->tuple[np.ndarray,dict[str,Any]]|None:
    """@luna-comment Produce and validate the one frozen raw P10 mask for a cached line.

    Input: Cached prediction YUV and valid support.
    Output: Bit-preserved raw mask and P10 record, or None for an allowed identity skip.
    Key Steps: Invoke P10 once, validate its low/high partition and known text polarity.
    Key Params: Existing frozen P10 parameter record is obtained from the V6 helper.
    State/Dependencies: Cache-only P10 segmentation; no OCR or model invocation.
    """
    raw,record=_p10_segment_text(base._rgb_to_luma(base._to_rgb(prediction)),base._params(),valid_mask=valid)
    if raw is None or not record.get('segmentation_accepted',False): return None
    raw=np.asarray(raw,bool); low=np.asarray(record['_p10_low_class'],bool); high=np.asarray(record['_p10_high_class'],bool); domain=np.asarray(record['_p10_valid'],bool)
    polarity=str(record.get('polarity',''))
    if not (np.array_equal(domain,valid) and np.array_equal(low|high,valid) and not (low&high).any() and not (raw&~valid).any() and raw.any() and (valid&~raw).any() and (polarity.startswith('dark_text_on_light') or polarity.startswith('light_text_on_dark'))): return None
    return raw,record

def fill_v9(prediction:np.ndarray,frame_crop:np.ndarray,valid:np.ndarray)->tuple[np.ndarray,dict[str,np.ndarray],dict[str,Any]]:
    """@luna-comment Apply V9 only on verified monotonic normal profiles then feather outward.

    Input: Cached prediction/frame crops and matching valid support.
    Output: Reconstructed crop, raw-mask-derived debug planes, and profile record.
    Key Steps: Locate maximum-gradient centres near raw boundaries, gate local profiles,
      map five transition samples through an endpoint-anchored S curve, and feather only
      raw-mask exterior rings seven through twenty toward the frame crop.
    Key Params: V9 fixes search radius 2, profile radius 4, gap .08-.55, exponent 1.6,
      and the raised-cosine exterior ring schedule.
    State/Dependencies: Uses prediction, same-coordinate frame, valid support, and one P10 call;
      U/V and all protected pixels stay prediction-exact.
    """
    if prediction.shape!=frame_crop.shape or prediction.ndim!=3 or valid.shape!=prediction.shape[1:]: raise ValueError('V9 crop/support shape mismatch')
    empty=np.zeros_like(valid,bool); evidence={'valid':valid,'raw':empty,'boundary':empty,'accepted':empty,'edge':empty,'protection':empty,'rejected':empty,'weight':np.zeros_like(prediction[0],np.float32)}
    report:dict[str,Any]={'status':'identity_skip','valid_pixels':int(valid.sum()),'accepted_profiles':0}
    segment=_p10(prediction,valid)
    if segment is None: report['identity_skip_reason']='p10_reject_or_raw_mask_contract_failure'; return frame_crop.copy(),evidence,report
    raw,record=segment; evidence['raw']=raw
    sign=1.0 if str(record['polarity']).startswith('light_text_on_dark') else -1.0
    # Sobel increases toward M; reverse it so r>0 points from text to background.
    gx=cv2.Sobel(raw.astype(np.float32),cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(raw.astype(np.float32),cv2.CV_32F,0,1,ksize=3); norm=np.hypot(gx,gy)
    boundary=raw & (cv2.dilate(raw.astype(np.uint8),K).astype(bool)!=cv2.erode(raw.astype(np.uint8),K).astype(bool)); boundary &= norm>1e-6
    evidence['boundary']=boundary
    sy,sx=np.nonzero(boundary)
    if not len(sx): report['identity_skip_reason']='no_raw_boundary'; return frame_crop.copy(),evidence,report
    nx=-gx[sy,sx]/norm[sy,sx]; ny=-gy[sy,sx]/norm[sy,sx]; y=prediction[0]
    offsets=np.arange(-2,3,dtype=np.float32); xx=sx[:,None]+nx[:,None]*offsets; yy=sy[:,None]+ny[:,None]*offsets
    left=_sample(y,xx-.5*nx[:,None],yy-.5*ny[:,None]); right=_sample(y,xx+.5*nx[:,None],yy+.5*ny[:,None]); grad=sign*(left-right)
    kindex=np.nanargmax(np.where(np.isfinite(grad),grad,-np.inf),axis=1); g=grad[np.arange(len(sx)),kindex]; k0=offsets[kindex]
    rr=np.arange(-4,5,dtype=np.float32); px=sx[:,None]+nx[:,None]*(k0[:,None]+rr); py=sy[:,None]+ny[:,None]*(k0[:,None]+rr); prof=_sample(y,px,py)
    f=np.nanmedian(prof[:,:2],axis=1); b=np.nanmedian(prof[:,-2:],axis=1); gap=np.abs(f-b)
    oriented=sign*(prof[:,:-1]-prof[:,1:]); reverse=(oriented<0).sum(1); reverse_size=np.where(oriented<0,-oriented,0).max(1)
    endpoints=np.maximum(np.abs(prof[:,2]-f),np.abs(prof[:,6]-b))
    ok=np.isfinite(prof).all(1)&np.isfinite(g)&(g>=.03)&(gap>=.08)&(gap<=.55)&(reverse<=1)&(reverse_size<=.015)&(endpoints<=.04)&((sign*(f-b))>0)
    rejected=boundary.copy(); rejected[sy[ok],sx[ok]]=False; evidence['rejected']=rejected
    p9=prediction.copy(); owner=np.full(valid.shape,-np.inf,np.float32); owner_seed=np.full(valid.shape,np.iinfo(np.int32).max,np.int32); edge=empty.copy(); protect=empty.copy(); accepted=empty.copy()
    for i in np.flatnonzero(ok):
        accepted[sy[i],sx[i]]=True
        for r in (-4.,-3.):
            ix=int(round(sx[i]+nx[i]*(k0[i]+r))); iy=int(round(sy[i]+ny[i]*(k0[i]+r)))
            if 0<=ix<valid.shape[1] and 0<=iy<valid.shape[0] and valid[iy,ix]: protect[iy,ix]=True
        for r in (-2.,-1.,0.,1.,2.):
            ix=int(round(sx[i]+nx[i]*(k0[i]+r))); iy=int(round(sy[i]+ny[i]*(k0[i]+r)))
            if not (0<=ix<valid.shape[1] and 0<=iy<valid.shape[0] and valid[iy,ix]): continue
            tie=i<owner_seed[iy,ix]
            if g[i]>owner[iy,ix] or (g[i]==owner[iy,ix] and tie):
                old=float(_sample(y,np.asarray([[sx[i]+nx[i]*(k0[i]+r)]],np.float32),np.asarray([[sy[i]+ny[i]*(k0[i]+r)]],np.float32))[0,0])
                t=np.clip(sign*(old-b[i])/gap[i],0,1); a=t**1.6; z=(1-t)**1.6; p9[0,iy,ix]=np.clip(b[i]+sign*gap[i]*a/max(a+z,1e-12),0,1); owner[iy,ix]=g[i]; owner_seed[iy,ix]=i; edge[iy,ix]=True
    rings=base._rings(raw,valid,20); core=raw.copy()
    for ring in rings[:6]: core|=ring
    output=frame_crop.copy(); output[:,core]=p9[:,core]
    weights=evidence['weight']
    for j,ring in enumerate(rings[6:],7):
        w=.5-.5*np.cos(np.pi*(j-6)/14.0); output[:,ring]=(1-w)*p9[:,ring]+w*frame_crop[:,ring]; weights[ring]=w
    output[1:,valid]=prediction[1:,valid]
    evidence.update({'accepted':accepted,'edge':edge,'protection':protect,'core':core})
    report.update({'status':'processed','polarity':str(record['polarity']),'raw_mask_pixels':int(raw.sum()),'boundary_seeds':int(boundary.sum()),'accepted_profiles':int(ok.sum()),'rejected_profiles':int((~ok).sum()),'edge_R_pixels':int(edge.sum()),'foreground_protection_pixels':int(protect.sum()),'foreground_feather_pixels':int((weights[raw]>0).sum()),'safety_ring_feather_pixels':int((weights[np.logical_or.reduce(rings[:6])]>0).sum()),'outer_feather_pixels':int((weights>0).sum()),'ring_weights':{str(j):(.5-.5*np.cos(np.pi*(j-6)/14.0)) for j in range(7,21)},'uv_preserved':bool(np.array_equal(output[1:],prediction[1:])),'profiles':[{'x':int(sx[i]),'y':int(sy[i]),'k0':float(k0[i]),'g':float(g[i]),'F':float(f[i]),'B':float(b[i]),'D':float(gap[i]),'endpoint_error':float(endpoints[i])} for i in np.flatnonzero(ok)[:16]]})
    return output,evidence,report

def _write_debug(d:Path,p:np.ndarray,c:np.ndarray,q:np.ndarray,e:dict[str,np.ndarray],report:dict[str,Any])->None:
    """@luna-comment Emit V9 review planes and profile diagnostics for one line.

    Input: Debug directory, prediction/frame/output crops, evidence planes, and profile record.
    Output: Required PNG/JSON evidence files.
    Key Steps: Render raw geometry, exterior rings, signed differences, and compact profile data.
    Key Params: Ring images are fixed to the V9 7--20 exterior schedule.
    State/Dependencies: Writes only the supplied experiment debug directory.
    """
    for name,key in [('mask_raw.png','raw'),('raw_boundary_seeds.png','boundary'),('accepted_edge_centers.png','accepted'),('edge_R.png','edge'),('foreground_protection.png','protection'),('profile_rejected.png','rejected')]: base._png(d/name,np.where(e[key],255,0).astype(np.uint8))
    rings=base._rings(e['raw'],e['valid'],20)
    for j in range(7,21): base._png(d/f'outward_feather_r{j:02d}.png',np.where(rings[j-1],255,0).astype(np.uint8))
    base._png(d/'outward_feather_weight.png',np.rint(e['weight']*255).astype(np.uint8)); base._rgb_png(d/'prediction_before.png',p); base._rgb_png(d/'cache_frame_crop.png',c); base._rgb_png(d/'reconstructed_line.png',q)
    for name,delta in [('prediction_minus_reconstruction_y_signed.png',p[0]-q[0]),('prediction_minus_frame_y_signed.png',p[0]-c[0]),('reconstruction_minus_frame_y_signed.png',q[0]-c[0])]: base._png(d/name,base._delta_image(delta))
    base._histogram_debug(d/'histogram_before_after.png',p[0,e['valid']],q[0,e['valid']],report); base._png(d/'normal_profiles.png',np.where(e['edge'],255,0).astype(np.uint8)); base._png(d/'normal_gradient_before_after.png',base._delta_image(q[0]-p[0])); (d/'normal_profiles.json').write_text(json.dumps(report.get('profiles',[]),ensure_ascii=False,indent=2))

def run(manifest_path:Path,cache_root:Path,output_root:Path)->None:
    """@luna-comment Execute V9 in ascending cached-line order with first-write ownership.

    Input: Frozen V9 manifest, verified line cache, and a new output directory.
    Output: Full frames, per-line V9 debug, routes, and cache-only receipts.
    Key Steps: Verify cache integrity, call fill_v9 once per line, then write valid support once.
    Key Params: Manifest binds the V9 mode and all numerical constants live in fill_v9.
    State/Dependencies: Immutable cache arrays only; no source image, OCR, or model is read.
    """
    exp=json.loads(manifest_path.read_text());
    if exp.get('authorized_postprocess')!=MODE: raise ValueError('manifest mode mismatch')
    cache=base.verify_cache(cache_root); output_root.mkdir(parents=True,exist_ok=True); summary={'experiment_id':exp['experiment_id'],'mode':MODE,'images':[]}
    for frame in cache['frames']:
        cf=np.load(cache_root/frame['directory']/'frame_yuv444.npy'); result=cf.copy(); taken=np.zeros(cf.shape[1:],bool); routes=[]
        for line in sorted(frame['lines'],key=lambda x:x['line_index']):
            d=cache_root/line['directory']; p=np.load(d/'prediction_yuv444.npy'); valid=np.load(d/'valid_support.npy').astype(bool); l,t,r,b=json.loads((d/'geometry.json').read_text())['crop_xyxy']; q,e,report=fill_v9(p,cf[:,t:b,l:r],valid); prior=taken[t:b,l:r].copy(); write=valid&~prior; result[:,t:b,l:r][:,write]=q[:,write]; taken[t:b,l:r][write]=True; dbg=output_root/'debug'/frame['image_id']/f"line_{line['line_index']:04d}"; _write_debug(dbg,p,cf[:,t:b,l:r],q,e,report); report.update({'line_index':line['line_index'],'write_pixels':int(write.sum()),'overlap_conflict_pixels':int((valid&prior).sum()),'mask_raw_bit_identical':True}); (dbg/'line_telemetry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); routes.append(report)
        base._rgb_png(output_root/f"{frame['image_id']}.png",result); (output_root/f"{frame['image_id']}.routes.json").write_text(json.dumps(routes,ensure_ascii=False,indent=2)); summary['images'].append({'image_id':frame['image_id'],'lines':len(routes)})
    summary['manifest_sha256']=base._sha256(manifest_path); (output_root/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); (output_root/'run_receipt.json').write_text(json.dumps({'manifest_sha256':summary['manifest_sha256'],'mode':MODE},indent=2)); (output_root/'command.log').write_text('offline V9 cached postprocess; OCR/model not executed\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--cache-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); run(a.manifest,a.cache_root,a.output_dir)
