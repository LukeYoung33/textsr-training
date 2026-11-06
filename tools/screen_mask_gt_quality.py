"""Native text-conditioned registration and recoverable-detail measurements."""
from pathlib import Path
import sys,json,argparse,time
import cv2
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_r1_width_bank import save,digest
SOURCE=Path('/AI_DNS/guang.yang/TextSR/data/vosr_s22_mask_repair_mix50_20261104')
OUT=SOURCE.parent/'mask_gt_filtered_trusted_20261105'
VERSION='native-text-local-registration-detail-v1'


def matching(image,mask,sl,sigmas):
    """@luna-comment Input:full highpass image/mask and native96px window. Output:peak correlation,zero-shift score,offset,template sigma. Key Steps:local multiscale template matching with8px search. Key Params:absolute polarity-invariant NCC;no image warp. State/Dependencies:OpenCV read-only float arrays."""
    yy,xx=sl;patch=image[yy.start-8:yy.stop+8,xx.start-8:xx.stop+8];m=mask[sl];best=None
    for sigma in sigmas:
        template=cv2.GaussianBlur(m,(0,0),sigma)-cv2.GaussianBlur(m,(0,0),8)
        if template.std()<1e-4:continue
        values=np.abs(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED));_,peak,_,location=cv2.minMaxLoc(values)
        record=dict(peak=float(peak),zero=float(values[8,8]),dx=int(location[0]-8),dy=int(location[1]-8),sigma=sigma)
        if best is None or peak>best['peak']:best=record
    return best or dict(peak=0.,zero=0.,dx=0,dy=0,sigma=0.)


def measure(row):
    """@luna-comment Input:source-bound original teacher/LQ/mask row. Output:native local detail/registration summaries and patch evidence. Key Steps:verify frozen hashes;spread text-supported windows;subtract background noise;multiscale match original mask. Key Params:96px windows,max64,both polarities;no OCR inference. State/Dependencies:immutable source images."""
    cv2.setNumThreads(1)
    for key in ['lq','gt','text_mask','valid_mask']:assert digest(row['files'][key])==row['output_sha256'][key]
    lq=cv2.imread(row['files']['lq']);gt=cv2.imread(row['files']['gt']);mask=cv2.imread(row['files']['text_mask'],0);valid=cv2.imread(row['files']['valid_mask'],0)
    assert lq is not None and gt is not None and mask is not None and valid is not None and lq.shape==gt.shape and lq.shape[:2]==mask.shape==valid.shape
    image=cv2.cvtColor(lq,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;teacher=cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;m=mask.astype(np.float32)/255;v=valid>=254
    lh=image-cv2.GaussianBlur(image,(0,0),8);gh=teacher-cv2.GaussianBlur(teacher,(0,0),8)
    fine=image-cv2.GaussianBlur(image,(0,0),1);mid=image-cv2.GaussianBlur(image,(0,0),4)
    region=cv2.dilate((m>.2).astype(np.uint8),np.ones((9,9),np.uint8))>0
    far=ndi.distance_transform_edt(m<=.2)>=8
    h,w=m.shape;windows=[]
    window=min(96,min(h,w)-16)
    if window>=40:
        ys=np.unique(np.r_[np.arange(8,h-window-7,72),h-window-8]);xs=np.unique(np.r_[np.arange(8,w-window-7,72),w-window-8])
        for y in ys:
            for x in xs:
                sl=np.s_[y:y+window,x:x+window]
                if v[sl].mean()>.90 and ((m[sl]>.5)&v[sl]).sum()>=24:windows.append((int(y),int(x)))
    if len(windows)>64:windows=[windows[i] for i in np.linspace(0,len(windows)-1,64,dtype=int)]
    patches=[]
    for y,x in windows:
        sl=np.s_[y:y+window,x:x+window];support=region[sl]&v[sl];background=far[sl]&v[sl]
        hf=float(np.sqrt(np.mean(fine[sl][support]**2)));mf=float(np.sqrt(np.mean(mid[sl][support]**2)))
        noise=float(np.sqrt(np.mean(fine[sl][background]**2))) if background.sum()>=32 else 0.
        coherent=float(np.sqrt(max(0,hf*hf-noise*noise)))*255
        geom=matching(gh,m,sl,[1.,2.5,4.]);focus=matching(lh,m,sl,[.6,1.2,2.4,4.8,7.])
        patches.append(dict(x=x,y=y,hf_dn=hf*255,coherent_hf_dn=coherent,mid_dn=mf*255,hf_ratio=hf/max(mf,1e-8),gt=geom,lq=focus,text_pixels=int(((m[sl]>.5)&v[sl]).sum())))
    def quantile(key,q):
        """@luna-comment Input:patch scalar key/quantile. Output:finite summary or0 when no valid text windows. Key Steps:numpy quantile. Key Params:no substitution of another scene. State/Dependencies:local measurements."""
        return float(np.quantile([p[key] for p in patches],q)) if patches else 0.
    bad=[p for p in patches if p['gt']['peak']>=.45 and np.hypot(p['gt']['dx'],p['gt']['dy'])>=1.75 and p['gt']['peak']-p['gt']['zero']>=.04]
    good=[p for p in patches if p['coherent_hf_dn']>=.6 and p['hf_ratio']>=.22 and p['lq']['peak']>=.30]
    return dict(version=VERSION,dataset=row['dataset'],sample_id=row['sample_id'],repair_mode=row['repair_mode'],shape_hw=[h,w],patch_count=len(patches),misaligned_fraction=len(bad)/max(len(patches),1),misaligned_text_fraction=sum(p['text_pixels'] for p in bad)/max(sum(p['text_pixels'] for p in patches),1),good_detail_fraction=len(good)/max(len(patches),1),hf_ratio_q50=quantile('hf_ratio',.5),hf_ratio_q75=quantile('hf_ratio',.75),coherent_hf_q75_dn=quantile('coherent_hf_dn',.75),mid_q75_dn=quantile('mid_dn',.75),gt_peak_q50=float(np.median([p['gt']['peak'] for p in patches])) if patches else 0.,patches=patches,source_hashes=row['output_sha256'])


def decide(record,thresholds):
    """@luna-comment Input:measured parent/frozen cutoffs. Output:retain/reject reasons and review flags. Key Steps:region-mask registration exemption;require sustained local mismatch;severe coherent-detail deficit;flag borderline evidence. Key Params:calibrated four user negatives,accepted07693 positive. State/Dependencies:pure metadata;no source deletion."""
    reasons=[];flags=[]
    if record['repair_mode']=='stroke_repair' and record['patch_count']>=4 and record['misaligned_fraction']>=thresholds['misaligned_patch_fraction'] and record['misaligned_text_fraction']>=thresholds['misaligned_text_fraction']:
        reasons.append('spatial_mask_image_misalignment')
    if record['patch_count'] and record['good_detail_fraction']<=thresholds['max_good_detail_fraction'] and (record['coherent_hf_q75_dn']<thresholds['minimum_coherent_hf_dn'] or record['mid_q75_dn']<thresholds['minimum_text_band_dn'] or record['hf_ratio_q75']<thresholds['minimum_hf_ratio']):
        reasons.append('insufficient_lq_text_detail')
    if not record['patch_count']:flags.append('insufficient_supported_text_windows')
    if not reasons and record['repair_mode']=='stroke_repair' and record['misaligned_fraction']>=.08:flags.append('borderline_registration_review')
    return dict(keep=not reasons,reasons=reasons,flags=flags)


def calibrate():
    """@luna-comment Input:four user exclusions,accepted07693 and representative sources. Output:actual calibration metrics. Key Steps:fixed references then deterministic stratified controls. Key Params:no threshold deployment before measurement. State/Dependencies:read-only data."""
    OUT.mkdir(exist_ok=True);rows=json.loads((SOURCE/'SOURCE_ROWS.json').read_text());selected=[7609,7612,7646,7648,7692]
    for dataset in sorted({r['dataset'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['dataset']==dataset];selected.extend(indices[k] for k in np.linspace(0,len(indices)-1,4,dtype=int))
    records=[]
    for i in dict.fromkeys(selected):
        r=measure(rows[i]);r['viewer_id']=f'{i+1:05d}';records.append(r);print(r['viewer_id'],r['dataset'],r['sample_id'],json.dumps({k:v for k,v in r.items() if k not in ['patches','source_hashes','dataset','sample_id','repair_mode','shape_hw','version']}),flush=True)
    save(OUT/'CALIBRATION.json',dict(passed=True,records=records))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--calibrate',action='store_true');args=parser.parse_args()
    if args.calibrate:calibrate()
