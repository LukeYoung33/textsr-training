"""P9: repair full V4 RGB correction transitions; two samples only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.postprocess_ppt_dehalo_20260924 import prepare, smoothstep, write_png
from tools.postprocess_ppt_normal_dehalo_v4 import normal_correct
from tools.postprocess_ppt_contour_v6 import read_image


def repair_patch(base, prepared):
    """@luna-comment Input: unrounded V4 and original buffers. Output: P9/debug.
    Key Steps: smooth full RGB correction with zero outer anchors, bound and blend.
    Key Params: sigma4, bound4, inner6/full14/fade28/outer40; luma allowance2.
    State/Dependencies: pure OpenCV/NumPy; neither source nor V4 modified.
    """
    src,y,roi,core,bg,_,d=prepared
    by=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
    likeness=smoothstep((y-by+12)/8)
    weight=(roi & ~core).astype(np.float32)*likeness
    den=cv2.GaussianBlur(weight,(0,0),4)
    full=src-base
    averaged=cv2.GaussianBlur(full*weight[...,None],(0,0),4)/np.maximum(den[...,None],1e-6)
    blend=smoothstep((d-6)/8)*(1-smoothstep((d-28)/12))
    blend*=roi*(~core)*smoothstep((den-.1)/.4)*likeness
    change=-blend[...,None]*4*np.tanh((averaged-full)/4)
    y4=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    dy=cv2.cvtColor(change,cv2.COLOR_BGR2GRAY)
    allowance=np.where(dy>=0,np.maximum(.5,by+2-y4),np.maximum(.5,y4-by+2))
    z=dy/allowance
    limiter=np.ones_like(z)
    np.divide(np.tanh(z),z,out=limiter,where=np.abs(z)>=1e-6)
    change*=limiter[...,None]
    floating=base+change
    result=np.clip(np.rint(floating),0,255).astype(np.uint8)
    protected=core | ~roi | (d<=6) | (d>=40) | (y-by<=-12)
    assert np.array_equal(result[protected],np.clip(np.rint(base),0,255).astype(np.uint8)[protected])
    assert np.abs(change).max()<=4+1e-5
    report=dict(max_change_float=float(np.abs(change).max()),protected_changed=0,
                clipping_count=int(np.count_nonzero((floating<0)|(floating>255))),
                limited_pixels=int(np.count_nonzero(np.abs(z)>=1e-6)),
                zero_anchor_pixels=int(np.count_nonzero((weight>0)&(d>=40))))
    return result, full, src-floating, change, report


def main():
    """@luna-comment Input: source and exclusive output CLI paths. Output: sample evidence.
    Key Steps: reproduce float V4; P9; compare full fields and boundary strips.
    Key Params: fixed two IDs, common debug gains, no batch/model.
    State/Dependencies: read-only baseline files; writes only new output directory.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(4)
    reports=[]
    crops=dict(title=(950,450,2750,760),practice=(950,740,2150,990),
               degrees=(1750,1260,2950,1480),color=(600,1690,2700,1890))
    for ident in ('044920599','052927799'):
        path=next(args.input.glob(f'*{ident}_preprocessInput.png'))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        src=read_image(path); prepared=prepare(src,robust_seed=True)
        v4,maps,_=normal_correct(prepared)
        base=prepared[0]+maps['correction_weight'][...,None]*(prepared[4]-prepared[0])
        name=path.stem+'_dehalo.png'; root=Path('results/ppt_dehalo_20260924')
        assert np.array_equal(v4,read_image(root/'sample_v4_normal'/name))
        result,d4,d7,change,report=repair_patch(base,prepared)
        v6=read_image(root/'sample_v6_field_blend'/name)
        write_png(args.output/name,result)
        assert np.array_equal(result,read_image(args.output/name))
        report.update(image=name,sigma=4,inner=6,outer=40,bound=4,metrics={})
        debugdir=args.output/'debug'/path.stem
        for label,field in [('full_D4',d4),('full_D7',d7),('adjustment',change)]:
            write_png(debugdir/(label+'_rgb.png'),np.clip(np.rint(128+field*4),0,255).astype(np.uint8))
            write_png(debugdir/(label+'_magnitude.png'),np.clip(np.rint(np.max(np.abs(field),axis=2)*8),0,255).astype(np.uint8))
        write_png(debugdir/'distance.png',np.clip(np.rint(prepared[6]*5),0,255).astype(np.uint8))
        for label,(x0,y0,x1,y1) in crops.items():
            panels=[]
            for caption,im in [('input',src),('V4',v4),('V6',v6),('V7 full RGB transition',result)]:
                panel=cv2.copyMakeBorder(im[y0:y1,x0:x1],26,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
                cv2.putText(panel,caption,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                panels.append(panel)
            write_png(args.output/'comparisons'/(path.stem+'_'+label+'.png'),np.concatenate(panels))
            # Same-domain full-field gradient diagnostics, not incremental deltas.
            metrics=[]; gradients=[]
            for field in (d4,d7):
                gray=cv2.cvtColor(field[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
                grad=np.hypot(*np.gradient(gray))
                metrics.append(float(grad.mean()))
                gradients.append(np.clip(np.rint(grad*32),0,255).astype(np.uint8))
            report['metrics'][label]=metrics
            write_png(debugdir/(label+'_full_gradient_D4_D7.png'),np.concatenate(gradients))
        # Horizontal strip across the gap between the large title characters.
        coords=dict(y=620,x0=1550,x1=2100)
        profile={**coords,'distance':prepared[6][620,1550:2100].tolist()}
        for label,im in [('input',src),('V4',v4),('V7',result)]:
            profile[label]=cv2.cvtColor(im.astype(np.float32),cv2.COLOR_BGR2GRAY)[620,1550:2100].tolist()
        (debugdir/'title_profile.json').write_text(json.dumps(profile),encoding='utf-8')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        reports.append(report)
        (args.output/'receipt.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
